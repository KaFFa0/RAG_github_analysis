import json
import operator
import os
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Annotated, TypedDict, Optional
import requests

import torch
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from utils import (
    Criterion,
    RepoAnalysisConfig,
    infer_kind,
    is_ignored_path,
    _read_binary_safe_text,
    compact_text,
    _join_cell_source,
    _join_output_text,
    make_doc,
    dedupe_docs,
    compact_doc_for_file,
    parse_notebook,
    parse_text_file,
    _text_hash,
    extract_json_blocks,
    safe_int,
    safe_float,
    unwrap_payload,
    extract_score_from_payload,
    parse_json_object,
    _find_first_match,
    recover_loose_payload,
    normalize_payload,
)

# Configuration

TEXT_EXTENSIONS = {
    ".py", ".md", ".txt", ".rst", ".toml", ".yaml", ".yml", ".json",
    ".js",".jsx",".ts",".tsx",".go",".rs",".java",".kt",".sh",".bash",
    ".c",".cc",".cpp",".h",".hpp",".sql",".html",".css",".ini",".cfg",
    ".env", ".ipynb",
}

DEFAULT_IGNORED_DIRS = {
    ".git",".idea",".vscode","node_modules","dist","build",
    "target","__pycache__",".mypy_cache",
    ".pytest_cache",".ruff_cache",".venv","venv",
}

@dataclass
class CriterionResult:
    criterion_id: str
    criterion_description: str
    score: int
    answer: str
    evidence: list[dict[str, Any]]
    confidence: float


@dataclass
class RepoCorpus:
    root: Path
    file_docs: list[Document]
    chunk_docs: list[Document]
    manifest: dict[str, Any] = field(default_factory=dict)
    pinned_docs: list[Document] = field(default_factory=list)


@dataclass
class RepositoryIndex:
    repo_path: Path
    corpus: RepoCorpus
    file_vectorstore: Any
    file_retriever: BaseRetriever
    chunk_vectorstore: Any
    chunk_retriever: BaseRetriever

class APIChat:
    def __init__(self, api_base_url: str, api_key: str, model_name: str) -> None:
        self.api_base_url = api_base_url.rstrip('/')
        self.api_key = api_key
        self.model_name = model_name

    def generate(self, system: str, user: str, max_new_tokens: int = 384, temperature: float = 0.1) -> str:
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "stream": False,
        }

        url = f"{self.api_base_url}/chat/completions"
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            raise RuntimeError(f"API call failed: {e}") from e


class SentenceTransformerEmbeddings:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        vector = self.model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
        return vector.tolist()


# Helpers

def load_repository_corpus(repo_path: str, config: RepoAnalysisConfig) -> RepoCorpus:
    root = Path(repo_path).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Repository path does not exist: {root}")

    ignored = set(DEFAULT_IGNORED_DIRS if config.ignored_dirs is None else config.ignored_dirs)
    ignored.discard(root.name)

    file_docs: list[Document] = []
    chunk_docs: list[Document] = []
    pinned_docs: list[Document] = []
    manifest_kinds: Counter[str] = Counter()
    manifest_paths: list[str] = []

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if is_ignored_path(rel, ignored):
            continue
        if path.suffix.lower() not in TEXT_EXTENSIONS and path.name not in {"Dockerfile", "Makefile", "LICENSE", "README", "README.md"}:
            continue

        try:
            if path.suffix.lower() == ".ipynb":
                file_doc, docs = parse_notebook(path, str(rel), config)
            else:
                file_doc, docs = parse_text_file(path, str(rel), config)
        except Exception:
            continue

        file_docs.append(file_doc)
        chunk_docs.extend(docs)
        manifest_paths.append(str(rel))
        manifest_kinds[file_doc.metadata.get("kind", "unknown")] += 1

        p = str(rel).lower()
        if "readme" in p:
            pinned_docs.append(file_doc)
        elif "/docs/" in p or p.startswith("docs/") or p.endswith(".md"):
            pinned_docs.append(file_doc)

    if not file_docs:
        raise ValueError(
            f"No readable text files found under {root}. "
            f"Check whether the repo is empty, fully binary, or filtered by ignored_dirs."
        )

    pinned_docs = dedupe_docs(pinned_docs)[: max(1, config.top_k_pinned)]

    manifest = {
        "num_files": len(file_docs),
        "num_chunks": len(chunk_docs),
        "kinds": dict(manifest_kinds),
        "paths": manifest_paths,
    }
    return RepoCorpus(root=root, file_docs=file_docs, chunk_docs=chunk_docs, manifest=manifest, pinned_docs=pinned_docs)


# Retrieval setup

def make_collection_name(prefix: str, repo_path: Path, suffix: str) -> str:
    repo_key = re.sub(r"[^a-zA-Z0-9_-]+", "_", repo_path.as_posix())[-60:]
    return f"{prefix}_{suffix}_{repo_key}"[:63].rstrip("._-")

def build_ensemble_retriever(
    docs: list[Document],
    embeddings: SentenceTransformerEmbeddings,
    collection_name: str,
    persist_dir: str,
    top_k_vector: int,
    top_k_bm25: int,
) -> tuple[Any, BaseRetriever]:
    from langchain_chroma import Chroma

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_dir,
    )

    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": top_k_vector})
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = top_k_bm25

    from langchain_classic.retrievers import EnsembleRetriever

    retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.65, 0.35],
    )
    return vectorstore, retriever


@dataclass(frozen=True)
class RetrievalBundle:
    pinned_docs: list[Document]
    file_hits: list[Document]
    chunk_hits: list[Document]
    selected_docs: list[Document]
    file_rank: dict[str, int]


def build_repository_index(corpus: RepoCorpus, config: RepoAnalysisConfig) -> RepositoryIndex:
    embeddings = SentenceTransformerEmbeddings(config.embedding_model_name)
    persist_root = config.persist_directory or os.path.join(".rag_cache", "chroma")
    os.makedirs(persist_root, exist_ok=True)

    file_collection = make_collection_name(config.chroma_collection_prefix, corpus.root, "files")
    chunk_collection = make_collection_name(config.chroma_collection_prefix, corpus.root, "chunks")

    file_vectorstore, file_retriever = build_ensemble_retriever(
        corpus.file_docs,
        embeddings,
        file_collection,
        os.path.join(persist_root, file_collection),
        config.top_k_file_vector,
        config.top_k_file_bm25,
    )
    chunk_vectorstore, chunk_retriever = build_ensemble_retriever(
        corpus.chunk_docs,
        embeddings,
        chunk_collection,
        os.path.join(persist_root, chunk_collection),
        config.top_k_chunk_vector,
        config.top_k_chunk_bm25,
    )

    return RepositoryIndex(
        repo_path=corpus.root,
        corpus=corpus,
        file_vectorstore=file_vectorstore,
        file_retriever=file_retriever,
        chunk_vectorstore=chunk_vectorstore,
        chunk_retriever=chunk_retriever,
    )


# Query rewriting

def rewrite_criterion_query(criterion: Criterion) -> str:
    desc = re.sub(r"\s+", " ", criterion.description.strip())
    return (
        f"Repository analysis question: {desc}. "
        f"Search for explicit evidence in README, docs, notebooks, code, tests, configs, "
        f"training scripts, experiment logs, and evaluation outputs "
        f"Prefer sources that directly mention the answer, and keep supporting or contradicting evidence"
    )


# Prompting and parsing

SYSTEM_PROMPT = """Ты аналитик GitHub-репозиториев
Твоя задача — оценить репозиторий строго по одному критерию

Правила:
- Отвечай только валидным JSON
- Не выдумывай факты, которых нет в контексте
- Если данных недостаточно, явно скажи, чего не хватает
- В evidence добавляй только то, что реально видно в контексте
- В criterion_description просто перепиши изначальный критерий
- score: целое число от 0 до 10
- confidence: число от 0 до 1

Формат JSON:
{
  "criterion_id": "...",
  "criterion_description": "...",
  "score": 0,
  "answer": "...",
  "evidence": [
    {"path": "...", "chunk_index": 0, "quote": "...", "why": "..."}
  ],
  "confidence": ...
}
"""

def format_context(docs: list[Document], max_chars_per_doc: int = 1400) -> str:
    blocks: list[str] = []
    for i, doc in enumerate(docs, start=1):
        md = doc.metadata
        snippet = compact_text(doc.page_content, max_chars_per_doc)
        path = md.get("path", "unknown")
        chunk_index = md.get("chunk_index", md.get("cell_index", "?"))
        part = md.get("part", "?")
        blocks.append(f"[SOURCE {i}] path={path} part={part} chunk={chunk_index}\n{snippet}")
    return "\n\n---\n\n".join(blocks)


def doc_kind_weight(doc: Document) -> float:
    md = doc.metadata
    kind = str(md.get("kind", "text"))
    part = str(md.get("part", ""))
    path = str(md.get("path", "")).lower()

    score = 0.0
    if "readme" in path:
        score += 5.0
    if "/docs/" in path or path.startswith("docs/") or kind == "docs":
        score += 4.0
    if kind == "notebook":
        score += 1.0
        if part == "markdown":
            score += 2.5
        elif part == "code":
            score += 1.0
        elif part == "output":
            score -= 2.0
        elif part == "file_summary":
            score += 1.5
    elif kind == "code":
        score += 1.5
    elif kind == "config":
        score += 1.0
    elif kind == "license":
        score -= 3.0
    elif kind == "ops":
        score += 0.5
    return score

def path_rank_bonus(path: str, file_rank: dict[str, int]) -> float:
    if path not in file_rank:
        return 0.0
    rank = file_rank[path]
    return max(0.0, 4.0 - rank * 0.6)


def select_pinned_docs(corpus: RepoCorpus, limit: int) -> list[Document]:
    candidates: list[Document] = []
    candidates.extend(corpus.pinned_docs)
    if not candidates:
        for doc in corpus.file_docs:
            md = doc.metadata
            if md.get("kind") == "notebook" and md.get("part") == "file_summary":
                candidates.append(doc)
                break
    return dedupe_docs(candidates, max_per_path=1)[:limit]


def select_context_documents(query: str, index: RepositoryIndex, config: RepoAnalysisConfig) -> RetrievalBundle:
    file_hits = index.file_retriever.invoke(query)
    chunk_hits = index.chunk_retriever.invoke(query)

    file_hits = dedupe_docs(file_hits, max_per_path=1)
    chunk_hits = dedupe_docs(chunk_hits, max_per_path=config.max_docs_per_path)

    file_rank: dict[str, int] = {}
    for i, doc in enumerate(file_hits):
        file_rank.setdefault(str(doc.metadata.get("path", "")), i)

    pinned_docs = select_pinned_docs(index.corpus, config.top_k_pinned)

    candidates: list[Document] = []
    candidates.extend(pinned_docs)
    candidates.extend(file_hits)
    candidates.extend(chunk_hits)

    scored: list[tuple[float, Document]] = []
    for idx, doc in enumerate(candidates):
        path = str(doc.metadata.get("path", ""))
        score = 0.0
        score += path_rank_bonus(path, file_rank)
        score += doc_kind_weight(doc)
        score += max(0.0, 2.5 - 0.05 * idx)
        if "readme" in path.lower():
            score += 2.0
        if "/docs/" in path.lower() or path.lower().startswith("docs/"):
            score += 1.5
        scored.append((score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    ordered = dedupe_docs([doc for _, doc in scored], max_per_path=config.max_docs_per_path)

    final_docs: list[Document] = []
    seen = set()
    for doc in pinned_docs + ordered:
        sig = (
            doc.metadata.get("path"),
            doc.metadata.get("part"),
            doc.metadata.get("chunk_index"),
            doc.metadata.get("cell_index"),
            doc.metadata.get("output_index"),
        )
        if sig in seen:
            continue
        seen.add(sig)
        final_docs.append(doc)
        if len(final_docs) >= config.top_k_final_context:
            break

    return RetrievalBundle(
        pinned_docs=pinned_docs,
        file_hits=file_hits,
        chunk_hits=chunk_hits,
        selected_docs=final_docs,
        file_rank=file_rank,
    )

def clone_repo(repo_url: str, target_dir: str) -> str:
    target = Path(target_dir)
    if target.exists():
        shutil.rmtree(target)
    subprocess.run(["git", "clone", "--depth", "1", repo_url, str(target)], check=True)
    return str(target)


class AnalysisState(TypedDict, total=False):
    criteria: list[Criterion]
    current_criterion: dict[str, str]
    results: Annotated[list[dict[str, Any]], operator.add]


class RepoAnalyzer:
    def __init__(self, config: RepoAnalysisConfig) -> None:
        self.config = config
        self.repo_path = Path(config.repo_path).resolve()
        self.llm = APIChat(
            api_base_url=config.api_base_url,
            api_key=config.api_key,
            model_name=config.api_model_name
        )

        corpus = load_repository_corpus(str(self.repo_path), config)
        self.index = build_repository_index(corpus, config)
        self.graph = self._build_graph()

    def _safe_build_answer(self, criterion: Criterion, raw: str, docs: list[Document]) -> CriterionResult:
        answer = raw.strip() or "Модель не вернула текстовый ответ"
        evidence: list[dict[str, Any]] = []
        for d in docs[:2]:
            evidence.append(
                {
                    "path": d.metadata.get("path"),
                    "part": d.metadata.get("part"),
                    "chunk_index": d.metadata.get("chunk_index", d.metadata.get("cell_index")),
                    "quote": compact_text(d.page_content, 240),
                    "why": "Fallback evidence: source was retrieved but final JSON parsing failed",
                }
            )

        return CriterionResult(
            criterion_id=criterion.id,
            criterion_description=criterion.description,
            score=0,
            answer=answer,
            evidence=evidence,
            confidence=0.0,

        )

    def _parse_or_recover(self, raw: str, criterion: Criterion) -> dict[str, Any]:
        """Parse model output, with a recovery path for malformed/truncated JSON"""
        try:
            return normalize_payload(parse_json_object(raw), criterion, raw)
        except Exception:
            recovered = recover_loose_payload(raw)
            if recovered is None:
                raise
            return normalize_payload(recovered, criterion, raw)

    def _build_graph(self):
        def fan_out(state: AnalysisState):
            return [Send("analyze_criterion", {"current_criterion": asdict(criterion)}) for criterion in state["criteria"]]

        def analyze_criterion(state: AnalysisState) -> dict[str, Any]:
            crit_raw = state["current_criterion"]
            criterion = Criterion(**crit_raw)
            query = rewrite_criterion_query(criterion)
            bundle = select_context_documents(query, self.index, self.config)
            docs = bundle.selected_docs

            context = format_context(docs)
            user_prompt = f"""Критерий:
ID: {criterion.id}
Описание: {criterion.description}

Запрос для поиска:
{query}

Контекст из репозитория:
{context}

Сделай оценку по этому критерию. Верни только JSON по заданной схеме
"""

            raw = self.llm.generate(
                system=SYSTEM_PROMPT,
                user=user_prompt,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
            )

            payload: dict[str, Any] | None = None
            try:
                payload = self._parse_or_recover(raw, criterion)
            except Exception:
                retry_prompt = user_prompt + "\n\nВажно: верни только один JSON-объект без markdown fences, без пояснений и без обертки по ключу критерия"
                raw_retry = self.llm.generate(
                    system=SYSTEM_PROMPT,
                    user=retry_prompt,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=0.0,
                )
                raw = raw_retry or raw
                try:
                    payload = self._parse_or_recover(raw_retry, criterion)
                except Exception:
                    return {"results": [self._safe_build_answer(criterion, raw, docs).__dict__]}

            if not isinstance(payload, dict):
                return {"results": [self._safe_build_answer(criterion, raw, docs).__dict__]}

            normalized = CriterionResult(
                criterion_id=str(payload.get("criterion_id", criterion.id)),
                criterion_description=str(payload.get("criterion_description", criterion.description)),
                score=safe_int(payload.get("score", 0), 0),
                answer=str(payload.get("answer", "")) or "Нет текстового ответа.",
                evidence=list(payload.get("evidence", [])) if isinstance(payload.get("evidence", []), list) else [],
                confidence=safe_float(payload.get("confidence", 0.0), 0.0),
            )
            return {"results": [normalized.__dict__]}

        graph = StateGraph(AnalysisState)
        graph.add_node("analyze_criterion", analyze_criterion)
        graph.add_conditional_edges(START, fan_out)
        graph.add_edge("analyze_criterion", END)
        return graph.compile()

    def run(self, criteria: list[dict[str, str]]) -> list[dict[str, Any]]:
        state: AnalysisState = {
            "criteria": [Criterion(**c) for c in criteria],
            "results": [],
        }
        result_state = self.graph.invoke(state)
        return result_state.get("results", [])


def build_markdown_report(results: list[dict[str, Any]]) -> str:
    lines = ["# Submit analysis report", ""]
    for item in results:
        lines.append(f"## {item.get('criterion_id', '')}: {item.get('criterion_description', '')}")
        lines.append(f"**Score:** {item.get('score', 0)}/10")
        lines.append(f"**Confidence:** {item.get('confidence', 0.0):.2f}")
        lines.append("")
        lines.append(item.get("answer", ""))
        lines.append("")
        evidence = item.get("evidence", [])
        if evidence:
            lines.append("**Evidence:**")
            for ev in evidence:
                lines.append(
                    f"- `{ev.get('path', '?')}` part={ev.get('part', '?')} chunk={ev.get('chunk_index', ev.get('cell_index', '?'))} — {ev.get('quote', '')}"
                )
        lines.append("")
    return "\n".join(lines).strip()

def load_submission_corpus(text: str, config: RepoAnalysisConfig, title: str = "submission") -> RepoCorpus:
    content_hash = _text_hash(text)
    root = Path(f".submission_cache/{title}_{content_hash}")
    safe_title = re.sub(r"[^a-zA-Z0-9_\-]+", "_", title)[:60]

    file_doc = make_doc(
        compact_doc_for_file(text, config.max_file_doc_chars),
        {
            "path": safe_title,
            "name": safe_title,
            "suffix": ".txt",
            "size": len(text.encode("utf-8")),
            "kind": "submission",
            "part": "file_summary",
        },
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_text(text)
    chunk_docs = []
    for idx, chunk in enumerate(chunks):
        chunk_docs.append(make_doc(
            chunk,
            {
                "path": safe_title,
                "name": safe_title,
                "suffix": ".txt",
                "size": len(text.encode("utf-8")),
                "kind": "submission",
                "part": "chunk",
                "chunk_index": idx,
                "chunk_total_hint": len(chunks),
            },
        ))

    pinned_docs = [file_doc]
    manifest = {
        "num_files": 1,
        "num_chunks": len(chunks),
        "kinds": {"submission": 1},
        "paths": [safe_title],
    }

    return RepoCorpus(
        root=root,
        file_docs=[file_doc],
        chunk_docs=chunk_docs,
        manifest=manifest,
        pinned_docs=pinned_docs,
    )

class SubmissionAnalyzer:
    """
    Analyzer for a single text submission.
    Uses the same retrieval and LLM logic as RepoAnalyzer.
    """
    def __init__(self, config: RepoAnalysisConfig, text_content: str, title: str = "Submission") -> None:
        self.config = config
        self.llm = APIChat(
            api_base_url=config.api_base_url,
            api_key=config.api_key,
            model_name=config.api_model_name
        )

        text = text_content.strip()
        if not text:
            raise ValueError("text_content must not be empty")

        corpus = load_submission_corpus(text, config, title=title)
        self.index = build_repository_index(corpus, config)

        self.graph = self._build_graph()

    def _build_graph(self):
        def fan_out(state: AnalysisState):
            return [Send("analyze_criterion", {"current_criterion": asdict(criterion)})
                    for criterion in state["criteria"]]

        def analyze_criterion(state: AnalysisState) -> dict[str, Any]:
            crit_raw = state["current_criterion"]
            criterion = Criterion(**crit_raw)
            query = rewrite_criterion_query(criterion)
            bundle = select_context_documents(query, self.index, self.config)
            docs = bundle.selected_docs

            context = format_context(docs)
            user_prompt = f"""Критерий:
ID: {criterion.id}
Описание: {criterion.description}

Запрос для поиска:
{query}

Контекст из репозитория:
{context}

Сделай оценку по этому критерию. Верни только JSON по заданной схеме
"""
            raw = self.llm.generate(
                system=SYSTEM_PROMPT,
                user=user_prompt,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
            )

            payload: dict[str, Any] | None = None
            try:
                payload = self._parse_or_recover(raw, criterion)
            except Exception:
                retry_prompt = user_prompt + "\n\nВажно: верни только один JSON-объект без markdown fences, без пояснений и без обертки по ключу критерия"
                raw_retry = self.llm.generate(
                    system=SYSTEM_PROMPT,
                    user=retry_prompt,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=0.0,
                )
                raw = raw_retry or raw
                try:
                    payload = self._parse_or_recover(raw_retry, criterion)
                except Exception:
                    return {"results": [self._safe_build_answer(criterion, raw, docs).__dict__]}

            if not isinstance(payload, dict):
                return {"results": [self._safe_build_answer(criterion, raw, docs).__dict__]}

            normalized = CriterionResult(
                criterion_id=str(payload.get("criterion_id", criterion.id)),
                criterion_description=str(payload.get("criterion_description", criterion.description)),
                score=safe_int(payload.get("score", 0), 0),
                answer=str(payload.get("answer", "")) or "Нет текстового ответа.",
                evidence=list(payload.get("evidence", [])) if isinstance(payload.get("evidence", []), list) else [],
                confidence=safe_float(payload.get("confidence", 0.0), 0.0),
            )
            return {"results": [normalized.__dict__]}

        graph = StateGraph(AnalysisState)
        graph.add_node("analyze_criterion", analyze_criterion)
        graph.add_conditional_edges(START, fan_out)
        graph.add_edge("analyze_criterion", END)
        return graph.compile()

    def _parse_or_recover(self, raw: str, criterion: Criterion) -> dict[str, Any]:
        """Parse model output, with a recovery path for malformed/truncated JSON"""
        try:
            return normalize_payload(parse_json_object(raw), criterion, raw)
        except Exception:
            recovered = recover_loose_payload(raw)
            if recovered is None:
                raise
            return normalize_payload(recovered, criterion, raw)

    def _safe_build_answer(self, criterion: Criterion, raw: str, docs: list[Document]) -> CriterionResult:
        answer = raw.strip() or "Модель не вернула текстовый ответ"
        evidence: list[dict[str, Any]] = []
        for d in docs[:2]:
            evidence.append(
                {
                    "path": d.metadata.get("path"),
                    "part": d.metadata.get("part"),
                    "chunk_index": d.metadata.get("chunk_index", d.metadata.get("cell_index")),
                    "quote": compact_text(d.page_content, 240),
                    "why": "Fallback evidence: source was retrieved but final JSON parsing failed",
                }
            )

        return CriterionResult(
            criterion_id=criterion.id,
            criterion_description=criterion.description,
            score=0,
            answer=answer,
            evidence=evidence,
            confidence=0.0,
        )

    def run(self, criteria: list[dict[str, str]]) -> list[dict[str, Any]]:
        state: AnalysisState = {
            "criteria": [Criterion(**c) for c in criteria],
            "results": [],
        }
        result_state = self.graph.invoke(state)
        return result_state.get("results", [])

def analyze_submission_text(
    title: str,
    text_content: str,
    criteria: list[dict[str, str]],
    config: RepoAnalysisConfig | None = None,
) -> list[dict[str, Any]]:
    analysis_config = config or RepoAnalysisConfig()
    analyzer = SubmissionAnalyzer(
        config=analysis_config,
        text_content=text_content,
        title=title,
    )
    return analyzer.run(criteria)


def analyze_repository_path(
    repo_path: str,
    criteria: list[dict[str, str]],
    config: RepoAnalysisConfig | None = None,
) -> list[dict[str, Any]]:
    if config is None:
        analysis_config = RepoAnalysisConfig(repo_path=repo_path)
    else:
        analysis_config = RepoAnalysisConfig(
            **{**asdict(config), "repo_path": repo_path},
        )

    analyzer = RepoAnalyzer(analysis_config)
    return analyzer.run(criteria)


def analyze_repository_url(
    repo_url: str,
    criteria: list[dict[str, str]],
    config: RepoAnalysisConfig | None = None,
) -> list[dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix="repo-analysis-") as temp_dir:
        repo_path = clone_repo(repo_url, temp_dir)
        return analyze_repository_path(repo_path=repo_path, criteria=criteria, config=config)
