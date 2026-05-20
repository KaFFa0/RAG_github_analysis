from pathlib import Path
from dataclasses import dataclass
from typing import Any
from langchain_core.documents import Document
from collections import Counter
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json
import re
import ast
import hashlib
from AI_review import (
    Criterion,
    RepoAnalysisConfig
    )

def infer_kind(path: Path) -> str:
    name = path.name.lower()
    suffix = path.suffix.lower()
    full = str(path).lower()

    if suffix == ".ipynb":
        return "notebook"
    if "readme" in name:
        return "readme"
    if suffix == ".md" or "/docs/" in full or full.startswith("docs/"):
        return "docs"
    if name in {"dockerfile", "makefile"}:
        return "ops"
    if suffix in {".toml", ".yaml", ".yml", ".json", ".ini", ".cfg", ".env"}:
        return "config"
    if suffix in {".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".java", ".kt", ".c", ".cc", ".cpp", ".h", ".hpp"}:
        return "code"
    if name in {"license", "copying", "notice"}:
        return "license"
    return "text"


def is_ignored_path(rel_path: Path, ignored_dirs: set[str]) -> bool:
    return any(part in ignored_dirs for part in rel_path.parts)


def _read_binary_safe_text(path: Path) -> str | None:
    try:
        data = path.read_bytes()
    except OSError:
        return None

    if b"\x00" in data:
        return None

    sample = data[:4096]
    if sample:
        printable = sum((32 <= b <= 126) or b in (9, 10, 13) for b in sample)
        if printable / len(sample) < 0.72:
            return None

    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return None


def compact_text(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head - 50
    return text[:head] + "\n... [TRUNCATED] ...\n" + text[-tail:]


def _join_cell_source(source: Any) -> str:
    if isinstance(source, list):
        return "".join(source)
    if isinstance(source, str):
        return source
    return ""


def _join_output_text(output: dict[str, Any]) -> str:
    parts: list[str] = []
    if "text" in output:
        txt = output["text"]
        if isinstance(txt, list):
            parts.extend(str(x) for x in txt)
        else:
            parts.append(str(txt))
    if "ename" in output or "evalue" in output:
        parts.append(f"{output.get('ename', '')}: {output.get('evalue', '')}".strip())
    if "traceback" in output and isinstance(output["traceback"], list):
        parts.extend(str(x) for x in output["traceback"])
    return "\n".join(p for p in parts if p).strip()


def make_doc(page_content: str, metadata: dict[str, Any]) -> Document:
    return Document(page_content=page_content, metadata=metadata)


def dedupe_docs(docs: list[Document], max_per_path: int | None = None) -> list[Document]:
    seen_keys: set[tuple[Any, ...]] = set()
    per_path_counter: Counter[str] = Counter()
    result: list[Document] = []

    for doc in docs:
        md = doc.metadata
        path = str(md.get("path", ""))
        key = (
            path,
            md.get("part"),
            md.get("chunk_index"),
            md.get("cell_index"),
            md.get("output_index"),
            hash(doc.page_content[:500]),
        )
        if key in seen_keys:
            continue
        if max_per_path is not None and per_path_counter[path] >= max_per_path:
            continue
        seen_keys.add(key)
        per_path_counter[path] += 1
        result.append(doc)
    return result


def compact_doc_for_file(page_content: str, max_chars: int) -> str:
    return compact_text(page_content, max_chars)


def parse_notebook(path: Path, rel_path: str, config: RepoAnalysisConfig) -> tuple[Document, list[Document]]:
    try:
        nb = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        raw = _read_binary_safe_text(path) or ""
        file_doc = make_doc(
            compact_doc_for_file(raw, config.max_file_doc_chars),
            {
                "path": rel_path,
                "name": path.name,
                "suffix": path.suffix.lower(),
                "size": path.stat().st_size,
                "kind": "notebook",
                "part": "file_summary",
            },
        )
        return file_doc, []

    cells = nb.get("cells", []) or []
    outline_lines: list[str] = []
    cell_docs: list[Document] = []

    for idx, cell in enumerate(cells):
        cell_type = cell.get("cell_type", "unknown")
        source = compact_text(_join_cell_source(cell.get("source", "")), config.max_notebook_cell_chars)

        if cell_type == "markdown":
            outline_lines.append(f"[MD {idx}] {source[:900]}")
            cell_docs.append(
                make_doc(
                    source,
                    {
                        "path": rel_path,
                        "name": path.name,
                        "suffix": ".ipynb",
                        "size": path.stat().st_size,
                        "kind": "notebook",
                        "part": "markdown",
                        "cell_type": "markdown",
                        "cell_index": idx,
                    },
                )
            )
        elif cell_type == "code":
            preview = source.splitlines()[:12]
            outline_lines.append(f"[CODE {idx}] {preview[0] if preview else ''}")
            if source.strip():
                cell_docs.append(
                    make_doc(
                        source,
                        {
                            "path": rel_path,
                            "name": path.name,
                            "suffix": ".ipynb",
                            "size": path.stat().st_size,
                            "kind": "notebook",
                            "part": "code",
                            "cell_type": "code",
                            "cell_index": idx,
                        },
                    )
                )

            for out_idx, output in enumerate(cell.get("outputs", []) or []):
                output_text = compact_text(_join_output_text(output), config.max_notebook_output_chars)
                if not output_text:
                    continue
                cell_docs.append(
                    make_doc(
                        output_text,
                        {
                            "path": rel_path,
                            "name": path.name,
                            "suffix": ".ipynb",
                            "size": path.stat().st_size,
                            "kind": "notebook",
                            "part": "output",
                            "cell_type": "code",
                            "cell_index": idx,
                            "output_index": out_idx,
                        },
                    )
                )
        else:
            if source.strip():
                outline_lines.append(f"[{cell_type.upper()} {idx}] {source[:900]}")
                cell_docs.append(
                    make_doc(
                        source,
                        {
                            "path": rel_path,
                            "name": path.name,
                            "suffix": ".ipynb",
                            "size": path.stat().st_size,
                            "kind": "notebook",
                            "part": cell_type,
                            "cell_type": cell_type,
                            "cell_index": idx,
                        },
                    )
                )

    file_summary = compact_text("\n\n".join(outline_lines), config.max_file_doc_chars)
    file_doc = make_doc(
        file_summary,
        {
            "path": rel_path,
            "name": path.name,
            "suffix": ".ipynb",
            "size": path.stat().st_size,
            "kind": "notebook",
            "part": "file_summary",
        },
    )
    return file_doc, cell_docs


def parse_text_file(path: Path, rel_path: str, config: RepoAnalysisConfig) -> tuple[Document, list[Document]]:
    text = _read_binary_safe_text(path)
    if not text:
        raise ValueError(f"Unreadable text file: {path}")

    if path.suffix.lower() == '.md':
        text = merge_markdown_tables(text)

    kind = infer_kind(path)
    file_doc = make_doc(
        compact_doc_for_file(text, config.max_file_doc_chars),
        {
            "path": rel_path,
            "name": path.name,
            "suffix": path.suffix.lower(),
            "size": path.stat().st_size,
            "kind": kind,
            "part": "file_summary",
        },
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_text(text)
    chunk_docs: list[Document] = []
    for idx, chunk in enumerate(chunks):
        chunk_docs.append(
            make_doc(
                chunk,
                {
                    "path": rel_path,
                    "name": path.name,
                    "suffix": path.suffix.lower(),
                    "size": path.stat().st_size,
                    "kind": kind,
                    "part": "chunk",
                    "chunk_index": idx,
                    "chunk_total_hint": len(chunks),
                },
            )
        )
    return file_doc, chunk_docs

def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

def extract_json_blocks(text: str) -> list[str]:
    blocks: list[str] = []
    start = None
    depth = 0
    in_string = False
    escape = False

    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    blocks.append(text[start : i + 1])
                    start = None

    return blocks


def safe_int(value: Any, default: int = 0) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        m = re.search(r"-?\d+", value)
        if m:
            return int(m.group())
    return default


def safe_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        m = re.search(r"-?(?:\d+(?:\.\d+)?|\.\d+)", value)
        if m:
            try:
                return float(m.group())
            except Exception:
                return default
    return default


def unwrap_payload(obj: Any) -> dict[str, Any]:
    if not isinstance(obj, dict):
        raise ValueError("Payload is not a dict")

    expected = {
        "criterion_id",
        "criterion_description",
        "score",
        "answer",
        "evidence",
        "confidence",
    }

    if expected.intersection(obj.keys()):
        return obj

    # single-key wrapper like {"id": {...}} or {"result": {...}}
    if len(obj) == 1:
        inner = next(iter(obj.values()))
        if isinstance(inner, dict):
            try:
                return unwrap_payload(inner)
            except Exception:
                pass

    # nested search
    for value in obj.values():
        if isinstance(value, dict):
            try:
                return unwrap_payload(value)
            except Exception:
                pass

    raise ValueError("No valid criterion payload found")


def extract_score_from_payload(payload: dict[str, Any], criterion: Criterion) -> int:
    if "score" in payload:
        return safe_int(payload.get("score"), 0)

    for key in (criterion.id, criterion.id.lower(), criterion.id.upper()):
        if key in payload:
            value = payload[key]
            if isinstance(value, dict) and "score" in value:
                return safe_int(value.get("score"), 0)
            return safe_int(value, 0)

    # common pattern where LLM uses integer-like string keys within a wrapper
    for key, value in payload.items():
        if isinstance(key, str) and key.lower().strip() == criterion.id.lower().strip():
            if isinstance(value, dict) and "score" in value:
                return safe_int(value.get("score"), 0)
            return safe_int(value, 0)

    return 0


def parse_json_object(text: str) -> dict[str, Any]:
    """Robust JSON extraction from messy LLM outputs

    Handles:
    - markdown fences
    - extra explanations before/after JSON
    - nested wrappers like {"c3": {...}}
    - trailing commas
    - single quotes (best-effort)
    - multiple JSON blocks
    - partially malformed output (returns the best balanced block found)
    """

    if not text or not text.strip():
        raise ValueError("Empty model output")

    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    blocks = extract_json_blocks(text)
    if not blocks:
        try:
            obj = ast.literal_eval(text)
            return unwrap_payload(obj)
        except Exception:
            raise ValueError("No complete JSON object found")

    last_error: Exception | None = None

    for block in sorted(blocks, key=len, reverse=True):
        candidate = block.strip()
        candidate = re.sub(r",(\s*[}\]])", r"\1", candidate)
        candidate = candidate.replace("\t", " ")

        try:
            return unwrap_payload(json.loads(candidate))
        except Exception as e:
            last_error = e

        try:
            repaired = candidate
            repaired = repaired.replace("None", "null")
            repaired = repaired.replace("True", "true")
            repaired = repaired.replace("False", "false")
            repaired = re.sub(r"(?<!\\)'", '"', repaired)
            return unwrap_payload(json.loads(repaired))
        except Exception as e:
            last_error = e

    loose = recover_loose_payload(text)
    if loose is not None:
        return loose

    raise ValueError(f"Could not parse JSON: {last_error}")


def _find_first_match(patterns: list[str], text: str, flags: int = 0) -> str | None:
    for pattern in patterns:
        m = re.search(pattern, text, flags)
        if m:
            return m.group(1).strip()
    return None


def recover_loose_payload(text: str) -> dict[str, Any] | None:

    criterion_id = _find_first_match([
        r'"criterion_id"\s*:\s*"([^"]+)"',
        r'"([a-zA-Z0-9_\-]+)"\s*:\s*\{',
        r'"([a-zA-Z0-9_\-]+)"\s*:\s*\d+',
    ], text)

    criterion_description = _find_first_match([
        r'"criterion_description"\s*:\s*"((?:\\.|[^"\\])*)"',
    ], text, flags=re.DOTALL)

    score = None
    score_text = _find_first_match([
        r'"score"\s*:\s*([0-9]+(?:\.[0-9]+)?)',
        r'"(?:c\d+|criterion_[^\"]+|[A-Za-z0-9_\-]+)"\s*:\s*([0-9]+(?:\.[0-9]+)?)',
    ], text)
    if score_text is not None:
        score = safe_int(score_text, 0)

    answer = _find_first_match([
        r'"answer"\s*:\s*"((?:\\.|[^"\\])*)"',
    ], text, flags=re.DOTALL)

    confidence_text = _find_first_match([
        r'"confidence"\s*:\s*([0-9]+(?:\.[0-9]+)?)',
    ], text)
    confidence = safe_float(confidence_text, 0.0) if confidence_text is not None else 0.0

    evidence: list[dict[str, Any]] = []
    for m in re.finditer(r'\{\s*"path"\s*:\s*"((?:\\.|[^"\\])*)".*?"quote"\s*:\s*"((?:\\.|[^"\\])*)".*?"why"\s*:\s*"((?:\\.|[^"\\])*)"', text, flags=re.DOTALL):
        evidence.append(
            {
                "path": m.group(1),
                "chunk_index": None,
                "quote": m.group(2),
                "why": m.group(3),
            }
        )
        if len(evidence) >= 3:
            break

    if not any([criterion_id, criterion_description, answer, evidence, score is not None, confidence > 0]):
        return None

    return {
        "criterion_id": criterion_id or "unknown",
        "criterion_description": criterion_description or "",
        "score": score if score is not None else 0,
        "answer": answer or "Нет текстового ответа",
        "evidence": evidence,
        "confidence": confidence,
    }


def normalize_payload(payload: dict[str, Any], criterion: Criterion, raw: str) -> dict[str, Any]:
    """Normalize a parsed payload into the canonical schema"""
    evidence = payload.get("evidence", [])
    if not isinstance(evidence, list):
        evidence = []

    normalized = {
        "criterion_id": str(payload.get("criterion_id", criterion.id)),
        "criterion_description": str(payload.get("criterion_description", criterion.description)),
        "score": extract_score_from_payload(payload, criterion),
        "answer": str(payload.get("answer") or "Нет текстового ответа."),
        "evidence": evidence,
        "confidence": safe_float(payload.get("confidence"), 0.0),
    }

    return normalized

def merge_markdown_tables(text: str) -> str:
    """
    Находит markdown-таблицы и заменяет внутренние переводы строк на ' ; ',
    чтобы RecursiveCharacterTextSplitter не разбил их на отдельные чанки
    Сохраняет одну пустую строку перед таблицей и одну после
    """
    lines = text.split('\n')
    out = []
    in_table = False
    table_buffer = []
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        is_table_line = bool(re.match(r'^\|.*\|$', stripped))
        is_separator = bool(re.match(r'^\|[\s\-:]+\|$', stripped))  
        
        if is_table_line and not is_separator:
            if not in_table:
                in_table = True
                table_buffer = [stripped]
            else:
                table_buffer.append(stripped)
        else:
            if in_table:
                merged = ' ; '.join(table_buffer)
                out.append('')          
                out.append(merged)
                out.append('')          
                in_table = False
                table_buffer = []
            out.append(line)
    
    if in_table:
        merged = ' ; '.join(table_buffer)
        out.append('')
        out.append(merged)
        out.append('')
    
    return '\n'.join(out)
