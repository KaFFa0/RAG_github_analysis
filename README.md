# RAG GitHub Analysis API

Сервис анализирует:

- текстовые документы по переданному `textContent`
- GitHub-репозитории по переданному `repo_url`

Оба endpoint'а возвращают одинаковый JSON-формат: список результатов по критериям.

## Установка

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Запуск API

Из корня проекта:

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Также можно запускать так:

```bash
python3 -m src.api
```

По умолчанию `python3 -m src.api` стартует на порту `8000`.

## Настройка порта

Через `uvicorn`:

```bash
uvicorn src.api:app --host 0.0.0.0 --port 9000
```

Через переменную окружения:

```bash
PORT=9000 python3 -m src.api
```

Дополнительно можно задать:

```bash
HOST=0.0.0.0
RELOAD=true
MODEL_NAME=Qwen/Qwen2.5-3B-Instruct
LOAD_IN_4BIT=true
MAX_NEW_TOKENS=512
TEMPERATURE=0.1
PERSIST_DIRECTORY=.rag_cache/chroma
```

## Endpoint'ы

### 1. Анализ текста

`POST /analyze/text`

Пример тела запроса:

```json
{
  "title": "Технические аспекты",
  "textContent": "текст из документа",
  "criteria": [
    {"id": "101", "description": "Какая модель и подход был использован"},
    {"id": "102", "description": "Оцените обоснованность решения"},
    {"id": "103", "description": "Есть ли метрики эффективности"},
    {"id": "104", "description": "Насколько подробно описан технический стек"}
  ]
}
```

Пример вызова:

```bash
curl -X POST http://localhost:8000/analyze/text \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Технические аспекты",
    "textContent": "текст из документа",
    "criteria": [
      {"id": "101", "description": "Какая модель и подход был использован"},
      {"id": "102", "description": "Оцените обоснованность решения"},
      {"id": "103", "description": "Есть ли метрики эффективности"},
      {"id": "104", "description": "Насколько подробно описан технический стек"}
    ]
  }'
```

### 2. Анализ репозитория

`POST /analyze/repository`

Пример тела запроса:

```json
{
  "repo_url": "https://github.com/owner/repo",
  "criteria": [
    {"id": "101", "description": "Какая модель и подход был использован"},
    {"id": "102", "description": "Оцените обоснованность решения"},
    {"id": "103", "description": "Есть ли метрики эффективности"},
    {"id": "104", "description": "Насколько подробно описан технический стек"}
  ]
}
```

Пример вызова:

```bash
curl -X POST http://localhost:8000/analyze/repository \
  -H "Content-Type: application/json" \
  -d '{
    "repo_url": "https://github.com/owner/repo",
    "criteria": [
      {"id": "101", "description": "Какая модель и подход был использован"},
      {"id": "102", "description": "Оцените обоснованность решения"},
      {"id": "103", "description": "Есть ли метрики эффективности"},
      {"id": "104", "description": "Насколько подробно описан технический стек"}
    ]
  }'
```

## Формат ответа

Оба endpoint'а возвращают массив:

```json
[
  {
    "criterion_id": "101",
    "criterion_description": "Какая модель и подход был использован",
    "score": 8,
    "answer": "Краткий вывод",
    "evidence": [
      {
        "path": "README.md",
        "chunk_index": 0,
        "quote": "Фрагмент текста",
        "why": "Почему это подтверждает вывод"
      }
    ],
    "confidence": 0.87
  }
]
```
