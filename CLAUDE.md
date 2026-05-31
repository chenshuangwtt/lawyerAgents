# lawyerAgents AI Development Guide

## Project Shape

`lawyerAgents` is a Python/FastAPI legal RAG application with a Vue frontend.
The main runtime path is:

- `run.py`: service bootstrap and component initialization.
- `app/api.py`: FastAPI routes, SSE streaming, admin endpoints.
- `app/rag_chain.py`: primary legal consultation RAG flow.
- `app/analysis_chain.py` and `app/analysis_graph.py`: structured case analysis flow.
- `app/document_chain.py` and `app/labor_arbitration.py`: labor arbitration document generation.
- `app/law_registry.yaml`: domain, intent, keyword, and document classification configuration.
- `app/classifier.py`: keyword/LLM domain and intent classification.
- `app/sanitizer.py`: input risk detection, sanitization, and blocking.
- `app/chat_history.py`: chat/session/feedback persistence.

## Common Commands

Use the project Conda environment unless a task states otherwise:

```powershell
& 'D:\DevTools\miniconda3\envs\myenv\python.exe' -m pytest
& 'D:\DevTools\miniconda3\envs\myenv\python.exe' -m pytest --cov=app --cov-report=term-missing
& 'D:\DevTools\miniconda3\envs\myenv\python.exe' -m py_compile app\api.py app\sanitizer.py
```

Frontend build:

```powershell
npm run build
```

## Security Rules

- Never print, log, or copy secrets from `.env`, API keys, database URLs, or authorization headers.
- Real environment files must stay untracked: `.env` and `.env.*` are ignored; `.env.example` is the tracked template.
- Do not bypass `app/sanitizer.py`. High-risk XSS or prompt-injection inputs must be blocked before reaching RAG/LLM chains.
- Admin API requests must use constant-time key comparison.
- In production, Admin API access must require HTTPS/TLS unless an explicit local override is set for development testing.

## Code Conventions

- Avoid adding new module-level mutable globals. Prefer `AppContext` / dependency injection for new service dependencies.
- New RAG, retrieval, classifier, or fallback behavior must include focused tests.
- SSE changes must cover normal `token -> done`, `error` without `done`, empty `done`, keepalive, and cancellation behavior.
- Keep public API behavior stable unless the change is a documented security fix.
- Avoid broad refactors in `rag_chain.py`, `chat_history.py`, or `MessageBubble.vue` unless covered by characterization tests first.

## Configuration Notes

- Domain and intent keyword changes belong in `app/law_registry.yaml`, not hard-coded in classifier logic.
- `document_classification.strong_keywords` controls high-priority document intent words.
- Official cases and legacy cases are configured separately; legacy cases should remain disabled by default.

## Testing Notes

- Security-sensitive changes require tests in `tests/test_sanitizer.py`, `tests/test_api.py`, or another focused test module.
- Classifier changes should update `tests/test_classifier.py`, `tests/test_intent.py`, or registry loader tests.
- Keep tests meaningful: prefer regression tests for real reported failures over coverage-only assertions.

