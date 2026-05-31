"""Small service container used to make FastAPI handlers easier to test."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class AppContext:
    rag_chain: Any = None
    retriever: Any = None
    llm: Any = None
    rag_components: dict[str, Any] | None = None
    semantic_cache: Any = None
    analysis_graph: Any = None

