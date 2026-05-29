"""Ashigaru-Search configuration — everything is env-driven so any OpenAI-compatible
endpoint/model can play orchestrator (Commander) or worker (Ashigaru scout)."""
from __future__ import annotations
import os
from dataclasses import dataclass, field

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # dotenv optional
    pass


def _b(name: str, default: bool) -> bool:
    v = os.getenv(name)
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Config:
    # ---- worker fleet (Ashigaru scouts): the model that runs search sub-tasks ----
    worker_base_url: str = field(default_factory=lambda: os.getenv("ASHIGARU_WORKER_BASE_URL", "http://localhost:8000/v1"))
    worker_model: str = field(default_factory=lambda: os.getenv("ASHIGARU_WORKER_MODEL", "lfm25-8b-a1b"))
    worker_api_key: str = field(default_factory=lambda: os.getenv("ASHIGARU_WORKER_API_KEY", "EMPTY"))

    # ---- orchestrator (Commander): plans & synthesizes. Defaults to the worker model. ----
    orch_base_url: str = field(default_factory=lambda: os.getenv("ASHIGARU_ORCH_BASE_URL", ""))
    orch_model: str = field(default_factory=lambda: os.getenv("ASHIGARU_ORCH_MODEL", ""))
    orch_api_key: str = field(default_factory=lambda: os.getenv("ASHIGARU_ORCH_API_KEY", ""))

    # ---- search backends ----
    searxng_url: str = field(default_factory=lambda: os.getenv("SEARXNG_URL", "http://localhost:8888"))
    rag_index: str = field(default_factory=lambda: os.getenv("ASHIGARU_RAG_INDEX", ""))  # path to built index (.pkl)

    # ---- behaviour knobs ----
    max_concurrency: int = field(default_factory=lambda: int(os.getenv("ASHIGARU_MAX_CONCURRENCY", "16")))
    max_subquestions: int = field(default_factory=lambda: int(os.getenv("ASHIGARU_MAX_SUBQUESTIONS", "6")))
    worker_max_steps: int = field(default_factory=lambda: int(os.getenv("ASHIGARU_WORKER_MAX_STEPS", "6")))
    search_results: int = field(default_factory=lambda: int(os.getenv("ASHIGARU_SEARCH_RESULTS", "6")))
    fetch_char_limit: int = field(default_factory=lambda: int(os.getenv("ASHIGARU_FETCH_CHAR_LIMIT", "6000")))
    temperature: float = field(default_factory=lambda: float(os.getenv("ASHIGARU_TEMPERATURE", "0.2")))
    request_timeout: float = field(default_factory=lambda: float(os.getenv("ASHIGARU_REQUEST_TIMEOUT", "120")))
    verbose: bool = field(default_factory=lambda: _b("ASHIGARU_VERBOSE", True))

    def __post_init__(self):
        # orchestrator falls back to the worker endpoint/model when unset
        self.orch_base_url = self.orch_base_url or self.worker_base_url
        self.orch_model = self.orch_model or self.worker_model
        self.orch_api_key = self.orch_api_key or self.worker_api_key

    @property
    def has_rag(self) -> bool:
        return bool(self.rag_index and os.path.exists(self.rag_index))
