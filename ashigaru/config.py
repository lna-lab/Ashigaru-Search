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
    emaki_tree: str = field(default_factory=lambda: os.getenv("ASHIGARU_EMAKI", ""))     # path to a built KURA-Emaki scroll dir

    # ---- behaviour knobs ----
    max_concurrency: int = field(default_factory=lambda: int(os.getenv("ASHIGARU_MAX_CONCURRENCY", "16")))
    max_subquestions: int = field(default_factory=lambda: int(os.getenv("ASHIGARU_MAX_SUBQUESTIONS", "6")))
    # nominal fleet size the Commander assumes for S/M/L density tags (S=10%, M=50%, L=100%)
    fleet_size: int = field(default_factory=lambda: int(os.getenv("ASHIGARU_FLEET_SIZE", "10")))
    worker_max_steps: int = field(default_factory=lambda: int(os.getenv("ASHIGARU_WORKER_MAX_STEPS", "6")))
    # quality gate: if a run's reports are thin, the Commander ESCALATES density and
    # re-runs (S->M->L; a numeric count -> ×3, capped at 12).
    escalate: bool = field(default_factory=lambda: _b("ASHIGARU_ESCALATE", True))
    max_escalations: int = field(default_factory=lambda: int(os.getenv("ASHIGARU_MAX_ESCALATIONS", "2")))
    quality_min_chars: int = field(default_factory=lambda: int(os.getenv("ASHIGARU_QUALITY_MIN_CHARS", "160")))
    quality_judge: bool = field(default_factory=lambda: _b("ASHIGARU_QUALITY_JUDGE", True))
    # mid-search check-ins: a scout reports a lead to the Commander, who orders
    # continue / regroup (file an interim note, re-launch on a new focus)
    supervise: bool = field(default_factory=lambda: _b("ASHIGARU_SUPERVISE", True))
    supervise_after: int = field(default_factory=lambda: int(os.getenv("ASHIGARU_SUPERVISE_AFTER", "1")))
    max_checkins: int = field(default_factory=lambda: int(os.getenv("ASHIGARU_MAX_CHECKINS", "5")))
    search_results: int = field(default_factory=lambda: int(os.getenv("ASHIGARU_SEARCH_RESULTS", "6")))
    # 足軽ターボ: hand the scout stable [Sn] source ids instead of raw URLs, and re-attach the
    # verbatim URLs in the harness. Stops a small model from corrupting/dropping long URLs.
    ref_id_sources: bool = field(default_factory=lambda: _b("ASHIGARU_REF_ID_SOURCES", True))
    # 足軽ターボ: seed the scout's loop with an automatic first search on the sub-question, instead
    # of trusting a small model to decide to search (tiny models often hallucinate a final instead).
    auto_first_search: bool = field(default_factory=lambda: _b("ASHIGARU_AUTO_FIRST_SEARCH", True))
    fetch_char_limit: int = field(default_factory=lambda: int(os.getenv("ASHIGARU_FETCH_CHAR_LIMIT", "6000")))
    # ---- X/Twitter & JS-walled reading via a reader proxy (renders JS the httpx fetch can't) ----
    # OFF by default. When on, x.com/twitter.com URLs are GET-prefixed through reader_base_url so
    # fetch_url gets rendered text instead of a login wall. DEFAULTS TO A LOCAL reader for sovereignty
    # (nothing leaves the box): run `docker run --rm -p 3000:8081 ghcr.io/jina-ai/reader:oss`.
    # Set ASHIGARU_READER_BASE_URL=https://r.jina.ai to opt INTO the remote service (the URL leaves the box).
    reader_enabled: bool = field(default_factory=lambda: _b("ASHIGARU_READER", False))
    reader_base_url: str = field(default_factory=lambda: os.getenv("ASHIGARU_READER_BASE_URL", "http://localhost:3000"))
    # comma-separated host substrings routed through the reader (only used when reader_all_js is False)
    reader_hosts: str = field(default_factory=lambda: os.getenv("ASHIGARU_READER_HOSTS", "x.com,twitter.com"))
    # route EVERY fetch through the reader (any JS-walled page), not just the host list above
    reader_all_js: bool = field(default_factory=lambda: _b("ASHIGARU_READER_ALL_JS", False))
    # optional bearer key (only meaningful for remote r.jina.ai: no-key=20 RPM, key=100/500 RPM)
    reader_api_key: str = field(default_factory=lambda: os.getenv("ASHIGARU_READER_API_KEY", ""))

    # ---- X/Twitter SEARCH as a POLITE HUMAN PROXY ("never get scolded, never be a nuisance") ----
    # OFF by default. When on, adds an `x_search` tool. X is just ONE source (web + 蔵 + X), used
    # occasionally. The GOVERNOR below keeps the fleet to a courteous human's scope & cadence so we stay
    # FAR under X's limits — we'd rather miss a result than knock too hard. Backend = twscrape (add ONE
    # warmed SECONDARY account to its pool; never your main).
    x_search_enabled: bool = field(default_factory=lambda: _b("ASHIGARU_X_SEARCH", False))
    x_backend: str = field(default_factory=lambda: os.getenv("ASHIGARU_X_BACKEND", "twscrape"))  # twscrape | (future) nitter | computeruse
    x_pool_db: str = field(default_factory=lambda: os.getenv("ASHIGARU_X_POOL_DB", ""))           # twscrape accounts.db path ("" = twscrape default)
    x_search_mode: str = field(default_factory=lambda: os.getenv("ASHIGARU_X_SEARCH_MODE", "Latest"))  # Latest | Top
    # --- the politeness governor (the whole point: behave like a courteous human, never a nuisance) ---
    x_max_results: int = field(default_factory=lambda: int(os.getenv("ASHIGARU_X_MAX_RESULTS", "15")))          # ~one human screenful; NO deep paging
    x_min_interval_s: float = field(default_factory=lambda: float(os.getenv("ASHIGARU_X_MIN_INTERVAL_S", "12")))  # min gap between searches
    x_jitter_s: float = field(default_factory=lambda: float(os.getenv("ASHIGARU_X_JITTER_S", "6")))             # +0..jitter random, so cadence looks human
    x_max_per_hour: int = field(default_factory=lambda: int(os.getenv("ASHIGARU_X_MAX_PER_HOUR", "20")))        # hard hourly courtesy ceiling
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

    @property
    def has_emaki(self) -> bool:
        return bool(self.emaki_tree and os.path.isdir(self.emaki_tree)
                    and os.path.exists(os.path.join(self.emaki_tree, "tree.json")))
