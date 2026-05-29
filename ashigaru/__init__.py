"""Ashigaru-Search — a fleet of small local LLMs (足軽) that fan out to search the web
(SearXNG) and local docs, then synthesize a cited answer. Any OpenAI-compatible model
can be the commander (大将) or the scouts.  https://github.com/lna-lab/Ashigaru-Search
"""
from .config import Config
from .orchestrator import research, ResearchResult
from .worker import WorkerResult

__version__ = "0.1.0"
__all__ = ["Config", "research", "ResearchResult", "WorkerResult", "__version__"]
