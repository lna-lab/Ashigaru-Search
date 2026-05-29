"""Ashigaru-Search — a fleet of small local LLMs (ashigaru = foot-soldiers) that fan out
to search the web (SearXNG) and local docs, then synthesize a cited answer. Any
OpenAI-compatible model can be the Commander or the scouts.
https://github.com/lna-lab/Ashigaru-Search
"""
from .config import Config
from .orchestrator import research, ResearchResult
from .worker import WorkerResult

__version__ = "0.1.0"
__all__ = ["Config", "research", "ResearchResult", "WorkerResult", "__version__"]
