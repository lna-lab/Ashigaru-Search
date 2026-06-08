"""x_search — search X/Twitter as a POLITE HUMAN PROXY, not a scraper.

Theme (Ken, 2026-06-08): 「決して怒られないように、ご迷惑をおかけしないように」 — never get scolded,
never be a nuisance. So this is built around a GOVERNOR that holds the fleet to a courteous human's
scope and cadence: a few queries, paced with jitter, shallow results (~one human screenful, NO deep
paging), a single warmed account, and immediate graceful backoff if X ever signals trouble. We would
rather miss a result than knock on X's door too hard. X is just ONE source among many (web + 蔵 + X),
used occasionally — the agent searches the way Ken himself would, on his behalf (代行).

Backend = twscrape: an account-credentialed client that calls X's internal `SearchTimeline` GraphQL
(the same op the x.com search box uses) and returns tweet JSON *including the full text*, so no separate
read step is needed. Results are emitted in the SAME `[Sn]` SourceRegistry contract as web_search, so a
scout cites/reads them identically; the URLs are real x.com/<user>/status/<id> permalinks (the reader
hook can also open them if ever wanted).

Turn on:  export ASHIGARU_X_SEARCH=1  — then add ONE warmed account to twscrape's pool
(`pip install twscrape`; see twscrape docs for `add_accounts` / cookie auth). Prefer a warmed SECONDARY
account, NEVER your main. Tune the courtesy with ASHIGARU_X_MIN_INTERVAL_S / _JITTER_S / _MAX_PER_HOUR /
_MAX_RESULTS — defaults are deliberately gentle (≤15 hits, ~12-18s apart, ≤20/hour).
"""
from __future__ import annotations

import asyncio
import random
import time

from ..config import Config
from ..registry import Tool


class Governor:
    """The politeness gate. Per-process. The point is NOT to evade X's limits — it is to stay so far
    under them that we never trouble the host. Enforces a human-like minimum spacing (+jitter) between
    searches and a courteous hourly ceiling."""

    def __init__(self, min_interval_s: float, jitter_s: float, max_per_hour: int):
        self.min_interval_s = max(0.0, min_interval_s)
        self.jitter_s = max(0.0, jitter_s)
        self.max_per_hour = max(1, max_per_hour)
        self._last = 0.0                 # monotonic time of the last search
        self._stamps: list[float] = []   # monotonic times of searches within the last hour

    def _prune(self, now: float) -> None:
        cutoff = now - 3600.0
        self._stamps = [t for t in self._stamps if t >= cutoff]

    def check_hourly(self) -> tuple[bool, float]:
        """(ok, seconds_until_a_slot_frees). Pure — does not mutate or sleep."""
        now = time.monotonic()
        self._prune(now)
        if len(self._stamps) < self.max_per_hour:
            return True, 0.0
        return False, max(0.0, 3600.0 - (now - self._stamps[0]))

    async def wait_turn(self) -> None:
        """Sleep just enough to honor the minimum spacing + human jitter before the next search."""
        gap = time.monotonic() - self._last
        need = self.min_interval_s + random.uniform(0.0, self.jitter_s)
        if gap < need:
            await asyncio.sleep(need - gap)

    def record(self) -> None:
        now = time.monotonic()
        self._last = now
        self._stamps.append(now)


async def _twscrape_search(cfg: Config, query: str, limit: int) -> list[dict]:
    """Return [{url, title, snippet}] via twscrape. Lazily imported so this module loads even without
    twscrape installed. Raises RuntimeError with a clear, actionable message if it can't run."""
    try:
        from twscrape import API, gather
    except Exception:
        raise RuntimeError("twscrape not installed — `pip install twscrape`, then add a warmed account "
                           "to its pool (see twscrape docs). Prefer a secondary account, never your main.")
    api = API(cfg.x_pool_db) if cfg.x_pool_db else API()
    # shallow, recent, human-scope: one page up to `limit`; never deep-paginate
    try:
        kv = {"product": cfg.x_search_mode} if cfg.x_search_mode in ("Latest", "Top", "Media") else None
        tweets = await gather(api.search(query, limit=limit, kv=kv))
    except TypeError:
        # older/newer twscrape signature without kv — fall back to a plain search
        tweets = await gather(api.search(query, limit=limit))
    out: list[dict] = []
    for t in tweets[:limit]:
        user = getattr(getattr(t, "user", None), "username", "") or ""
        tid = getattr(t, "id", "") or getattr(t, "id_str", "")
        url = getattr(t, "url", "") or (f"https://x.com/{user}/status/{tid}" if user and tid else "")
        text = getattr(t, "rawContent", None) or getattr(t, "content", "") or ""
        out.append({"url": url, "title": (f"@{user}" if user else "tweet"),
                    "snippet": " ".join(text.split())})
    return out


def make_x_search_tools(cfg: Config, client=None, sources=None) -> list[Tool]:
    """An `x_search` tool that behaves like a courteous human proxy. `client` is accepted for a uniform
    factory signature but unused (twscrape manages its own session). Emits the [Sn] SourceRegistry
    contract so X hits are cited/read exactly like web_search hits."""
    gov = Governor(cfg.x_min_interval_s, cfg.x_jitter_s, cfg.x_max_per_hour)

    async def x_search(args: dict) -> str:
        query = str(args.get("query") or args.get("q") or "").strip()
        if not query:
            return "ERROR: x_search needs a 'query'."
        # hard cap to human scope no matter what the model asks for
        n = max(1, min(int(args.get("num") or args.get("k") or cfg.x_max_results), cfg.x_max_results))

        ok, wait = gov.check_hourly()
        if not ok:
            return (f"x_search is resting to stay a polite guest (hourly courtesy cap {cfg.x_max_per_hour} "
                    f"reached). Try again in ~{int(wait)}s — use web_search or the 蔵 meanwhile.")
        await gov.wait_turn()                      # human-paced spacing + jitter
        attempted = False
        try:
            backend = (cfg.x_backend or "twscrape").lower()
            if backend == "twscrape":
                attempted = True
                hits = await _twscrape_search(cfg, query, n)
            else:
                return (f"ERROR: x_backend '{cfg.x_backend}' not implemented yet (only 'twscrape' for now; "
                        f"'computeruse' is the planned last-resort backend).")
        except RuntimeError as e:                  # not installed / no accounts — a clear setup issue
            return f"ERROR: x_search unavailable — {e}"
        except Exception as e:
            # politeness first: on ANY snag (incl. a rate-limit signal) back off and degrade — never hammer
            return (f"x_search backed off after a snag ({type(e).__name__}) to avoid being a nuisance. "
                    f"Fall back to web_search or the 蔵; retry X later.")
        finally:
            if attempted:
                gov.record()                       # count only real X calls against the courtesy budget

        if not hits:
            return f"No X results for: {query}"
        out = [f"X results for: {query} (polite human-scope, top {len(hits)}):"]
        for i, h in enumerate(hits, 1):
            url, title = h.get("url", ""), h.get("title", "")
            snippet = (h.get("snippet") or "")[:300]
            if sources is not None and url:
                ref = sources.register(url, title, snippet)
                out.append(f"{sources.label(ref)}\n    {snippet}")
            else:
                out.append(f"[{i}] {title}\n    {url}\n    {snippet}")
        if sources is not None:
            out.append("(To read a source, call fetch_url with its id, e.g. {\"id\":\"S1\"}. "
                        "Cite sources by id, e.g. [S1].)")
        return "\n".join(out)

    desc = ("Search X/Twitter as a polite human proxy — occasional, human-scope, recent posts. Returns "
            "tweets as [S1] @user + snippet; cite/read by id. Use sparingly, alongside web_search and the 蔵.")
    usage = '<tool>{"name":"x_search","arguments":{"query":"...", "num":10}}</tool>'
    return [Tool("x_search", desc, usage, x_search)]
