"""Web tools: web_search (self-hosted SearXNG JSON API) + fetch_url (readable text)."""
from __future__ import annotations
import httpx

from ..config import Config
from ..registry import Tool


def _extract_text(html: str, limit: int) -> str:
    """Best-effort main-text extraction. Uses trafilatura if available, else BeautifulSoup."""
    try:
        import trafilatura
        # keep tables: spec / pricing / benchmark / comparison rows are exactly the dense facts
        # a scout wants, and they live in <table> — dropping them silently loses the best content.
        txt = trafilatura.extract(html, include_comments=False, include_tables=True)
        if txt:
            return txt[:limit]
    except Exception:
        pass
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer", "noscript", "svg"]):
            tag.decompose()
        text = " ".join(soup.get_text(" ").split())
        return text[:limit]
    except Exception:
        # last resort: crude strip
        import re
        return re.sub(r"<[^>]+>", " ", html)[:limit]


def _reader_url(cfg: Config, url: str) -> str | None:
    """If `url` should be read through the reader proxy, return the prefixed reader URL, else None.
    The reader renders JS (e.g. x.com) that the plain httpx fetch can't get past a login wall.
    Sovereign by default: reader_base_url points at a self-hosted reader (nothing leaves the box)."""
    if not getattr(cfg, "reader_enabled", False) or not cfg.reader_base_url:
        return None
    if not cfg.reader_all_js:
        hosts = [h.strip().lower() for h in (cfg.reader_hosts or "").split(",") if h.strip()]
        try:
            from urllib.parse import urlsplit
            host = (urlsplit(url).hostname or "").lower()
        except Exception:
            host = url.lower()
        if not any(host == h or host.endswith("." + h) for h in hosts):
            return None
    # r.jina.ai / jina reader:oss contract: GET {base}/{full-url}
    return f"{cfg.reader_base_url.rstrip('/')}/{url}"


def make_web_tools(cfg: Config, client: httpx.AsyncClient, sources=None) -> list[Tool]:
    """Web tools. When `sources` (a SourceRegistry) is given, results are handed to the model
    as stable `[Sn]` ids with NO raw URL (足軽ターボ) — the model fetches and cites by id and
    the harness re-attaches verbatim URLs. With `sources=None` it falls back to printing URLs."""
    async def web_search(args: dict) -> str:
        query = str(args.get("query") or args.get("q") or "").strip()
        if not query:
            return "ERROR: web_search needs a 'query'."
        n = int(args.get("num") or args.get("k") or cfg.search_results)
        params = {"q": query, "format": "json", "safesearch": "0"}
        if args.get("lang"):
            params["language"] = str(args["lang"])
        try:
            r = await client.get(f"{cfg.searxng_url.rstrip('/')}/search", params=params)
            r.raise_for_status()
        except httpx.HTTPError as e:
            return (f"ERROR: can't reach SearXNG at {cfg.searxng_url} ({type(e).__name__}). "
                    f"Is it running?  Start it with: "
                    f"docker compose -f docker/docker-compose.yml up -d searxng")
        results = r.json().get("results", [])[:n]
        if not results:
            return f"No results for: {query}"
        out = [f"Search results for: {query}"]
        for i, res in enumerate(results, 1):
            title = res.get("title", "").strip()
            url = res.get("url", "").strip()
            snippet = " ".join((res.get("content") or "").split())[:300]
            if sources is not None:
                ref = sources.register(url, title, snippet)
                # show the stable id + title + domain — never the full path/query string
                out.append(f"{sources.label(ref)}\n    {snippet}")
            else:
                out.append(f"[{i}] {title}\n    {url}\n    {snippet}")
        if sources is not None:
            out.append("(To read a source, call fetch_url with its id, e.g. {\"id\":\"S1\"}. "
                        "Cite sources in your report by id, e.g. [S1] — do NOT write URLs.)")
        return "\n".join(out)

    async def fetch_url(args: dict) -> str:
        # 足軽ターボ: prefer fetching by source id so the model never handles a raw URL
        ref = str(args.get("id") or args.get("ref") or args.get("source") or "").strip()
        url = str(args.get("url") or "").strip()
        shown = url
        if sources is not None and ref:
            resolved = sources.resolve(ref)
            if not resolved:
                return f"ERROR: unknown source id '{ref}'. Use an id from web_search results (e.g. S1)."
            url, shown = resolved, sources.label(ref)
        if not url:
            hint = "fetch_url needs a source 'id' from web_search (e.g. {\"id\":\"S1\"})." if sources is not None \
                   else "fetch_url needs a 'url'."
            return f"ERROR: {hint}"
        reader = _reader_url(cfg, url)
        if reader is not None:
            headers = {}
            if cfg.reader_api_key:
                headers["Authorization"] = f"Bearer {cfg.reader_api_key}"
            r = await client.get(reader, headers=headers)
            r.raise_for_status()
            # the reader returns clean markdown/text already — pass through _extract_text
            # (trafilatura is a no-op on plain text, so it just truncates to the limit)
            text = _extract_text(r.text, cfg.fetch_char_limit)
            return f"Content of {shown} (via reader, truncated to {cfg.fetch_char_limit} chars):\n{text}"
        r = await client.get(url)
        r.raise_for_status()
        # lower-case before the substring check: a server sending "TEXT/HTML" or "Text/Html"
        # would otherwise be wrongly rejected as non-text.
        ctype = r.headers.get("content-type", "").lower()
        if "html" not in ctype and "text" not in ctype and "xml" not in ctype:
            return f"(non-text content: {ctype}) {shown}"
        text = _extract_text(r.text, cfg.fetch_char_limit)
        return f"Content of {shown} (truncated to {cfg.fetch_char_limit} chars):\n{text}"

    if sources is not None:
        search_desc = "Search the web via SearXNG. Returns ranked results as [S1] Title (domain) + snippet; cite/fetch them by id."
        fetch_desc = "Read a search result's full text by its source id (from web_search). The system tracks the real URL."
        fetch_usage = '<tool>{"name":"fetch_url","arguments":{"id":"S1"}}</tool>'
    else:
        search_desc = "Search the web via SearXNG. Returns ranked titles, URLs and snippets."
        fetch_desc = "Fetch a URL and return its readable main text (use after web_search to read a source)."
        fetch_usage = '<tool>{"name":"fetch_url","arguments":{"url":"https://..."}}</tool>'

    return [
        Tool("web_search", search_desc,
             '<tool>{"name":"web_search","arguments":{"query":"...", "num":6}}</tool>',
             web_search),
        Tool("fetch_url", fetch_desc, fetch_usage, fetch_url),
    ]
