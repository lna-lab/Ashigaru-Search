"""Web tools: web_search (self-hosted SearXNG JSON API) + fetch_url (readable text)."""
from __future__ import annotations
import httpx

from ..config import Config
from ..registry import Tool


def _extract_text(html: str, limit: int) -> str:
    """Best-effort main-text extraction. Uses trafilatura if available, else BeautifulSoup."""
    try:
        import trafilatura
        txt = trafilatura.extract(html, include_comments=False, include_tables=False)
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


def make_web_tools(cfg: Config, client: httpx.AsyncClient) -> list[Tool]:
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
            out.append(f"[{i}] {title}\n    {url}\n    {snippet}")
        return "\n".join(out)

    async def fetch_url(args: dict) -> str:
        url = str(args.get("url") or "").strip()
        if not url:
            return "ERROR: fetch_url needs a 'url'."
        r = await client.get(url)
        r.raise_for_status()
        ctype = r.headers.get("content-type", "")
        if "html" not in ctype and "text" not in ctype and "xml" not in ctype:
            return f"(non-text content: {ctype}) {url}"
        text = _extract_text(r.text, cfg.fetch_char_limit)
        return f"Content of {url} (truncated to {cfg.fetch_char_limit} chars):\n{text}"

    return [
        Tool("web_search",
             "Search the web via SearXNG. Returns ranked titles, URLs and snippets.",
             '<tool>{"name":"web_search","arguments":{"query":"...", "num":6}}</tool>',
             web_search),
        Tool("fetch_url",
             "Fetch a URL and return its readable main text (use after web_search to read a source).",
             '<tool>{"name":"fetch_url","arguments":{"url":"https://..."}}</tool>',
             fetch_url),
    ]
