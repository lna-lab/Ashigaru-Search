"""足軽ターボ (reference-id sources) tests — no network.
Run:  PYTHONPATH=. python3 tests/test_sources.py
"""
import asyncio

from ashigaru.sources import SourceRegistry


def test_registry():
    reg = SourceRegistry()
    # a deliberately nasty URL: long path + query string (the kind a 1.2B mangles)
    u1 = "https://www.reddit.com/r/privacy/comments/18w6zjk/which_is_better/?tl=ja"
    u2 = "https://ja.wikipedia.org/wiki/富士山"
    r1 = reg.register(u1, "Reddit privacy", "snippet one")
    r2 = reg.register(u2, "富士山 - Wikipedia", "snippet two")
    assert (r1, r2) == ("S1", "S2"), (r1, r2)
    # dedup by URL -> same ref, no new id
    assert reg.register(u1, "dup") == "S1"
    # empty url -> no ref
    assert reg.register("") == ""
    # resolve is byte-for-byte verbatim, query string intact
    assert reg.resolve("S1") == u1 and reg.resolve("[S1]") == u1 and reg.resolve("S2") == u2
    assert reg.resolve("S9") is None
    # label shown to the model carries domain, NEVER the path/query
    lbl = reg.label("S1")
    assert "reddit.com" in lbl and "?tl=ja" not in lbl and "/comments/" not in lbl, lbl
    assert lbl.startswith("[S1] Reddit privacy"), lbl
    # refs_in: ordered, deduped, registry-gated (ignores unregistered S-tokens)
    txt = "Per [S2] it is famous; see also [S1] and [S1] again, but S9 is not real."
    assert reg.refs_in(txt) == ["S2", "S1"], reg.refs_in(txt)
    # verbatim_sources resolves cited refs to exact URLs
    assert reg.verbatim_sources(["S2", "S1"]) == [u2, u1]
    # source_map is the harness-built, verbatim Sources block
    sm = reg.source_map(["S1"])
    assert sm == f"- [S1] {u1} — Reddit privacy", sm
    print("✓ SourceRegistry: register/dedup/resolve/label/refs_in/source_map pass")


def test_worker_turbo_resolves_and_ignores_model_urls():
    """End-to-end at the worker level: the scout fetches+cites by [Sn]; even if it ALSO
    blurts a corrupted URL in prose, the harness returns the VERBATIM registry URLs."""
    import ashigaru.worker as worker
    from ashigaru.llm import LLMClient
    from ashigaru import Config

    GOOD = "https://www.reddit.com/r/privacy/comments/18w6zjk/which_is_better/?tl=ja"

    # a toolbox double that carries a real registry and registers a source on web_search
    class TurboBox:
        def __init__(self):
            self.sources = SourceRegistry()
            self.start_hint = "start"
        def render_docs(self): return "- web_search: ...\n- fetch_url: ..."
        async def run(self, name, args):
            if name == "web_search":
                ref = self.sources.register(GOOD, "Reddit privacy", "searx vs searxng")
                return f"Search results\n{self.sources.label(ref)}\n    searx vs searxng"
            if name == "fetch_url":
                return "Content of [S1]: SearXNG is a privacy-respecting metasearch engine."
            return "RESULT"
        async def aclose(self): pass

    box = TurboBox()

    # scripted scout: search -> fetch by id -> final citing [S1] but ALSO writing a WRONG url
    turns = iter([
        '<tool>{"name":"web_search","arguments":{"query":"searxng"}}</tool>',
        '<tool>{"name":"fetch_url","arguments":{"id":"S1"}}</tool>',
        # note the mangled URL in prose (dropped ?tl=ja, truncated path) — must be IGNORED
        "<final>SearXNG is a privacy metasearch engine [S1].\n"
        "Sources:\n- https://www.reddit.com/r/privacy/comments/18w6/which [S1]</final>",
    ])
    async def fake_chat(self, messages, **kw):
        return next(turns)

    orig = LLMClient.chat
    LLMClient.chat = fake_chat
    try:
        cfg = Config(); cfg.worker_max_steps = 5; cfg.supervise = False
        llm = LLMClient("http://x/v1", "m", "EMPTY", 10)
        res = asyncio.run(worker.run_ashigaru(llm, box, "what is searxng", cfg))
    finally:
        LLMClient.chat = orig

    # the returned sources are the VERBATIM registry URL — not the model's mangled one
    assert res.sources == [GOOD], res.sources
    # the harness rewrote the report's Sources block to the exact URL
    assert f"- [S1] {GOOD}" in res.findings, res.findings
    # the model's corrupted URL did not survive into the canonical source list
    assert "18w6/which" not in "".join(res.sources), res.findings
    print("✓ worker turbo: cites [S1] -> verbatim URL resolved, mangled model URL ignored")


if __name__ == "__main__":
    test_registry()
    test_worker_turbo_resolves_and_ignores_model_urls()
    print("\nALL SOURCE TESTS PASSED ✅")
