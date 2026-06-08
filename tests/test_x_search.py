"""x_search — offline verification of the POLITE HUMAN-PROXY governor + the [Sn] contract.

Never touches X (the backend is stubbed) — not touching X in a test is itself the good-guest theme.
Run:  PYTHONPATH=. python tests/test_x_search.py
"""
import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ashigaru.tools.x_search as xs          # noqa: E402
from ashigaru.config import Config            # noqa: E402
from ashigaru.sources import SourceRegistry   # noqa: E402


def _cfg(**over):
    c = Config()
    c.x_search_enabled = True
    c.x_backend = "twscrape"
    c.x_max_results = 15
    c.x_min_interval_s = 0.0
    c.x_jitter_s = 0.0
    c.x_max_per_hour = 50
    for k, v in over.items():
        setattr(c, k, v)
    return c


def main():
    print("\n── Governor: courteous hourly ceiling ──")
    g = xs.Governor(min_interval_s=0.0, jitter_s=0.0, max_per_hour=3)
    for _ in range(3):
        ok, _ = g.check_hourly(); assert ok; g.record()
    ok, wait = g.check_hourly()
    assert not ok and wait > 0, (ok, wait)
    print(f"  ✓ after 3/3 used, 4th is gated (wait ~{int(wait)}s) — stays under the cap")

    print("\n── Governor: human-paced spacing (+jitter) ──")
    g2 = xs.Governor(min_interval_s=0.15, jitter_s=0.0, max_per_hour=50)
    async def two_calls():
        await g2.wait_turn(); g2.record()
        t0 = time.monotonic()
        await g2.wait_turn(); g2.record()     # must wait ~0.15s
        return time.monotonic() - t0
    elapsed = asyncio.run(two_calls())
    assert elapsed >= 0.13, elapsed
    print(f"  ✓ back-to-back searches spaced by {elapsed:.2f}s (≥ min interval) — never bursts")

    print("\n── x_search: [Sn] contract + human-scope cap (backend stubbed, X untouched) ──")
    seen_limits = []
    async def fake(cfg, q, limit):
        seen_limits.append(limit)
        return [{"url": f"https://x.com/Tono_Ken3/status/{i}", "title": "@Tono_Ken3",
                 "snippet": f"post {i} 足軽 local LLM freedom"} for i in range(1, limit + 1)]
    xs._twscrape_search = fake
    src = SourceRegistry()
    x = xs.make_x_search_tools(_cfg(), None, src)[0]
    out = asyncio.run(x.run({"query": "local llm", "num": 3}))
    assert "[S1]" in out and "@Tono_Ken3" in out, out
    assert src.resolve("S1") == "https://x.com/Tono_Ken3/status/1", src.resolve("S1")
    assert seen_limits[-1] == 3
    print("  ✓ emits [S1] @user; registry resolves S1 → real permalink; num=3 honored")

    asyncio.run(x.run({"query": "anything", "num": 100}))
    assert seen_limits[-1] == 15, seen_limits
    print(f"  ✓ a request for num=100 is hard-capped to x_max_results=15 (human scope)")

    print("\n── x_search: graceful, never-a-nuisance failure modes ──")
    async def missing(cfg, q, limit):
        raise RuntimeError("twscrape not installed — pip install twscrape ...")
    xs._twscrape_search = missing
    out = asyncio.run(xs.make_x_search_tools(_cfg(), None, SourceRegistry())[0].run({"query": "q"}))
    assert out.startswith("ERROR: x_search unavailable"), out
    print("  ✓ no account/lib → clear actionable ERROR (not a crash)")

    async def boom(cfg, q, limit):
        raise ValueError("rate-limit-ish")
    xs._twscrape_search = boom
    out = asyncio.run(xs.make_x_search_tools(_cfg(), None, SourceRegistry())[0].run({"query": "q"}))
    assert "backed off" in out, out
    print("  ✓ any snag → back off + suggest web_search/蔵 (degrade, never hammer)")

    out = asyncio.run(xs.make_x_search_tools(_cfg(x_backend="nitter"), None, SourceRegistry())[0].run({"query": "q"}))
    assert "not implemented" in out, out
    print("  ✓ unknown backend → honest 'not implemented' (computeruse is the planned last resort)")

    print("\n── build_toolbox wires x_search only when enabled ──")
    from ashigaru.registry import build_toolbox
    box_off = build_toolbox(Config())                       # default OFF
    box_on = build_toolbox(_cfg())                          # enabled
    names_off = list(box_off._tools.keys())
    names_on = list(box_on._tools.keys())
    asyncio.run(box_off.aclose()); asyncio.run(box_on.aclose())
    assert "x_search" not in names_off, names_off
    assert "x_search" in names_on, names_on
    print(f"  ✓ OFF → {names_off};  ON → x_search present")

    print("\n✅ x_search verified offline — polite human-proxy governor + [Sn] contract intact, X untouched.\n")


if __name__ == "__main__":
    main()
