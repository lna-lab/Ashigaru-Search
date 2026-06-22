"""web_search throttle — offline check that the shared SearXNG min-interval spaces the fleet's
concurrent calls. Never touches the network (the httpx client is stubbed).

Run:  PYTHONPATH=. python tests/test_web_throttle.py
"""
import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ashigaru.config import Config            # noqa: E402
from ashigaru.tools.web import make_web_tools  # noqa: E402


class _Resp:
    def raise_for_status(self):
        pass

    def json(self):
        return {"results": []}


class _FakeClient:
    """Records the monotonic time of every GET so we can measure the spacing."""

    def __init__(self):
        self.calls = []

    async def get(self, url, params=None, **kw):
        self.calls.append(time.monotonic())
        return _Resp()


def _web_search(min_interval):
    cfg = Config()
    cfg.searxng_min_interval_s = min_interval
    client = _FakeClient()
    tools = {t.name: t for t in make_web_tools(cfg, client)}  # sources=None → legacy URL path
    return tools["web_search"], client


async def _burst(ws, n):
    await asyncio.gather(*[ws.run({"query": f"q{i}"}) for i in range(n)])


def main():
    print("\n── throttle OFF (default 0) ──")
    ws, client = _web_search(0.0)
    t0 = time.monotonic()
    asyncio.run(_burst(ws, 5))
    dt = time.monotonic() - t0
    assert dt < 0.15, f"unthrottled burst should be near-instant, took {dt:.2f}s"
    print(f"  ✓ 5 concurrent calls in {dt * 1000:.0f}ms — no spacing when off")

    print("\n── throttle 0.2s/call (Ken's setting: ~5 req/s) ──")
    ws, client = _web_search(0.2)
    t0 = time.monotonic()
    asyncio.run(_burst(ws, 5))
    dt = time.monotonic() - t0
    gaps = [b - a for a, b in zip(client.calls, client.calls[1:])]
    assert dt >= 0.8, f"5 calls @0.2s should take >=0.8s, took {dt:.2f}s"
    assert all(g >= 0.19 for g in gaps), f"consecutive GETs must be ~0.2s apart, got {gaps}"
    print(f"  ✓ 5 calls took {dt:.2f}s; gaps={[round(g, 2) for g in gaps]} — human-paced")

    print("\n✅ web throttle OK")


if __name__ == "__main__":
    main()
