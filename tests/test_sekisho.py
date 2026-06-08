"""関所 (sekisho) — one full turn of the loop, as a test AND a readable demo.

    選別 (the gate sieves by kind) → 蓄積 (only verified assets enter) →
    上書き阻止 (a failing update can't poison a good asset) → 想起 (genre-routed recall)

Run:  PYTHONPATH=. python tests/test_sekisho.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ashigaru.sekisho import Sekisho, Asset  # noqa: E402


SMA_V1 = "def sma(xs, n):\n    return [sum(xs[i-n+1:i+1])/n for i in range(n-1, len(xs))]"
SMA_V2 = ("def sma(xs, n):\n    if n <= 0 or n > len(xs):\n        return []\n"
          "    return [sum(xs[i-n+1:i+1])/n for i in range(n-1, len(xs))]")   # improved: guards n
SMA_BAD = "def sma(xs, n):\n    return [sum(xs[i:i+n])/n for i in range(len(xs))]"  # off-by-one
SMA_TEST = "assert sma([1,2,3,4], 2) == [1.5, 2.5, 3.5], sma([1,2,3,4], 2)"


def main():
    fd, path = tempfile.mkstemp(suffix="_sekisho_demo.jsonl")
    os.close(fd); os.remove(path)
    k = Sekisho(path)

    def submit(a, **kw):
        v = k.submit(a, **kw)
        flag = {"pass": "✓ PASS  ", "reject": "✗ REJECT", "duplicate": "= DUP   "}[v.status]
        extra = f"  ({v.reason})" if v.reason else ("  (supersedes older)" if v.supersedes else "")
        print(f"  {flag} [{a.realm or '-':<14}] {a.id:<12}{extra}")
        return v

    print("\n── ①選別: assets queue at the 関所 ─────────────────────────────")
    # moving average: simultaneously CODE and TRADE — content on a gradient
    submit(Asset("sma", "code", SMA_V1, realm="code,trade", source="fleet", tests=SMA_TEST))
    # a life-preference about Ken — life-log memory, also touches the trade realm
    submit(Asset("pref-local", "preference", "Ken keeps anything touching trading logic local-only.",
                 realm="life,trade", source="exchange 2026-06-08", why="logic leak = lost edge"))
    submit(Asset("fact-bls", "fact", "BLS greedy temp=0 is its sweet spot for code generation.",
                 realm="knowledge", source="bench 2026-06-08"))
    # things the 関所 must turn away:
    submit(Asset("buggy-rsi", "code", SMA_BAD, realm="code,trade", source="fleet", tests=SMA_TEST))
    submit(Asset("untested", "code", SMA_V1, realm="code", source="fleet"))             # no test
    submit(Asset("guess", "note", "Ken probably likes dark mode.", realm="life"))       # no source/why

    live_ids = {r["id"] for r in k.live()}
    assert live_ids == {"sma", "pref-local", "fact-bls"}, live_ids
    print(f"  → 蔵 now holds only the verified: {sorted(live_ids)}")

    print("\n── ④戻す + 上書き阻止: a better version supersedes; a broken one can't ──")
    submit(Asset("sma", "code", SMA_V2, realm="code,trade", source="fleet", tests=SMA_TEST))  # v2 passes
    assert k.live("code")[0]["text"] == SMA_V2
    print("  → sma upgraded to v2 (guards n)")
    submit(Asset("sma", "code", SMA_BAD, realm="code,trade", source="fleet", tests=SMA_TEST))  # v3 broken
    assert k.live("code")[0]["text"] == SMA_V2, "poison got through!"
    print("  → broken v3 rejected; the good v2 still stands (poison-proof)")
    assert submit(Asset("sma", "code", SMA_V2, realm="code,trade", source="fleet", tests=SMA_TEST)).status == "duplicate"

    print("\n── ③想起: same 蔵, different context → different ranking ──────────")
    def show(label, ctx):
        rows = k.recall(ctx)
        print(f"  {label:<26} → " + " · ".join(r["id"] for r in rows))
        return [r["id"] for r in rows]

    coding = show("coding task (SAZANAMI)", {"task": "coding"})
    macbook = show("just talking (MacBook)", {"node": "macbook"})
    trading = show("trading task", {"task": "trading"})

    assert coding[0] == "sma", coding                       # code surfaces first when coding
    assert macbook[0] == "pref-local", macbook              # from his Mac, the life memory leads
    assert {"sma", "pref-local"} <= set(trading[:2]), trading  # gradient: trade pulls BOTH code & life

    print("\n✅ 選別→蓄積→上書き阻止→ジャンル想起 — one full turn, all invariants held.\n")


if __name__ == "__main__":
    main()
