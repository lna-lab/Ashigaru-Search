"""recall_commander — the idle Commander reads our OWN 蔵 (memory) concurrent with the web
scouts, and its findings join the synthesis pool. Fully mocked (no network, no real LLM).

Run:  PYTHONPATH=. python tests/test_recall_commander.py
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ashigaru.orchestrator as orch          # noqa: E402
from ashigaru import Config                   # noqa: E402
from ashigaru.llm import LLMClient            # noqa: E402


def _run(recall_commander: bool, has_doc_search: bool):
    async def fake_chat(self, messages, **kw):
        sysmsg = messages[0]["content"]
        if "JSON array" in sysmsg:                              # planner
            return '["What is X?", "How does X compare to Y?"]'
        if "scouts each investigated" in sysmsg:                # synthesizer
            return "SYNTHESIZED ANSWER about X.\nSources:\n- https://example.com/x"
        if "蔵" in sysmsg:                                      # the 蔵-recall pass (memory)
            return ("<final>From our own notes we already studied X in 2026-05; prior "
                    "conclusion still holds. Sources:\n- journal/private#42 — prior study</final>")
        if "Ashigaru" in sysmsg:                                # a web scout
            if not any("tool_result" in m.get("content", "") for m in messages):
                return '<tool>{"name":"web_search","arguments":{"query":"X"}}</tool>'
            return "<final>X is a thing. Sources:\n- https://example.com/x — defines X</final>"
        return "<final>fallback</final>"

    class FakeBox:
        start_hint = "web first"

        def render_docs(self):
            return "- web_search: ...\n- doc_search: ..."

        def get(self, name):
            ok = {"web_search"} | ({"doc_search"} if has_doc_search else set())
            return (lambda *a, **k: None) if name in ok else None  # truthy stand-in tool

        async def run(self, name, args):
            return f"RESULT[{name}]: canned evidence for {args}"

        async def aclose(self):
            pass

    orig_chat, orig_build = LLMClient.chat, orch.build_toolbox
    LLMClient.chat = fake_chat
    orch.build_toolbox = lambda cfg: FakeBox()
    try:
        cfg = Config()
        cfg.max_subquestions = 4
        cfg.worker_max_steps = 3
        cfg.recall_commander = recall_commander
        return asyncio.run(orch.research("Tell me about X.", cfg))
    finally:
        LLMClient.chat = orig_chat
        orch.build_toolbox = orig_build


def main():
    print("\n── recall_commander OFF (default) ──")
    res = _run(False, has_doc_search=True)
    assert len(res.workers) == 2, res.workers
    assert not any("蔵" in w.task for w in res.workers)
    print(f"  ✓ {len(res.workers)} web scouts, no 蔵 pass")

    print("\n── recall_commander ON + toolbox has doc_search ──")
    res = _run(True, has_doc_search=True)
    kura = [w for w in res.workers if "蔵" in w.task]
    assert len(res.workers) == 3, res.workers
    assert len(kura) == 1, res.workers
    assert "already studied X" in kura[0].findings, kura[0].findings
    assert "SYNTHESIZED ANSWER" in res.answer, res.answer
    print(f"  ✓ {len(res.workers)} workers incl. the 蔵 pass; memory findings reach synthesis")

    print("\n── recall_commander ON but NO local tool → no-op ──")
    res = _run(True, has_doc_search=False)
    assert len(res.workers) == 2, res.workers
    assert not any("蔵" in w.task for w in res.workers)
    print("  ✓ no doc_search/graph tool → 蔵-recall is a no-op (web-only)")

    print("\n✅ recall_commander OK")


if __name__ == "__main__":
    main()
