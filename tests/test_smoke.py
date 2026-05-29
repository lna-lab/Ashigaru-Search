"""No-network smoke tests: tool-call parser + a fully-mocked orchestrator run.
Run:  PYTHONPATH=. python3 tests/test_smoke.py
"""
import asyncio

from ashigaru.toolproto import parse_action, strip_think


def test_parser():
    a = parse_action('<tool>{"name":"web_search","arguments":{"query":"x"}}</tool>')
    assert a.kind == "tool" and a.name == "web_search" and a.args["query"] == "x", a

    a = parse_action('<think>let me search</think>\n<tool>{"name":"fetch_url","arguments":{"url":"u"}}</tool>')
    assert a.kind == "tool" and a.name == "fetch_url" and a.args["url"] == "u", a

    a = parse_action('```json\n{"tool":"web_search","arguments":{"query":"y"}}\n```')
    assert a.kind == "tool" and a.name == "web_search" and a.args["query"] == "y", a

    a = parse_action('<|tool_call_start|>[web_search(query="z", num=3)]<|tool_call_end|>')
    assert a.kind == "tool" and a.name == "web_search" and a.args["query"] == "z" and a.args["num"] == 3, a

    a = parse_action("<final>The answer is 42.</final>")
    assert a.kind == "final" and a.text == "The answer is 42.", a

    a = parse_action("<think>musing</think>\nJust a plain answer with no tags.")
    assert a.kind == "final" and "plain answer" in a.text and "musing" not in a.text, a

    assert strip_think("<think>a</think>visible") == "visible"
    print("✓ parser: 7/7 cases pass")


def test_scout_count():
    from ashigaru.orchestrator import parse_scout_count as p
    cases = [
        # (question, default, fleet) -> (n, cleaned, explicit)
        (("3 compare X and Y", 6, 10), (3, "compare X and Y", True)),
        (("5 何々について調べたい", 6, 10), (5, "何々について調べたい", True)),
        (("99 too many", 6, 10), (12, "too many", True)),           # clamp to 12
        (("S quick check", 6, 10), (1, "quick check", True)),       # 10% of 10 = 1
        (("M latest on X", 6, 10), (5, "latest on X", True)),       # 50% of 10 = 5
        (("L deep survey", 6, 10), (10, "deep survey", True)),      # 100% of 10 = 10
        (("l lowercase tag", 6, 10), (10, "lowercase tag", True)),  # case-insensitive
        (("S tiny fleet", 6, 6), (1, "tiny fleet", True)),          # ceil(0.6)=1, min 1
        (("M big fleet", 6, 16), (8, "big fleet", True)),           # ceil(8.0)=8
        (("plain question", 6, 10), (6, "plain question", False)),  # no tag -> default
        (("5G networks", 6, 10), (6, "5G networks", False)),        # no space -> not a tag
    ]
    for (args, exp) in cases:
        got = p(*args)
        assert got == exp, f"{args} -> {got} (expected {exp})"
    print(f"✓ scout-count/SML: {len(cases)}/{len(cases)} cases pass")


def test_orchestrator_mocked():
    import ashigaru.orchestrator as orch
    import ashigaru.worker as worker
    from ashigaru.llm import LLMClient

    # --- mock the LLM: script plan / worker-step / synth by prompt content ---
    async def fake_chat(self, messages, **kw):
        sys = messages[0]["content"]
        if "JSON array" in sys:                              # planner
            return '["What is X?", "How does X compare to Y?"]'
        if "scouts each investigated" in sys:                # synthesizer (also says "Ashigaru scouts")
            return "SYNTHESIZED ANSWER about X and Y.\nSources:\n- https://example.com/x"
        if "Ashigaru" in sys:                                # worker scout
            has_result = any("<tool_result" in m.get("content", "") for m in messages)
            if not has_result:
                return '<think>search first</think><tool>{"name":"web_search","arguments":{"query":"X"}}</tool>'
            return "<final>X is a thing. Sources:\n- https://example.com/x — defines X</final>"
        return "<final>fallback</final>"

    # --- mock the toolbox (no network) ---
    class FakeBox:
        def render_docs(self): return "- web_search: ...\n- fetch_url: ..."
        async def run(self, name, args): return f"RESULT[{name}]: canned evidence for {args}"
        async def aclose(self): pass

    orig_chat = LLMClient.chat
    orig_build = orch.build_toolbox
    LLMClient.chat = fake_chat
    orch.build_toolbox = lambda cfg: FakeBox()
    try:
        from ashigaru import Config
        cfg = Config(); cfg.max_subquestions = 4; cfg.worker_max_steps = 3
        res = asyncio.run(orch.research("Tell me about X.", cfg))
    finally:
        LLMClient.chat = orig_chat
        orch.build_toolbox = orig_build

    assert len(res.subquestions) == 2, res.subquestions
    assert len(res.workers) == 2, res.workers
    assert all(w.findings for w in res.workers), res.workers
    assert "SYNTHESIZED ANSWER" in res.answer, res.answer
    assert "https://example.com/x" in res.sources, res.sources
    print(f"✓ orchestrator (mocked): {len(res.subquestions)} sub-Qs, {len(res.workers)} scouts, "
          f"{len(res.sources)} source(s), synth OK")


if __name__ == "__main__":
    test_parser()
    test_scout_count()
    test_orchestrator_mocked()
    print("\nALL SMOKE TESTS PASSED ✅")
