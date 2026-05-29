"""CLI:  ashigaru "your research question"   [--json] [--quiet] [-k N]"""
from __future__ import annotations
import argparse
import asyncio
import json
import sys

from .config import Config
from .orchestrator import research

# ANSI dims for progress (stderr)
_D, _G, _Y, _C, _R = "\033[2m", "\033[32m", "\033[33m", "\033[36m", "\033[0m"


def _make_reporter(quiet: bool):
    def report(stage: str, info: dict):
        if quiet:
            return
        if stage == "plan":
            req = f" (requested {info['requested']} scouts)" if info.get("requested") else ""
            print(f"{_C}Commander: planned {len(info['subquestions'])} sub-questions{req}:{_R}", file=sys.stderr)
            for i, q in enumerate(info["subquestions"]):
                print(f"  {_D}{i+1}. {q}{_R}", file=sys.stderr)
        elif stage == "worker_start":
            print(f"{_Y}Ashigaru #{info['index']+1} ▶ {info['task'][:70]}{_R}", file=sys.stderr)
        elif stage == "worker_tool":
            a = info.get("args", {})
            arg = a.get("query") or a.get("url") or a.get("id") or ""
            print(f"  {_D}Ashigaru #{info['index']+1} step{info['step']} {info['tool']}({str(arg)[:60]}){_R}", file=sys.stderr)
        elif stage == "worker_done":
            tag = " (forced)" if info.get("forced") else ""
            print(f"{_G}Ashigaru #{info['index']+1} ✓ done in {info['steps']} steps{tag}{_R}", file=sys.stderr)
        elif stage == "synthesize":
            print(f"{_C}Commander: synthesizing {info['workers']} scout reports…{_R}", file=sys.stderr)
    return report


def main():
    ap = argparse.ArgumentParser(
        prog="ashigaru", description="Ashigaru-Search: local search-agent fleet.",
        epilog='Tip: lead the question with a count or density tag (half-width + space), '
               'overrides -k: "3 ..." = 3 scouts; "S ..."/"M ..."/"L ..." = 10%%/50%%/100%% '
               'of the fleet (ASHIGARU_FLEET_SIZE, default 10 -> S=1, M=5, L=10).')
    ap.add_argument("question", help='research question; a leading "<n> " or "S|M|L " sets the scout count')
    ap.add_argument("--json", action="store_true", help="emit full result as JSON")
    ap.add_argument("--quiet", action="store_true", help="no progress on stderr")
    ap.add_argument("-k", "--subquestions", type=int, default=None, help="max sub-questions (scout count)")
    a = ap.parse_args()

    cfg = Config()
    if a.subquestions:
        cfg.max_subquestions = a.subquestions

    res = asyncio.run(research(a.question, cfg, on_event=_make_reporter(a.quiet)))

    if a.json:
        print(json.dumps({
            "question": res.question,
            "answer": res.answer,
            "subquestions": res.subquestions,
            "sources": res.sources,
            "scouts": [{"task": w.task, "findings": w.findings, "sources": w.sources, "steps": w.steps}
                       for w in res.workers],
        }, ensure_ascii=False, indent=2))
    else:
        print("\n" + "=" * 70 + f"\n{res.question}\n" + "=" * 70)
        print(res.answer)
        if res.sources:
            print("\n— sources —")
            for s in res.sources:
                print(f"  • {s}")


if __name__ == "__main__":
    main()
