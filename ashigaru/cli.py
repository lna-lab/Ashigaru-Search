"""CLI:  ashigaru "your research question"   [--json] [--quiet] [-k N]"""
from __future__ import annotations
import argparse
import asyncio
import json
import os
import sys

import httpx

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
        elif stage == "worker_checkin":
            act = info["action"]
            if act == "regroup":
                print(f"  {_Y}Ashigaru #{info['index']+1} ⟲ Commander: regroup → {info['focus']}{_R}", file=sys.stderr)
            elif act == "return":
                print(f"  {_Y}Ashigaru #{info['index']+1} ⏎ Commander: return & report{_R}", file=sys.stderr)
            else:
                print(f"  {_D}Ashigaru #{info['index']+1} ↪ Commander: continue digging{_R}", file=sys.stderr)
        elif stage == "worker_done":
            tag = " (recalled)" if info.get("recalled") else (" (forced)" if info.get("forced") else "")
            print(f"{_G}Ashigaru #{info['index']+1} ✓ done in {info['steps']} steps{tag}{_R}", file=sys.stderr)
        elif stage == "escalate":
            print(f"{_Y}Commander: thin results ({info['good']}/{info['needed']} solid) "
                  f"→ escalating {info['from']}→{info['to']} scouts, redo ↑{_R}", file=sys.stderr)
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

    # a scroll path set but unusable is otherwise silently ignored — say so, don't mislead
    if cfg.emaki_tree and not cfg.has_emaki:
        if not os.path.isdir(cfg.emaki_tree):
            print(f"[warn] ASHIGARU_EMAKI={cfg.emaki_tree} does not exist — scroll navigation OFF "
                  f"(falling back to web/BM25).", file=sys.stderr)
        else:
            print(f"[warn] ASHIGARU_EMAKI={cfg.emaki_tree} has no tree.json — not a built scroll. "
                  f"Build one first: ashigaru-emaki <corpus> <out>. Navigation OFF.", file=sys.stderr)

    try:
        res = asyncio.run(research(a.question, cfg, on_event=_make_reporter(a.quiet)))
    except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout, httpx.PoolTimeout,
            httpx.HTTPStatusError) as e:
        lines = [f"\nCouldn't reach a required service ({type(e).__name__}).",
                 f"  • LLM fleet expected at: {cfg.worker_base_url}  (is your vLLM / llama.cpp server up?)",
                 f"  • SearXNG expected at:   {cfg.searxng_url}  "
                 f"(try: docker compose -f docker/docker-compose.yml up -d searxng)"]
        if isinstance(e, httpx.HTTPStatusError) and e.response is not None \
                and e.response.status_code in (400, 404):
            lines.append(f"  • {e.response.status_code} from the LLM — check ASHIGARU_WORKER_MODEL "
                         f"matches your server's --served-model-name and the base URL ends in /v1.")
        lines.append("  No infra handy? Try the offline demo: "
                     "ashigaru-emaki . ./demo_scroll --no-llm --graph")
        print("\n".join(lines), file=sys.stderr)
        sys.exit(1)

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
