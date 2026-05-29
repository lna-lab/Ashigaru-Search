"""Model-agnostic tool-call protocol.

Workers/orchestrators talk in a tiny, strict format that *any* instruct model can
follow, while we also transparently accept LFM2.5's native Pythonic tool calls.

A turn is parsed into one Action:
  - ("tool", name, args)   -> the model wants to call a tool
  - ("final", None, text)  -> the model is done; `text` is the answer

Accepted tool-call syntaxes (first match wins):
  1. <tool>{"name": "web_search", "arguments": {"query": "..."}}</tool>     (canonical)
  2. ```json {"tool": "web_search", "arguments": {...}} ```                  (fenced)
  3. <|tool_call_start|>[web_search(query="...")]<|tool_call_end|>           (LFM2.5 Pythonic)
And a final answer:
  - <final> ... </final>   (canonical) OR everything that isn't a tool call.
"""
from __future__ import annotations
import ast
import json
import re
from dataclasses import dataclass, field

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
TOOL_TAG_RE = re.compile(r"<tool>\s*(\{.*?\})\s*</tool>", re.DOTALL | re.IGNORECASE)
FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
FINAL_RE = re.compile(r"<final>\s*(.*?)\s*</final>", re.DOTALL | re.IGNORECASE)
LFM_RE = re.compile(r"<\|tool_call_start\|>\s*\[(.*?)\]\s*<\|tool_call_end\|>", re.DOTALL)
PYCALL_RE = re.compile(r"^\s*([a-zA-Z_]\w*)\s*\((.*)\)\s*$", re.DOTALL)


@dataclass
class Action:
    kind: str               # "tool" | "final"
    name: str | None = None
    args: dict = field(default_factory=dict)
    text: str = ""


def strip_think(text: str) -> str:
    """Drop <think>…</think> CoT (LFM2.5 / R1-style) before parsing the action."""
    out = THINK_RE.sub("", text)
    # also handle an unclosed trailing <think> (truncated generations)
    if "<think>" in out and "</think>" not in out:
        out = out.split("<think>")[0]
    return out.strip()


def _coerce_args(name: str, payload: dict) -> Action:
    args = payload.get("arguments", payload.get("args", {})) or {}
    if not isinstance(args, dict):
        args = {"query": str(args)}
    return Action("tool", name=name, args=args)


def _parse_pythonic(inner: str) -> Action | None:
    m = PYCALL_RE.match(inner.strip().split("\n")[0] if "\n" not in inner else inner.strip())
    m = PYCALL_RE.match(inner.strip())
    if not m:
        return None
    name, argstr = m.group(1), m.group(2).strip()
    args: dict = {}
    if argstr:
        try:
            call = ast.parse(f"_f({argstr})", mode="eval").body  # type: ignore
            for kw in getattr(call, "keywords", []):
                args[kw.arg] = ast.literal_eval(kw.value)
            pos = [ast.literal_eval(a) for a in getattr(call, "args", [])]
            if pos and not args:
                args["query"] = pos[0]
        except Exception:
            args = {"query": argstr.strip("'\"")}
    return Action("tool", name=name, args=args)


def parse_action(text: str) -> Action:
    body = strip_think(text)

    # explicit final marker wins if present
    fm = FINAL_RE.search(body)
    if fm:
        return Action("final", text=fm.group(1).strip())

    for rx in (TOOL_TAG_RE, FENCE_RE):
        m = rx.search(body)
        if m:
            try:
                payload = json.loads(m.group(1))
                name = payload.get("name") or payload.get("tool")
                if name:
                    return _coerce_args(name, payload)
            except Exception:
                pass

    lm = LFM_RE.search(body)
    if lm:
        act = _parse_pythonic(lm.group(1))
        if act:
            return act

    # no tool call -> treat the whole (think-stripped) message as the final answer
    return Action("final", text=body)


def tool_result_message(name: str, result: str) -> dict:
    """How a tool's output is fed back to the model (as a user turn for max
    compatibility across models that lack a native `tool` role parser)."""
    return {"role": "user", "content": f"<tool_result name=\"{name}\">\n{result}\n</tool_result>\n\n"
                                        "Use this. Call another tool the same way, or give your answer in <final>…</final>."}
