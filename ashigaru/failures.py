"""Failure taxonomy + recovery recipes — deterministic, recover-once-then-escalate fault handling.

Clean-room port of Lna-Lab AIOS's `core/failure_taxonomy.py` (itself after claw-code's
"recovery before escalation"). Turns ad-hoc try/except into: classify the error by pattern →
try ONE recovery recipe → escalate. Pure pattern matching, no LLM. Log classified failures so a
later instinct-distiller can learn from recurring failure→recovery chains.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from enum import Enum

log = logging.getLogger("ashigaru.failures")


class FailureClass(str, Enum):
    MODEL_TIMEOUT = "model_timeout"
    MODEL_REFUSAL = "model_refusal"
    TOOL_ERROR = "tool_error"
    SEARCH_FAILURE = "search_failure"      # SearXNG / web search
    PARSE_ERROR = "parse_error"            # JSON/XML parse of model output
    OOM = "oom"
    SERVICE_DOWN = "service_down"          # a backend (local 口 / SearXNG) is unreachable
    UNKNOWN = "unknown"


@dataclass
class Failure:
    cls: FailureClass
    message: str
    source: str = ""
    recoverable: bool = True
    recovery_attempted: bool = False
    recovery_succeeded: bool = False

    def to_dict(self) -> dict:
        return {"class": self.cls.value, "message": self.message, "source": self.source,
                "recoverable": self.recoverable, "recovery_attempted": self.recovery_attempted,
                "recovery_succeeded": self.recovery_succeeded}


def classify(error: Exception | str, source: str = "") -> Failure:
    """Classify an error into a failure class by pattern (instant, mechanical)."""
    msg = str(error).lower()
    if "timeout" in msg or "timed out" in msg:
        return Failure(FailureClass.MODEL_TIMEOUT, str(error), source)
    # SERVICE_DOWN before REFUSAL — "connection refused" is infra, not a model refusal.
    if "connection" in msg or "refused" in msg or "unreachable" in msg or "503" in msg or "502" in msg:
        return Failure(FailureClass.SERVICE_DOWN, str(error), source)
    if "refus" in msg or "cannot assist" in msg or "i'm sorry" in msg or "i cannot" in msg:
        return Failure(FailureClass.MODEL_REFUSAL, str(error), source)
    if "json" in msg and ("decode" in msg or "parse" in msg or "expecting" in msg):
        return Failure(FailureClass.PARSE_ERROR, str(error), source)
    if "out of memory" in msg or "oom" in msg or "cuda" in msg:
        return Failure(FailureClass.OOM, str(error), source, recoverable=False)
    if "searx" in msg or "search" in msg:
        return Failure(FailureClass.SEARCH_FAILURE, str(error), source)
    if "tool" in msg or "command" in msg or "permission" in msg:
        return Failure(FailureClass.TOOL_ERROR, str(error), source)
    return Failure(FailureClass.UNKNOWN, str(error), source)


# Recovery recipes — tried once before escalation. Recipes signal/wait; the CALLER re-tries.
_RECIPES = {}


def recovery_for(cls: FailureClass):
    def deco(fn):
        _RECIPES[cls] = fn
        return fn
    return deco


async def attempt_recovery(failure: Failure) -> Failure:
    """Try the one registered recovery for this failure class. Recover once, then escalate."""
    if not failure.recoverable:
        log.warning("not recoverable: %s — %s", failure.cls.value, failure.message)
        return failure
    recipe = _RECIPES.get(failure.cls)
    if recipe is None:
        return failure
    failure.recovery_attempted = True
    try:
        await recipe(failure)
        failure.recovery_succeeded = True
    except Exception as e:  # a recovery that itself fails just escalates
        log.warning("recovery failed: %s — %s", failure.cls.value, e)
    return failure


@recovery_for(FailureClass.MODEL_TIMEOUT)
async def _r_timeout(_f):
    await asyncio.sleep(8)            # give the local 口 a beat to catch up; caller retries


@recovery_for(FailureClass.SERVICE_DOWN)
async def _r_service(_f):
    await asyncio.sleep(5)            # a brief wait in case the backend is restarting


@recovery_for(FailureClass.SEARCH_FAILURE)
async def _r_search(_f):
    log.info("search recovery: honor the SearXNG throttle and fall back to local doc_search")


@recovery_for(FailureClass.PARSE_ERROR)
async def _r_parse(_f):
    log.info("parse recovery: run the JSON-salvage path before retrying")
