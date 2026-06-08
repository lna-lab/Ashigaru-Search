"""関所 (sekisho) — the gate at the mouth of the 蔵.

Nothing becomes durable memory unless it *earns* its place. The 関所 is a 蔵-agnostic
quality gate: it sieves submitted **token-assets**, lets only the verified ones into the
storehouse, retires older versions when a better one passes, and keeps a full audit trail.
It guards every 蔵 the same way — the code 蔵, the knowledge 蔵, and the self/life 蔵 (the
journal) — so a companion's memory stays *true* instead of drifting into free-floating belief.

    選別 (sieve) → 蓄積 (store) → 想起 (recall) → ④ 検証で生まれた良いものを蔵へ戻す

The whole point is the gate criterion is **objective per kind**, not a matter of taste:

    policy "code"     : the asset's tests must pass in a sandboxed run        (verification)
    policy "grounded" : the asset must carry a `source`                       (provenance)
    policy "reasoned" : the asset must carry a `source` AND a `why`           (auditable judgment)

Lived memory (an event, a preference learned about someone, a decision in their life) is
`grounded`/`reasoned`: it must trace back to what actually happened — never confabulated.
That is the same rule the journal enforces with `sources`, generalised to any 蔵.

Recall is poison-proof: `live()` returns only the *latest passing* version per id, so a
superseded chunk is never served, and a failing update can never overwrite a good asset.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field


# kind -> gate policy. Add new kinds here; the policy decides what "good" means.
POLICY = {
    "code": "code",            # provable: ships only if its tests pass
    "fact": "grounded",        # a claim about the world: must cite a source
    "event": "grounded",       # something that happened (lived memory): must trace to when/where
    "memory": "grounded",      # a remembered observation: must trace to its origin
    "note": "reasoned",        # a written note: must say where it came from AND why it matters
    "decision": "reasoned",    # a choice made: must record its rationale
    "preference": "reasoned",  # a learned preference (e.g. about Ken): grounded + why, not a guess
}
DEFAULT_POLICY = "reasoned"


# ---- genre routing (③想起 side) -----------------------------------------
# Content is a GRADIENT, not discrete bins: an asset's `realm` may name several genres
# ("code,trade"), and routing is SOFT — context PREFERS some realms but never excludes the
# rest, because a life-companion cross-pollinates (a life note can matter mid-coding).
NODE_REALMS = {                              # which body she's reached from is a context cue
    "wakashio": ["dialogue", "life"],        # the Mac / her room / the self gateway
    "macbook":  ["dialogue", "life"],
    "sazanami": ["code", "knowledge"],       # the 大将 GPU box
    "shiosai":  ["code", "knowledge"],       # the 足軽 fleet box
}
TASK_REALMS = {
    "coding":   ["code", "knowledge"],
    "trading":  ["trade", "code", "knowledge"],
    "chat":     ["dialogue", "life"],
    "research": ["knowledge", "dialogue"],
    "lifelog":  ["life", "dialogue"],
}


def _realms_of(rec: dict) -> set:
    """An asset can live across several genres (the gradient). '' means realm-less (always eligible)."""
    return {r.strip() for r in (rec.get("realm") or "").split(",") if r.strip()}


def route(context: dict | None) -> list[str]:
    """Context cues -> preferred realms, most-relevant first (deduped). Cues: `task`, `node`."""
    context = context or {}
    pref: list[str] = []
    for src in (TASK_REALMS.get((context.get("task") or "").lower(), []),
                NODE_REALMS.get((context.get("node") or "").lower(), [])):
        for r in src:
            if r not in pref:
                pref.append(r)
    return pref


@dataclass
class Asset:
    """A unit that wants into the 蔵."""
    id: str                              # stable identity — a newer version with the same id supersedes the old
    kind: str                            # asset TYPE (see POLICY): code / fact / event / note / preference ...
    text: str                            # the content itself (the token-asset)
    realm: str = ""                      # GENRE/domain, orthogonal to kind: "code"|"trade"|"dialogue"|"life"|"knowledge"
    source: str = ""                     # provenance: a file path, journal#n, a URL, "exchange 2026-06-08", a model name
    why: str = ""                        # rationale / significance (required by the "reasoned" policy)
    tests: str = ""                      # python asserts that must pass (required by the "code" policy)
    meta: dict = field(default_factory=dict)

    def hash(self) -> str:
        h = hashlib.sha256()
        h.update((self.id + "\x00" + self.text).encode("utf-8"))
        return h.hexdigest()[:16]


@dataclass
class Verdict:
    ok: bool
    status: str          # "pass" | "reject" | "duplicate"
    reason: str = ""
    supersedes: str = "" # the id this new version retires (set when an older live asset shared the id)


def _run_code_tests(code: str, tests: str, timeout: float) -> tuple[bool, str]:
    """Run code+tests in a throwaway subprocess. Pass = exit 0. Basic sandbox (process + timeout
    only); for untrusted code add resource limits / a container. Here the code is our own fleet's."""
    src = code + "\n\n# --- 関所 tests ---\n" + tests + "\n"
    fd, path = tempfile.mkstemp(suffix="_sekisho.py", prefix="kura_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(src)
        try:
            p = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return False, f"timed out after {timeout}s"
        if p.returncode == 0:
            return True, ""
        tail = (p.stderr or p.stdout or "").strip().splitlines()
        return False, (tail[-1] if tail else f"exit {p.returncode}")
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def inspect(asset: Asset, *, require_tests: bool = True, timeout: float = 8.0) -> Verdict:
    """Pure judgment — decides pass/reject for one asset, touches no storage."""
    if not (asset.id or "").strip():
        return Verdict(False, "reject", "no id (can't supersede/dedup without one)")
    if not (asset.text or "").strip():
        return Verdict(False, "reject", "empty text")

    policy = POLICY.get(asset.kind, DEFAULT_POLICY)

    if policy == "code":
        if not (asset.tests or "").strip():
            if require_tests:
                return Verdict(False, "reject", "code with no test can't prove itself")
            return Verdict(True, "pass", "admitted UNVERIFIED (no test; require_tests=False)")
        ok, err = _run_code_tests(asset.text, asset.tests, timeout)
        return Verdict(ok, "pass" if ok else "reject", "" if ok else f"tests failed: {err}")

    # provenance-based policies: keep lived memory honest, never free-floating belief
    if not (asset.source or "").strip():
        return Verdict(False, "reject", "no source — would be free-floating belief, not grounded memory")
    if policy == "reasoned" and not (asset.why or "").strip():
        return Verdict(False, "reject", "no 'why' — a kept judgment must carry its rationale")
    return Verdict(True, "pass")


class Sekisho:
    """The 関所 + its ledger. Append-only JSONL audit trail on disk; `live()` reconstructs the
    current admitted 蔵 (latest passing version per id). Mirrors JournalStore's crash-safe append."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    # ---- storage (②蓄積) -------------------------------------------------
    def _records(self) -> list[dict]:
        if not os.path.exists(self.path):
            return []
        out: list[dict] = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue                      # drop only the corrupt line
                if isinstance(obj, dict) and obj.get("id"):
                    out.append(obj)
        return out

    def _append(self, rec: dict) -> None:
        line = (json.dumps(rec, ensure_ascii=False) + "\n").encode("utf-8")
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            size = os.fstat(fd).st_size
            if size > 0:
                os.lseek(fd, size - 1, os.SEEK_SET)
                if os.read(fd, 1) != b"\n":
                    os.write(fd, b"\n")
            os.write(fd, line)
            os.fsync(fd)
        finally:
            os.close(fd)

    # ---- the gate (①選別) ------------------------------------------------
    def submit(self, asset: Asset, *, require_tests: bool = True, timeout: float = 8.0,
               t: float | None = None) -> Verdict:
        """Run `asset` through the 関所. On pass it enters the 蔵 (and retires any older same-id
        version); on reject the existing good version is untouched (poison can't overwrite)."""
        records = self._records()

        # content-level dedup: identical (id+text) already admitted -> no churn
        h = asset.hash()
        live_now = self._live_from(records)
        if any(a["hash"] == h for a in live_now.values()):
            return Verdict(True, "duplicate", "identical asset already in the 蔵")

        verdict = inspect(asset, require_tests=require_tests, timeout=timeout)
        if verdict.ok and asset.id in live_now:
            verdict.supersedes = live_now[asset.id]["hash"]

        rec = {
            "n": max((r.get("n", 0) for r in records), default=0) + 1,
            "t": time.time() if t is None else t,
            "hash": h, "id": asset.id, "kind": asset.kind, "realm": asset.realm,
            "status": verdict.status, "reason": verdict.reason,
            "source": asset.source, "why": asset.why, "text": asset.text,
            "supersedes": verdict.supersedes, "meta": asset.meta,
        }
        self._append(rec)
        return verdict

    # ---- recall surface (③想起) -----------------------------------------
    @staticmethod
    def _live_from(records: list[dict]) -> dict[str, dict]:
        """Latest *passing* record per id. Rejects/duplicates don't displace a prior good one,
        so a failing update leaves the last good version standing (poison-proof recall)."""
        live: dict[str, dict] = {}
        for r in records:
            if r.get("status") == "pass":
                live[r["id"]] = r            # later pass for same id supersedes the earlier
        return live

    def live(self, realm: str | None = None) -> list[dict]:
        """The current admitted 蔵 — exactly what recall should see. Optional hard genre filter
        (gradient-aware: matches an asset tagged across several realms)."""
        rows = sorted(self._live_from(self._records()).values(), key=lambda r: r["id"])
        if realm:
            rows = [r for r in rows if realm in _realms_of(r)]
        return rows

    def recall(self, context: dict | None = None, *, strict: bool = False) -> list[dict]:
        """Genre-routed recall. Assets in the context's preferred realms rank first; the rest
        still follow (soft routing — a companion cross-pollinates) unless strict=True. Realm-less
        and multi-realm assets are handled by their best-matching genre."""
        live = self.live()
        pref = route(context)
        if not pref:
            return live

        def rank(r: dict) -> int:
            rs = _realms_of(r)
            hits = [pref.index(p) for p in pref if p in rs]
            return min(hits) if hits else len(pref)   # realm-less / off-genre sort after preferred

        ranked = sorted(live, key=lambda r: (rank(r), r["id"]))
        if strict:
            want = set(pref)
            ranked = [r for r in ranked if want & _realms_of(r)]
        return ranked

    def export_chunks(self, realm: str | None = None) -> list[dict]:
        """Feed for the indexer/絵巻: [{id, source, text}] of only the live, vetted assets.
        Hand these to ashigaru.rag_index / emaki so only good tokens ever get indexed."""
        return [{"id": r["id"], "source": r.get("source", ""), "text": r["text"]} for r in self.live(realm)]

    def export_to_folder(self, folder: str) -> int:
        """Materialise the live 蔵 as one .md per asset (provenance in front-matter) so a normal
        `ashigaru-index <folder>` run indexes the curated set. Returns the file count."""
        os.makedirs(folder, exist_ok=True)
        live = self.live()
        for r in live:
            safe = r["id"].replace("/", "_").replace("\\", "_")
            with open(os.path.join(folder, f"{safe}.md"), "w", encoding="utf-8") as f:
                f.write(f"<!-- source: {r.get('source','')} | kind: {r.get('kind','')} -->\n")
                if r.get("why"):
                    f.write(f"<!-- why: {r['why']} -->\n")
                f.write(r["text"].rstrip() + "\n")
        return len(live)

    def ledger(self) -> list[dict]:
        """Full audit trail — every decision the 関所 ever made, in order."""
        return self._records()
