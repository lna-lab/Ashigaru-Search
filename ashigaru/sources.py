"""SourceRegistry — the 足軽's "turbo" source handling.

A small (1.2B-class) scout is excellent at grounding its answer in retrieved snippets, but
it is unreliable at *reproducing URLs*: long paths and query strings get truncated or
altered, and some cited sources never get a URL printed at all. The fix is architectural,
not a bigger model — never make the LLM write a URL.

Instead, every search result is registered here under a short, stable reference id (S1, S2,
…). The model only ever cites `[S1]`; the harness re-attaches the **verbatim** URL
afterwards by looking the id back up. This removes both failure modes at once:
  - corruption — there is no long string for the model to mis-copy (it never sees the path)
  - omission   — every `[Sn]` the model cites resolves to its exact source URL

The registry is created once per research run and shared across the whole fleet, so ids are
globally unique and the Commander can resolve any scout's `[Sn]`.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from urllib.parse import urlparse

# matches a citation id: S1, [S1], (S1).  The id must look like S<digits>.
REF_RE = re.compile(r"\[?\(?\bS(\d+)\b\)?\]?")


@dataclass
class Source:
    ref: str            # "S1"
    url: str            # the verbatim source URL (never shown to the model)
    title: str = ""
    snippet: str = ""


@dataclass
class SourceRegistry:
    _by_ref: dict[str, Source] = field(default_factory=dict)
    _by_url: dict[str, str] = field(default_factory=dict)   # url -> ref (dedup)
    _n: int = 0

    # -- registration (called by tools as they surface sources) --
    def register(self, url: str, title: str = "", snippet: str = "") -> str:
        """Register a URL and return its stable ref id (deduped by URL)."""
        url = (url or "").strip()
        if not url:
            return ""
        if url in self._by_url:
            return self._by_url[url]
        self._n += 1
        ref = f"S{self._n}"
        self._by_ref[ref] = Source(ref, url, (title or "").strip(), (snippet or "").strip())
        self._by_url[url] = ref
        return ref

    # -- lookup --
    @staticmethod
    def _norm(ref: str) -> str:
        m = REF_RE.search(str(ref))
        return f"S{m.group(1)}" if m else str(ref).strip().strip("[]()")

    def get(self, ref: str) -> Source | None:
        return self._by_ref.get(self._norm(ref))

    def resolve(self, ref: str) -> str | None:
        s = self.get(ref)
        return s.url if s else None

    def refs_in(self, text: str) -> list[str]:
        """Ordered, de-duplicated list of registered refs cited in `text`."""
        seen: set[str] = set()
        out: list[str] = []
        for m in REF_RE.finditer(text or ""):
            ref = f"S{m.group(1)}"
            if ref in self._by_ref and ref not in seen:
                seen.add(ref)
                out.append(ref)
        return out

    def verbatim_sources(self, refs) -> list[str]:
        """Resolve cited refs to their exact URLs (ordered, deduped)."""
        out: list[str] = []
        for r in refs:
            u = self.resolve(r)
            if u and u not in out:
                out.append(u)
        return out

    # -- rendering --
    def label(self, ref: str) -> str:
        """How a source is shown TO THE MODEL: `[S1] Title (domain)` — no full URL/path,
        so there is nothing for it to mis-copy. Domain is kept for credibility judgement."""
        s = self.get(ref)
        if not s:
            return f"[{self._norm(ref)}]"
        dom = urlparse(s.url).netloc.removeprefix("www.")
        head = f"[{s.ref}] {s.title}".rstrip()
        return f"{head}  ({dom})" if dom else head

    def source_map(self, refs) -> str:
        """Deterministic, harness-built 'Sources' block mapping each cited ref to its
        VERBATIM URL. This is appended to the scout's report by the harness — the model
        never writes it, so the URLs are guaranteed byte-for-byte correct."""
        lines = []
        for r in refs:
            s = self.get(r)
            if s:
                tail = f" — {s.title}" if s.title else ""
                lines.append(f"- [{s.ref}] {s.url}{tail}")
        return "\n".join(lines)
