"""The one true tokenizer — CJK-aware, dependency-free (pure ``re``).

Ashigaru's primary language is Japanese, but a plain ``\\w+`` tokenizer is *greedy* over
Unicode word characters: a space-less Japanese sentence ("これは公開してよい…") collapses into
ONE token, so a query ("公開") never intersects it and **local recall / TF-IDF clustering is
dead for the platform's primary language**.

We split instead: ASCII / digit / other-language runs stay whole (so "LM", "Studio", "FUSE"
survive), while each CJK ideograph / kana is its own single-character token (so 公開 →
``["公","開"]`` at BOTH index and query time, and they intersect). Coarse but consistent —
which is all BM25 / cosine-TF-IDF needs. Used identically by ``tools/rag.py`` (recall) and
``emaki/cluster.py`` (topic tree); they MUST share one tokenizer or scores are meaningless.
"""
from __future__ import annotations

import re

# Unicode ranges treated as "CJK" (split per-character): Hiragana + Katakana, Katakana phonetic
# extensions, CJK Extension A, CJK Unified Ideographs, CJK Compatibility Ideographs, Halfwidth
# Katakana.
_CJK_RANGES = (
    "぀-ヿ"  # Hiragana + Katakana
    "ㇰ-ㇿ"  # Katakana phonetic extensions
    "㐀-䶿"  # CJK Extension A
    "一-鿿"  # CJK Unified Ideographs
    "豈-﫿"  # CJK Compatibility Ideographs
    "ｦ-ﾟ"  # Halfwidth Katakana
)
_CJK_RE = re.compile(f"[{_CJK_RANGES}]")
# A token is EITHER a single CJK character OR a maximal run of non-CJK word characters.
_TOKEN_RE = re.compile(f"[{_CJK_RANGES}]|[^\\W{_CJK_RANGES}]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Lower-case ``text`` and return CJK-aware tokens (the one true tokenizer).

    Each token is either a single CJK character (Han / kana) or a maximal run of non-CJK word
    characters (ASCII letters + digits + ``_`` and other-language letters). So
    ``"今日はLM Studio"`` → ``["今","日","は","lm","studio"]`` — embedded ASCII stays a findable
    token instead of being glued to the surrounding Japanese. A non-string is coerced via
    ``str`` so a stray non-text payload cannot crash an index build.
    """
    return _TOKEN_RE.findall(str(text).lower())


def is_cjk_char(token: str) -> bool:
    """True if ``token`` is a single CJK character (so callers that drop short ASCII noise can
    still keep meaningful single-ideograph tokens)."""
    return len(token) == 1 and bool(_CJK_RE.match(token))
