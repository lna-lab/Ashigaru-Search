"""Build a local BM25 index from a folder of documents.

    ashigaru-index <folder> <out.pkl> [--chunk 512] [--overlap 64]

Supports .txt / .md / .rst out of the box, and .pdf if `pypdf` is installed.
"""
from __future__ import annotations
import argparse
import os
import pickle
import sys

TEXT_EXT = {".txt", ".md", ".rst", ".markdown", ".text"}


def _read(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        try:
            from pypdf import PdfReader
            return "\n".join((p.extract_text() or "") for p in PdfReader(path).pages)
        except Exception as e:
            print(f"  skip pdf {path}: {e}", file=sys.stderr)
            return ""
    if ext in TEXT_EXT:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    return ""


def _chunk(words: list[str], size: int, overlap: int):
    step = max(1, size - overlap)
    for start in range(0, len(words), step):
        piece = words[start:start + size]
        if piece:
            yield " ".join(piece)
        if start + size >= len(words):
            break


def build_index(folder: str, out: str, chunk: int = 512, overlap: int = 64) -> int:
    chunks = []
    for root, _, files in os.walk(folder):
        for fn in sorted(files):
            path = os.path.join(root, fn)
            text = _read(path)
            if not text.strip():
                continue
            stem = os.path.relpath(path, folder)
            for n, piece in enumerate(_chunk(text.split(), chunk, overlap)):
                chunks.append({"id": f"{stem}#{n}", "source": stem, "text": piece})
    with open(out, "wb") as f:
        pickle.dump({"chunks": chunks, "folder": os.path.abspath(folder)}, f)
    return len(chunks)


def main():
    ap = argparse.ArgumentParser(description="Build a BM25 index for Ashigaru-Search local RAG.")
    ap.add_argument("folder")
    ap.add_argument("out")
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=64)
    a = ap.parse_args()
    n = build_index(a.folder, a.out, a.chunk, a.overlap)
    print(f"indexed {n} chunks from {a.folder} -> {a.out}")


if __name__ == "__main__":
    main()
