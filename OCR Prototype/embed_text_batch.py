"""Gemini TEXT embeddings for the OCR .txt files next to a silhouette folder's
images - separate files from `embed_batch.py`'s IMAGE embeddings, on purpose:
same folder, two different things embedded, so they never collide or get
mixed up when read back.

WHICH MODEL, AND WHY IT DIFFERS FROM embed_batch.py's `gemini-embedding-2`:
measured directly (see embed_batch.py's own docstring) that `gemini-embedding-2`
takes image input but `gemini-embedding-001` rejects it ("The text content is
empty.") - meaning -001 is the TEXT-only model, the opposite of what
embed_batch.py needs. Confirmed here the other direction: -001 embeds a plain
string fine, 3072-dim, L2-normalised (cosine similarity is a plain dot product).

    venv/bin/python embed_text_batch.py --dir <silhouette folder>
    venv/bin/python embed_text_batch.py --all             # both silhouette folders
    venv/bin/python embed_text_batch.py --all --compare    # ...then pair them up

Uses the personal GEMINI_API_KEY in .env, same as embed_batch.py - there is no
embed capability on the ONDC intelligence server to route this through instead.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from dotenv import dotenv_values
from google import genai

MODEL = "gemini-embedding-001"
MAX_RETRIES = 4
OUT_NPZ = "embeddings_text.npz"
OUT_JSON = "embeddings_text.json"
OUT_INDEX = "embeddings_text_index.json"


def embed(client: genai.Client, text: str) -> list[float]:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = client.models.embed_content(model=MODEL, contents=[text])
            return list(r.embeddings[0].values)
        except Exception as exc:
            s = str(exc)
            transient = any(m in s for m in ("429", "RESOURCE_EXHAUSTED", "500", "503", "INTERNAL", "UNAVAILABLE"))
            if not transient or attempt == MAX_RETRIES:
                raise
            wait = 2 ** attempt
            print(f"      transient error, retrying in {wait}s ({attempt}/{MAX_RETRIES - 1}): {s[:90]}")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def run_dir(client: genai.Client, folder: Path, overwrite: bool) -> None:
    texts = sorted(
        (p for p in folder.glob("*.txt")),
        key=lambda p: int("".join(c for c in p.stem if c.isdigit()) or 0),
    )
    out_json = folder / OUT_JSON
    existing: dict[str, list[float]] = {}
    if out_json.exists() and not overwrite:
        existing = json.loads(out_json.read_text(encoding="utf-8"))["embeddings"]

    print(f"\n{folder}\n  {len(texts)} text file(s)  ·  model={MODEL}  ·  {len(existing)} already embedded")

    done, skipped, failed = 0, 0, 0
    for i, p in enumerate(texts, 1):
        if p.stem in existing:
            skipped += 1
            continue
        content = p.read_text(encoding="utf-8", errors="ignore").strip()
        if not content:
            print(f"  [{i}/{len(texts)}] {p.name} -> SKIP (empty text)")
            failed += 1
            continue
        try:
            existing[p.stem] = embed(client, content)
            print(f"  [{i}/{len(texts)}] {p.name} -> ok  ({len(existing)} embedded)")
            done += 1
        except Exception as exc:
            print(f"  [{i}/{len(texts)}] {p.name} -> FAIL {type(exc).__name__}: {exc}")
            failed += 1

    print(f"  -- {done} embedded, {skipped} already had one, {failed} failed")
    if done == 0:
        return

    stems = sorted(existing.keys(), key=lambda s: int("".join(c for c in s if c.isdigit()) or 0))
    matrix = np.array([existing[s] for s in stems], dtype=np.float32)
    np.savez(folder / OUT_NPZ, stems=np.array(stems), embeddings=matrix)
    (folder / OUT_JSON).write_text(
        json.dumps({"model": MODEL, "dimensions": matrix.shape[1] if len(stems) else 0,
                    "count": len(stems), "embeddings": existing}, ensure_ascii=False),
        encoding="utf-8",
    )
    (folder / OUT_INDEX).write_text(
        json.dumps({"model": MODEL, "dimensions": matrix.shape[1] if len(stems) else 0,
                    "count": len(stems), "normalized": True,
                    "note": "Vectors are L2-normalised by the API, so cosine similarity is a plain dot product.",
                    "source_folder": folder.name, "stems": stems}, indent=2),
        encoding="utf-8",
    )
    print(f"  -> {OUT_NPZ}, {OUT_JSON}, {OUT_INDEX}")


def compare(dirs: list[Path]) -> None:
    if len(dirs) != 2:
        return
    a, b = (np.load(d / OUT_NPZ, allow_pickle=True) for d in dirs)
    va = dict(zip(a["stems"], a["embeddings"]))
    vb = dict(zip(b["stems"], b["embeddings"]))
    common = sorted(set(va) & set(vb), key=lambda s: int("".join(c for c in s if c.isdigit()) or 0))
    print(f"\n{dirs[0].name} vs {dirs[1].name} - {len(common)} matched text pair(s)")
    for s in common:
        x, y = va[s].astype(np.float64), vb[s].astype(np.float64)
        sim = float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))
        print(f"  {s}: {sim*100:.1f}%")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, action="append", default=[], help="repeatable")
    ap.add_argument("--compare", action="store_true", help="cosine-compare the (exactly 2) --dir folders")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if not args.dir:
        raise SystemExit("usage: embed_text_batch.py --dir <folder> [--dir <folder2> --compare]")

    env_path = Path(__file__).resolve().parent / ".env"
    key = dotenv_values(env_path).get("GEMINI_API_KEY")
    if not key:
        raise SystemExit(f"GEMINI_API_KEY is not set in {env_path}")
    client = genai.Client(api_key=key)

    for d in args.dir:
        run_dir(client, d, args.overwrite)
    if args.compare:
        compare(args.dir)


if __name__ == "__main__":
    main()
