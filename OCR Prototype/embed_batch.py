"""Gemini image embeddings for a folder of silhouette cutouts.

Built for the raw-vs-enhanced comparison: embed the product cut out of the ORIGINAL
photo and the same product cut out of the GEMINI-ENHANCED image, then measure how far
apart they sit. A high similarity is evidence the enhancement kept the product it was
given; a low one is a flag to go look at that pair.

    venv/bin/python embed_batch.py --dir <silhouette folder>
    venv/bin/python embed_batch.py --all           # both silhouette folders
    venv/bin/python embed_batch.py --all --compare # ...then pair them up

WHICH MODEL, AND WHY IT MATTERS

`gemini-embedding-2`. This is not interchangeable with the other embedding models on
the account, and the difference is not documented in the model list — it was measured:

    gemini-embedding-2      image input -> OK, 3072-dim
    gemini-embedding-001    image input -> 400 INVALID_ARGUMENT "The text content is
                                           empty." (text-only, despite listing
                                           `embedContent` exactly like the other)

So a plausible-looking swap to `gemini-embedding-001` does not degrade quality, it
fails outright. Verified further that the vectors genuinely respond to pixels rather
than hashing a blank: the same file twice returns cosine 1.0000, two bottles
(Heineken/Sprite) sit at 0.77, and either bottle against a juice carton sits at
0.64-0.68. Semantically ordered, so the model is looking at the image.

Two API details worth recording because both are invisible until measured:
  * Vectors come back ALREADY L2-normalised (norm = 1.0 exactly), so cosine
    similarity is a plain dot product and re-normalising is wasted work.
  * `task_type` is ACCEPTED for image input and then ignored - SEMANTIC_SIMILARITY
    and RETRIEVAL_DOCUMENT both return a vector identical to the default at cosine
    1.0000. It is therefore not passed; passing it would imply a control that does
    not exist.

PREPROCESSING IS DELIBERATELY IDENTICAL FOR EVERY FOLDER

The whole point is comparing two folders, so any preprocessing difference between
them would show up as a similarity difference and be indistinguishable from a real
one. Every image, both folders, gets exactly: alpha composited onto white, longest
side capped at EMBED_MAX_DIM, JPEG q90.

The alpha composite is not cosmetic. These PNGs carry a transparent background whose
hidden RGB values are still the ORIGINAL shelf pixels - `detect_batch.py` zeroes alpha
without zeroing colour. A plain `.convert("RGB")` silently brings that background
back, so the "silhouette" embedding would encode the shelf too, and every product
photographed against the same shelf would look artificially alike.
"""

from __future__ import annotations

import argparse
import json
import time
from io import BytesIO
from pathlib import Path

import numpy as np
from dotenv import dotenv_values
from google import genai
from google.genai import types
from PIL import Image

MODEL = "gemini-embedding-2"
EMBED_MAX_DIM = 512   # longest side handed to the API; the model downsamples anyway,
                      # and a consistent cap keeps every call the same shape of work
JPEG_QUALITY = 90
MAX_RETRIES = 4

# The two folders this exists to compare. Named here rather than passed so `--all`
# cannot accidentally point at one folder twice.
SILHOUETTE_DIRS = [
    "image_enhancement_processed_silhouette",       # cut from the ORIGINAL photos
    "image_enhancement_output_nb_silhouette",       # cut from the ENHANCED images
]


def find_image_enhancement_dir() -> Path:
    """Locate the checkout by content - its directory name has already changed once
    in this project, and a second checkout sits beside it under a different name."""
    root = Path(__file__).resolve().parent.parent
    for sibling in sorted(p for p in root.iterdir() if p.is_dir()):
        candidate = sibling / "image_enhancement" / "image_enhancement_input"
        if candidate.is_dir():
            return candidate.parent
    raise SystemExit(f"no image_enhancement/ folder found in any sibling of {root}")


def prepare(path: Path) -> bytes:
    """Silhouette PNG -> JPEG bytes, transparency flattened onto white.

    See the module docstring: `.convert("RGB")` on one of these files restores the
    original background instead of dropping it, because only the alpha channel was
    zeroed when the cutout was made.
    """
    with Image.open(path) as opened:
        has_alpha = opened.mode in ("RGBA", "LA") or (
            opened.mode == "P" and "transparency" in opened.info
        )
        if has_alpha:
            rgba = opened.convert("RGBA")
            white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            img = Image.alpha_composite(white, rgba).convert("RGB")
        else:
            img = opened.convert("RGB")
        img.thumbnail((EMBED_MAX_DIM, EMBED_MAX_DIM), Image.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=JPEG_QUALITY)
        return buf.getvalue()


def embed(client: genai.Client, jpeg: bytes) -> list[float]:
    """One embedding call, retrying the two failures that are worth retrying.

    429 (rate limit) and 5xx (provider hiccup) are transient and back off; anything
    else - a 400 for the wrong model, say - is a real error and is raised at once
    rather than retried four times into the same wall.
    """
    part = types.Part.from_bytes(data=jpeg, mime_type="image/jpeg")
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = client.models.embed_content(model=MODEL, contents=[part])
            return list(r.embeddings[0].values)
        except Exception as exc:
            text = str(exc)
            transient = "429" in text or "RESOURCE_EXHAUSTED" in text or "500" in text \
                or "503" in text or "INTERNAL" in text or "UNAVAILABLE" in text
            if not transient or attempt == MAX_RETRIES:
                raise
            wait = 2 ** attempt
            print(f"      transient error, retrying in {wait}s "
                  f"({attempt}/{MAX_RETRIES - 1}): {text[:90]}")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def run_dir(client: genai.Client, folder: Path, overwrite: bool) -> dict:
    """Embed every PNG in `folder`, writing the results INTO that same folder."""
    images = sorted(folder.glob("*.png"),
                    key=lambda p: (int("".join(c for c in p.stem if c.isdigit()) or 0), p.stem))
    if not images:
        raise SystemExit(f"no .png files in {folder}")

    out_npz = folder / "embeddings.npz"
    out_json = folder / "embeddings.json"
    out_index = folder / "embeddings_index.json"

    existing: dict[str, list[float]] = {}
    if out_json.exists() and not overwrite:
        try:
            existing = json.loads(out_json.read_text())["embeddings"]
        except Exception:
            existing = {}

    print(f"\n{'=' * 76}")
    print(f"{folder.name}")
    print(f"  {len(images)} image(s)  ·  model={MODEL}  ·  "
          f"{len(existing)} already embedded")
    print(f"{'=' * 76}")

    vectors: dict[str, list[float]] = dict(existing)
    started = time.perf_counter()
    done = skipped = failed = 0

    for path in images:
        if path.stem in vectors and not overwrite:
            skipped += 1
            continue
        try:
            vec = embed(client, prepare(path))
        except Exception as exc:
            print(f"  {path.name:16} ERROR  {type(exc).__name__}: {str(exc)[:110]}")
            failed += 1
            continue
        vectors[path.stem] = vec
        done += 1
        if done % 10 == 0 or done == 1:
            print(f"  {path.name:16} ok  ({done} embedded)")

    stems = sorted(vectors, key=lambda s: (int("".join(c for c in s if c.isdigit()) or 0), s))
    matrix = np.asarray([vectors[s] for s in stems], dtype=np.float32)

    np.savez_compressed(out_npz, stems=np.array(stems), embeddings=matrix)
    out_json.write_text(json.dumps({
        "model": MODEL,
        "dimensions": int(matrix.shape[1]) if matrix.size else 0,
        "count": len(stems),
        "embeddings": {s: vectors[s] for s in stems},
    }), encoding="utf-8")
    out_index.write_text(json.dumps({
        "model": MODEL,
        "dimensions": int(matrix.shape[1]) if matrix.size else 0,
        "count": len(stems),
        "normalized": True,
        "note": "Vectors are returned L2-normalised by the API (norm=1.0), so cosine "
                "similarity is a plain dot product.",
        "preprocessing": {
            "alpha_composited_onto": "white",
            "max_dimension": EMBED_MAX_DIM,
            "format": f"JPEG q{JPEG_QUALITY}",
        },
        "source_folder": folder.name,
        "stems": stems,
        "files": {
            "embeddings.npz": "float32 matrix + stem array, load with numpy.load",
            "embeddings.json": "same vectors as plain JSON",
            "embeddings_index.json": "this file - metadata only, no vectors",
        },
    }, indent=2), encoding="utf-8")

    elapsed = time.perf_counter() - started
    print(f"  -- {done} embedded, {skipped} already had one, {failed} failed "
          f"({elapsed:.1f}s)")
    print(f"  -> {out_npz.name}, {out_json.name}, {out_index.name}")
    return vectors


def compare(a: dict, b: dict, label_a: str, label_b: str) -> None:
    """Pair the two folders by stem and report cosine similarity per product.

    Vectors are unit-norm from the API, so the dot product IS the cosine - no
    normalisation step, and none hidden in a helper where it could silently differ
    between the two sides.
    """
    shared = sorted(set(a) & set(b),
                    key=lambda s: (int("".join(c for c in s if c.isdigit()) or 0), s))
    only_a, only_b = sorted(set(a) - set(b)), sorted(set(b) - set(a))
    if not shared:
        print("\nno stems in common - nothing to compare")
        return

    va = np.asarray([a[s] for s in shared], dtype=np.float64)
    vb = np.asarray([b[s] for s in shared], dtype=np.float64)
    sims = np.einsum("ij,ij->i", va, vb)

    print(f"\n{'=' * 76}")
    print(f"RAW vs ENHANCED  ·  cosine similarity per product  ·  {len(shared)} pairs")
    print(f"  A = {label_a}")
    print(f"  B = {label_b}")
    print(f"{'=' * 76}")
    order = np.argsort(sims)
    print(f"\n  lowest 10 (most changed by the enhancement):")
    for i in order[:10]:
        print(f"    {shared[i]:10} {sims[i]:.4f}")
    print(f"\n  highest 5 (best preserved):")
    for i in order[-5:][::-1]:
        print(f"    {shared[i]:10} {sims[i]:.4f}")
    print(f"\n  mean {sims.mean():.4f}   median {np.median(sims):.4f}   "
          f"min {sims.min():.4f}   max {sims.max():.4f}")
    if only_a:
        print(f"\n  only in A ({len(only_a)}): {', '.join(only_a)}")
    if only_b:
        print(f"  only in B ({len(only_b)}): {', '.join(only_b)}")

    out = Path(__file__).resolve().parent / "similarity_raw_vs_enhanced.json"
    out.write_text(json.dumps({
        "model": MODEL,
        "metric": "cosine (dot product of L2-normalised vectors)",
        "folder_a": label_a,
        "folder_b": label_b,
        "pairs": {shared[i]: round(float(sims[i]), 6) for i in range(len(shared))},
        "summary": {
            "count": len(shared),
            "mean": round(float(sims.mean()), 6),
            "median": round(float(np.median(sims)), 6),
            "min": round(float(sims.min()), 6),
            "max": round(float(sims.max()), 6),
        },
        "only_in_a": only_a,
        "only_in_b": only_b,
    }, indent=2), encoding="utf-8")
    print(f"\n  -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=None, help="one silhouette folder")
    ap.add_argument("--all", action="store_true", help="both silhouette folders")
    ap.add_argument("--compare", action="store_true",
                    help="with --all, also pair the folders by stem and report cosine")
    ap.add_argument("--overwrite", action="store_true", help="re-embed everything")
    args = ap.parse_args()

    env_path = Path(__file__).resolve().parent / ".env"
    key = dotenv_values(env_path).get("GEMINI_API_KEY")
    if not key:
        raise SystemExit(f"GEMINI_API_KEY is not set in {env_path}")
    client = genai.Client(api_key=key)

    if args.dir:
        run_dir(client, args.dir, args.overwrite)
        return

    ie = find_image_enhancement_dir()
    folders = [ie / name for name in SILHOUETTE_DIRS]
    missing = [f for f in folders if not f.is_dir()]
    if missing:
        raise SystemExit("missing folder(s): " + ", ".join(str(m) for m in missing))

    results = [run_dir(client, f, args.overwrite) for f in folders]
    if args.compare:
        compare(results[0], results[1], folders[0].name, folders[1].name)
    print()


if __name__ == "__main__":
    main()
