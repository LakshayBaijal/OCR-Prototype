"""How often does the enhancer actually preserve the product? Run the whole batch.

`verify_app.py` answers the question for one image, interactively. This answers it for
a directory of already-enhanced pairs, non-interactively, which is the form you need
before deciding whether an enhancement step is safe to put in front of a catalog at
all. One image looking fine proves nothing; a rejection rate does.

It reads pairs by matching STEM, so it works directly against the batches
`ondc-intelligence/image_enhancement/` already produced:

    venv/bin/python verify_batch.py \\
        --original ../ondc-intelligence/image_enhancement/image_enhancement_input \\
        --enhanced ../ondc-intelligence/image_enhancement/image_enhancement_output_nb \\
        --n 10

Everything is OCR-only and CPU-only - no VLM, no API key, no GPU, nothing billable -
so a full-directory run costs only time. `--no-artwork` skips the ORB feature match
and colour histogram, which is the slowest part, for a fast first pass.
"""

from __future__ import annotations

import argparse
import json
import warnings
from collections import Counter
from pathlib import Path

warnings.filterwarnings("ignore")

from verify import extract, report

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def pairs(original_dir: Path, enhanced_dir: Path) -> list[tuple[Path, Path]]:
    """Match on stem, so img1.jpg pairs with img1.png - the enhancer's output format
    follows whatever the model returned, not the input's extension."""
    out = []
    for o in sorted(p for p in original_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS):
        matches = [e for e in enhanced_dir.glob(f"{o.stem}.*")
                   if e.suffix.lower() in IMAGE_EXTENSIONS]
        if matches:
            out.append((o, matches[0]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--original", type=Path, required=True)
    ap.add_argument("--enhanced", type=Path, required=True)
    ap.add_argument("--n", type=int, default=0, help="cap the number of pairs (0 = all)")
    ap.add_argument("--lang", default="auto",
                    help="OCR recognition script: auto (default), ch (Latin/digits), "
                         "el (Greek), devanagari, ta, te, ka, cyrillic, arabic")
    ap.add_argument("--no-artwork", action="store_true",
                    help="skip ORB feature matching and the colour histogram (faster)")
    ap.add_argument("--json-out", type=Path, help="write the full per-pair result here")
    args = ap.parse_args()

    found = pairs(args.original, args.enhanced)
    if not found:
        raise SystemExit(f"no stem-matched pairs between {args.original} and {args.enhanced}")
    if args.n:
        found = found[: args.n]

    print(f"\n{'=' * 78}")
    print(f"ENHANCEMENT FIDELITY - {len(found)} pairs · OCR-only "
          f"(artwork check {'off' if args.no_artwork else 'on'})")
    print(f"{'=' * 78}\n")

    verdicts: Counter[str] = Counter()
    status_totals: Counter[str] = Counter()
    critical_hits: Counter[str] = Counter()
    rows = []

    for o_path, e_path in found:
        o = extract.load_rgb_bytes(o_path.read_bytes())
        e = extract.load_rgb_bytes(e_path.read_bytes())
        rep = report.run_verification(o, e, check_artwork=not args.no_artwork,
                                      lang=args.lang)

        verdicts[rep.verdict] += 1
        for c in rep.checks:
            status_totals[c.status] += 1
            if c.status in ("CHANGED", "LOST") and c.critical:
                critical_hits[c.name] += 1

        headline = rep.reasons[0] if rep.reasons else ""
        counts = rep.counts()
        print(f"  {o_path.name:22} {rep.verdict:7} "
              f"kept={counts['MATCH']:2} changed={counts['CHANGED']:2} "
              f"lost={counts['LOST']:2}  {headline[:66]}")

        rows.append({
            "original": str(o_path), "enhanced": str(e_path),
            "verdict": rep.verdict, "counts": counts, "reasons": rep.reasons,
            "checks": [{"name": c.name, "status": c.status, "detail": c.detail,
                        "original": c.original, "enhanced": c.enhanced,
                        "critical": c.critical} for c in rep.checks],
        })

    n = len(found)
    print(f"\n{'-' * 78}")
    for v in ("PASS", "REVIEW", "REJECT"):
        print(f"  {v:7} {verdicts[v]:3}/{n}  ({verdicts[v] / n * 100:.0f}%)")
    print(f"\n  field outcomes: " + "  ".join(f"{k}={v}" for k, v in status_totals.items()))
    if critical_hits:
        print(f"\n  critical fields changed or lost, by field:")
        for field_name, count in critical_hits.most_common():
            print(f"    {field_name:20} {count:3}/{n}")
    print()

    if args.json_out:
        args.json_out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"  wrote {args.json_out}\n")


if __name__ == "__main__":
    main()
