"""Run the PRODUCTION image.enhance pipeline over the detect_batch.py outputs.

Drives `app/` — the real thing — not the standalone `image_enhancement/` CLI. Every
piece that decides what the model sees is imported from the app rather than
reimplemented:

    app/prompts/image_enhance/system.txt   the system instruction
    app/prompts/image_enhance/user.txt     the user turn
    app/domain/image_ops.preprocess        the Pillow preprocessing pass
    app/agents/gemini.GeminiClient         the call itself, incl. its empty-response retry

Only the input plumbing is local: the app reads its images from an upload envelope
(`AgentInvokeRequest` + file refs) and this reads them from a folder, which is the
one thing a batch run has to do differently. `EnhancementAgent.build_prompt` and
`_call_model` are two lines each and are mirrored exactly here — see `enhance_one`.

Why this is NOT `image_enhancement/image_enhancement.py` with a new --input-dir.
That script is a separate, older implementation of the same idea, and it differs from
the deployed pipeline in five measurable ways:

    prompt delivery  CLI: one string prepended to `contents`, no system instruction.
                     App: `system_instruction=system.txt`, and the user turn is a
                     SECOND message after the image.
    content order    CLI: [text, image].   App: [image, text].
    sharpening       CLI: OpenCV GaussianBlur + addWeighted, amount 1.2/0.5.
                     App: Pillow UnsharpMask(radius=1.0, percent=120/50, threshold=3).
    exposure         CLI: CLAHE on L in LAB (LOCAL, per-tile).
                     App: ImageOps.autocontrast(cutoff=1) (GLOBAL). image_ops' own
                     docstring calls this out as the one step it cannot reproduce.
    blur score       CLI: variance of cv2.Laplacian.
                     App: variance of Pillow FIND_EDGES with a 3px border crop.
                     Different scales entirely, so the 100.0 threshold selects the
                     strong/weak sharpen on different images in each.
    retry            CLI: none — a refusal is a lost image.
                     App: up to 3 attempts, judged on whether IMAGE bytes came back
                     rather than on whether text did.

Enhancing through the CLI would measure the CLI. `image_enhancement_output_nb/` was
produced by it, so that folder is the CLI's baseline; these two new folders are the
app's, over app inputs.

Usage (from OCR Prototype/):
    venv/bin/python enhance_batch.py --source both --limit 5 --dry-run
    venv/bin/python enhance_batch.py --source both --limit 5

--dry-run runs everything including prompt assembly and preprocessing, writes the
exact bytes that would be uploaded into _preprocessed/, and stops before the one
billable line.

Writes, per source, a sibling of image_enhancement_input/:
    image_enhancement_output_nb_processed_bounding box/
    image_enhancement_output_nb_processed_silhouette/
holding <stem>.<ext> and <stem>.json. The sidecar's first eight keys are exactly
image_enhancement.py's schema in its order, so build_image_comparison.py keeps
working; app-specific fields are appended after.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from io import BytesIO
from pathlib import Path

from dotenv import dotenv_values
from PIL import Image, ImageOps

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

SOURCES = {
    "bbox": ("image_enhancement_processed_bounding box",
             "image_enhancement_output_nb_processed_bounding box"),
    "silhouette": ("image_enhancement_processed_silhouette",
                   "image_enhancement_output_nb_processed_silhouette"),
}


def find_repo() -> Path:
    """The ondc-intelligence checkout. Located by content, not by name — the
    directory has already been renamed once in this project (…-fea-enhancement-evals),
    and there is a second checkout beside it whose name differs again."""
    root = Path(__file__).resolve().parent.parent
    for sibling in sorted(p for p in root.iterdir() if p.is_dir()):
        if (sibling / "app" / "prompts" / "image_enhance" / "system.txt").is_file():
            return sibling
    raise SystemExit(f"no ondc-intelligence checkout with app/prompts/ found under {root}")


def load_app(repo: Path, need_key: bool):
    """Import the app package, with the API key placed where it reads it from.

    `app.config.Settings.gemini_api_key` is `os.getenv("GEMINI_API_KEY")` evaluated at
    import, and this checkout has no .env of its own — the premium key lives in
    OCR Prototype/.env. So the environment is set BEFORE the import, not after; doing
    it after would leave `settings.gemini_api_key` as None and the client would
    report a missing credential while the key sat right there.
    """
    if need_key:
        env_path = Path(__file__).resolve().parent / ".env"
        key = dotenv_values(env_path).get("GEMINI_API_KEY")
        if not key:
            raise SystemExit(f"GEMINI_API_KEY is not set in {env_path}")
        os.environ["GEMINI_API_KEY"] = key

    sys.path.insert(0, str(repo))
    mods = {
        name: importlib.import_module(name)
        for name in ("app.config", "app.domain.image_ops", "app.prompts", "app.agents.gemini")
    }
    return mods["app.config"].settings, mods["app.domain.image_ops"], \
        mods["app.prompts"], mods["app.agents.gemini"]


def read_flattened(path: Path) -> tuple[bytes, bool]:
    """Image bytes with any transparency composited onto WHITE, ready for
    `image_ops.preprocess`.

    This is the one step the app cannot do for us, and it is not a nicety.
    `image_ops.preprocess` calls `.convert("RGB")`, which DISCARDS the alpha channel
    and keeps the RGB sitting underneath it. `detect_batch.py`'s `_cutout` only zeroes
    alpha — it never zeroes the colour of the hidden pixels — so the "removed" shelf
    background is still fully present in those values. Verified on img1.png: pixels at
    alpha=0 hold real colours like (12,8,7) and (225,222,215), not black, not white.

    Handed straight to the app, every silhouette would reach Gemini with its
    background completely restored — byte-for-byte the same job as the bounding-box
    crop. The API calls would succeed, the folder would fill up, and the "silhouette"
    column of the comparison would be a duplicate of the other one with nothing to
    show for it.

    White, because the prompt asks for a #FFFFFF background anyway: this agrees with
    the target instead of fighting it, and leaves no dark fringe along the cut edge
    for the unsharp pass to bite into.

    Re-encoded as PNG, not JPEG — lossless, so the only lossy step in the chain stays
    the single JPEG `image_ops.preprocess` writes at the end, exactly as for an image
    that needed no compositing.
    """
    with Image.open(path) as opened:
        has_alpha = opened.mode in ("RGBA", "LA") or (
            opened.mode == "P" and "transparency" in opened.info
        )
        if not has_alpha:
            return path.read_bytes(), False
        rgba = opened.convert("RGBA")
        white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        flat = Image.alpha_composite(white, rgba).convert("RGB")
        # exif_transpose is left to image_ops, which does it itself; PNG carries no
        # EXIF orientation anyway.
        buf = BytesIO()
        flat.save(buf, format="PNG")
        return buf.getvalue(), True


def enhance_one(gemini_mod, prompts_mod, client, model: str, prepared):
    """`EnhancementAgent.build_prompt` + `_call_model` for the image capability,
    mirrored. Both are short enough that mirroring them is safer than constructing an
    `AgentInvokeRequest` and a file-ref envelope around a folder read — but they are
    the two places this file could drift from the app, so they sit together here:

        build_prompt:  Prompt(system=<image_enhance/system>, user=<image_enhance/user>,
                              image_parts=[the preprocessed image])
        _call_model:   image_model_client().generate_image(prompt, model=...)

    The client's own retry (up to `gemini_empty_response_retries + 1` attempts, judged
    on image bytes rather than text) comes along with it — that behaviour lives in
    `generate_image`, so calling it is what inherits it.
    """
    prompt = gemini_mod.Prompt(
        system=prompts_mod.load_prompt("image_enhance", "system"),
        user=prompts_mod.load_prompt("image_enhance", "user"),
        image_parts=[gemini_mod.ImagePart(data=prepared.data, mime_type=prepared.mime_type)],
    )
    return client.generate_image(prompt, model=model)


def natural_key(path: Path):
    """img2 before img10 — plain sorted() gives img1, img10, img11, img12, img13, so
    "--limit 5" would mean an arbitrary five rather than the first five."""
    digits = "".join(c for c in path.stem if c.isdigit())
    return (int(digits) if digits else 0, path.stem)


def run_source(source: str, repo: Path, ctx, args) -> None:
    settings, image_ops, prompts_mod, gemini_mod, client, model = ctx
    in_name, out_name = SOURCES[source]
    base = repo / "image_enhancement"
    input_dir, output_dir = base / in_name, base / out_name
    if not input_dir.is_dir():
        raise SystemExit(f"input dir not found: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(
        (p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS),
        key=natural_key,
    )
    if args.limit:
        images = images[: args.limit]

    print(f"\n{'=' * 78}")
    print(f"{source.upper()}  ·  app image.enhance  ·  model={model}  ·  {len(images)} image(s)"
          + ("  ·  DRY RUN, no API calls" if args.dry_run else ""))
    print(f"  in:  {input_dir.name}")
    print(f"  out: {output_dir.name}")
    print(f"{'=' * 78}")

    preview_dir = output_dir / "_preprocessed" if args.dry_run else None
    if preview_dir:
        preview_dir.mkdir(exist_ok=True)

    done = skipped = failed = 0
    for path in images:
        # The model chooses the output format, so the extension isn't known ahead of
        # time — same "any extension" cache check the CLI uses.
        if [p for p in output_dir.glob(f"{path.stem}.*") if p.suffix != ".json"] \
                and not args.overwrite:
            print(f"  {path.name:16} cached")
            skipped += 1
            continue

        started = time.perf_counter()
        try:
            raw, composited = read_flattened(path)
            source_w, source_h = image_ops.dimensions(raw)
            prepared = image_ops.preprocess(raw)
        except Exception as exc:
            print(f"  {path.name:16} ERROR preprocessing  {type(exc).__name__}: {exc}")
            failed += 1
            continue
        preprocess_time = time.perf_counter() - started

        note = (f"blur={prepared.blur_score:.1f} {source_w}x{source_h}"
                f"->{prepared.width}x{prepared.height}"
                + (" alpha->white" if composited else ""))

        if args.dry_run:
            (preview_dir / f"{path.stem}.jpg").write_bytes(prepared.data)
            print(f"  {path.name:16} dry-run  {note}  "
                  f"{len(prepared.data) / 1024:.0f}KB -> _preprocessed/{path.stem}.jpg")
            done += 1
            continue

        try:
            gen = enhance_one(gemini_mod, prompts_mod, client, model, prepared)
        except Exception as exc:
            # LlmError already carries the finish_reason and, on a refusal, the text
            # the model answered with instead of pixels — that is the diagnosable part,
            # so it is printed whole rather than summarised.
            print(f"  {path.name:16} ERROR  {type(exc).__name__}: {exc}")
            failed += 1
            continue

        latency = time.perf_counter() - started
        gemini_time = latency - preprocess_time
        data = gen.image_bytes
        ext = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}.get(
            gen.image_mime, ".png"
        )
        out_path = output_dir / f"{path.stem}{ext}"
        out_path.write_bytes(data)
        out_w, out_h = image_ops.dimensions(data)

        (output_dir / f"{path.stem}.json").write_text(json.dumps({
            # --- image_enhancement.py's schema, same keys, same order --------
            "input": path.name,
            "output": out_path.name,
            "model": model,
            "preprocess": True,
            "preprocess_time_s": round(preprocess_time, 3),
            "gemini_time_s": round(gemini_time, 3),
            "total_time_s": round(latency, 3),
            "output_bytes": len(data),
            # --- appended: what the app records that the CLI does not --------
            # Appended, not interleaved, so anything reading the eight keys above by
            # name is unaffected. Without `pipeline`/`source_variant` the three output
            # folders' sidecars are indistinguishable from each other.
            "pipeline": "app.image.enhance",
            "source_variant": source,
            "alpha_composited_onto_white": composited,
            "prompt_system_sha": prompts_mod.template_hash("image_enhance", "system"),
            "prompt_user_sha": prompts_mod.template_hash("image_enhance", "user"),
            "blur_score": round(prepared.blur_score, 2),
            "source_width": source_w,
            "source_height": source_h,
            "preprocessed_width": prepared.width,
            "preprocessed_height": prepared.height,
            "width": out_w,
            "height": out_h,
            # gemini_time_s is wall clock and so covers every retry; latency_ms is the
            # winning attempt alone. They differ exactly when attempts > 1.
            "attempts": getattr(gen, "attempts", 1),
            "last_attempt_latency_ms": gen.latency_ms,
            "input_tokens": gen.input_tokens,
            "output_tokens": gen.output_tokens,
            "finish_reason": gen.finish_reason,
        }, indent=2), encoding="utf-8")

        tries = getattr(gen, "attempts", 1)
        print(f"  {path.name:16} OK -> {out_path.name}  {len(data) / 1024:.0f}KB "
              f"{out_w}x{out_h}  ({note}; pre {preprocess_time:.2f}s, "
              f"gemini {gemini_time:.2f}s{f', {tries} attempts' if tries > 1 else ''})")
        done += 1

    print(f"  -- {done} done, {skipped} cached, {failed} failed")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["bbox", "silhouette", "both"], default="both")
    ap.add_argument("--limit", type=int, default=5,
                    help="first N images in natural order (0 = all). Defaults to 5, "
                         "not 0: every image is a paid API call")
    ap.add_argument("--dry-run", action="store_true",
                    help="preprocess and write _preprocessed/ previews; make NO API call")
    ap.add_argument("--overwrite", action="store_true",
                    help="re-enhance images that already have output (pays again)")
    ap.add_argument("--model", default=None,
                    help="default: settings.image_enhancement_model")
    args = ap.parse_args()

    repo = find_repo()
    settings, image_ops, prompts_mod, gemini_mod = load_app(repo, need_key=not args.dry_run)
    model = args.model or settings.image_enhancement_model
    client = None if args.dry_run else gemini_mod.GeminiClient()

    ctx = (settings, image_ops, prompts_mod, gemini_mod, client, model)
    for source in (["bbox", "silhouette"] if args.source == "both" else [args.source]):
        run_source(source, repo, ctx, args)
    print()


if __name__ == "__main__":
    main()
