# Cloud batch-eval: models too big (or too runtime-broken) for the local Mac

This runs one of the candidate models your senior asked about — the ones that don't fit the
dev Mac's 16GB total RAM, or that hit a real conversion bug in `mlx-vlm`/`llama.cpp` — on
**free cloud GPU compute**, scoring against the exact same `ground_truth_100` set every other
mode in this project (Gemini, local Gemma-3, heuristic+LayoutLM) has already been benchmarked
against.

This is a one-off/periodic batch job, not a live service. It runs in a Kaggle Notebook (or
Colab), produces `rowN.json` files in the same shape as everything else, and you download the
results to compare by eye.

## Why Kaggle over Colab

Kaggle's free GPU quota (30 hrs/week) gives you **two T4s = 32GB combined VRAM**, comfortably
fitting a 27-30B model at 4-bit. Colab's free tier is a single 16GB T4 — fine for the smaller
candidates (Phi-4-Reasoning-Vision-15B, Kimi-VL-A3B ~16B, Nemotron Nano 2 VL 12.6B), tight or
not enough for Qwen3.6-27B. Use Kaggle unless you're specifically testing one of the
single-GPU-sized ones.

## One-time setup

1. **Kaggle account + GPU access**: sign up at kaggle.com, verify your phone number under
   Settings (required before GPU quota is unlocked — this step is entirely manual, I can't do
   it for you).

2. **Package the eval data** (run locally, on this machine):
   ```bash
   cd "OCR Prototype"
   venv/bin/python cloud_eval/package_dataset.py
   ```
   This writes `cloud_eval/cloud_eval_data.zip` — the same 100 products' images (matching
   `ground_truth_100`) + `Ground Truth.csv` + the DigiCatalog schema, zipped up. Expect
   roughly a few hundred MB (not the full 3.1GB Dataset/ — just the eval subset).

3. **Upload as a Kaggle Dataset**: kaggle.com → Datasets → New Dataset → upload
   `cloud_eval_data.zip` (Kaggle unzips it automatically) → give it any name, e.g.
   `ocr-eval-100`.

4. **Create a Notebook**: kaggle.com → Code → New Notebook. In the right sidebar:
   - **Accelerator**: GPU T4 x2
   - **Data**: Add Input → your `ocr-eval-100` dataset (mounts at
     `/kaggle/input/ocr-eval-100/`)

5. **Install deps** (first cell):
   ```
   !pip install -q -U transformers accelerate bitsandbytes pillow
   ```

6. **Upload `run_batch.py`**: either use the Notebook's file upload (drag it into the file
   browser on the left) or paste its contents into a cell — it's a single self-contained file,
   no other project files needed on the Kaggle side.

## Running it

Start small — 5 products, not 100 — to prove the model/adapter actually works before spending
your GPU-hours on a full run:

```
!python run_batch.py --model qwen3.6-27b \
    --data /kaggle/input/ocr-eval-100 \
    --start 0 --n 5 --out /kaggle/working/results
```

`--model` must be one of the keys in `run_batch.py`'s `MODEL_REGISTRY` — currently
`qwen3.6-27b`, `phi4-reasoning-vision-15b`, `kimi-vl-a3b-thinking`. **Read that entry's
`notes` field before running** — some are marked UNVALIDATED (never run against real images
yet), meaning the first run is also the first real test of whether the chat-template format
even matches that model's processor. If it errors on the image content-block shape, that's
the one place to patch — see `_call_model()`'s docstring in `run_batch.py`.

If the small run looks reasonable, scale up:
```
!python run_batch.py --model qwen3.6-27b \
    --data /kaggle/input/ocr-eval-100 \
    --start 0 --n 100 --out /kaggle/working/results
```

## Getting results back

The script zips its own output (`/kaggle/working/results.zip`). Download it from the
Notebook's Output pane, then locally:

```bash
mkdir -p "Output/pipeline_ground_truth_cloud/qwen3.6-27b"
unzip results.zip -d "Output/pipeline_ground_truth_cloud/qwen3.6-27b"
```

Compare a handful of `rowN.json` files by eye against
`../Image to OCR to Json Export/ground_truth_100/rowN.json` — the same manual-comparison
approach already used for the Gemini prompt-tuning work. There's no automated scorer against
`ground_truth_100` yet in this project; that'd be a reasonable separate follow-up if you want
it, not something this batch job does for you.

## Adding another model from the list

Add one entry to `MODEL_REGISTRY` in `run_batch.py` (repo id + whether it needs
`trust_remote_code`), then run it the same way. Everything else — prompt, schema, JSON
repair, output shape — is shared. Two names from the original list are deliberately **not**
in the registry:

- **DiffusionGemma** — genuinely diffusion-based (not autoregressive), so the standard
  `.generate()` loop in `_call_model()` almost certainly doesn't apply; it needs its own
  sampling code. Treat as a separate, bigger effort, not a quick registry addition.
- **GLM-4.7-Flash / Nemotron Cascade 2** — confirmed text-only, no vision input at all
  (see the research from earlier this session) — not applicable to an image-extraction task
  regardless of compute.
