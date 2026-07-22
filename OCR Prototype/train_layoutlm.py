"""Fine-tune LayoutLMv3 for Stage 2 field extraction, with WEAK SUPERVISION.

We have no token-level annotations - only product-level field values in Ground Truth.csv. This
script manufactures token labels by aligning those values back onto the OCR tokens that spell
them (stage2.layoutlm.weak_labels), then fine-tunes `microsoft/layoutlmv3-base` as a token
classifier and saves the checkpoint to `layoutlmv3_finetuned/`. Once that exists,
`stage2.layoutlm.available()` flips True and the Stage 2 pipeline / eval_stage2.py --layoutlm
will use it.

Runs Stage 0 (fast heuristic) + Stage 1 OCR to get each image's words + boxes; OCR is cached,
so re-runs are quick. Training is on MPS if available, else CPU.

Usage:
  venv/bin/python train_layoutlm.py --n 300 --epochs 3        # a real run
  venv/bin/python train_layoutlm.py --n 20 --epochs 1         # a fast smoke test (checks the
                                                              #   whole loop end-to-end)
"""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

os.environ.setdefault("BGREMOVE", "0")  # fast Stage-0 for data collection
warnings.filterwarnings("ignore")

import pandas as pd

import products
from ocr import router
from ocr.rapid import _quiet_rapidocr
from preprocessing import run_pipeline
from stage2.layoutlm import (
    CHECKPOINT_DIR, ID2LABEL, LABEL2ID, LABELS, MODEL_ID,
    line_to_words, normalise_box, weak_labels,
)

ROOT = Path(__file__).parent
DATASET = ROOT / "Dataset"
TARGET_FIELDS_GT = [
    "Product Name", "Brand", "Mrp", "Net Quantity", "Fssai No", "Manufacturer",
    "Country Of Origin",
    # Newer label classes (see stage2/layoutlm.py LABELS) - CSV-backed fields added to widen
    # coverage beyond the original 7.
    "Flavours & Spices", "Mineral Source", "Recommended Age", "Theme/ Occasion Type",
    "Hair Type", "Caffeine Content", "Pack Quantity",
]


def _gt(row: pd.Series) -> dict:
    out = {}
    for f in TARGET_FIELDS_GT:
        v = row.get(f)
        out[f] = None if (v is None or pd.isna(v) or not str(v).strip()) else str(v).strip()
    return out


def collect_examples(n: int) -> list[dict]:
    """Build weak-labelled training examples: one per crop image, each
    {image, words, boxes(0-1000), label_ids}. Skips images with no labelled token (all "O"
    carries no field signal to learn from)."""
    _quiet_rapidocr()
    df = pd.read_csv(ROOT / "Ground Truth.csv", low_memory=False)
    cols = [c for c in df.columns if "Filename" in c]
    all_names = sorted(p.name for p in DATASET.glob("*.jpg"))
    sample = df.sample(n=min(n, len(df)), random_state=13)  # 13: a different split from the evals

    examples: list[dict] = []
    for _, row in sample.iterrows():
        gt = _gt(row)
        img_names = products.group_for(
            [str(row[c]).strip() for c in cols if isinstance(row[c], str) and str(row[c]).strip()],
            all_names,
        )
        for nm in img_names:
            p = DATASET / nm
            if not p.exists():
                continue
            _, crops = run_pipeline(str(p), {})
            for crop in crops:
                h, w = crop.shape[:2]
                words, boxes = [], []
                for line in router.read(crop).lines:
                    ws, bs = line_to_words(line.text, line.bbox)
                    for word, box in zip(ws, bs):
                        words.append(word)
                        boxes.append(normalise_box(box, w, h))
                if not words:
                    continue
                labels = weak_labels(words, gt)
                if all(l == "O" for l in labels):
                    continue  # nothing to learn from this crop
                examples.append({
                    "image": crop, "words": words, "boxes": boxes,
                    "label_ids": [LABEL2ID[l] for l in labels],
                })
    return examples


def main(n: int, epochs: int, batch_size: int, lr: float) -> None:
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor

    print(f"Collecting weak-labelled examples from {n} products (Stage 0/1, OCR cached)...")
    examples = collect_examples(n)
    if not examples:
        print("No labelled examples found - nothing to train on.")
        return
    labelled_tokens = sum(sum(1 for i in e["label_ids"] if i != 0) for e in examples)
    print(f"  {len(examples)} crop examples, {labelled_tokens} weakly-labelled tokens.\n")

    processor = LayoutLMv3Processor.from_pretrained(MODEL_ID, apply_ocr=False)

    class DS(Dataset):
        def __len__(self):
            return len(examples)

        def __getitem__(self, i):
            e = examples[i]
            enc = processor(
                Image.fromarray(e["image"]).convert("RGB"), e["words"], boxes=e["boxes"],
                word_labels=e["label_ids"], truncation=True, padding="max_length",
                max_length=512, return_tensors="pt",
            )
            return {k: v.squeeze(0) for k, v in enc.items()}

    loader = DataLoader(DS(), batch_size=batch_size, shuffle=True)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        MODEL_ID, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total = 0.0
        for step, batch in enumerate(loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            out = model(**batch)
            out.loss.backward()
            optimizer.step()
            total += out.loss.item()
            if step % 20 == 0:
                print(f"  epoch {epoch+1}/{epochs}  step {step}/{len(loader)}  loss {total/step:.4f}")
        print(f"epoch {epoch+1} done - mean loss {total/len(loader):.4f}")

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    model.save_pretrained(str(CHECKPOINT_DIR))
    processor.save_pretrained(str(CHECKPOINT_DIR))
    print(f"\nSaved fine-tuned LayoutLMv3 to {CHECKPOINT_DIR}/")
    print("stage2.layoutlm.available() is now True; run: eval_stage2.py --n 100 --layoutlm")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="products to build training data from")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-5)
    a = ap.parse_args()
    main(a.n, a.epochs, a.batch_size, a.lr)
