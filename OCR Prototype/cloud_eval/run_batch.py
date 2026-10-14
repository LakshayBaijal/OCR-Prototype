"""Batch-run one of the "too big / runtime-blocked for the local 16GB Mac" candidate models
on a free cloud GPU (Kaggle's dual-T4, or a single-GPU Colab for the smaller ones), scoring
against the exact same `ground_truth_100` eval set every other extraction mode in this
project has already been benchmarked against.

SELF-CONTAINED ON PURPOSE: this file has zero imports from `ocr/`/`stage2/` (those are
MLX/Apple-Silicon-only) and duplicates the small amount of pure-Python prompt/schema logic
those modules already contain (see PROMPT SECTION and SCHEMA SECTION below - copied nearly
verbatim from stage2/vlm_direct.py and stage2/schema.py). Kaggle/Colab run NVIDIA CUDA, a
genuinely different runtime, not something to force-share code with - this project already
keeps its Gemini/local-VLM paths separate for the same reason.

Setup (see cloud_eval/README.md for the full step-by-step):
    1. Upload cloud_eval_data.zip (made by cloud_eval/package_dataset.py) as a Kaggle Dataset,
       or unzip it into a Google-Drive-mounted folder for Colab.
    2. In the notebook: !pip install -q transformers accelerate bitsandbytes pillow
    3. Upload/paste this file, then:
           !python run_batch.py --model qwen3.6-27b --data /kaggle/input/<dataset-name> \
               --start 0 --n 5 --out /kaggle/working/results
       (start with --n 5 to prove the model/adapter works before a full 100-row run)
    4. Zip and download /kaggle/working/results, unzip locally into
       Output/pipeline_ground_truth_cloud/<model>/, diff a few rowN.json files by eye against
       Image to OCR to Json Export/ground_truth_100/rowN.json.

Adding a new model: add one MODEL_REGISTRY entry (see the module docstring above each
existing one for what's been validated vs. still a guess) - everything else (prompt,
schema, CLI, retry/repair, output shape) is shared.
"""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from dataclasses import dataclass, field as _dc_field
from pathlib import Path

import pandas as pd
from PIL import Image

# ============================================================== SCHEMA SECTION
# Copied from stage2/schema.py (ProductExtraction / to_ground_truth_dict / GT_FIELD_ORDER) -
# the OCR Prototype/ import isn't usable here since that package isn't on this cloud VM.


@dataclass
class Field:
    value: str | None = None
    confidence: float = 0.0
    source: str = ""


@dataclass
class ProductExtraction:
    fields: dict[str, Field] = _dc_field(default_factory=dict)
    bullets: list[str] = _dc_field(default_factory=list)

    def set(self, name: str, value: str | None, confidence: float, source: str) -> None:
        if value is None or not str(value).strip():
            return
        existing = self.fields.get(name)
        if existing is None or confidence > existing.confidence:
            self.fields[name] = Field(value=str(value).strip(), confidence=confidence, source=source)

    def add_bullet(self, text: str) -> None:
        text = (text or "").strip()
        if text and text not in self.bullets:
            self.bullets.append(text)

    def to_ground_truth_dict(self) -> dict:
        f = self.fields

        def v(name: str) -> str | None:
            return f[name].value if name in f else None

        def safe_int(s: str | None) -> int | None:
            if not s:
                return None
            try:
                return int(float(s))
            except ValueError:
                return None

        net_qty_raw = v("Net Quantity")
        net_qty, uom = None, None
        if net_qty_raw:
            parts = net_qty_raw.split(None, 1)
            if parts:
                net_qty = safe_int(parts[0])
                uom = parts[1].strip() if len(parts) > 1 else None

        return {
            "Category": "Grocery",
            "Sub_Category": None,
            "Product_Name": v("Product Name"),
            "Net_Quantity": net_qty,
            "UOM": uom,
            "Product_Description": v("Product_Description"),
            "Bullet_Point": list(self.bullets),
            "Manufacturer": v("Manufacturer"),
            "Country_Of_Origin": v("Country Of Origin"),
            "Images": [],
            "Brand": v("Brand"),
            "Pack_Quantity": safe_int(v("Pack_Quantity")),
            "Pack_Size": None,
            "UPC_or_EAN": v("Upc / Ean"),
            "FSSAI_no": v("Fssai No"),
            "Dimension": v("Dimension"),
            "Preservatives": v("Preservatives"),
            "Preservatives_Details": v("Preservatives_Details"),
            "Flavours_or_Spices": v("Flavours_or_Spices"),
            "Diet_Type": v("Diet_Type"),
            "Ready_to_cook": v("Ready_to_cook"),
            "Ready_to_eat": v("Ready_to_eat"),
            "Ingredients": v("Ingredients"),
            "Nutritional_Details": v("Nutritional_Details"),
            "Usage_Details": v("Usage_Details"),
            "Storage_Instructions": v("Storage_Instructions"),
            "Recommended_Age": v("Recommended_Age"),
            "Baby_Weight": v("Baby_Weight"),
            "Absorption_Duration": v("Absorption_Duration"),
            "Scented_Flavoured": ("Yes" if v("Flavours_or_Spices") else None),
            "Herbal_or_Ayurvedic": v("Herbal_or_Ayurvedic"),
            "Theme_or_Occasion_Type": v("Theme_or_Occasion_Type"),
            "Hair_Type": v("Hair_Type"),
            "Mineral_Source": v("Mineral_Source"),
            "Caffeine_Content": v("Caffeine_Content"),
            "Capacity": (v("Capacity") or net_qty_raw or None),
            "Variants": [],
        }


GT_FIELD_ORDER = [
    "Category", "Sub_Category", "Product_Name", "Net_Quantity", "UOM",
    "Product_Description", "Bullet_Point", "Manufacturer", "Country_Of_Origin",
    "Images", "Brand", "Pack_Quantity", "Pack_Size", "UPC_or_EAN", "FSSAI_no",
    "Dimension", "Preservatives", "Preservatives_Details", "Flavours_or_Spices",
    "Diet_Type", "Ready_to_cook", "Ready_to_eat", "Ingredients", "Nutritional_Details",
    "Usage_Details", "Storage_Instructions", "Recommended_Age", "Baby_Weight",
    "Absorption_Duration", "Scented_Flavoured", "Herbal_or_Ayurvedic",
    "Theme_or_Occasion_Type", "Hair_Type", "Mineral_Source", "Caffeine_Content",
    "Capacity", "Variants",
]

# GT-shaped keys -> the internal ProductExtraction key, same mapping as
# stage2/vlm_direct.py's _GT_TO_INTERNAL_KEY (kept in sync by hand - small and stable).
_GT_TO_INTERNAL_KEY = {
    "Product_Name": "Product Name",
    "Country_Of_Origin": "Country Of Origin",
    "UPC_or_EAN": "Upc / Ean",
    "FSSAI_no": "Fssai No",
}
_SKIP_FIELDS = {"Category", "Sub_Category", "Images", "Variants"}


# ============================================================== PROMPT SECTION
# Copied from stage2/vlm_direct.py's _SYSTEM_PROMPT - same field-confusion rules, same
# casing/thoroughness fixes already validated against real products via the Gemini path this
# session. Keep these two files in sync by hand if the prompt changes again.

_SYSTEM_PROMPT = (
    "You extract ONE structured product record from the attached product image(s) - there "
    "may be several images (front, back, nutrition panel) of the SAME product; use all of "
    "them together, since different facts sit on different faces of the pack.\n"
    "Fill a field ONLY when the image(s) actually show it; otherwise set it to null. Never "
    "guess, infer from world knowledge, or auto-complete - most of all codes, numbers, dates "
    "and country of origin. Copy numbers, units, codes, and long text fields (Ingredients, "
    "Nutritional_Details, Usage_Details, Storage_Instructions, Product_Description, "
    "Bullet_Point) EXACTLY as printed, not paraphrased.\n"
    "\n"
    "FIELD-CONFUSION RULES - these specific mix-ups are common, watch for them:\n"
    "- Manufacturer is the legal company that MAKES or MARKETS the product, printed in fine "
    "print (usually starting 'Manufactured by:', 'Marketed by:', or similar), e.g. 'Pushpam "
    "Foods & Beverages Pvt. Ltd.' or 'Hindustan Unilever Ltd.'. It is NEVER the Brand name, "
    "and NEVER a celebrity/athlete/endorser's name printed as part of marketing copy (e.g. "
    "if a cricketer's name and signature appear as an endorsement, that is NOT the "
    "Manufacturer - leave Manufacturer null unless a real company name is separately printed).\n"
    "- Net_Quantity + UOM is the TOTAL contents of THIS package as sold (e.g. '330 ml', "
    "'400 g'), usually printed near the barcode or on the front of pack as the pack size. It "
    "is NEVER a per-serving number from the nutrition/serving-size line (e.g. 'Serving Size: "
    "30 g' or 'Per 100 ml' is NOT the Net_Quantity - those describe a portion, not the whole "
    "package). If you cannot find a total-package quantity distinct from serving-size "
    "numbers, leave Net_Quantity, UOM, and Capacity null rather than reusing a serving-size "
    "figure.\n"
    "- Pack_Size and Pack_Quantity describe MULTI-UNIT packs (e.g. 'Pack of 6' -> "
    "Pack_Quantity 6). They are NEVER a nutritional value or calorie count - if you're about "
    "to put something like a kcal number in Pack_Size, that is wrong; leave it null instead. "
    "A '13 servings per pack' or 'servings per container' line on a nutrition panel is NOT "
    "Pack_Quantity either - that describes how many servings the contents divide into, not "
    "how many physical units are in the box. Leave Pack_Quantity null unless the pack "
    "explicitly says something like 'Pack of N' or shows N distinct physical units.\n"
    "- Bullet_Point is ONLY short marketing claims/callouts about the product itself (e.g. "
    "'Zero Sugar', 'No Added Preservatives', 'High Protein'). It is NEVER a licence/FSSAI "
    "number, a company name, or an address fragment. BE THOROUGH: scan every face of the "
    "pack for every distinct short claim, badge, stat callout (e.g. 'High Caffeine (99mg/330ml "
    "Can)'), usage tip (e.g. 'Tastes best when chilled'), and hashtag/slogan (e.g. "
    "'#PLAYTHEDAY') - most products carry somewhere between 3 and 8 of these, not 1-2. Missing "
    "one is worse than listing one extra genuine claim; only skip a phrase if it is a near-"
    "duplicate of one you already listed.\n"
    "- Mineral_Source is ONLY for a product explicitly MARKETED as a source of that mineral "
    "(e.g. a bottled water's front-of-pack 'Rich in Magnesium' claim). It is NEVER inferred "
    "just because a mineral compound (e.g. 'Magnesium Carbonate') appears somewhere in the "
    "Ingredients list - an ingredient is not a marketing claim. Leave it null unless the pack "
    "itself calls the mineral out as a feature.\n"
    "- Caffeine_Content is ONLY the short caffeine amount as printed (e.g. '99mg/330ml Can', "
    "'80 mg'), never the surrounding marketing sentence or badge text around it (that fuller "
    "phrase belongs in Bullet_Point instead, not duplicated here).\n"
    "- Recommended_Age covers BOTH a baby/child suitability age AND an adult minimum-age / "
    "age-gate warning (e.g. energy drinks or supplements often print '18+' or 'not "
    "recommended for children' near the caution text) - check caution/warning text for this, "
    "not only dedicated baby-product age labelling.\n"
    "\n"
    "CASING: for SHORT categorical fields (Brand, Product_Name, Country_Of_Origin, "
    "Flavours_or_Spices, Diet_Type, Caffeine_Content, Mineral_Source, and similar single-"
    "value fields), normalize to natural title/sentence case (e.g. 'India', not 'INDIA'; "
    "'Mixed Fruit Flavour', not 'MIXED FRUIT FLAVOUR') even if the pack prints them in all "
    "caps - packaging typography is not the same as correct casing. For LONG free-text "
    "fields (Ingredients, Nutritional_Details, Usage_Details, Storage_Instructions, "
    "Product_Description, Bullet_Point), keep the printed casing/wording verbatim instead - "
    "do not paraphrase or re-case those.\n"
    "Read brand/product names carefully for embedded digits or stylized characters (e.g. "
    "'one8', 'Mi3') - do not drop or substitute a digit that's part of the name.\n"
    "\n"
    "For a Yes/No field (Preservatives, Ready_to_cook, Ready_to_eat, Herbal_or_Ayurvedic): "
    "only answer 'Yes' or 'No' when the pack's printed text actually states it either way. "
    "If the pack is silent on it, use null - do not default to 'No' as a guess just because "
    "the claim isn't printed; 'not shown' and 'shown as No' are different things.\n"
    "Bullet_Point is a JSON array of short printed claims/callouts (e.g. 'No Added "
    "Preservatives', 'Zero Sugar') - use [] if none are shown, not null.\n"
    "Respond with ONLY the JSON object - no commentary, no markdown code fence, no "
    "explanation before or after it."
)

_NOT_A_BULLET_RE = re.compile(
    r"\b(?:lic\.?\s*no\.?|fssai|licence\s*no|license\s*no)\b|\d{10,}", re.I
)


def _looks_like_bullet(text: str) -> bool:
    if _NOT_A_BULLET_RE.search(text):
        return False
    return len(text.split()) <= 12


def _load_schema_description(schema_path: Path) -> str:
    try:
        raw = json.loads(schema_path.read_text(encoding="utf-8"))
        lines = []
        for name, spec in raw.get("properties", {}).items():
            if name in _SKIP_FIELDS:
                continue
            desc = spec.get("description", "")
            enum = spec.get("enum")
            extra = f" One of: {enum}." if enum else ""
            lines.append(f"- {name} ({spec.get('type', 'string')}): {desc}{extra}")
        return "\n".join(lines)
    except (OSError, json.JSONDecodeError):
        return "\n".join(f"- {f}" for f in GT_FIELD_ORDER if f not in _SKIP_FIELDS)


def _build_user_prompt(schema_path: Path) -> str:
    return (
        "Extract one product record as a JSON object with EXACTLY these fields:\n\n"
        + _load_schema_description(schema_path)
        + "\n\nRespond with only the JSON object."
    )


def _apply_vlm_dict(extraction: ProductExtraction, vlm_dict: dict, confidence: float) -> None:
    net_qty = vlm_dict.get("Net_Quantity")
    uom = vlm_dict.get("UOM")
    if net_qty is not None:
        combined = f"{net_qty} {uom}".strip() if uom else str(net_qty)
        extraction.set("Net Quantity", combined, confidence, "cloud_vlm")

    for bullet in (vlm_dict.get("Bullet_Point") or []):
        bullet = str(bullet).strip()
        if bullet and _looks_like_bullet(bullet):
            extraction.add_bullet(bullet)

    handled = _SKIP_FIELDS | {"Net_Quantity", "UOM", "Bullet_Point"}
    for key, value in vlm_dict.items():
        if key in handled:
            continue
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        internal_key = _GT_TO_INTERNAL_KEY.get(key, key)
        extraction.set(internal_key, str(value), confidence, "cloud_vlm")


# =========================================================== JSON REPAIR SECTION
# Copied from ocr/local_vlm.py - same "salvage a response cut off mid-object" logic.

def _strip_fence(text: str) -> str:
    fenced = text.strip()
    if fenced.startswith("```"):
        fenced = fenced.split("\n", 1)[1] if "\n" in fenced else fenced
        if fenced.endswith("```"):
            fenced = fenced[:-3]
        fenced = fenced.strip()
        if fenced.startswith("json"):
            fenced = fenced[4:].strip()
    return fenced


def _repair_truncated_json(text: str) -> dict | None:
    text = text.strip()
    if not text.startswith("{"):
        return None
    for cut in range(len(text) - 1, 0, -1):
        candidate = text[:cut].rstrip().rstrip(",") + "}"
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and len(parsed) >= 3:
            return parsed
    return None


def _parse_model_output(text: str) -> dict | None:
    fenced = _strip_fence(text)
    try:
        parsed = json.loads(fenced)
    except json.JSONDecodeError:
        parsed = _repair_truncated_json(fenced)
    return parsed if isinstance(parsed, dict) else None


# ============================================================== PRODUCT GROUPING
# Copied from products.py - same barcode-token grouping used everywhere else in this
# project, so the resolved image set for row N matches ground_truth_100/rowN.json exactly.

_VIEW = re.compile(r"^Image_(\d+)_(.+)\.jpg$", re.IGNORECASE)


def _product_key(filename: str) -> str | None:
    m = _VIEW.match(Path(filename).name)
    return m.group(2) if m else None


def _view_number(filename: str) -> int:
    m = _VIEW.match(Path(filename).name)
    return int(m.group(1)) if m else 0


def _group_for(seed_filenames: list[str], all_names: list[str]) -> list[str]:
    seed = next((n for n in seed_filenames if _product_key(n)), None)
    if seed is None:
        return list(seed_filenames)
    group = sorted((n for n in all_names if _product_key(n) == _product_key(seed)), key=_view_number)
    return group or [seed]


# ==================================================================== MODEL REGISTRY
# One entry per candidate. `notes` records what's actually been validated vs. still a guess -
# be honest here, don't let an unverified entry look as solid as a tested one.

MODEL_REGISTRY = {
    "qwen3.6-27b": {
        "repo_id": "Qwen/Qwen3.6-27B",
        "trust_remote_code": True,
        "notes": (
            "PRIORITY 1 - first model to validate this whole script end-to-end. Confirmed "
            "vision-capable with community MLX/GGUF builds already existing (implies the "
            "base HF checkpoint itself is solid), no smaller sibling fits the local 16GB "
            "Mac. NOT yet run against real images by this script - the chat-template/image "
            "content-block format below is the standard modern transformers VLM pattern, "
            "but hasn't been verified against this exact model. Needs Kaggle's dual-T4 "
            "(~13.5-15GB at 4-bit alone leaves little headroom on a single T4)."
        ),
    },
    "phi4-reasoning-vision-15b": {
        "repo_id": "microsoft/Phi-4-reasoning-vision-15B",  # verify exact repo id before use
        "trust_remote_code": True,
        "notes": (
            "PRIORITY 2 - the clearest case where cloud+transformers should succeed where "
            "local mlx-vlm/llama.cpp failed: every GGUF/MLX conversion found drops the "
            "vision weights, but that's a conversion bug, not a base-model problem. Fits a "
            "single 16GB GPU (~9GB at 4-bit) - Colab is fine, doesn't need Kaggle's dual-T4. "
            "UNVALIDATED - confirm the real HF repo id first (SigLIP-2 + Phi-4-reasoning)."
        ),
    },
    "kimi-vl-a3b-thinking": {
        "repo_id": "moonshotai/Kimi-VL-A3B-Thinking-2506",
        "trust_remote_code": True,
        "notes": (
            "PRIORITY 3 - 'A3B' is ACTIVE params only; total is ~16B (~8GB at 4-bit), fits a "
            "single 16GB GPU. GGUF/MLX both lack a working vision path for this MoE "
            "architecture, so this MUST go through transformers directly. UNVALIDATED."
        ),
    },
    # PRIORITY 4 (GLM-4.7-Flash's real vision sibling, GLM-4.6V-Flash) and PRIORITY 5
    # (Nemotron Nano 2 VL) already fit the LOCAL Mac per this session's research - only worth
    # adding here if a local mlx-vlm/llama.cpp attempt hits a runtime bug, same reasoning as
    # phi4/kimi-vl above.
    # DiffusionGemma deliberately not registered yet: genuinely diffusion (not autoregressive)
    # architecture - standard `.generate()` below almost certainly does not apply. Needs its
    # own sampling loop; treat as a separate, higher-effort follow-up, not a registry entry.
}


def _load_model(model_key: str):
    """Deferred heavy imports - keep this file importable (e.g. for --help) even before
    `pip install transformers accelerate bitsandbytes` has run."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    cfg = MODEL_REGISTRY[model_key]
    print(f"Loading {cfg['repo_id']} (4-bit, device_map=auto)... this can take several minutes "
          f"on first run (downloading weights).")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4",
    )
    processor = AutoProcessor.from_pretrained(cfg["repo_id"], trust_remote_code=cfg["trust_remote_code"])
    model = AutoModelForImageTextToText.from_pretrained(
        cfg["repo_id"], quantization_config=bnb_config, device_map="auto",
        trust_remote_code=cfg["trust_remote_code"],
    )
    return model, processor


def _call_model(model, processor, image_paths: list[str], system: str, user: str) -> str:
    """Standard modern transformers VLM chat-template pattern. If a specific model's
    processor expects a different image content-block shape (some want {"type": "image",
    "url": ...} or raw PIL objects inline), this is the one place to special-case it - add an
    `if cfg.get("custom_call"): ...` branch keyed off the registry entry rather than
    forking the whole function."""
    images = [Image.open(p).convert("RGB") for p in image_paths]
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": [*({"type": "image", "image": img} for img in images),
                                      {"type": "text", "text": user}]},
    ]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=images, return_tensors="pt").to(model.device)
    output_ids = model.generate(**inputs, max_new_tokens=3000, do_sample=False)
    new_tokens = output_ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()


def main(model_key: str, data_dir: Path, start: int, n: int, out_dir: Path) -> None:
    gt_csv = data_dir / "Ground Truth.csv"
    dataset_dir = data_dir / "images"
    schema_path = data_dir / "DigiCatalog_groceries_schema_format.json"

    df = pd.read_csv(gt_csv, low_memory=False)
    cols = [c for c in df.columns if "Filename" in c]
    all_names = sorted(p.name for p in dataset_dir.glob("*.jpg"))
    user_prompt = _build_user_prompt(schema_path)

    model, processor = _load_model(model_key)
    out_dir.mkdir(parents=True, exist_ok=True)

    end = min(start + n, len(df))
    written, failed = 0, 0
    for i in range(start, end):
        row = df.iloc[i]
        seeds = [str(row[c]).strip() for c in cols if isinstance(row[c], str) and str(row[c]).strip()]
        group = _group_for(seeds, all_names)
        paths = [str(dataset_dir / nm) for nm in group if (dataset_dir / nm).exists()]
        if not paths:
            print(f"row{i}: no resolvable images, skipping")
            continue

        try:
            raw_text = _call_model(model, processor, paths, _SYSTEM_PROMPT, user_prompt)
            vlm_dict = _parse_model_output(raw_text)
        except Exception as e:  # noqa: BLE001 - a batch job should keep going past one bad row
            print(f"row{i}: call FAILED ({type(e).__name__}: {e})")
            failed += 1
            continue

        extraction = ProductExtraction()
        if vlm_dict:
            _apply_vlm_dict(extraction, vlm_dict, confidence=0.85)
        else:
            print(f"row{i}: model responded but output wasn't parseable JSON - writing an empty record")

        record = {k: extraction.to_ground_truth_dict().get(k) for k in GT_FIELD_ORDER}
        with open(out_dir / f"row{i}.json", "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
            f.write("\n")
        written += 1
        print(f"row{i}: done ({written}/{end - start})")

    zip_path = out_dir.parent / f"{out_dir.name}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in out_dir.glob("row*.json"):
            zf.write(p, arcname=p.name)
    print(f"wrote {written} record(s), {failed} hard failure(s), to {out_dir} - zipped to {zip_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODEL_REGISTRY), help="which registered model to run")
    ap.add_argument("--data", type=Path, required=True,
                    help="the unzipped cloud_eval_data.zip folder (contains images/, Ground Truth.csv, the schema json)")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=5, help="start small (5) to prove the model works before a full run")
    ap.add_argument("--out", type=Path, default=Path("results"))
    a = ap.parse_args()
    main(a.model, a.data, a.start, a.n, a.out)
