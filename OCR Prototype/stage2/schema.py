"""Stage 2 output contract: the structured ProductExtraction, plus its scoring view.

Stage 0 cleaned the pixels, Stage 1 read them into text + boxes + confidence, and the
barcode pipeline decoded the one field that has its own checksum. Stage 2 turns all of that
into named product FIELDS - the thing a catalog actually wants.

The field keys are the EXACT column names from Ground Truth.csv, so `to_scoring_dict()`
lines up 1:1 with a ground-truth row and `json_metrics.score(gt, pred)` can grade it with no
remapping. Every field also keeps its confidence and provenance (which rule/engine produced
it), because Stage 2 feeds a confidence-routing / human-review step downstream - a value with
no idea how sure we are is not actually useful there.
"""

from __future__ import annotations

from dataclasses import dataclass, field as _dc_field

# The fields Stage 2 tries to extract, keyed by their Ground Truth.csv column name. Ordered
# roughly front-of-pack first. Deliberately a SUBSET of the CSV's ~40 columns: these are the
# ones that are physically printed on a pack and so are actually recoverable from an image
# (unlike catalog-only columns like "Sub Domain Name" or "Uploaded On").
TARGET_FIELDS = [
    "Product Name",
    "Brand",
    "Mrp",
    "Net Quantity",
    "Upc / Ean",
    "Fssai No",
    "Manufacturer",
    "Country Of Origin",
]


@dataclass
class Field:
    """One extracted value, with how sure we are and where it came from."""

    value: str | None = None
    confidence: float = 0.0
    source: str = ""  # e.g. "regex:mrp", "barcode", "layout:largest-text", "layoutlmv3"


@dataclass
class ProductExtraction:
    """The structured record for one product, as a map of field name -> Field."""

    fields: dict[str, Field] = _dc_field(default_factory=dict)
    # Bullet_Point is the one target field that is a LIST, not a single scored value (a pack
    # carries several independent short claims, not one "best" claim) - kept separate from
    # `fields` rather than distorting Field's single-value/confidence shape for one field.
    bullets: list[str] = _dc_field(default_factory=list)

    def set(self, name: str, value: str | None, confidence: float, source: str) -> None:
        """Record a field, but never let a blank overwrite an existing value, and keep the
        higher-confidence answer when two rules fire on the same field."""
        if value is None or not str(value).strip():
            return
        existing = self.fields.get(name)
        if existing is None or confidence > existing.confidence:
            self.fields[name] = Field(value=str(value).strip(), confidence=confidence, source=source)

    def add_bullet(self, text: str) -> None:
        text = (text or "").strip()
        if text and text not in self.bullets:
            self.bullets.append(text)

    def to_scoring_dict(self) -> dict:
        """Flat {GT column name: value} view for json_metrics.score(). Missing fields are
        None so an abstention scores as an abstention, not as a wrong guess."""
        return {k: (self.fields[k].value if k in self.fields else None) for k in TARGET_FIELDS}

    def provenance(self) -> dict:
        """{field: "value  (conf, source)"} - for display / the routing step."""
        out = {}
        for k in TARGET_FIELDS:
            f = self.fields.get(k)
            out[k] = f"{f.value}   ({f.confidence:.2f}, {f.source})" if f else "—"
        return out

    def to_ground_truth_dict(self) -> dict:
        """Same JSON shape as `ground_truth_100/rowN.json` (the ONDC DigiCatalog Grocery
        product schema) - so this pipeline's output can be dropped next to, and diffed
        against, the hand/image-derived ground truth.

        Two fields are structurally always-empty here, not extraction bugs:
          * `Category` is a fixed constant ("Grocery") for this whole dataset.
          * `Sub_Category` and `Images` are catalog/eval-harness metadata, not something
            printed on a pack, so they're not this pipeline's job to fill (left null / []).
          * `Variants` and `Pack_Size` have no reliable OCR signal distinguishing them from
            Net_Quantity/Capacity, so they stay null/[] rather than guess.
        """
        f = self.fields
        def v(name: str) -> str | None:
            return f[name].value if name in f else None

        def safe_int(s: str | None) -> int | None:
            """Best-effort int coercion. A field's value isn't always the heuristic's own
            clean digit-only regex match - it might be LayoutLMv3's guess instead (e.g. it
            can mislabel a nutrition-panel percentage like "0.1%" as Pack_Quantity), which is
            noisy by construction (see stage2/layoutlm.py). Malformed input becomes an honest
            None, never a crash and never a fabricated number."""
            if not s:
                return None
            try:
                return int(float(s))
            except ValueError:
                return None

        net_qty_raw = v("Net Quantity")  # e.g. "330 ml" - split into amount + unit
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
            # Prefer an explicit Capacity guess (e.g. LayoutLMv3's PACK_QUANTITY-adjacent
            # signal, if ever wired to this key) but fall back to mirroring Net_Quantity+UOM,
            # which is the reliable, already-working source for the common case.
            "Capacity": (v("Capacity") or net_qty_raw or None),
            "Variants": [],
        }


# The full ground_truth_100/rowN.json key order, exported so callers (e.g. the export
# script) can assert/round-trip against it without hard-coding the list a second time.
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


@dataclass
class PositionedLine:
    """One OCR line with enough geometry for the layout heuristics.

    `bbox` is the 4 corner points in the crop's own pixel space; `img_w`/`img_h` are that
    crop's dimensions, so a line's size and position can be judged RELATIVE to its image
    (a brand name is "big relative to its pack", not "big in absolute pixels").
    """

    text: str
    bbox: list[tuple[int, int]]
    confidence: float | None
    img_w: int
    img_h: int

    @property
    def height_frac(self) -> float:
        """Line height as a fraction of image height - a font-size proxy for 'is this a big,
        prominent line' (brand / product name) vs 'fine print' (ingredients, licence)."""
        ys = [p[1] for p in self.bbox]
        return (max(ys) - min(ys)) / self.img_h if self.img_h else 0.0

    @property
    def top_frac(self) -> float:
        """Vertical position of the line's top, 0 (top of image) .. 1 (bottom)."""
        return min(p[1] for p in self.bbox) / self.img_h if self.img_h else 0.0
