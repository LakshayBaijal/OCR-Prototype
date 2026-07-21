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

    def set(self, name: str, value: str | None, confidence: float, source: str) -> None:
        """Record a field, but never let a blank overwrite an existing value, and keep the
        higher-confidence answer when two rules fire on the same field."""
        if value is None or not str(value).strip():
            return
        existing = self.fields.get(name)
        if existing is None or confidence > existing.confidence:
            self.fields[name] = Field(value=str(value).strip(), confidence=confidence, source=source)

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
