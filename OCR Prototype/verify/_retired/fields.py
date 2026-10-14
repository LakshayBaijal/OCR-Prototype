"""Which fields we verify, and how much a change to each one actually matters.

Not every field carries the same weight. If the enhancement quietly rewrites the
marketing blurb, that's cosmetic. If it rewrites the FSSAI licence number or flips
the veg mark, that is a compliance defect that would flow straight into the ONDC
catalog. Treating those two the same - one flat "N fields changed" number - would
bury the second in the first.

So every field carries a TIER, and the tier decides the overall verdict
(`verify.report.overall_verdict`), not the raw count of differences:

  CRITICAL  a mismatch is disqualifying. Regulated or identity-bearing: the barcode
            IS the product's identity, the FSSAI number is a licence, MRP is the
            legally printed price, and Diet_Type is the veg/non-veg mark that Indian
            food labelling law mandates. One wrong digit here is a wrong product.
  IMPORTANT a mismatch means a human should look before this image is used. Real
            catalog content (name, manufacturer, ingredients) where OCR itself is
            noisy enough that an automatic reject would be wrong too often.
  MINOR     informational. Marketing copy and soft keyword fields, where the
            heuristic extractors are lowest-confidence by their own admission
            (see stage2/heuristic.py's `_extract_keyword_field` docstring).

Field keys are the INTERNAL `ProductExtraction` keys that `stage2.heuristic` writes -
a mix of "Net Quantity"-style and "Diet_Type"-style names, because that's genuinely
what that module emits (see stage2/vlm_direct.py's `_GT_TO_INTERNAL_KEY` for the same
seam). Comparing on those keys means both extraction paths land on one key per field
with no remapping layer to drift out of sync.
"""

from __future__ import annotations

from enum import Enum


class Tier(str, Enum):
    CRITICAL = "CRITICAL"
    IMPORTANT = "IMPORTANT"
    MINOR = "MINOR"


# (internal key, display label, tier). Ordered most-consequential first, which is
# also the order the UI table renders - a reviewer reads top-down and the things that
# can sink the image are the first things they see.
FIELD_CATALOGUE: list[tuple[str, str, Tier]] = [
    ("Upc / Ean",             "Barcode (UPC/EAN)",      Tier.CRITICAL),
    ("Fssai No",              "FSSAI licence no.",      Tier.CRITICAL),
    ("Mrp",                   "MRP",                    Tier.CRITICAL),
    ("Net Quantity",          "Net quantity",           Tier.CRITICAL),
    ("Diet_Type",             "Veg / Non-veg",          Tier.CRITICAL),
    ("Brand",                 "Brand",                  Tier.CRITICAL),

    ("Product Name",          "Product name",           Tier.IMPORTANT),
    ("Manufacturer",          "Manufacturer",           Tier.IMPORTANT),
    ("Country Of Origin",     "Country of origin",      Tier.IMPORTANT),
    ("Ingredients",           "Ingredients",            Tier.IMPORTANT),
    ("Nutritional_Details",   "Nutritional details",    Tier.IMPORTANT),
    ("Pack_Quantity",         "Pack quantity",          Tier.IMPORTANT),
    ("Storage_Instructions",  "Storage instructions",   Tier.IMPORTANT),
    ("Usage_Details",         "Usage / directions",     Tier.IMPORTANT),

    ("Product_Description",   "Description",            Tier.MINOR),
    ("Flavours_or_Spices",    "Flavour",                Tier.MINOR),
    ("Preservatives",         "Preservatives",          Tier.MINOR),
    ("Ready_to_cook",         "Ready to cook",          Tier.MINOR),
    ("Ready_to_eat",          "Ready to eat",           Tier.MINOR),
    ("Herbal_or_Ayurvedic",   "Herbal / ayurvedic",     Tier.MINOR),
    ("Caffeine_Content",      "Caffeine content",       Tier.MINOR),
    ("Recommended_Age",       "Recommended age",        Tier.MINOR),
    ("Dimension",             "Dimensions",             Tier.MINOR),
    ("Mineral_Source",        "Mineral source",         Tier.MINOR),
    ("Theme_or_Occasion_Type", "Theme / occasion",      Tier.MINOR),
    ("Hair_Type",             "Hair type",              Tier.MINOR),
]

FIELD_KEYS = [k for k, _, _ in FIELD_CATALOGUE]
LABELS = {k: label for k, label, _ in FIELD_CATALOGUE}
TIERS = {k: tier for k, _, tier in FIELD_CATALOGUE}


def label(key: str) -> str:
    return LABELS.get(key, key)


def tier(key: str) -> Tier:
    return TIERS.get(key, Tier.MINOR)
