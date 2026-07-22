"""Stage 2 - Layout / Field Parser: OCR text + boxes + barcode -> structured ProductExtraction.

Public surface:
  extract_product(image_paths, ...)     -> ProductExtraction   (end-to-end for one product,
                                                                 LayoutLMv3 on by default)
  heuristic.parse(text_lines, layout_lines, barcode) -> ProductExtraction (parser only, on
                                            ready lines - text_lines for regex/keyword/block
                                            extractors, layout_lines for the brand/name
                                            position heuristic; see heuristic.parse's docstring)
  ProductExtraction.to_scoring_dict()   -> narrow 8-field dict, for json_metrics against
                                            Ground Truth.csv (unchanged, backward-compatible)
  ProductExtraction.to_ground_truth_dict() -> full ground_truth_100/rowN.json-shaped dict
  schema.TARGET_FIELDS / GT_FIELD_ORDER / ProductExtraction / PositionedLine
"""

from . import heuristic, schema
from .pipeline import collect_lines_and_barcode, extract_product
from .schema import GT_FIELD_ORDER, PositionedLine, ProductExtraction, TARGET_FIELDS

__all__ = [
    "extract_product",
    "collect_lines_and_barcode",
    "heuristic",
    "schema",
    "PositionedLine",
    "ProductExtraction",
    "TARGET_FIELDS",
    "GT_FIELD_ORDER",
]
