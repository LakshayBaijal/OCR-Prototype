"""Stage 2 - Layout / Field Parser: OCR text + boxes + barcode -> structured ProductExtraction.

Public surface:
  extract_product(image_paths, ...)  -> ProductExtraction   (end-to-end for one product)
  heuristic.parse(lines, barcode)    -> ProductExtraction   (parser only, on ready lines)
  schema.TARGET_FIELDS / ProductExtraction / PositionedLine
"""

from . import heuristic, schema
from .pipeline import collect_lines_and_barcode, extract_product
from .schema import PositionedLine, ProductExtraction, TARGET_FIELDS

__all__ = [
    "extract_product",
    "collect_lines_and_barcode",
    "heuristic",
    "schema",
    "PositionedLine",
    "ProductExtraction",
    "TARGET_FIELDS",
]
