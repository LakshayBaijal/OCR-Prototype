"""Group a product's associated images together.

Every file in Dataset/ is named `Image_<view>_<barcode>.0.jpg` - e.g.

    Image_1_8901030834691.0.jpg   (front)
    Image_2_8901030834691.0.jpg   (back)
    Image_3_8901030834691.0.jpg   (side / nutrition panel)

The `<view>` number differs per shot; the `<barcode>` token is the SAME across every
image of one product, and it is exactly what the ground-truth CSV rows group on. So the
barcode token is the product key: images that share it are the same product, photographed
from different angles, and belong together (one has the front, another the back panel with
MRP/FSSAI/barcode, etc.).

This is why scoring - and reading - must be done per PRODUCT, not per image: the FSSAI
number lives on exactly one face, so a single front-of-pack shot can never contain it.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

# Image_<view>_<barcode>.jpg  ->  (view number, product key). The product key keeps the
# ".0" the dataset appends; it is only ever compared to itself, never parsed as a number.
_VIEW = re.compile(r"^Image_(\d+)_(.+)\.jpg$", re.IGNORECASE)


def product_key(filename: str) -> str | None:
    """The shared barcode token that identifies a product, or None for an odd filename."""
    m = _VIEW.match(Path(filename).name)
    return m.group(2) if m else None


def view_number(filename: str) -> int:
    """The per-shot view index (1..5), used only to order a product's images front-first."""
    m = _VIEW.match(Path(filename).name)
    return int(m.group(1)) if m else 0


def group_index(all_names: list[str]) -> dict[str, list[str]]:
    """Map every product key -> its image filenames, ordered by view number.

    Built once over the whole dataset listing; O(n) and cheap to cache.
    """
    idx: dict[str, list[str]] = defaultdict(list)
    for n in all_names:
        key = product_key(n)
        if key is not None:
            idx[key].append(n)
    return {k: sorted(v, key=view_number) for k, v in idx.items()}


def associated(filename: str, all_names: list[str]) -> list[str]:
    """Every image belonging to the same product as `filename`, view-ordered.

    Falls back to just the file itself if the name does not match the dataset pattern, so
    a caller always gets a non-empty group.
    """
    key = product_key(filename)
    name = Path(filename).name
    if key is None:
        return [name]
    group = sorted(
        (n for n in all_names if product_key(n) == key), key=view_number
    )
    return group or [name]


def group_for(seed_filenames: list[str], all_names: list[str]) -> list[str]:
    """Full product group for whichever of `seed_filenames` names a real product.

    Used by the benchmarks: given the image filenames a ground-truth CSV row lists, return
    every image of that product (view-ordered) via the same barcode-token grouping the app
    uses - so there is ONE grouping definition across the whole codebase. Verified
    equivalent to the row's own Filename columns on this dataset. Falls back to the seeds
    themselves if none match the dataset pattern.
    """
    seed = next((n for n in seed_filenames if product_key(n)), None)
    return associated(seed, all_names) if seed else list(seed_filenames)
