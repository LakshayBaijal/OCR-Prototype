


"""Best-effort free-memory check (macOS), used to warn before a run that could get the
whole process OOM-killed - the exact failure mode already hit twice on this machine: the
MLX server (3-4GB resident) plus Streamlit/RapidOCR running at once, with no in-app error,
no traceback - the process just disappears. A crash with no warning is worse than a warning
that is occasionally too cautious, so this errs toward warning early.

Not a general-purpose tool: parses `vm_stat`, which is macOS-only. On any other platform
`free_memory_gb()` returns None and callers should skip the check rather than guess.
"""

from __future__ import annotations

import re
import subprocess

_PAGE_RE = re.compile(r"page size of (\d+) bytes")
_FIELD_RE = re.compile(r"^(Pages [a-z ]+?):\s+(\d+)\.", re.MULTILINE)


def free_memory_gb() -> float | None:
    """Free + inactive + purgeable pages, in GB - macOS's own rough proxy for memory that
    is either already free or reclaimable without swapping. None if vm_stat isn't available
    (non-macOS, or the command failed) - callers should treat that as "unknown", not "zero".
    """
    try:
        out = subprocess.run(
            ["vm_stat"], capture_output=True, text=True, timeout=3, check=True
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None

    page_match = _PAGE_RE.search(out)
    page_size = int(page_match.group(1)) if page_match else 4096

    fields = {k.strip(): int(v) for k, v in _FIELD_RE.findall(out)}
    reclaimable = (
        fields.get("Pages free", 0)
        + fields.get("Pages inactive", 0)
        + fields.get("Pages purgeable", 0)
    )
    return reclaimable * page_size / 1e9
