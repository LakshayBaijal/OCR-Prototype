"""Persist every benchmark's report to the Evaluation/ folder.

The benchmark scripts print a human-readable report to the terminal. Wrapping that
print block in `capture(name)` ALSO writes it verbatim to

    Evaluation/<name>_latest.txt

overwritten on every run, so there is exactly one results file per benchmark - the most
recent one - rather than a pile of timestamped history. The saved file's header still
records WHEN it was run and the exact command, so you never lose that.

Only stdout (the `print()` report) is captured - RapidOCR's logging goes to stderr, so
the saved file stays clean of engine noise.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

EVAL_DIR = Path(__file__).parent / "Evaluation"


class _Tee:
    """Write to several streams at once (the real terminal AND the results file)."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data: str) -> int:
        for s in self._streams:
            s.write(data)
        return len(data)

    def flush(self) -> None:
        for s in self._streams:
            s.flush()


@contextmanager
def capture(name: str):
    """Tee everything printed inside the block to Evaluation/<name>_latest.txt.

    Overwrites any previous run of the same benchmark. Yields the Path written to. The
    terminal output is unchanged; the file gets a small header (timestamp + the exact
    command) followed by the verbatim report.
    """
    EVAL_DIR.mkdir(exist_ok=True)
    ts = datetime.now()
    path = EVAL_DIR / f"{name}_latest.txt"

    with open(path, "w") as f:
        f.write(f"# {name}\n")
        f.write(f"# run at : {ts:%Y-%m-%d %H:%M:%S}\n")
        f.write(f"# command: {' '.join(sys.argv)}\n\n")
        f.flush()

        terminal = sys.stdout
        sys.stdout = _Tee(terminal, f)
        try:
            yield path
        finally:
            sys.stdout = terminal

    print(f"\n[saved] {path.relative_to(EVAL_DIR.parent)}")
