#!/usr/bin/env python3
"""Compile the Beamer deck with tectonic."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "slides" / "slides_letemps_parlacap.tex"


def main() -> None:
    if shutil.which("tectonic") is None:
        raise SystemExit("tectonic is required (https://tectonic-typesetting.github.io).")
    subprocess.run(["tectonic", "-X", "compile", str(TEX.relative_to(ROOT))],
                   cwd=ROOT, check=True)
    shutil.copy2(TEX.with_suffix(".pdf"), ROOT / "slides_letemps_parlacap.pdf")


if __name__ == "__main__":
    main()
