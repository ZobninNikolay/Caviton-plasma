#!/usr/bin/env python3
"""Verify the 32 files fixed in the 2026-08-10 registration listing."""

from __future__ import annotations

import hashlib
from pathlib import Path


FILES = (
    "plasma_level1.py", "plasma_level2.py", "plasma_level3.py",
    "plasma_level4.py", "plasma_level5.py", "plasma_extrema.py",
    "plasma_level45.py", "plasma_level46.py", "plasma_level47.py",
    "smilei_level47_full.py", "plasma_level48.py", "config.yaml",
    "config_level2.yaml", "config_level3.yaml", "config_level4.yaml",
    "config_level5.yaml", "config_extrema.yaml", "config_level45.yaml",
    "config_level46.yaml", "config_level47.yaml", "config_level48.yaml",
    "tests/test_sanity.py", "tests/test_level2.py", "tests/test_level3.py",
    "tests/test_level4.py", "tests/test_level5.py", "tests/test_extrema.py",
    "tests/test_level45.py", "tests/test_level46.py", "tests/test_level47.py",
    "tests/test_level48.py", "requirements.txt",
)
EXPECTED = "bfc4f949a1d08f720941599c768e24edfcd256412e3362120c69c8f4e582aad1"


def main() -> int:
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for relative in FILES:
        data = (root / relative).read_bytes()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(data)
        digest.update(b"\0")
    actual = digest.hexdigest()
    print(f"registered files: {len(FILES)}")
    print(f"aggregate SHA-256: {actual}")
    if actual != EXPECTED:
        print(f"ERROR: expected {EXPECTED}")
        return 1
    print("OK: registered source tree is unchanged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
