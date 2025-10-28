# SPDX-License-Identifier: CECILL-2.1
# Copyright (c) 2025 Synchrotron SOLEIL

"""
_version.py — library version
"""

from importlib.metadata import PackageNotFoundError, version

__all__ = ["__version__"]

try:
    __version__ = version("barc4beams")
except PackageNotFoundError:

    try:
        import tomllib
        from pathlib import Path

        root = Path(__file__).resolve().parents[2]
        data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
        __version__ = data.get("project", {}).get("version", "0+unknown")
    except Exception:
        __version__ = "0+unknown"