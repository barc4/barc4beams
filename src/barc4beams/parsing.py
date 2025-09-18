# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2025 Synchrotron SOLEIL

"""
parsing.py - parsing helpers for SHADOW outputs 
"""

from __future__ import annotations

import re
from typing import Optional, Sequence

import numpy as np


def parse_shadow_sys_info(text: str, labels: Optional[Sequence[str]] = None) -> dict:
    """
    Parse SHADOW optical-element positions from text, ignoring primed rows (e.g., 3').
    Optionally attach user-provided labels and infer element *kind*.

    Kinds (codes)
    -------------
    - 'SRC' : source (index 0 is always the source)
    - 'O'   : screen / observation point
    - 'M'   : mirror
    - 'G'   : grating
    - 'C'   : crystal
    - 'S'   : slit
    - 'E'   : empty element (coordinate break / rotation / placeholder)
    - 'F'   : focus (designated focal plane)
    - 'X'   : experiment (endstation / sample environment)

    Label conventions (case-insensitive)
    ------------------------------------
    • Empty string → 'E'
    • Startswith: 'M'→M, 'G'→G, 'C'→C, 'S'→S, 'F' or 'FOC'→F, 'X' or 'EXP'→X, 'O' or 'SCR'→O
    • Anything else (non-empty) → 'O'
    • Element 0 is always 'SRC' regardless of label.

    Returns
    -------
    dict
        {
          "x": np.ndarray,
          "y": np.ndarray,
          "z": np.ndarray,
          "elements": {
            "labels": list[str],   # as provided (trimmed) or "" if None
            "kinds":  list[str],   # codes in {'SRC','O','M','G','C','S','E','F','X'}
          }
        }
    """
    # Match only UNPRIMED indices: number not immediately followed by a prime.
    pattern = r"^\s*(\d+)(?!')\s+([-.\dEe+]+)\s+([-.\dEe+]+)\s+([-.\dEe+]+)\s*$"
    matches = re.findall(pattern, text, re.MULTILINE)

    if not matches:
        return {"x": np.array([]), "y": np.array([]), "z": np.array([]),
                "elements": {"labels": [], "kinds": []}}

    x, y, z = [], [], []
    for _, xx, yy, zz in matches:
        # SHADOW -> our convention (invert X)
        x.append(-1.0 * float(xx))
        y.append(float(yy))
        z.append(float(zz))

    n = len(matches)

    # Validate/normalize labels
    if labels is None:
        labels_list = [""] * n
    else:
        if len(labels) != n:
            raise ValueError(f"`labels` must have length {n} (unprimed elements), got {len(labels)}.")
        labels_list = [str(lbl).strip() for lbl in labels]

    def classify(idx: int, lbl: str) -> str:
        if idx == 0:
            return "SRC"
        s = lbl.strip().upper()
        if s == "":
            return "E"
        # synonyms / startswith
        if s.startswith("M"):       return "M"
        if s.startswith("G"):       return "G"
        if s.startswith("C"):       return "C"
        if s.startswith("S"):       return "S"
        if s.startswith("F") or s.startswith("FOC"): return "F"
        if s.startswith("X") or s.startswith("EXP"): return "X"
        if s.startswith("O") or s.startswith("SCR"): return "O"
        # default non-empty → observation point
        return "O"

    kinds = [classify(i, lbl) for i, lbl in enumerate(labels_list)]

    return {
        "x": np.asarray(x),
        "y": np.asarray(y),
        "z": np.asarray(z),
        "elements": {"labels": labels_list, "kinds": kinds},
    }