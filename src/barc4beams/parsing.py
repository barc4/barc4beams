# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2025 Synchrotron SOLEIL

"""
parsing.py - 
"""

from __future__ import annotations

import re
from typing import Optional, Sequence

import numpy as np


def parse_shadow_info(text: str, labels: Optional[Sequence[str]] = None) -> dict:
    """
    Parse SHADOW optical element positions from text, ignoring primed rows (e.g., 3').
    Optionally attach user-provided labels and inferred type codes.

    Parameters
    ----------
    text : str
        SHADOW output text containing optical element positions.
    labels : sequence of str, optional
        Custom labels for each UNPRIMED element (must match the number of unprimed rows).
        Conventions:
          - Element 0 is the source (type 'SRC'), label can be 'SRC', 'U52', etc.
          - Empty string "" → empty element (type 'E').
          - Labels starting with 'M' → mirror (type 'M').
          - Labels starting with 'G' → grating (type 'G').
          - Labels starting with 'C' → crystal (type 'C').
          - Labels starting with 'S' → slit (type 'S').
          - Any other non-empty label → observation point (type 'O').

    Returns
    -------
    dict
        {
          "x": np.ndarray,
          "y": np.ndarray,
          "z": np.ndarray,
          "oe": {
            "labels": list[str],      # length = number of UNPRIMED elements
            "type":   list[str],      # each in {'SRC','E','M','G','C','S','O'}
          }
        }
    """
    # Match only UNPRIMED indices: number not immediately followed by a prime.
    pattern = r"^\s*(\d+)(?!')\s+([-.\dEe+]+)\s+([-.\dEe+]+)\s+([-.\dEe+]+)\s*$"
    matches = re.findall(pattern, text, re.MULTILINE)

    if not matches:
        return {"x": np.array([]), "y": np.array([]), "z": np.array([]), "oe": {"labels": [], "type": []}}

    x, y, z = [], [], []
    for _, xx, yy, zz in matches:
        x.append(-1*float(xx)); y.append(float(yy)); z.append(float(zz))

    n = len(matches)

    # Validate/prepare labels
    if labels is None:
        labels_list = [""] * n
    else:
        if len(labels) != n:
            raise ValueError(f"`labels` must have length {n} (unprimed elements), got {len(labels)}.")
        labels_list = [str(lbl).strip() for lbl in labels]

    def classify(idx: int, lbl: str) -> str:
        if idx == 0:
            return "SRC"  # first element is always the source (e.g., 'SRC', 'U52', etc.)
        if lbl == "":
            return "E"
        if lbl.startswith("M"):
            return "M"
        if lbl.startswith("G"):
            return "G"
        if lbl.startswith("C"):
            return "C"
        if lbl.startswith("S"):
            return "S"
        return "O"

    types = [classify(i, lbl) for i, lbl in enumerate(labels_list)]

    return {
        "x": np.array(x),
        "y": np.array(y),
        "z": np.array(z),
        "oe": {"labels": labels_list, "type": types},
    }