# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2025 Synchrotron SOLEIL
"""
schema.py — definition and validation of the standard beam format.
"""

from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Core schema definition
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS: tuple[str, ...] = (
    "X",          # [m] horizontal position
    "Y",          # [m] vertical position
    "dX",         # [rad] horizontal angle
    "dY",         # [rad] vertical angle
    "intensity",  # relative intensity [0..1], 0 for lost rays
    "wavelength", # [m]; if missing, adapters should compute from energy
)

OPTIONAL_COLUMNS: tuple[str, ...] = (
    "Z",          # [m] longitudinal position
    "dZ",         # [rad] longitudinal angle
    "energy",     # [eV]; if not present, derived from wavelength
    "lost_ray_flag",  # 1 = lost, 0 = alive
)

COLUMN_ORDER: tuple[str, ...] = (
    "energy",
    "X", "Y", "Z",
    "dX", "dY", "dZ",
    "wavelength",
    "intensity",
    "lost_ray_flag",
)

SCHEMA_VERSION: str = "1.0"

# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_beam(beam: pd.DataFrame | dict | np.ndarray, 
                  required: Sequence[str] = REQUIRED_COLUMNS,
                  strict: bool = False) -> None:
    """
    Validate that a beam object follows the standard schema.

    Parameters
    ----------
    beam : DataFrame or dict-like
        The beam to validate. Must expose columns or keys.
    required : sequence of str, optional
        Columns that must be present. Defaults to REQUIRED_COLUMNS.
    strict : bool, optional
        If True, require that columns appear exactly in COLUMN_ORDER.
        If False, extra columns are tolerated.

    Raises
    ------
    ValueError
        If required columns are missing or ordering is wrong in strict mode.
    """
    if isinstance(beam, pd.DataFrame):
        cols = list(beam.columns)
    elif isinstance(beam, dict):
        cols = list(beam.keys())
    else:
        raise TypeError("validate_beam expects a DataFrame or dict-like object")

    missing = [c for c in required if c not in cols]
    if missing:
        raise ValueError(f"Beam is missing required columns: {missing}")

    optional = [c for c in OPTIONAL_COLUMNS if c not in cols]
    if optional:
        warnings.warn(f"Optional columns missing: {optional}", UserWarning)

    if strict:
        order = [c for c in COLUMN_ORDER if c in cols]
        if cols != order:
            raise ValueError(
                f"Beam columns not in standard order. Expected {order}, got {cols}"
            )

    if "intensity" in cols:
        if np.nanmax(beam["intensity"]) > 1.0:
            raise ValueError("Intensity values must be ≤ 1.0")
        if np.nanmin(beam["intensity"]) < 0.0:
            raise ValueError("Intensity values must be ≥ 0.0")

    if "lost_ray_flag" in cols:
        bad_flags = set(np.unique(beam["lost_ray_flag"])) - {0, 1}
        if bad_flags:
            raise ValueError(f"Invalid lost_ray_flag values: {bad_flags}")