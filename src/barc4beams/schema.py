# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2025 Synchrotron SOLEIL
"""
schema.py - definition and validation of the standard beam format.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Core schema definition
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS: tuple[str, ...] = (
    "energy",
    "X", "Y",
    "dX", "dY",
    "wavelength",
    "intensity",
    "intensity_s-pol",
    "intensity_p-pol",
    "lost_ray_flag",  # internal convention: 1 = lost, 0 = alive
)

SCHEMA_VERSION: str = "1.0"

# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_beam(df: pd.DataFrame) -> None:
    """
    Validate a standardized beam DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Beam after `to_standard_beam(...)`. Columns may appear in any order.

    Raises
    ------
    ValueError
        If required columns are missing, intensity columns are out of [0, 1],
        lost-ray flags are not {0, 1}, or a lost ray has nonzero intensity.
    """
    # presence
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Beam is missing required columns: {missing}")

    for name in ("intensity", "intensity_s-pol", "intensity_p-pol"):
        x = pd.to_numeric(df[name], errors="coerce").to_numpy()
        if np.nanmin(x) < 0.0 or np.nanmax(x) > 1.0:
            raise ValueError(f"{name} must be within [0, 1].")

    flag = pd.to_numeric(df["lost_ray_flag"], errors="coerce").to_numpy()
    uniq = set(np.unique(flag[~np.isnan(flag)]))
    if not uniq.issubset({0.0, 1.0}):
        raise ValueError("lost_ray_flag must be {0, 1} (0 = alive, 1 = lost).")

    lost = flag == 1.0
    if np.any(pd.to_numeric(df.loc[lost, "intensity"], errors="coerce").to_numpy() > 0.0):
        raise ValueError("Lost rays must have zero 'intensity'.")