# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2025 Synchrotron SOLEIL

"""
adapters.py - conversion of external beam data (PyOptiX, SHADOW3, SHADOW4)
into the standard beam schema.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from . import misc, schema

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def to_standard_beam(beam, code: Optional[str] = None) -> pd.DataFrame:

    """
    Convert an external beam representation into the standard schema.

    Supported inputs
    ----------------
    - PyOptiX beam (pd.DataFrame): typically from:
      OpticalElement.get_diagram(...)
      OpticalElement.get_impacts(...).

    - Shadow3 beam (code in {'shadow3','s3'}):
        Uses beam.getshcol([...]) with indices:
          [11, 1, 3, 2, 4, 6, 5, 19, 23, 10, 24, 25]
        and returns a DataFrame with columns:
          ["energy","X","Y","Z","dX","dY","dZ","wavelength",
           "intensity","lost_ray_flag","intensity_s-pol","intensity_p-pol"]

    - Shadow4 beam (code in {'shadow4','s4','shadow'}):
        Uses beam.get_columns([...]) with indices:
          [26, 1, 3, 2, 4, 6, 5, 19, 23, 10, 24, 25]
        with the same output columns and clipping as Shadow3.

    Parameters
    ----------
    beam : object
        PyOptiX DataFrame OR Shadow3/4 beam object.
    code : {"pyoptix", "shadow3", "shadow4", "s3", "s4"}, optional
        Explicit backend. If None, auto-detect.

    Returns
    -------
    pandas.DataFrame
        pd.DataFrame
    """

    if code is None:
        code = _detect_backend(beam)

    if code in {"pyoptix", "optix", "pyx"}:
        df = _from_pyoptix(beam)
    elif code in {"shadow3", "s3"}:
        df = _from_shadow3(beam)
    elif code in {"shadow4", "s4", "shadow"}:
        df = _from_shadow4(beam)
    else:
        raise ValueError(f"Unsupported code: {code}")

    schema.validate_beam(df, strict=False)

    return df

def merge_standard_beams(beams: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Merge multiple standard photon beams (pandas DataFrames) into a single combined beam.
    Each input beam should already be standardized via `to_standard_beam`,
    The merge is performed by simple row-wise concatenation of all beams.

    Args:
        beams (list of pd.DataFrame): List of DataFrames to merge.

    Returns:
        pd.DataFrame: A single DataFrame containing all rows from the input beams.
                      Columns are the union of all input columns;
    """
    if not beams:
        raise ValueError("merge_beams: no beams provided")

    if not all(isinstance(b, pd.DataFrame) for b in beams):
        raise TypeError("merge_beams: all inputs must be pandas DataFrames")

    merged = pd.concat(beams, ignore_index=True, sort=False)
    schema.validate_beam(merged, strict=False)

    return merged

# ---------------------------------------------------------------------------
# internal helpers
# ---------------------------------------------------------------------------

def _detect_backend(beam) -> str:
    """Heuristic backend detection."""
    if isinstance(beam, pd.DataFrame):
        if {"X", "Y", "dX", "dY"}.issubset(beam.columns):
            return "pyoptix"
    if hasattr(beam, "getshcol"):
        return "shadow3"
    if hasattr(beam, "get_columns"):
        return "shadow4"
    raise ValueError("Could not detect backend for beam")

def _from_pyoptix(df: pd.DataFrame) -> pd.DataFrame:
    """Adapt a PyOptiX beam (already tabular). Assumes columns close to standard."""
    out = df.copy()
    df["intensity"] = np.clip(df["intensity"].to_numpy(), 0.0, 1.0) 
    out.insert(0, "energy", misc.energy_wavelength(out["wavelength"], "m"))
    if "intensity_s-pol" not in out.columns:
        out["intensity_s-pol"] = out["intensity"]
    if "intensity_p-pol" not in out.columns:
        out["intensity_p-pol"] = out["intensity"]
    out["lost_ray_flag"] = (out["intensity"].to_numpy() == 0.0).astype(int)
    return out

def _from_shadow3(beam) -> pd.DataFrame:
    """Adapt a SHADOW3 beam object to the standard schema."""
    cols = [11, 1, 3, 2, 4, 6, 5, 19, 23, 24, 25, 10]
    return _from_shadow(beam, cols, "getshcol")

def _from_shadow4(beam) -> pd.DataFrame:
    """Adapt a SHADOW4 beam object to the standard schema."""
    cols = [26, 1, 3, 2, 4, 6, 5, 19, 23, 24, 25, 10]
    return _from_shadow(beam, cols, "get_columns")

def _from_shadow(beam, cols, getter_name: str) -> pd.DataFrame:
    """
    Generic SHADOW adapter (used by S3/S4 wrappers).

    Parameters
    ----------
    beam : object
        SHADOW beam-like object exposing the given getter.
    cols : sequence of int
        Column indices to extract, in the order matching `headers` below.
    getter_name : {"getshcol", "get_columns"}
        Name of the method on `beam` used to fetch columns.

    Returns
    -------
    pandas.DataFrame
        Standardized frame with columns:
        ["energy","X","Y","Z","dX","dY","dZ","wavelength",
         "intensity","intensity_s-pol","intensity_p-pol","lost_ray_flag"]

    Notes
    -----
    SHADOW convention is (+1 alive, -1 lost). We convert to barc4beams
    convention: `lost_ray_flag` = 1 for lost, 0 for alive. Intensities
    for lost rays are zeroed. Only values > 1.0 are clipped.
    """
    getter = getattr(beam, getter_name, None)
    if getter is None:
        raise AttributeError(f"Shadow beam must provide .{getter_name}(indices).")

    data = np.asarray(getter(cols)).T
    headers = [
        "energy",
        "X", "Y", "Z",
        "dX", "dY", "dZ",
        "wavelength",
        "intensity",
        "intensity_s-pol",
        "intensity_p-pol",
        "lost_ray_flag",
    ]
    df = pd.DataFrame(data, columns=headers)

    raw = pd.to_numeric(df["lost_ray_flag"], errors="coerce")
    df["lost_ray_flag"] = (raw == -1).astype(int)

    lost = df["lost_ray_flag"] == 1
    for col in ("intensity", "intensity_s-pol", "intensity_p-pol"):
        df.loc[lost, col] = 0.0
    df["intensity"] = np.minimum(pd.to_numeric(df["intensity"], errors="coerce"), 1.0)

    return df