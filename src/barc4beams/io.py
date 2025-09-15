# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2025 Synchrotron SOLEIL

"""
io.py - saving and loading of standardised beams and stats.
"""

from __future__ import annotations

import json
import math
import os
import time
from typing import Any, Dict, Optional

import h5py
import numpy as np
import pandas as pd

from . import adapters, schema

_DEFAULT_UNITS = {
    "X": "m", "Y": "m", "Z": "m",
    "dX": "rad", "dY": "rad", "dZ": "rad",
    "wavelength": "m", "energy": "eV",
    "intensity": "arb", "intensity_s-pol": "arb", "intensity_p-pol": "arb",
    "lost_ray_flag": "0=alive,1=lost",
}

# -----------------------------------------------------------------------------
# Beam I/O (h5)
# -----------------------------------------------------------------------------

def save_beam(
    obj: Any,
    path: str,
    *,
    code: Optional[str] = None,
    overwrite: bool = True,
    chunks: bool = True,
) -> None:
    """
    Save a standardized beam to an HDF5 file.

    Parameters
    ----------
    obj : Any
        Beam-like object. If a pandas.DataFrame, it is assumed to be already
        standardized. Otherwise it is converted via
        ``adapters.to_standard_beam(obj, code=code)``.
    path : str
        Output HDF5 file path.
    code : str, optional
        Backend hint for conversion (passed to ``to_standard_beam``).
    overwrite : bool, optional
        If False and `path` exists, raise an error. Default is True.
    chunks : bool, optional
        Enable chunked datasets (recommended). Default True.

    Raises
    ------
    ValueError
        If standardization/validation fails or no numeric columns found.
    FileExistsError
        If `overwrite` is False and file already exists.
    """
    if isinstance(obj, pd.DataFrame):
        df = obj.copy()
    else:
        df = adapters.to_standard_beam(obj, code=code)

    schema.validate_beam(df)

    def _is_numeric_series(s: pd.Series) -> bool:
        return pd.api.types.is_numeric_dtype(s.dtype)

    numeric_cols = [c for c in df.columns if _is_numeric_series(df[c])]
    if not numeric_cols:
        raise ValueError("No numeric columns to save in beam DataFrame.")

    if os.path.exists(path) and not overwrite:
        raise FileExistsError(f"File exists and overwrite=False: {path}")

    with h5py.File(path, "w") as h5:
        h5.attrs["format"] = "barc4beams/beam"
        h5.attrs["convention.lost_ray_flag"] = "0=alive,1=lost"
        h5.attrs["library"] = "barc4beams"
        g = h5.require_group("beam")

        for col in numeric_cols:
            data = pd.to_numeric(df[col], errors="coerce").to_numpy()

            if col == "lost_ray_flag":
                arr = np.nan_to_num(data, nan=0.0).astype(np.uint8)
            else:
                arr = data.astype(np.float64)
            ds = g.create_dataset(col, data=arr, chunks=chunks)

            if col in _DEFAULT_UNITS:
                ds.attrs["units"] = _DEFAULT_UNITS[col]

        units_map = {c: _DEFAULT_UNITS.get(c, "") for c in numeric_cols}
        # g.attrs["units"] = str(units_map)
        g.attrs["units_json"] = json.dumps(units_map)
        g.attrs["column_order"] = json.dumps(numeric_cols)

def read_beam(path: str) -> pd.DataFrame:
    """
    Read a standardized beam from an HDF5 file.

    Parameters
    ----------
    path : str
        Input HDF5 file path.

    Returns
    -------
    pandas.DataFrame
        Beam as a DataFrame.

    Raises
    ------
    KeyError
        If the 'beam' group or required datasets are missing.
    ValueError
        If the reconstructed DataFrame fails validation.
    """
    with h5py.File(path, "r") as h5:
        if "beam" not in h5:
            raise KeyError("HDF5 file missing group '/beam'.")
        g = h5["beam"]
        order = None
        if "column_order" in g.attrs:
            try:
                order = json.loads(g.attrs["column_order"])
            except Exception:
                order = None
        data = {}
        for col in g.keys():
            arr = g[col][()]
            if col == "lost_ray_flag":
                data[col] = np.asarray(arr, dtype=np.uint8)
            else:
                data[col] = np.asarray(arr, dtype=np.float64)
        df = pd.DataFrame(data)
        if order is None:
            canonical = [
                "energy", "X", "Y", "Z", "dX", "dY", "dZ",
                "wavelength", "intensity", "intensity_s-pol",
                "intensity_p-pol", "lost_ray_flag",
            ]
            order = [c for c in canonical if c in df.columns] + \
                    [c for c in df.columns if c not in canonical]

        df = df[order]
    schema.validate_beam(df)
    return df

# -----------------------------------------------------------------------------
# Stats I/O (json)
# -----------------------------------------------------------------------------

def _sanitize(o: Any) -> Any:
    """Make dict JSON-safe: numpy/pandas → Python; NaN/Inf → None."""
    if isinstance(o, dict):
        return {k: _sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sanitize(v) for v in o]
    if isinstance(o, np.ndarray):
        return [_sanitize(v) for v in o.tolist()]
    if isinstance(o, pd.Series):
        return _sanitize(o.to_dict())
    if isinstance(o, pd.DataFrame):
        return [_sanitize(r) for r in o.to_dict(orient="records")]
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return v if math.isfinite(v) else None
    if isinstance(o, (np.integer, int)):
        return int(o)
    return o

def save_json_stats(stats: Dict[str, Any], path: str, *, meta: Optional[Dict[str, Any]] = None) -> str:
    """
    Save a get_statistics() dictionary to JSON.

    Parameters
    ----------
    stats : dict
        The dictionary returned by get_statistics(...).
    path : str
        Target JSON file path.
    meta : dict, optional
        Extra metadata to embed (e.g., n_seeds, n_rays, code_sha, settings).

    Returns
    -------
    str
        The path written.
    """
    record = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "meta": _sanitize(meta or {}),
        "stats": _sanitize(stats),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    return path


def read_json_stats(path: str) -> Dict[str, Any]:
    """
    Load a JSON file saved by save_json_stats.

    Returns
    -------
    dict
        Record with keys: created_utc, meta, stats
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)