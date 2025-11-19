# SPDX-License-Identifier: CECILL-2.1
# Copyright (c) 2025 Synchrotron SOLEIL

"""
sampling.py — generate a standardized beam by sampling 2D intensity maps.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from . import misc, schema

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def beam_from_intensity(
    *,
    far_field: dict,
    near_field: Optional[dict] = None,
    n_rays: int,
    energy: float | None = None,
    wavelength: float | None = None,
    jitter: bool = True,
    threshold: float | None = None,
    seed: int | None = 42,
    z0: float = 0.0,
    polarization_degree: float = 1.0,
) -> pd.DataFrame:
    """
    Build a standard beam by sampling intensity maps (SI units only).

    Parameters
    ----------
    far_field : dict
        Flat dict with keys: {"intensity", "x_axis", "y_axis"}.
        - "intensity": 2D array Iff[y, x] (already per pixel; no unit conversion here)
        - "x_axis": 1D array of xp in radians (strictly monotonic)
        - "y_axis": 1D array of yp in radians (strictly monotonic)
        This is REQUIRED and defines (dX, dY).

    near_field : dict or None, optional
        Flat dict with keys: {"intensity", "x_axis", "y_axis"}.
        - "intensity": 2D array Inf[y, x] (already per pixel)
        - "x_axis": 1D array of x in meters (strictly monotonic)
        - "y_axis": 1D array of y in meters (strictly monotonic)
        If provided, defines (X, Y); otherwise X=Y=0 (point source at z0).

    n_rays : int
        Number of rays to sample.
    energy, wavelength : float, optional
        Exactly one must be provided. The other is computed using
        misc.energy_wavelength(...). If 'energy' is given, wavelength is
        computed in meters; if 'wavelength' is given, energy is computed in eV.
    jitter : bool, optional
        Sub-pixel jitter for both FF and NF maps (if NF present).
    threshold : float or None, optional
        Relative cutoff in [0, 1] applied independently to each map.
    seed : int | None, optional
        RNG seed. If both FF and NF are sampled, NF uses (seed+1) for decorrelation.
    z0 : float, default 0.0
        Source plane position assigned to all rays (Z=z0, dZ=0).
    polarization_degree : float, default 1.0
        Value in [0,1] is the s-polarization fraction: Is = pdeg, Ip = 1-pdeg

    Returns
    -------
    pandas.DataFrame
        Standard beam DataFrame ready for plotting/propagation/caustic.

    Notes
    -----
    The sampling algorithm implemented here follows the strategy described in
    Rebuffi *et al.*, *J. Synchrotron Rad.* **27**, 1108–1120 (2020).

    """
    if (energy is None) == (wavelength is None):
        raise ValueError("Provide exactly one of (energy, wavelength).")
    if energy is not None:
        energy = float(energy)
        wavelength = float(misc.energy_wavelength(energy, "eV"))   
    else:
        wavelength = float(wavelength)
        energy = float(misc.energy_wavelength(wavelength, "m"))

    for k in ("intensity", "x_axis", "y_axis"):
        if k not in far_field:
            raise KeyError(f"far_field missing key: {k!r}")
    dX, dY = _sample_from_intensity(
        far_field["intensity"], far_field["x_axis"], far_field["y_axis"],
        n_rays, jitter=jitter, threshold=threshold, seed=seed,
    )

    if near_field is not None:
        for k in ("intensity", "x_axis", "y_axis"):
            if k not in near_field:
                raise KeyError(f"near_field missing key: {k!r}")
        X, Y = _sample_from_intensity(
            near_field["intensity"], near_field["x_axis"], near_field["y_axis"],
            n_rays, jitter=jitter, threshold=threshold,
            seed=(None if seed is None else seed + 1),
        )
    else:
        X = np.zeros(n_rays, dtype=float)
        Y = np.zeros(n_rays, dtype=float)

    Z  = np.full(n_rays, float(z0), dtype=float)
    dZ = np.zeros(n_rays, dtype=float)
    E  = np.full(n_rays, energy, dtype=float)
    W  = np.full(n_rays, wavelength, dtype=float)

    pdeg = float(np.clip(polarization_degree, 0.0, 1.0))
    I  = np.ones(n_rays, dtype=float)
    Is = np.full(n_rays, pdeg, dtype=float)
    Ip = np.full(n_rays, 1.0 - pdeg, dtype=float)

    df = pd.DataFrame(
        {
            "energy": E,
            "X": X, "Y": Y, "Z": Z,
            "dX": dX, "dY": dY, "dZ": dZ,
            "wavelength": W,
            "intensity": I,
            "intensity_s-pol": Is,
            "intensity_p-pol": Ip,
            "lost_ray_flag": np.zeros(n_rays, dtype=np.uint8),
        }
    )

    schema.validate_beam(df)
    return df

# ---------------------------------------------------------------------------
# internal helpers
# ---------------------------------------------------------------------------

def _sample_from_intensity(intensity, x_axis, y_axis, n, jitter=True, threshold=None, seed=42):
    """
    Randomly sample (x, y) coordinates from a 2D intensity distribution,
    with *optional* ub-pixel jitter for de-gridding and relative thresholding in [0, 1].

    Parameters
    ----------
    intensity : 2D ndarray
        Intensity or power density array I[y, x].
    x_axis, y_axis : 1D ndarray
        Coordinates corresponding to the columns (x) and rows (y).
        These can be linear positions or small-angle coordinates
        (e.g. arctan(x / propagation_distance)).
    n : int
        Number of samples (rays) to draw.
    jitter : bool, optional
        If True, adds uniform sub-pixel jitter in the range
        [-0.5*dx, 0.5*dx] and [-0.5*dy, 0.5*dy] to xs and ys.
    threshold : float or None, optional
        Relative cutoff in [0, 1]. Keeps pixels with I >= threshold * max(I).
        - None: no thresholding
        - 0.0:   no effect
        - 1.0:   keeps only pixels at the global maximum
    seed : int | None, optional
        Seed for reproducibility. Use an int for deterministic draws;
        use None for non-deterministic sampling.

    Returns
    -------
    xs, ys : 1D ndarray
        Sampled coordinates following the normalized intensity distribution.
    """

    rng = np.random.default_rng(seed)

    I = np.asarray(intensity, dtype=float)
    I = np.nan_to_num(I, nan=0.0, posinf=0.0, neginf=0.0)

    if I.ndim != 2:
        raise ValueError("intensity must be 2D (I[y, x]).")
    if I.size == 0:
        raise ValueError("intensity is empty.")
    
    ny, nx = I.shape
    if x_axis.ndim != 1 or y_axis.ndim != 1:
        raise ValueError("x_axis and y_axis must be 1D arrays (bin centers).")
    if x_axis.size != nx or y_axis.size != ny:
        raise ValueError("Axis lengths must match intensity shape (ny, nx).")
    if not (np.all(np.diff(x_axis) > 0) or np.all(np.diff(x_axis) < 0)):
        raise ValueError("x_axis must be strictly monotonic.")
    if not (np.all(np.diff(y_axis) > 0) or np.all(np.diff(y_axis) < 0)):
        raise ValueError("y_axis must be strictly monotonic.")

    if threshold is not None:
        thr = float(threshold)
        if thr < 0.0 or thr > 1.0:
            raise ValueError("`threshold` must be in [0, 1] (relative to max intensity).")
        maxI = np.max(I)
        if not np.isfinite(maxI) or maxI <= 0.0:
            raise ValueError("Intensity max is non-finite or non-positive; cannot apply threshold.")
        I = np.where(I >= thr * maxI, I, 0.0)

    prob = I.ravel()
    total = float(prob.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("Thresholding left no positive intensity to sample from.")
    prob /= total

    idx = rng.choice(prob.size, size=n, p=prob, replace=True)
    iy, ix = np.unravel_index(idx, intensity.shape)

    x_axis = np.asarray(x_axis, dtype=float)
    y_axis = np.asarray(y_axis, dtype=float)
   
    def centers_to_edges(c):
        e = np.empty(c.size + 1, dtype=float)
        e[1:-1] = 0.5 * (c[1:] + c[:-1])
        e[0] = c[0] - 0.5 * (c[1] - c[0])
        e[-1] = c[-1] + 0.5 * (c[-1] - c[-2])
        return e

    x_edges = centers_to_edges(x_axis)
    y_edges = centers_to_edges(y_axis)

    xs = x_axis[ix].copy()
    ys = y_axis[iy].copy()

    if jitter:
        xs = rng.uniform(x_edges[ix], x_edges[ix + 1])
        ys = rng.uniform(y_edges[iy], y_edges[iy + 1])

    return xs, ys