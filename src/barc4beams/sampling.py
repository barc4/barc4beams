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


def beam_from_wavefront(
    *,
    wavefront: dict,
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
    Build a standard beam by sampling a spatial wavefront map (SI units only).

    The ray positions (X, Y) are sampled from the wavefront intensity map.
    The local ray angles (dX, dY) are obtained from the phase gradient of the
    complex wavefront:
        dX = (1 / k) * dphi/dx
        dY = (1 / k) * dphi/dy
    with k = 2*pi / wavelength.

    Parameters
    ----------
    wavefront : dict
        Flat dict with keys:
            REQUIRED
            - "intensity": 2D array I[y, x] (already per pixel)
            - "x_axis": 1D array of x in meters (strictly monotonic)
            - "y_axis": 1D array of y in meters (strictly monotonic)

            OPTIONAL
            - "Re": 2D array of real part of the complex field
            - "Im": 2D array of imaginary part of the complex field
            - "phase": 2D array of wavefront phase in radians

        The complex field is built as follows:
            - if both "Re" and "Im" are present, they are used directly
            - otherwise, "phase" is required and the complex field is
              reconstructed from:
                  U = sqrt(intensity) * exp(1j * phase)

    n_rays : int
        Number of rays to sample.
    energy, wavelength : float, optional
        Exactly one must be provided. The other is computed using
        misc.energy_wavelength(...). If 'energy' is given, wavelength is
        computed in meters; if 'wavelength' is given, energy is computed in eV.
    jitter : bool, optional
        Sub-pixel jitter for the sampled (X, Y) coordinates.
    threshold : float or None, optional
        Relative cutoff in [0, 1] applied to the intensity map used for
        sampling and for masking unreliable low-intensity regions in the
        complex-field gradient estimation.
    seed : int | None, optional
        RNG seed for reproducibility. Use an int for deterministic draws;
        use None for non-deterministic sampling.
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
    - This function samples ray positions from a spatial intensity map and
      assigns local propagation angles from the phase gradient of the complex
      wavefront.
    - If the complex field is reconstructed from ("intensity", "phase"),
      the phase may be wrapped; the complex representation remains valid.
    - Low-intensity regions are masked when estimating phase gradients in
      order to avoid unstable divisions by small |U| values.
    - Rays landing in invalid slope regions are rejected and re-sampled.

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
        if k not in wavefront:
            raise KeyError(f"wavefront missing key: {k!r}")

    intensity = np.asarray(wavefront["intensity"], dtype=float)
    intensity = np.nan_to_num(intensity, nan=0.0, posinf=0.0, neginf=0.0)

    x_axis = np.asarray(wavefront["x_axis"], dtype=float)
    y_axis = np.asarray(wavefront["y_axis"], dtype=float)

    if intensity.ndim != 2:
        raise ValueError("wavefront['intensity'] must be 2D (I[y, x]).")
    if x_axis.ndim != 1 or y_axis.ndim != 1:
        raise ValueError("wavefront['x_axis'] and wavefront['y_axis'] must be 1D arrays.")
    if x_axis.size != intensity.shape[1] or y_axis.size != intensity.shape[0]:
        raise ValueError("Axis lengths must match wavefront intensity shape (ny, nx).")
    if not (np.all(np.diff(x_axis) > 0) or np.all(np.diff(x_axis) < 0)):
        raise ValueError("wavefront['x_axis'] must be strictly monotonic.")
    if not (np.all(np.diff(y_axis) > 0) or np.all(np.diff(y_axis) < 0)):
        raise ValueError("wavefront['y_axis'] must be strictly monotonic.")

    has_reim = ("Re" in wavefront) and ("Im" in wavefront)
    if has_reim:
        re = np.asarray(wavefront["Re"], dtype=float)
        im = np.asarray(wavefront["Im"], dtype=float)
        if re.shape != intensity.shape or im.shape != intensity.shape:
            raise ValueError("'Re' and 'Im' must have the same shape as 'intensity'.")
        U = re + 1j * im
    else:
        if "phase" not in wavefront:
            raise KeyError("wavefront must contain either ('Re', 'Im') or 'phase'.")
        phase = np.asarray(wavefront["phase"], dtype=float)
        if phase.shape != intensity.shape:
            raise ValueError("'phase' must have the same shape as 'intensity'.")
        amp = np.sqrt(np.clip(intensity, 0.0, None))
        U = amp * np.exp(1j * phase)

    # ------------------------------------------------------------------
    # valid mask for stable gradient / slope estimation
    # ------------------------------------------------------------------
    valid = np.isfinite(intensity) & (intensity > 0.0)

    if threshold is not None:
        thr = float(threshold)
        if thr < 0.0 or thr > 1.0:
            raise ValueError("`threshold` must be in [0, 1] (relative to max intensity).")
        maxI = np.max(intensity)
        if not np.isfinite(maxI) or maxI <= 0.0:
            raise ValueError("Intensity max is non-finite or non-positive; cannot apply threshold.")
        valid &= intensity >= thr * maxI

    absU2 = np.abs(U) ** 2
    eps = np.finfo(float).eps
    valid &= np.isfinite(absU2) & (absU2 > eps)

    if not np.any(valid):
        raise ValueError("No valid wavefront support remains after masking.")

    # ------------------------------------------------------------------
    # phase gradients from complex field
    # dphi/dx = Im[(dU/dx) / U], dphi/dy = Im[(dU/dy) / U]
    # ------------------------------------------------------------------
    dU_dy, dU_dx = np.gradient(U, y_axis, x_axis)

    grad_x = np.full(intensity.shape, np.nan, dtype=float)
    grad_y = np.full(intensity.shape, np.nan, dtype=float)

    ratio_x = np.full(intensity.shape, np.nan + 0j, dtype=complex)
    ratio_y = np.full(intensity.shape, np.nan + 0j, dtype=complex)

    ratio_x[valid] = dU_dx[valid] / U[valid]
    ratio_y[valid] = dU_dy[valid] / U[valid]

    grad_x[valid] = np.imag(ratio_x[valid])
    grad_y[valid] = np.imag(ratio_y[valid])

    k = 2.0 * np.pi / wavelength
    slope_x = grad_x / k
    slope_y = grad_y / k

    # ------------------------------------------------------------------
    # sampling with reject-and-resample
    # ------------------------------------------------------------------
    rng = np.random.default_rng(seed)
    max_rounds = 100
    batch_min = max(1024, n_rays)

    X_parts = []
    Y_parts = []
    dX_parts = []
    dY_parts = []

    n_kept = 0
    rounds = 0

    while n_kept < n_rays:
        rounds += 1
        if rounds > max_rounds:
            raise RuntimeError(
                "Could not sample enough valid rays from the wavefront. "
                "Try lowering `threshold`, disabling `jitter`, or enlarging the "
                "valid-support region of the wavefront."
            )

        n_need = n_rays - n_kept
        n_try = max(batch_min, int(np.ceil(1.5 * n_need)))

        batch_seed = None if seed is None else int(rng.integers(0, np.iinfo(np.int64).max))

        X_try, Y_try = _sample_from_intensity(
            intensity,
            x_axis,
            y_axis,
            n_try,
            jitter=jitter,
            threshold=threshold,
            seed=batch_seed,
        )

        dX_try = _interp2d_regular(x_axis, y_axis, slope_x, X_try, Y_try)
        dY_try = _interp2d_regular(x_axis, y_axis, slope_y, X_try, Y_try)

        good = np.isfinite(dX_try) & np.isfinite(dY_try)
        if not np.any(good):
            continue

        X_parts.append(X_try[good])
        Y_parts.append(Y_try[good])
        dX_parts.append(dX_try[good])
        dY_parts.append(dY_try[good])

        n_kept += int(np.count_nonzero(good))

    X = np.concatenate(X_parts)[:n_rays]
    Y = np.concatenate(Y_parts)[:n_rays]
    dX = np.concatenate(dX_parts)[:n_rays]
    dY = np.concatenate(dY_parts)[:n_rays]

    Z = np.full(n_rays, float(z0), dtype=float)
    dZ = np.zeros(n_rays, dtype=float)
    E = np.full(n_rays, energy, dtype=float)
    W = np.full(n_rays, wavelength, dtype=float)

    pdeg = float(np.clip(polarization_degree, 0.0, 1.0))
    I_ray = np.ones(n_rays, dtype=float)
    Is = np.full(n_rays, pdeg, dtype=float)
    Ip = np.full(n_rays, 1.0 - pdeg, dtype=float)

    df = pd.DataFrame(
        {
            "energy": E,
            "X": X, "Y": Y, "Z": Z,
            "dX": dX, "dY": dY, "dZ": dZ,
            "wavelength": W,
            "intensity": I_ray,
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

def _interp2d_regular(
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    values: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
) -> np.ndarray:
    """
    Bilinear interpolation on a regular rectilinear grid.

    Parameters
    ----------
    x_axis, y_axis : 1D ndarray
        Grid coordinates corresponding to columns (x) and rows (y).
        Must be strictly monotonic.
    values : 2D ndarray
        Array values[y, x].
    xs, ys : 1D ndarray
        Query coordinates.

    Returns
    -------
    1D ndarray
        Interpolated values at (xs, ys). Points involving NaN neighbors return NaN.
    """
    x_axis = np.asarray(x_axis, dtype=float)
    y_axis = np.asarray(y_axis, dtype=float)
    values = np.asarray(values, dtype=float)
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)

    if values.shape != (y_axis.size, x_axis.size):
        raise ValueError("`values` shape must be (len(y_axis), len(x_axis)).")

    if x_axis[0] > x_axis[-1]:
        x_axis = x_axis[::-1]
        values = values[:, ::-1]
    if y_axis[0] > y_axis[-1]:
        y_axis = y_axis[::-1]
        values = values[::-1, :]

    ix1 = np.searchsorted(x_axis, xs, side="right")
    iy1 = np.searchsorted(y_axis, ys, side="right")

    ix1 = np.clip(ix1, 1, x_axis.size - 1)
    iy1 = np.clip(iy1, 1, y_axis.size - 1)

    ix0 = ix1 - 1
    iy0 = iy1 - 1

    x0 = x_axis[ix0]
    x1 = x_axis[ix1]
    y0 = y_axis[iy0]
    y1 = y_axis[iy1]

    tx = (xs - x0) / (x1 - x0)
    ty = (ys - y0) / (y1 - y0)

    v00 = values[iy0, ix0]
    v10 = values[iy0, ix1]
    v01 = values[iy1, ix0]
    v11 = values[iy1, ix1]

    out = (
        (1.0 - tx) * (1.0 - ty) * v00 +
        tx * (1.0 - ty) * v10 +
        (1.0 - tx) * ty * v01 +
        tx * ty * v11
    )

    bad = (
        ~np.isfinite(v00) |
        ~np.isfinite(v10) |
        ~np.isfinite(v01) |
        ~np.isfinite(v11)
    )
    out[bad] = np.nan
    return out