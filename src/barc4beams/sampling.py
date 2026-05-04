# SPDX-License-Identifier: CECILL-2.1
# Copyright (c) 2026 ESRF - the European Synchrotron

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
    The local ray angles (dX, dY) are obtained from the phase gradient:

        dX = (1 / k) * dphi/dx
        dY = (1 / k) * dphi/dy

    with k = 2*pi / wavelength.

    Parameters
    ----------
    wavefront : dict
        Flat dict with keys:
            - "intensity": 2D array I[y, x].
            - "phase": 2D array of unwrapped phase in radians.
            - "x_axis": 1D array of x in meters.
            - "y_axis": 1D array of y in meters.

    n_rays : int
        Number of rays to sample.
    energy, wavelength : float, optional
        Exactly one must be provided. The other is computed using
        misc.energy_wavelength(...). If 'energy' is given, wavelength is
        computed in meters; if 'wavelength' is given, energy is computed in eV.
    jitter : bool, optional
        Sub-pixel jitter for the sampled (X, Y) coordinates.
    threshold : float or None, optional
        Relative cutoff in [0, 1]. Pixels below threshold * max(intensity)
        are excluded both from position sampling and from phase-gradient support.
    seed : int | None, optional
        RNG seed for reproducibility. Use an int for deterministic draws;
        use None for non-deterministic sampling.
    z0 : float, default 0.0
        Source plane position assigned to all rays (Z=z0, dZ=0).
    polarization_degree : float, default 1.0
        Value in [0, 1] is the s-polarization fraction: Is = pdeg, Ip = 1-pdeg.

    Returns
    -------
    pandas.DataFrame
        Standard beam DataFrame ready for plotting/propagation/caustic.

    Notes
    -----
    This function samples ray positions from a spatial intensity map and
    assigns local propagation angles from the phase gradient.

    Low-intensity or invalid pixels are masked internally before gradient
    evaluation. This avoids differentiating noisy finite phase values outside
    the useful wavefront support.
    """
    if (energy is None) == (wavelength is None):
        raise ValueError("Provide exactly one of (energy, wavelength).")

    if energy is not None:
        energy = float(energy)
        wavelength = float(misc.energy_wavelength(energy, "eV"))
    else:
        wavelength = float(wavelength)
        energy = float(misc.energy_wavelength(wavelength, "m"))

    intensity, phase, x_axis, y_axis = _validate_wavefront_map(wavefront)

    valid = (
        np.isfinite(intensity)
        & (intensity > 0.0)
        & np.isfinite(phase)
    )

    if threshold is not None:
        thr = float(threshold)
        if thr < 0.0 or thr > 1.0:
            raise ValueError("`threshold` must be in [0, 1] (relative to max intensity).")

        maxI = np.max(intensity)
        if not np.isfinite(maxI) or maxI <= 0.0:
            raise ValueError("Intensity max is non-finite or non-positive; cannot apply threshold.")

        valid &= intensity >= thr * maxI

    if not np.any(valid):
        raise ValueError("No valid wavefront support remains after masking.")

    intensity_sample = np.where(valid, intensity, 0.0)
    phase_masked = np.where(valid, phase, np.nan)

    dphi_dy, dphi_dx = np.gradient(phase_masked, y_axis, x_axis)

    grad_x = np.where(valid, dphi_dx, np.nan)
    grad_y = np.where(valid, dphi_dy, np.nan)

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

        batch_seed = None if seed is None else int(
            rng.integers(0, np.iinfo(np.int64).max)
        )

        X_try, Y_try = _sample_from_intensity(
            intensity_sample,
            x_axis,
            y_axis,
            n_try,
            jitter=jitter,
            threshold=None,
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
            "X": X,
            "Y": Y,
            "Z": Z,
            "dX": dX,
            "dY": dY,
            "dZ": dZ,
            "wavelength": W,
            "intensity": I_ray,
            "intensity_s-pol": Is,
            "intensity_p-pol": Ip,
            "lost_ray_flag": np.zeros(n_rays, dtype=np.uint8),
        }
    )

    schema.validate_beam(df)
    return df

def apply_wavefront(
    *,
    standard_beam: pd.DataFrame,
    wavefront: dict,
    energy: float | None = None,
    wavelength: float | None = None,
    threshold: float | None = None,
) -> pd.DataFrame:
    """
    Apply a spatial wavefront map to an existing standard beam.

    The input beam is copied. Ray positions are preserved. The wavefront
    intensity is interpolated at each ray position and applied as a
    multiplicative transmission factor. The wavefront phase gradient is
    interpolated at each ray position and applied as an angular kick:

        dX_out = dX_in + (1 / k) * dphi/dx
        dY_out = dY_in + (1 / k) * dphi/dy

    with k = 2*pi / wavelength.

    Rays outside the wavefront grid, rays landing on zero/invalid
    transmission, and rays landing where the phase-gradient support is invalid
    are marked as lost and assigned zero intensity.

    Parameters
    ----------
    standard_beam : pandas.DataFrame
        Standard beam DataFrame.
    wavefront : dict
        Flat dict with keys:
            - "intensity": 2D array I[y, x].
            - "phase": 2D array of unwrapped phase in radians.
            - "x_axis": 1D array of x in meters.
            - "y_axis": 1D array of y in meters.
    energy, wavelength : float, optional
        Exactly one must be provided. The other is computed using
        misc.energy_wavelength(...). If 'energy' is given, wavelength is
        computed in meters; if 'wavelength' is given, energy is computed in eV.
    threshold : float or None, optional
        Relative cutoff in [0, 1]. Pixels below threshold * max(intensity)
        are treated as invalid support.

    Returns
    -------
    pandas.DataFrame
        New standard beam DataFrame with updated intensity, slopes, and
        lost-ray flags.

    Raises
    ------
    ValueError
        If the input beam or wavefront is invalid.
    """
    schema.validate_beam(standard_beam)

    if (energy is None) == (wavelength is None):
        raise ValueError("Provide exactly one of (energy, wavelength).")

    if energy is not None:
        energy = float(energy)
        wavelength = float(misc.energy_wavelength(energy, "eV"))
    else:
        wavelength = float(wavelength)
        energy = float(misc.energy_wavelength(wavelength, "m"))

    intensity, phase, x_axis, y_axis = _validate_wavefront_map(wavefront)

    valid = (
        np.isfinite(intensity)
        & (intensity > 0.0)
        & np.isfinite(phase)
    )

    if threshold is not None:
        thr = float(threshold)
        if thr < 0.0 or thr > 1.0:
            raise ValueError("`threshold` must be in [0, 1] (relative to max intensity).")

        maxI = np.max(intensity)
        if not np.isfinite(maxI) or maxI <= 0.0:
            raise ValueError("Intensity max is non-finite or non-positive; cannot apply threshold.")

        valid &= intensity >= thr * maxI

    if not np.any(valid):
        raise ValueError("No valid wavefront support remains after masking.")

    maxI = np.max(np.where(valid, intensity, 0.0))
    if not np.isfinite(maxI) or maxI <= 0.0:
        raise ValueError("No positive wavefront intensity remains after masking.")

    transmission_map = np.where(valid, intensity / maxI, 0.0)
    phase_masked = np.where(valid, phase, np.nan)

    dphi_dy, dphi_dx = np.gradient(phase_masked, y_axis, x_axis)

    k = 2.0 * np.pi / wavelength
    slope_x_map = np.where(valid, dphi_dx / k, np.nan)
    slope_y_map = np.where(valid, dphi_dy / k, np.nan)

    beam = standard_beam.copy(deep=True)

    xs = beam["X"].to_numpy(dtype=float)
    ys = beam["Y"].to_numpy(dtype=float)

    transmission = _interp2d_regular(x_axis, y_axis, transmission_map, xs, ys)
    delta_dX = _interp2d_regular(x_axis, y_axis, slope_x_map, xs, ys)
    delta_dY = _interp2d_regular(x_axis, y_axis, slope_y_map, xs, ys)

    inside = (
        (xs >= min(x_axis[0], x_axis[-1]))
        & (xs <= max(x_axis[0], x_axis[-1]))
        & (ys >= min(y_axis[0], y_axis[-1]))
        & (ys <= max(y_axis[0], y_axis[-1]))
    )

    lost = beam["lost_ray_flag"].to_numpy(dtype=np.uint8).copy()

    alive = (
        inside
        & np.isfinite(transmission)
        & (transmission > 0.0)
        & np.isfinite(delta_dX)
        & np.isfinite(delta_dY)
        & (lost == 0)
    )

    beam.loc[alive, "dX"] = beam.loc[alive, "dX"].to_numpy(dtype=float) + delta_dX[alive]
    beam.loc[alive, "dY"] = beam.loc[alive, "dY"].to_numpy(dtype=float) + delta_dY[alive]

    for key in ("intensity", "intensity_s-pol", "intensity_p-pol"):
        values = beam[key].to_numpy(dtype=float).copy()
        values[alive] *= transmission[alive]
        values[~alive] = 0.0
        beam[key] = values

    beam.loc[~alive, "lost_ray_flag"] = 1

    schema.validate_beam(beam)
    return beam

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


def _validate_wavefront_map(wavefront: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Validate and normalize a flat wavefront dictionary.

    Parameters
    ----------
    wavefront : dict
        Flat dict with keys:
            - "intensity": 2D array I[y, x].
            - "phase": 2D array of unwrapped phase in radians.
            - "x_axis": 1D array of x in meters.
            - "y_axis": 1D array of y in meters.

    Returns
    -------
    intensity, phase, x_axis, y_axis : tuple of numpy.ndarray
        Validated arrays with float dtype.

    Raises
    ------
    KeyError
        If a required key is missing.
    ValueError
        If array dimensions, shapes, or axes are invalid.
    """
    for key in ("intensity", "phase", "x_axis", "y_axis"):
        if key not in wavefront:
            raise KeyError(f"wavefront missing key: {key!r}")

    intensity = np.asarray(wavefront["intensity"], dtype=float)
    phase = np.asarray(wavefront["phase"], dtype=float)
    x_axis = np.asarray(wavefront["x_axis"], dtype=float)
    y_axis = np.asarray(wavefront["y_axis"], dtype=float)

    if intensity.ndim != 2:
        raise ValueError("wavefront['intensity'] must be 2D (I[y, x]).")
    if phase.shape != intensity.shape:
        raise ValueError("wavefront['phase'] must have the same shape as 'intensity'.")
    if x_axis.ndim != 1 or y_axis.ndim != 1:
        raise ValueError("wavefront['x_axis'] and wavefront['y_axis'] must be 1D arrays.")
    if x_axis.size != intensity.shape[1] or y_axis.size != intensity.shape[0]:
        raise ValueError("Axis lengths must match wavefront intensity shape (ny, nx).")
    if not (np.all(np.diff(x_axis) > 0) or np.all(np.diff(x_axis) < 0)):
        raise ValueError("wavefront['x_axis'] must be strictly monotonic.")
    if not (np.all(np.diff(y_axis) > 0) or np.all(np.diff(y_axis) < 0)):
        raise ValueError("wavefront['y_axis'] must be strictly monotonic.")

    intensity = np.nan_to_num(intensity, nan=0.0, posinf=0.0, neginf=0.0)

    return intensity, phase, x_axis, y_axis


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