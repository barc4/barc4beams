# SPDX-License-Identifier: CECILL-2.1
# Copyright (c) 2026 ESRF - the European Synchrotron

"""
propagation.py - free space ray tracing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import schema, stats

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def propagate(
    beam: pd.DataFrame,
    z_offset: float,
) -> pd.DataFrame:
    """
    Compute free-space propagation for a standard beam

    Performs:
        X <- X + z_offset * dX
        Y <- Y + z_offset * dY

    Parameters
    ----------
    beam : pandas.DataFrame
        Standardized beam. Validated via `schema.validate_beam(beam)`.
    z_offset : float
        Propagation distance in meters (positive downstream).

    Returns
    -------
    pandas.DataFrame
        Standard beam at the propagated plane.

    """
    schema.validate_beam(beam)

    if not np.isfinite(z_offset):
        raise ValueError("z_offset must be a finite float (meters).")

    df = beam.copy()

    X, Y = _propagate_xy(df["X"].to_numpy(), df["Y"].to_numpy(),
                         df["dX"].to_numpy(), df["dY"].to_numpy(), z_offset)

    df["X"], df["Y"] = X, Y

    if "Z" in df.columns:
        df["Z"] = df["Z"] + z_offset
    else:
        df["Z"] = np.full(len(df), z_offset, dtype=float)

    return df


def caustic(
    beam: pd.DataFrame,
    *,
    n_points: int = 501,
    start: float = -0.5,
    finish: float = 0.5,
) -> dict:
    """
    Compute free-space caustics for a standard beam.

    Ray positions are propagated in free space. Ray intensity weights are not
    modified by propagation and are used for all per-plane statistics.
    """
    if n_points < 2:
        raise ValueError("n_points must be >= 2")
    if not (finish > start):
        raise ValueError("`finish` must be strictly larger than `start`")

    schema.validate_beam(beam)

    df = beam
    if "lost_ray_flag" in df.columns:
        df = df.loc[df["lost_ray_flag"] == 0]

    z = np.linspace(start, finish, n_points)

    if df.shape[0] == 0:
        nan_line = np.full(n_points, np.nan)
        return {
            "caustic": {},
            "optical_axis": z,
            "moments": {
                "x": {"mean": nan_line, "std": nan_line, "skewness": nan_line, "kurtosis": nan_line},
                "y": {"mean": nan_line, "std": nan_line, "skewness": nan_line, "kurtosis": nan_line},
            },
            "fwhm": {"x": nan_line, "y": nan_line},
            "focal_length": {"x": nan_line, "y": nan_line},
        }

    X0 = df["X"].to_numpy(dtype=float)
    Y0 = df["Y"].to_numpy(dtype=float)
    dX = df["dX"].to_numpy(dtype=float)
    dY = df["dY"].to_numpy(dtype=float)
    weights = df["intensity"].to_numpy(dtype=float)

    X, Y = _propagate_xy(X0, Y0, dX, dY, z)
    intensity = np.tile(weights[np.newaxis, :], (n_points, 1))

    mu_x = np.empty(n_points)
    sig_x = np.empty(n_points)

    mu_y = np.empty(n_points)
    sig_y = np.empty(n_points)

    fwhm_x = np.empty(n_points)
    fwhm_y = np.empty(n_points)


    for i in range(n_points):
        mx, sx, _, _ = stats.calc_moments_from_particle_distribution(
            X[i],
            weights=weights,
        )
        my, sy, _, _ = stats.calc_moments_from_particle_distribution(
            Y[i],
            weights=weights,
        )

        mu_x[i], sig_x[i],= mx, sx
        mu_y[i], sig_y[i],= my, sy

        fwhm_x[i] = stats.calc_fwhm_from_particle_distribution(
            X[i],
            weights=weights,
            bins=None,
        )
        fwhm_y[i] = stats.calc_fwhm_from_particle_distribution(
            Y[i],
            weights=weights,
            bins=None,
        )

    return {
        "caustic": {
            "X": X,
            "Y": Y,
            "intensity": intensity,
        },
        "optical_axis": z,
        "moments": {
            "x": {"mean": mu_x, "std": sig_x},
            "y": {"mean": mu_y, "std": sig_y},
        },
        "fwhm": {
            "x": fwhm_x,
            "y": fwhm_y,
        },
    }

# ---------------------------------------------------------------------------
# internal helpers
# ---------------------------------------------------------------------------

def _propagate_xy(
    X0: np.ndarray,
    Y0: np.ndarray,
    dX: np.ndarray,
    dY: np.ndarray,
    z: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Internal helper for free-space propagation.

    Parameters
    ----------
    X0, Y0 : np.ndarray
        Initial positions.
    dX, dY : np.ndarray
        Direction cosines (small angles in radians).
    z : float or np.ndarray
        Propagation distance(s) in meters.
        If 1D array, returns broadcasted 2D arrays [len(z), len(X0)].

    Returns
    -------
    X, Y : np.ndarray
        Propagated positions at each z.
    """
    X0, Y0, dX, dY = map(np.asarray, (X0, Y0, dX, dY))
    z = np.asarray(z)

    if z.ndim == 0:
        X = X0 + z * dX
        Y = Y0 + z * dY
    else:
        X = X0[np.newaxis, :] + z[:, np.newaxis] * dX[np.newaxis, :]
        Y = Y0[np.newaxis, :] + z[:, np.newaxis] * dY[np.newaxis, :]

    return X, Y
