# SPDX-License-Identifier: CECILL-2.1
# Copyright (c) 2025 Synchrotron SOLEIL
"""
caustics.py - free space ray tracing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import schema, stats

def compute_caustic(
    beam: pd.DataFrame,
    *,
    n_points: int = 501,
    start: float = -0.5,
    finish: float = 0.5,
    return_points: bool = True,
) -> dict:
    """
    Compute free-space caustics for a standard beam.

    Geometry
    --------
    X(z) = X0 + z * dX
    Y(z) = Y0 + z * dY
    where dX, dY are slopes (radians).

    Parameters
    ----------
    beam : pandas.DataFrame
        Standard barc4beams beam (see schema). Must contain
        X, Y, dX, dY, intensity, lost_ray_flag.
    n_points : int, default 501
        Number of planes (z-samples) along the optical axis.
    start : float, default -0.5
        First z (meters) relative to current beam plane (0 at the beam plane).
        May be negative.
    finish : float, default 0.5
        Last z. MUST be strictly larger than `start`.
    return_points : bool, default True
        If True, return the full X(z), Y(z) arrays. If False, omit them and
        return only the optical axis and the per-plane moments.

    Returns
    -------
    dict
        {
          "caustic": { "X": (P,N), "Y": (P,N) }      # optional (see return_points)
          "optical_axis": (P,),                      # z grid [m]
          "moments": {                               # per-plane arrays
            "x": {"mean":(P,), "std":(P,), "skew":(P,), "kurtosis_excess":(P,)},
            "y": {"mean":(P,), "std":(P,), "skew":(P,), "kurtosis_excess":(P,)}
          },
          "fwhm": { "x": (P,), "y": (P,) },          # FWHM at each plane [same units as X,Y]
          "focal_length": { "x": (P,), "y": (P,) }   # focus distance from each plane [m]
        }
    """
    # --- input checks & schema ----
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
        out = {
            "caustic": {},
            "optical_axis": z,
            "moments": {
                "x": {"mean": nan_line, "std": nan_line, "skew": nan_line, "kurtosis_excess": nan_line},
                "y": {"mean": nan_line, "std": nan_line, "skew": nan_line, "kurtosis_excess": nan_line},
            },
            "fwhm": {"x": nan_line, "y": nan_line},
            "focal_length": {"x": nan_line, "y": nan_line},
        }
        return out

    X0 = df["X"].to_numpy(dtype=float)
    Y0 = df["Y"].to_numpy(dtype=float)
    dX = df["dX"].to_numpy(dtype=float)
    dY = df["dY"].to_numpy(dtype=float)
    N = X0.size

    X = X0[np.newaxis, :] + z[:, np.newaxis] * dX[np.newaxis, :]
    Y = Y0[np.newaxis, :] + z[:, np.newaxis] * dY[np.newaxis, :]

    mu_x  = np.empty(n_points); sig_x = np.empty(n_points)
    sk_x  = np.empty(n_points); ku_x  = np.empty(n_points)
    mu_y  = np.empty(n_points); sig_y = np.empty(n_points)
    sk_y  = np.empty(n_points); ku_y  = np.empty(n_points)

    fwhm_x = np.empty(n_points)
    fwhm_y = np.empty(n_points)

    fx = np.empty(n_points)
    fy = np.empty(n_points)

    for i in range(n_points):
        mx, sx, gx, kx = stats.calc_moments_from_particle_distribution(X[i])
        my, sy, gy, ky = stats.calc_moments_from_particle_distribution(Y[i])
        mu_x[i], sig_x[i], sk_x[i], ku_x[i] = mx, sx, gx, kx
        mu_y[i], sig_y[i], sk_y[i], ku_y[i] = my, sy, gy, ky

        fwhm_x[i] = stats.calc_fwhm_from_particle_distribution(X[i], bins=None)
        fwhm_y[i] = stats.calc_fwhm_from_particle_distribution(Y[i], bins=None)

        fx[i] = stats.calc_focal_distance_from_particle_distribution(X[i], dX)
        fy[i] = stats.calc_focal_distance_from_particle_distribution(Y[i], dY)

    caustic_block = {}
    if return_points:
        caustic_block["X"] = X
        caustic_block["Y"] = Y

    return {
        "caustic": caustic_block,
        "optical_axis": z,
        "moments": {
            "x": {"mean": mu_x, "std": sig_x, "skewness": sk_x, "kurtosis": ku_x},
            "y": {"mean": mu_y, "std": sig_y, "skewness": sk_y, "kurtosis": ku_y},
        },
        "fwhm": {"x": fwhm_x, "y": fwhm_y},
        "focal_length": {"x": fx, "y": fy},
    }