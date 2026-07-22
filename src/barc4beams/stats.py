# SPDX-License-Identifier: CECILL-2.1
# Copyright (c) 2026 ESRF - the European Synchrotron

"""
stats.py - beam statistics and 1D profile metrics.
"""

from __future__ import annotations


import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import warnings

# ---------------------------------------------------------------------------
# main statistics function - Public API
# ---------------------------------------------------------------------------

def get_statistics(
    beams: Union[pd.DataFrame, List[pd.DataFrame]],
    *,
    verbose: bool = False,
) -> dict:
    """
    Compute intensity-weighted beam statistics for X/Y, dX/dY, and energy.

    Lost rays are removed via ``lost_ray_flag == 0`` before computing stats.
    All statistical observables are weighted by the ``intensity`` column.

    With unit intensities, the weighted formulas reduce to the unweighted case.

    Inputs
    ------
    beams : pd.DataFrame or list[pd.DataFrame]
        One or multiple barc4beams beam DataFrames.

    Keyword Args
    ------------
    verbose : bool, optional
        If True, print a human-readable summary.

    Returns
    -------
    dict
        {
          "meta": {
             "n_rays": int,
             "n_repetitions": int,
             "good_rays": [mean, std],
             "transmission": [mean%, std%],
          },
          "energy": {"mean":[val,std], "std":[val,std], "fwhm":[val,std]},
          "X":  {"mean":[val,std], "std":[val,std],
                 "fwhm":[val,std], "skewness":[val,std], "kurtosis":[val,std]},
          "Y":  { ... same keys as X ... },
          "dX": { ... same keys as X ... },
          "dY": { ... same keys as X ... },
          "fx": [val,std],
          "fy": [val,std],
        }

        Only keys for columns actually present in the input are included.
    """

    runs = _as_list(beams)
    if not all(isinstance(df, pd.DataFrame) for df in runs):
        raise TypeError("get_statistics: all inputs must be pandas DataFrames")

    if len(runs) == 0:
        raise ValueError("get_statistics: empty input")

    if not all("intensity" in df.columns for df in runs):
        raise ValueError("get_statistics: all input DataFrames must contain an 'intensity' column")

    totals = [int(df.shape[0]) for df in runs]
    n_rays = totals[0]
    if any(t != n_rays for t in totals):
        warnings.warn(
            "Input DataFrames have different lengths; using the first for n_rays.",
            UserWarning,
        )

    cleaned: List[pd.DataFrame] = []
    transmission_pct: List[float] = []
    good_counts: List[int] = []

    for df in runs:
        total = int(df.shape[0])
        tmp = df.loc[df["lost_ray_flag"] == 0] if "lost_ray_flag" in df.columns else df

        good_counts.append(int(tmp.shape[0]))

        intensity = float(
            pd.to_numeric(tmp["intensity"], errors="coerce")
            .to_numpy(dtype=float)
            .sum()
        )
        transmission_pct.append(100.0 * intensity / total if total > 0 else np.nan)

        cleaned.append(tmp.copy())

    n_reps = len(cleaned)
    good_mean = float(np.nanmean(good_counts))
    good_std = float(np.nanstd(good_counts, ddof=0)) if n_reps > 1 else 0.0

    trans_mean = float(np.nanmean(transmission_pct))
    trans_std = float(np.nanstd(transmission_pct, ddof=0)) if n_reps > 1 else 0.0

    result: dict = {
        "meta": {
            "n_rays": n_rays,
            "n_repetitions": n_reps,
            "good_rays": [good_mean, good_std],
            "transmission": [trans_mean, trans_std],
        }
    }

    coord_cols = [c for c in ("X", "Y") if c in cleaned[0].columns]
    div_cols = [c for c in ("dX", "dY") if c in cleaned[0].columns]

    for col in coord_cols + div_cols:
        per_run = [_per_run_weighted_stats(df, col) for df in cleaned]
        result[col] = _aggregate_dicts(per_run)

    energy_runs = []
    for df in cleaned:
        energy = df["energy"].to_numpy(dtype=float)
        weights = df["intensity"].to_numpy(dtype=float)

        mu, std, skew, kurt = _weighted_moments(energy, weights)
        fwhm = _weighted_fwhm(energy, weights)

        energy_runs.append(
            {
                "mean": mu,
                "std": std,
                "fwhm": fwhm,
            }
        )

    if energy_runs:
        metrics = {"mean": [], "std": [], "fwhm": []}
        for d in energy_runs:
            for k in metrics:
                metrics[k].append(d[k])

        result["energy"] = {
            k: [float(np.nanmean(v)), float(np.nanstd(v, ddof=0))]
            for k, v in metrics.items()
        }

    fx_runs = np.array([_weighted_fl(df, "X") for df in cleaned], dtype=float)
    fy_runs = np.array([_weighted_fl(df, "Y") for df in cleaned], dtype=float)

    if np.any(np.isfinite(fx_runs)):
        result["fx"] = [
            float(np.nanmean(fx_runs)),
            float(np.nanstd(fx_runs, ddof=0)),
        ]

    if np.any(np.isfinite(fy_runs)):
        result["fy"] = [
            float(np.nanmean(fy_runs)),
            float(np.nanstd(fy_runs, ddof=0)),
        ]

    if verbose:
        _print_statistics_summary(result)

    return result

# ---------------------------------------------------------------------------
# independent functions - Public API
# ---------------------------------------------------------------------------

def get_focal_distance(
    beam: pd.DataFrame,
    verbose: bool = False,
    direction: str = "both",
    eps: float = 1e-16,
    ridge: float = 0.0,
    huge_m: float = 1e23,
) -> Dict[str, float]:
    """
    Calculate the intensity-weighted focal distance along X and Y.

    Lost rays are removed via ``lost_ray_flag == 0`` before computing the focal
    distance. If the ``intensity`` column is present, it is used as the ray
    weight. Otherwise, all rays are assigned unit weight.

    Uses the closed-form weighted least-squares expression:

        x* = -Cov_w(axis, d_axis) / (Var_w(d_axis) + ridge)

    Parameters
    ----------
    beam : pandas.DataFrame
        Beam dataframe containing X/dX and/or Y/dY columns.
    verbose : bool, optional
        If True, prints diagnostic information.
    direction : {'x', 'y', 'both'}, optional
        Direction to optimize.
    eps : float, optional
        Numerical floor for denominator stability.
    ridge : float, optional
        Extra ridge term added to the divergence variance.
    huge_m : float, optional
        Finite surrogate distance for an effectively infinite focus.

    Returns
    -------
    dict
        Dictionary with keys ``'fx'`` and ``'fy'``.
    """
    if direction not in ("x", "y", "both"):
        warnings.warn(
            f"Invalid direction '{direction}' provided; falling back to 'both'.",
            UserWarning,
        )
        direction = "both"

    if not isinstance(beam, pd.DataFrame):
        raise TypeError("get_focal_distance: beam must be a pandas DataFrame.")

    beam = beam.loc[beam["lost_ray_flag"] == 0] if "lost_ray_flag" in beam.columns else beam

    if "intensity" in beam.columns:
        weights = beam["intensity"].to_numpy(dtype=float)
    else:
        weights = np.ones(len(beam), dtype=float)

    results = {"fx": np.nan, "fy": np.nan}

    if direction in ("x", "both") and {"X", "dX"}.issubset(beam.columns):
        fx = calc_focal_distance_from_particle_distribution(
            beam["X"].to_numpy(dtype=float),
            beam["dX"].to_numpy(dtype=float),
            weights=weights,
            eps=eps,
            ridge=ridge,
            huge_m=huge_m,
        )
        results["fx"] = fx
        if verbose and np.isfinite(fx):
            print(f"Focal distance along X (m): {fx:.6g}")

    if direction in ("y", "both") and {"Y", "dY"}.issubset(beam.columns):
        fy = calc_focal_distance_from_particle_distribution(
            beam["Y"].to_numpy(dtype=float),
            beam["dY"].to_numpy(dtype=float),
            weights=weights,
            eps=eps,
            ridge=ridge,
            huge_m=huge_m,
        )
        results["fy"] = fy
        if verbose and np.isfinite(fy):
            print(f"Focal distance along Y (m): {fy:.6g}")

    return results


def calc_focal_distance_from_particle_distribution(
    position: np.ndarray,
    divergence: np.ndarray,
    weights: Optional[np.ndarray] = None,
    *,
    eps: float = 1e-16,
    ridge: float = 0.0,
    huge_m: float = 1e26,
) -> float:
    """
    Compute the signed focal distance minimizing the weighted variance of
    ``position + x * divergence``.

    If ``weights`` is None, all particles are assigned unit weight.

    Parameters
    ----------
    position : np.ndarray
        1D array of transverse positions, e.g. X or Y [m].
    divergence : np.ndarray
        1D array of corresponding angular components, e.g. dX or dY [rad].
    weights : np.ndarray, optional
        1D array of statistical ray weights. If None, unit weights are used.
    eps : float, optional
        Numerical floor for denominator stability.
    ridge : float, optional
        Extra ridge term added to the divergence variance.
    huge_m : float, optional
        Finite surrogate distance for an effectively infinite focus.

    Returns
    -------
    float
        Signed focal distance in meters.
    """
    pos = np.asarray(position, dtype=float)

    if weights is None:
        weights = np.ones_like(pos, dtype=float)

    return _weighted_focal_distance(
        position,
        divergence,
        weights,
        eps=eps,
        ridge=ridge,
        huge_m=huge_m,
    )


def calc_fwhm_from_particle_distribution(
    profile: np.ndarray,
    weights: Optional[np.ndarray] = None,
    bins: Union[int, None] = None,
) -> float:
    """
    Calculate the FWHM of a 1D particle distribution.

    If ``weights`` is None, all particles are assigned unit weight. If weights are
    provided, the FWHM is computed from a weighted histogram.

    Parameters
    ----------
    profile : np.ndarray
        1D array representing particle positions, divergences, or energies.
    weights : np.ndarray, optional
        1D array of statistical ray weights. If None, unit weights are used.
    bins : int or None, optional
        Number of histogram bins. If None, an adaptive rule is used.

    Returns
    -------
    float
        FWHM value in the same units as ``profile``. Returns -1.0 if computation
        fails.
    """
    x = np.asarray(profile, dtype=float)

    if weights is None:
        weights = np.ones_like(x, dtype=float)

    return _weighted_fwhm(x, weights, bins=bins)


def calc_centroid_from_particle_distribution(
    profile: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> float:
    """
    Compute the centroid (center of mass) of a 1D particle-position distribution.

    This treats `profile` as Monte Carlo samples of positions (each sample has equal
    weight by default). If `weights` are provided (e.g., per-ray intensities), a
    weighted centroid is computed.

    Args
    ----
    profile : np.ndarray
        1D array of particle positions (e.g., X or Y).
    weights : np.ndarray, optional
        Non-negative weights for each sample (same shape as `profile`).
        If None, all samples are equally weighted.

    Returns
    -------
    float
        Centroid in the same units as `profile`, or np.nan if it cannot be computed
        (e.g., empty or all-nonfinite inputs, or zero total weight).
    """
    x = np.asarray(profile, dtype=float)
    finite = np.isfinite(x)
    if weights is None:
        x = x[finite]
        if x.size == 0:
            return np.nan
        return float(np.mean(x))

    w = np.asarray(weights, dtype=float)
    if w.shape != x.shape:
        raise ValueError("weights must have the same shape as profile")

    finite &= np.isfinite(w)
    x, w = x[finite], w[finite]
    if x.size == 0:
        return np.nan

    w = np.clip(w, 0.0, np.inf)  # guard against negative/NaN weights
    tw = w.sum()
    if tw <= 0 or not np.isfinite(tw):
        return np.nan

    return float(np.dot(x, w) / tw)


def calc_moments_from_particle_distribution(
    profile: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> Tuple[float, float, float, float]:
    """
    Return (mean, std, skewness, kurtosis_excess) using population definitions.

    If ``weights`` is None, all particles are assigned unit weight. If weights are
    provided, the moments are intensity-weighted.

    Parameters
    ----------
    profile : np.ndarray
        1D array representing particle positions, divergences, or energies.
    weights : np.ndarray, optional
        1D array of statistical ray weights. If None, unit weights are used.

    Returns
    -------
    tuple[float, float, float, float]
        Mean, standard deviation, skewness, and excess kurtosis.
    """
    x = np.asarray(profile, dtype=float)

    if weights is None:
        weights = np.ones_like(x, dtype=float)

    return _weighted_moments(x, weights)


def calc_envelope_from_moments(
    mean: float,
    std: float,
    skewness: float,
    kurtosis_excess: float,
    axis: np.ndarray,
    method: str = "edgeworth",
    clip_negative: bool = True,
    maxent_iters: int = 2000,
    maxent_lr: float = 1e-3,
    seed: int | None = None
) -> Dict:
    """
    Construct an approximate 1D PDF envelope consistent with the first four moments.

    Parameters
    ----------
    mean : float
        Mean (μ) of the target distribution.
    std : float
        Standard deviation (σ > 0).
    skewness : float
        Standardized 3rd moment γ1.
    kurtosis_excess : float
        Excess kurtosis γ2 (kurtosis − 3).
    axis : np.ndarray
        1D grid where the envelope is evaluated.
    method : {'edgeworth', 'pearson', 'maxent'}, default 'edgeworth'
        - 'edgeworth': Gram–Charlier/Edgeworth expansion around N(μ, σ²) up to H6.
        - 'pearson'  : Pearson Type III (Gamma) matched to γ1; reflect for γ1<0.
        - 'maxent'   : Discrete maximum-entropy pdf ~ exp(a0 + a1 x + ... + a4 x^4)
                       fitted so moments up to order 4 match on the given axis.
    clip_negative : bool, default True
        If True, clip negative pdf values to 0 and renormalize (useful for 'edgeworth').
    maxent_iters : int, default 2000
        Iterations for the 'maxent' solver.
    maxent_lr : float, default 1e-3
        Learning rate for the 'maxent' solver.
    seed : int or None
        RNG seed (used only by 'maxent' initialization).

    Returns
    -------
    dict
        {'envelope': pdf_vals, 'axis': axis}

    Notes
    -----
    * **Edgeworth (default)** is fastest and smooth for |skew|≲1 and |excess kurtosis|≲2.
      Negative lobes can appear when moments are large; keep `clip_negative=True`
      to force a valid PDF.
    * **Pearson Type III** guarantees positivity and matches mean/σ/skewness exactly.
      The implied excess kurtosis is γ₂ = 1.5·γ₁²; the supplied kurtosis is ignored
      if inconsistent.
    * **Maximum entropy** finds the least-assumptive distribution subject to the
      moment constraints. It needs a wide, dense `axis` (e.g. μ±6σ) and is slower;
      increase `maxent_iters` or reduce `maxent_lr` if convergence is poor.
    * The four moments do **not uniquely define** a distribution. The returned
      envelope is only one of many possible PDFs consistent with them.
    """
    axis = np.asarray(axis, dtype=float)
    if axis.ndim != 1:
        raise ValueError("axis must be a 1D array.")
    if std <= 0:
        raise ValueError("std must be positive.")

    def _normalize(y):
        Z = np.trapz(np.clip(y, 0, None), axis)
        return (y / Z) if Z > 0 else np.full_like(y, 1.0 / len(y))

    z = (axis - mean) / std
    phi = np.exp(-0.5 * z**2) / (np.sqrt(2*np.pi))

    if method.lower() in {"edgeworth", "gram-charlier", "gram_charlier"}:
        g1 = float(skewness)
        g2 = float(kurtosis_excess)
        H3 = z**3 - 3*z
        H4 = z**4 - 6*z**2 + 3
        H6 = z**6 - 15*z**4 + 45*z**2 - 15
        corr = 1.0 + (g1/6.0)*H3 + (g2/24.0)*H4 + (g1**2/72.0)*H6
        pdf = (phi / std) * corr
        if clip_negative:
            pdf = np.clip(pdf, 0, None)
        return {"envelope": _normalize(pdf), "axis": axis}

    elif method.lower() == "pearson":
        g1 = float(skewness)
        if abs(g1) < 1e-8:
            return {"envelope": _normalize(phi / std), "axis": axis}
        k = 4.0 / (g1**2)
        theta = std / np.sqrt(k)
        loc = mean - k*theta
        from math import lgamma
        if g1 > 0:
            t = axis - loc
            base = np.where(t > 0, t**(k-1) * np.exp(-t/theta), 0.0)
        else:
            t = loc - axis
            base = np.where(t > 0, t**(k-1) * np.exp(-t/theta), 0.0)
        const = np.exp(-lgamma(k)) / (theta**k)
        pdf = const * base
        return {"envelope": _normalize(pdf), "axis": axis}

    elif method.lower() == "maxent":
        rng = np.random.default_rng(seed)
        x = axis
        mu, s = float(mean), float(std)
        g1, g2 = float(skewness), float(kurtosis_excess)
        m1 = mu
        m2 = mu**2 + s**2
        m3 = mu**3 + 3*mu*s**2 + g1*(s**3)
        m4 = mu**4 + 6*(mu**2)*(s**2) + 3*(s**4) + g2*(s**4)
        targets = np.array([1.0, m1, m2, m3, m4])
        a = np.zeros(5)
        a[1:] = rng.normal(scale=1e-3, size=4)

        def eval_pdf(params):
            y = np.exp(params[0] + params[1]*x + params[2]*x**2 +
                       params[3]*x**3 + params[4]*x**4)
            return _normalize(y)

        pdf = _normalize(np.ones_like(x))
        for _ in range(maxent_iters):
            m0 = np.trapz(pdf, x)
            m1c = np.trapz(pdf * x, x)
            m2c = np.trapz(pdf * x**2, x)
            m3c = np.trapz(pdf * x**3, x)
            m4c = np.trapz(pdf * x**4, x)
            cur = np.array([m0, m1c, m2c, m3c, m4c])
            err = cur - targets
            rel = np.array([0.0,
                            abs(err[1])/max(1.0, abs(targets[1])),
                            abs(err[2])/max(1.0, abs(targets[2])),
                            abs(err[3])/max(1.0, abs(targets[3])),
                            abs(err[4])/max(1.0, abs(targets[4]))])
            if np.all(rel[1:] < 1e-3):
                break
            a[1:] -= maxent_lr * err[1:]
            expo = a[1]*x + a[2]*x**2 + a[3]*x**3 + a[4]*x**4
            un = np.exp(expo - np.max(expo))
            Z = np.trapz(un, x)
            a[0] = -np.log(Z) + np.max(expo)
            pdf = eval_pdf(a)

        return {"envelope": _normalize(np.clip(pdf, 0, None)), "axis": axis}

    else:
        raise ValueError("method must be one of {'edgeworth', 'pearson', 'maxent'}.")
    
# ---------------------------------------------------------------------------
# internal helpers
# ---------------------------------------------------------------------------

def _finite(a):
    return a[np.isfinite(a)]

def _as_list(obj):
    return list(obj) if isinstance(obj, (list, tuple)) else [obj]

def _aggregate_dicts(dicts):
    metrics = list(dicts[0].keys())
    out = {}
    for m in metrics:
        vals = np.array([d[m] for d in dicts], dtype=float)
        out[m] = [float(np.nanmean(vals)), float(np.nanstd(vals, ddof=0))]
    return out
    
def _weighted_moments(
    profile: np.ndarray,
    weights: np.ndarray,
) -> Tuple[float, float, float, float]:
    """
    Return intensity-weighted (mean, std, skewness, kurtosis_excess).

    Non-finite profile or weight entries are excluded pairwise.
    Entries with zero weight do not contribute.
    """
    x = np.asarray(profile, dtype=float)
    w = np.asarray(weights, dtype=float)

    if x.shape != w.shape:
        raise ValueError("profile and weights must have the same shape")

    valid = np.isfinite(x) & np.isfinite(w) & (w > 0.0)
    x = x[valid]
    w = w[valid]

    if x.size == 0:
        return (np.nan, np.nan, np.nan, np.nan)

    w_sum = float(np.sum(w))
    if w_sum <= 0.0 or not np.isfinite(w_sum):
        return (np.nan, np.nan, np.nan, np.nan)

    mu = float(np.sum(w * x) / w_sum)
    xc = x - mu

    m2 = float(np.sum(w * xc**2) / w_sum)
    if not np.isfinite(m2) or m2 <= 0.0:
        return (mu, 0.0, np.nan, np.nan)

    sigma = float(np.sqrt(m2))
    m3 = float(np.sum(w * xc**3) / w_sum)
    m4 = float(np.sum(w * xc**4) / w_sum)

    skew = m3 / sigma**3
    kurt_excess = m4 / sigma**4 - 3.0

    return (mu, sigma, skew, kurt_excess)


def _weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantiles,
) -> np.ndarray:
    """Return linearly interpolated quantiles of a weighted sample."""
    order = np.argsort(values)
    x = np.asarray(values, dtype=float)[order]
    w = np.asarray(weights, dtype=float)[order]
    cumulative = (np.cumsum(w) - 0.5 * w) / np.sum(w)
    return np.interp(quantiles, cumulative, x, left=x[0], right=x[-1])


def _weighted_fwhm(
    profile: np.ndarray,
    weights: np.ndarray,
    bins: Union[int, None] = None,
) -> float:
    """
    Calculate weighted FWHM of a 1D particle distribution.

    Uses a weighted histogram plus linear interpolation around the half-maximum.
    With unit weights, this reduces to the unweighted histogram result.
    """
    x = np.asarray(profile, dtype=float)
    w = np.asarray(weights, dtype=float)

    if x.shape != w.shape:
        raise ValueError("profile and weights must have the same shape")

    valid = np.isfinite(x) & np.isfinite(w) & (w > 0.0)
    x = x[valid]
    w = w[valid]

    if x.size < 2:
        return -1.0

    histogram_range = None
    if bins is None:
        weight_sum = float(np.sum(w))
        effective_n = weight_sum**2 / float(np.sum(w**2))
        q25, q75 = _weighted_quantile(x, w, [0.25, 0.75])
        iqr = q75 - q25

        tail_probability = 0.5 / max(1.0, effective_n)
        lo, hi = _weighted_quantile(
            x,
            w,
            [tail_probability, 1.0 - tail_probability],
        )
        histogram_range = (float(lo), float(hi))
        span = hi - lo

        if iqr > 0:
            h = 2.0 * iqr / (effective_n ** (1.0 / 3.0))
            bins = max(2, int(np.ceil(span / h))) if h > 0 and span > 0 else 2
        else:
            bins = max(2, int(np.sqrt(effective_n)))

    bins = max(2, int(bins))

    counts, edges = np.histogram(
        x,
        bins=bins,
        range=histogram_range,
        weights=w,
        density=False,
    )

    if not np.any(np.isfinite(counts)) or counts.max() <= 0:
        return -1.0

    centers = 0.5 * (edges[:-1] + edges[1:])
    target = 0.5 * counts.max()

    above = counts >= target
    flips = np.flatnonzero(above[:-1] ^ above[1:])

    if flips.size == 0 and np.all(above):
        return float(edges[-1] - edges[0])

    if flips.size == 0:
        return -1.0

    def interp_cross(i: int) -> float:
        y1, y2 = counts[i], counts[i + 1]
        x1, x2 = centers[i], centers[i + 1]

        if y2 == y1:
            return float(0.5 * (x1 + x2))

        return float(x1 + (target - y1) * (x2 - x1) / (y2 - y1))

    x_cross = np.array([interp_cross(i) for i in flips], dtype=float)

    if x_cross.size < 2 or not np.all(np.isfinite(x_cross)):
        return -1.0

    x_peak = centers[int(np.argmax(counts))]
    left = x_cross[x_cross <= x_peak]
    right = x_cross[x_cross >= x_peak]

    if left.size == 0 or right.size == 0:
        return float(edges[-1] - edges[0])

    width = float(right[0] - left[-1])
    return width if np.isfinite(width) and width > 0 else -1.0

def _weighted_focal_distance(
    position: np.ndarray,
    divergence: np.ndarray,
    weights: np.ndarray,
    *,
    eps: float = 1e-16,
    ridge: float = 0.0,
    huge_m: float = 1e26,
) -> float:
    """
    Compute the signed focal distance minimizing the weighted variance of
    position + x * divergence.

    x* = -Cov_w(position, divergence) / (Var_w(divergence) + ridge)
    """
    pos = np.asarray(position, dtype=float)
    div = np.asarray(divergence, dtype=float)
    w = np.asarray(weights, dtype=float)

    if pos.shape != div.shape or pos.shape != w.shape:
        raise ValueError("position, divergence, and weights must have the same shape")

    valid = np.isfinite(pos) & np.isfinite(div) & np.isfinite(w) & (w > 0.0)
    pos = pos[valid]
    div = div[valid]
    w = w[valid]

    if pos.size == 0:
        return np.nan

    w_sum = float(np.sum(w))
    if w_sum <= 0.0 or not np.isfinite(w_sum):
        return np.nan

    pos_mean = float(np.sum(w * pos) / w_sum)
    div_mean = float(np.sum(w * div) / w_sum)

    posc = pos - pos_mean
    divc = div - div_mean

    var_d = float(np.sum(w * divc**2) / w_sum)
    cov_pd = float(np.sum(w * posc * divc) / w_sum)

    denom = (var_d if np.isfinite(var_d) else 0.0) + max(eps, ridge)

    x_star = -cov_pd / denom if np.isfinite(cov_pd) else np.nan

    sign = -1.0 if (np.isfinite(cov_pd) and cov_pd > 0) else 1.0
    return float(x_star) if np.isfinite(x_star) else float(sign * huge_m)

def _weighted_fl(df: pd.DataFrame, col: str) -> float:
    """
    Weighted focal distance helper for dataframe columns.

    col='X' uses X/dX.
    col='Y' uses Y/dY.
    """
    dcol = f"d{col}"

    if {col, dcol, "intensity"}.issubset(df.columns):
        return _weighted_focal_distance(
            df[col].to_numpy(dtype=float),
            df[dcol].to_numpy(dtype=float),
            df["intensity"].to_numpy(dtype=float),
        )

    return np.nan


def _print_statistics_summary(result: dict) -> None:
    """
    Print a human-readable statistics summary with stable display units.
    """
    meta = result["meta"]
    n_reps = meta["n_repetitions"]
    n_rays = meta["n_rays"]

    good_mean, good_std = meta["good_rays"]
    t_mean, t_std = meta["transmission"]
    good_pct = 100.0 * good_mean / n_rays if n_rays > 0 else np.nan
    good_pct_std = 100.0 * good_std / n_rays if n_rays > 0 else np.nan

    print(f"\n\n{n_reps} x {n_rays} rays ")
    print(
        f"> good rays: "
        f"{_format_with_unc(good_pct, good_pct_std, 'percent', n_reps)}"
    )
    print(
        f"> intensity transmission: "
        f"{_format_with_unc(t_mean, t_std, 'percent', n_reps)}\n"
    )

    if "energy" in result:
        e = result["energy"]
        energy = _format_with_unc(e["mean"][0], e["mean"][1], "energy", n_reps)
        if e["std"][0] < 1e-6:
            print(f"Beam energy: {energy} (monochromatic)")
        else:
            spread = _format_with_unc(
                e["std"][0],
                e["std"][1],
                "energy_spread",
                n_reps,
            )
            fwhm = _format_with_unc(
                e["fwhm"][0],
                e["fwhm"][1],
                "energy_spread",
                n_reps,
            )
            print(f"Beam energy: {energy} (RMS: {spread}, FWHM: {fwhm})")

    for axis in ("X", "Y"):
        if axis not in result:
            continue

        direction = "horizontal" if axis == "X" else "vertical"
        print(f"\n------------------ {direction}-plane:")

        key_f = "fx" if axis == "X" else "fy"
        if key_f in result:
            f_mean, f_std = result[key_f]
            print(f"> Beam focusing at {_format_with_unc(f_mean, f_std, 'focal', n_reps)}")

        stats_axis = result[axis]
        stats_div = result["dX" if axis == "X" else "dY"]

        size = _format_with_unc(
            stats_axis["std"][0],
            stats_axis["std"][1],
            "beam_um",
            n_reps,
            scale=1e6,
        )
        size_fwhm = _format_with_unc(
            stats_axis["fwhm"][0],
            stats_axis["fwhm"][1],
            "beam_um",
            n_reps,
            scale=1e6,
        )
        divergence = _format_with_unc(
            stats_div["std"][0],
            stats_div["std"][1],
            "divergence_urad",
            n_reps,
            scale=1e6,
        )
        divergence_fwhm = _format_with_unc(
            stats_div["fwhm"][0],
            stats_div["fwhm"][1],
            "divergence_urad",
            n_reps,
            scale=1e6,
        )
        centroid = _format_with_unc(
            stats_axis["mean"][0],
            stats_axis["mean"][1],
            "beam_um",
            n_reps,
            scale=1e6,
        )

        print(f">> RMS beam size: {size} (FWHM: {size_fwhm})")
        print(f">> Divergence: {divergence} (FWHM: {divergence_fwhm})")
        print(f">> Centroid: {centroid}")
        print(
            f">> Skewness: "
            f"{_format_with_unc(stats_axis['skewness'][0], stats_axis['skewness'][1], 'shape', n_reps)}"
        )
        print(
            f">> Kurtosis: "
            f"{_format_with_unc(stats_axis['kurtosis'][0], stats_axis['kurtosis'][1], 'shape', n_reps)}"
        )


def _format_with_unc(
    val: float,
    unc: float,
    kind: str,
    n_reps: int,
    *,
    scale: float = 1.0,
) -> str:
    """
    Format a scalar or scalar +- repetition uncertainty.
    """
    v = val * scale
    u = unc * scale
    unit = _format_unit(kind)

    if not np.isfinite(v):
        return f"{v:.6g}{unit}"

    if n_reps <= 1 or not np.isfinite(u) or u == 0:
        return f"{_format_scalar(v, kind)}{unit}"

    decimals = _uncertainty_decimals(u, kind)
    if decimals is None:
        return f"{_format_scalar(v, kind)} +- {_format_scalar(u, kind)}{unit}"

    return f"{v:.{decimals}f} +- {u:.{decimals}f}{unit}"


def _format_scalar(val: float, kind: str) -> str:
    """
    Format one already-scaled value according to its physical quantity.
    """
    if not np.isfinite(val):
        return f"{val:.6g}"

    if kind in {"beam_um", "divergence_urad"}:
        return _format_beam_scale(val)

    if kind == "focal":
        return _format_focal_m(val)

    if kind == "percent":
        return _format_percent(val)

    if kind == "shape":
        return f"{val:.3f}"

    if kind == "energy":
        return f"{val:.6g}"

    if kind == "energy_spread":
        return _format_energy_spread(val)

    return f"{val:.6g}"


def _format_beam_scale(val: float) -> str:
    aval = abs(val)
    if aval >= 10000:
        return f"{val:.4e}"
    if aval < 1:
        return f"{val:.3f}"
    if aval < 10:
        return f"{val:.2f}"
    if aval < 100:
        return f"{val:.1f}"
    return f"{val:.0f}"


def _format_focal_m(val: float) -> str:
    aval = abs(val)
    if aval < 10e-9:
        return "0"
    if aval < 100e-6:
        return f"{val:.4e}"
    if aval < 100e-3:
        return f"{val:.6f}"
    if aval < 1:
        return f"{val:.4f}"
    if aval < 100:
        return f"{val:.3f}"
    if aval < 1000:
        return f"{val:.2f}"
    if aval < 10000:
        return f"{val:.1f}"
    return f"{val:.4e}"


def _format_percent(val: float) -> str:
    aval = abs(val)
    if aval < 1:
        return f"{val:.3f}"
    if aval < 10:
        return f"{val:.2f}"
    return f"{val:.1f}"


def _format_energy_spread(val: float) -> str:
    aval = abs(val)
    if aval < 1:
        return f"{val:.3f}"
    if aval < 10:
        return f"{val:.2f}"
    if aval < 100:
        return f"{val:.1f}"
    return f"{val:.0f}"


def _uncertainty_decimals(unc: float, kind: str) -> Optional[int]:
    """
    Pick shared decimal places for value +- uncertainty.
    """
    if not np.isfinite(unc) or unc == 0:
        return None

    if kind in {"beam_um", "divergence_urad"}:
        return min(max(_significant_unc_decimals(unc), 0), 3)

    if kind == "percent":
        return min(max(_significant_unc_decimals(unc), 0), 3)

    if kind == "shape":
        return min(max(_significant_unc_decimals(unc), 0), 4)

    if kind == "energy_spread":
        return min(max(_significant_unc_decimals(unc), 0), 3)

    return None


def _significant_unc_decimals(unc: float) -> int:
    """
    Decimal places needed to keep one significant digit of an uncertainty.
    """
    if unc == 0 or not np.isfinite(unc):
        return 0

    exp = int(np.floor(np.log10(abs(unc))))
    return max(-exp, 0)


def _format_unit(kind: str) -> str:
    if kind == "beam_um":
        return " µm"
    if kind == "divergence_urad":
        return " µrad"
    if kind == "focal":
        return " m"
    if kind == "percent":
        return "%"
    if kind == "energy":
        return " eV"
    if kind == "energy_spread":
        return " eV"
    return ""


def _per_run_weighted_stats(df: pd.DataFrame, col: str) -> dict:
    """
    Compute weighted statistics for one dataframe column in one run.
    """
    arr = df[col].to_numpy(dtype=float)
    weights = df["intensity"].to_numpy(dtype=float)

    mu, std, skew, kurt = _weighted_moments(arr, weights)
    fwhm = _weighted_fwhm(arr, weights)

    return {
        "mean": mu,
        "std": std,
        "fwhm": fwhm,
        "skewness": skew,
        "kurtosis": kurt,
    }
