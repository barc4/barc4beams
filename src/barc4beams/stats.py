# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2025 Synchrotron SOLEIL

"""
stats.py - beam statistics and 1D profile metrics.
"""

from __future__ import annotations


import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
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
    Compute beam statistics for X/Y and their divergences dX/dY, plus energy.

    - Lost rays are removed via (intensity != 0) before computing stats.

    Inputs
    ------
    beams : pd.DataFrame or list[pd.DataFrame]
        One or multiple PyOptiX beam DataFrames (e.g., from get_diagram/get_impacts).

    Keyword Args
    ------------

    verbose : bool, optional
        If True, print a human-readable summary.

    Returns
    -------
    dict
        {
          "meta": {
             "n_rays": int,                 # total rays per run (before filtering)
             "n_repetitions": int,          # number of runs
             "transmission": [mean%, std%], # mean/std of 100*(good/total)
          },
          "energy": {"mean":[val,std], "std":[val,std], "fwhm":[val,std]},
          "X":  {"mean":[val,std], "centroid":[val,std], "std":[val,std],
                 "fwhm":[val,std], "skewness":[val,std], "kurtosis":[val,std]},
          "Y":  { ... same keys as X ... },
          "dX": { ... same keys as X ... },
          "dY": { ... same keys as X ... },
          "fx": [val,std],   # from calc_focal_distance_from_particle_distribution
          "fy": [val,std],
        }
        Only keys for columns actually present in the input are included.
    """

    runs = _as_list(beams)
    if not all(isinstance(df, pd.DataFrame) for df in runs):
        raise TypeError("get_beam_statistics: all inputs must be pandas DataFrames")

    if len(runs) == 0:
        raise ValueError("get_beam_statistics: empty input")

    totals = [int(df.shape[0]) for df in runs]
    n_rays = totals[0]
    if any(t != n_rays for t in totals):
        warnings.warn("Input DataFrames have different lengths; using the first for n_rays.", UserWarning)

    cleaned: List[pd.DataFrame] = []
    transmission_pct: List[float] = []
    good_counts: List[int] = []

    for df in runs:
        total = int(df.shape[0])
        tmp = df.loc[df["lost_ray_flag"] == 0] if "lost_ray_flag" in df.columns else df
        good_counts.append(int(tmp.shape[0]))
        intensity = float(
            pd.to_numeric(tmp["intensity"], errors="coerce").to_numpy(dtype=float).sum()
        )
        transmission_pct.append(100.0 * intensity / total if total > 0 else np.nan)

        cleaned.append(tmp.copy())

    n_reps = len(cleaned)
    good_mean = float(np.nanmean(good_counts))
    good_std  = float(np.nanstd(good_counts, ddof=0)) if n_reps > 1 else 0.0

    trans_mean = float(np.nanmean(transmission_pct))
    trans_std  = float(np.nanstd(transmission_pct, ddof=0)) if n_reps > 1 else 0.0

    result: dict = {
        "meta": {
            "n_rays": n_rays,
            "n_repetitions": n_reps,
            "good_rays": [good_mean, good_std],
            "transmission": [trans_mean, trans_std],
        }
    }

    coord_cols = [c for c in ("X", "Y") if c in cleaned[0].columns]
    div_cols   = [c for c in ("dX", "dY") if c in cleaned[0].columns]

    for col in coord_cols + div_cols:
        per_run = [_per_run_stats(df, col) for df in cleaned]
        result[col] = _aggregate_dicts(per_run)

    energy_runs = []
    for df in cleaned:
        e = df["energy"].to_numpy(dtype=float)
        e = e[np.isfinite(e)]
        if e.size == 0:
            energy_runs.append({"mean": np.nan, "std": np.nan, "fwhm": np.nan})
        else:
            mu, std, *_ = _moments(e)
            f = _fwhm(e)
            energy_runs.append({"mean": mu, "std": std, "fwhm": f})

    if energy_runs:
        metrics = {"mean": [], "std": [], "fwhm": []}
        for d in energy_runs:
            for k in metrics:
                metrics[k].append(d[k])
        result["energy"] = {k: [float(np.nanmean(v)), float(np.nanstd(v, ddof=0))]
                            for k, v in metrics.items()}

    fx_runs = np.array([_fl(df, 'X') for df in cleaned], dtype=float)
    fy_runs = np.array([_fl(df, 'Y') for df in cleaned], dtype=float)

    if np.any(np.isfinite(fx_runs)):
        result["fx"] = [float(np.nanmean(fx_runs)), float(np.nanstd(fx_runs, ddof=0))]
    if np.any(np.isfinite(fy_runs)):
        result["fy"] = [float(np.nanmean(fy_runs)), float(np.nanstd(fy_runs, ddof=0))]

    if verbose:
        def fmt_with_unc(val, unc, scale = 1.0, unit = "") -> str:
            """
            Format value +- uncertainty with 1 significant figure for the uncertainty.
            Scales by `scale` (e.g. 1e6 for um) and appends `unit`.
            """
            if not np.isfinite(val) or not np.isfinite(unc):
                return f"{val:.6g}{unit}"
            if unc == 0:
                return f"{val*scale:.6g}{unit}"
            v, u = val * scale, unc * scale

            exp = int(np.floor(np.log10(abs(u)))) if u != 0 else 0
            dec = -exp
            u_rounded = round(u, -exp)
            v_rounded = round(v, -exp)
            return f"{v_rounded:.{max(dec,0)}f} +- {u_rounded:.{max(dec,0)}f}{unit}"

        t_mean, t_std = result["meta"]["transmission"]
        print(f"\n\n{n_reps} x {n_rays} rays ")
        print(f"> good rays: {fmt_with_unc(good_mean/n_rays, good_std/n_rays, scale=100, unit='%')}")
        print(f"> intensity transmission: {fmt_with_unc(t_mean/100, t_std/100, scale=100, unit='%')}\n")

        if "energy" in result:
            e = result["energy"]
            if e['std'][0] < 1e-6:  # tolerance for "monochromatic"
                print(f"Beam energy: {e['mean'][0]:.6g} eV (monochromatic)")
            else:
                print(f"Beam energy: {e['mean'][0]:.6g} +- {e['std'][0]:.3g} eV "
                    f"(FWHM: {e['fwhm'][0]:.3g} eV)")

        for axis in ("X", "Y"):
            if axis not in result:
                continue
            direction = "horizontal" if axis == "X" else "vertical"
            print(f"\n------------------ {direction}-plane:")

            key_f = "fx" if axis == "X" else "fy"
            if key_f in result:
                f_mean, f_std = result[key_f]
                print(f"> Beam focusing at {fmt_with_unc(f_mean, f_std, unit=' m')}")

            stats_axis = result[axis]
            stats_div  = result["dX" if axis == "X" else "dY"]
            cen_mean, cen_std = stats_axis["centroid"]

            print(f">> RMS beam size: "
                  f"{fmt_with_unc(stats_axis['std'][0], stats_axis['std'][1], scale=1e6, unit=' um')} "
                  f"(FWHM: {fmt_with_unc(stats_axis['fwhm'][0], stats_axis['fwhm'][1], scale=1e6, unit=' um')})")
                #   f"1/e2: {fmt_with_unc(stats_axis['e2'][0], stats_axis['e2'][1], scale=1e6, unit=' um')})")
            print(f">>> Centroid: {fmt_with_unc(cen_mean, cen_std, scale=1e6, unit=' um')}")
            print(f">> Divergence: "
                  f"{fmt_with_unc(stats_div['std'][0], stats_div['std'][1], scale=1e6, unit=' urad')} "
                  f"(FWHM: {fmt_with_unc(stats_div['fwhm'][0], stats_div['fwhm'][1], scale=1e6, unit=' urad')})")            
            print(f">> Skewness: {fmt_with_unc(stats_axis['skewness'][0], stats_axis['skewness'][1])}")
            print(f">> Kurtosis: {fmt_with_unc(stats_axis['kurtosis'][0], stats_axis['kurtosis'][1])}")

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
    huge_m: float = 1e23
) -> Dict[str, float]:
    """
    Calculate the focalization distance along X and Y axes based on phase space data.

    Uses a closed-form least-squares expression instead of iterative minimization:
        x* = -Cov(axis, d_axis) / (Var(d_axis) + ridge)
    where axis is {X, Y} and d_axis is {dX, dY}.
    The ridge and eps prevent numerical instabilities.

    Args:
        beam (pandas.DataFrame): 
            PyOptiX beam (pd.DataFrame): typically from:
                OpticalElement.get_diagram(...)
                OpticalElement.get_impacts(...).
        verbose (bool, optional): If True, prints diagnostic information. Defaults to False.
        direction (str, optional): Direction to optimize ('x', 'y', or 'both'). Defaults to 'both'.
        eps (float, optional): Regularization floor for denominator stability. Defaults to 1e-16.
        ridge (float, optional): Extra ridge term added to denominator to stabilize variance. Defaults to 0.0.
        huge_m (float, optional): Threshold distance beyond which the beam is flagged as near-collimated (effectively ∞). Defaults to 1e6.

    Returns:
        Dict[str, float]: Dictionary with keys 'fx' and 'fy' for focalization distances [m].
                          If a direction is not computed, its value is np.nan.
    """
    if direction not in ("x", "y", "both"):
        warnings.warn(
            f"Invalid direction '{direction}' provided; falling back to 'both'.",
            UserWarning
        )
        direction = "both"

    if not isinstance(beam, pd.DataFrame):
        raise TypeError("get_beam_focal_distance: beam must be a pandas DataFrame.")

    beam = beam.loc[beam["lost_ray_flag"] == 0] if "lost_ray_flag" in beam.columns else beam

    results = {"fx": np.nan, "fy": np.nan}

    if direction in ("x", "both") and {"X", "dX"}.issubset(beam.columns):
        fx = calc_focal_distance_from_particle_distribution(
            beam["X"].to_numpy(dtype=float),
            beam["dX"].to_numpy(dtype=float),
            eps=eps, ridge=ridge, huge_m=huge_m
        )
        results["fx"] = fx
        if verbose and np.isfinite(fx):
            print(f"Focal distance along X (m): {fx:.6g}")

    if direction in ("y", "both") and {"Y", "dY"}.issubset(beam.columns):
        fy = calc_focal_distance_from_particle_distribution(
            beam["Y"].to_numpy(dtype=float),
            beam["dY"].to_numpy(dtype=float),
            eps=eps, ridge=ridge, huge_m=huge_m
        )
        results["fy"] = fy
        if verbose and np.isfinite(fy):
            print(f"Focal distance along Y (m): {fy:.6g}")

    return results

def calc_focal_distance_from_particle_distribution(
    position: np.ndarray,
    divergence: np.ndarray,
    *,
    eps: float = 1e-16,
    ridge: float = 0.0,
    huge_m: float = 1e26,
) -> float:
    """
    Compute the signed focal distance x* minimizing std(position + x * divergence).

    Closed-form least-squares:
        x* = -Cov(position, divergence) / (Var(divergence) + ridge)

    Args:
        position (np.ndarray): 1D array of transverse positions (e.g., X or Y) [m].
        divergence (np.ndarray): 1D array of corresponding angular components (e.g., dX or dY) [rad].
        eps (float): Numerical floor added to denominator for stability.
        ridge (float): Additional ridge term to stabilize Var(divergence).
        huge_m (float): Distance used as finite surrogate when focus is effectively at infinity.

    Returns:
        float: Signed focal distance in meters. If the optimum is effectively at
               infinity, returns ±huge_m with the sign inferred from covariance.
               Returns np.nan only if inputs are unusable (no finite samples).
    """
    # sanitize
    pos = np.asarray(position, dtype=float)
    div = np.asarray(divergence, dtype=float)
    finite = np.isfinite(pos) & np.isfinite(div)
    if not np.any(finite):
        return np.nan

    pos, div = pos[finite], div[finite]
    if pos.size == 0 or div.size == 0:
        return np.nan

    # center
    posc = pos - pos.mean()
    divc = div - div.mean()

    # moments
    var_d  = float(np.dot(divc, divc) / divc.size)
    cov_pd = float(np.dot(posc, divc) / divc.size)
    denom  = (var_d if np.isfinite(var_d) else 0.0) + max(eps, ridge)

    x_star = -cov_pd / denom if np.isfinite(cov_pd) else np.nan

    # near-collimated / overflow handling
    sign = -1.0 if (np.isfinite(cov_pd) and cov_pd > 0) else 1.0
    return float(x_star) if np.isfinite(x_star) else float(sign * huge_m)


def calc_fwhm_from_particle_distribution(profile: np.ndarray, bins: Union[int, None] = None) -> float:
    """
    Calculate the Full Width at Half Maximum (FWHM) of a 1D beam profile
    using histogram + linear interpolation (no splines).

    - If `bins` is None, uses Freedman–Diaconis rule to set bin width; falls back to sqrt(N) if IQR=0.
    - The algorithm finds the two half-maximum crossing points that straddle the global peak.
    - Returns width (>0) in profile units, or the data span when the histogram
      never goes below half max (flat-top/cropped case). Returns -1.0 only on failure.

    Args:
        profile (np.ndarray): 1D array representing particle positions (e.g., X or Y).
        bins (int, optional): Number of bins for the histogram. If None, adaptive rules are applied.

    Returns:
        float: FWHM value in the same units as `profile`. Returns -1.0 if computation fails.
    """
    x = np.asarray(profile, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return -1.0

    # choose bins
    if bins is None:
        q75, q25 = np.percentile(x, [75, 25])
        iqr = q75 - q25
        if iqr > 0:
            h = 2.0 * iqr / (x.size ** (1/3))
            bins = max(2, int(np.ceil((x.max() - x.min()) / h))) if h > 0 else int(np.sqrt(x.size))
        else:
            bins = int(np.sqrt(x.size))
    bins = max(2, int(bins))

    # histogram
    counts, edges = np.histogram(x, bins=bins, density=True)
    if not np.any(np.isfinite(counts)) or counts.max() <= 0:
        return -1.0

    centers = 0.5 * (edges[:-1] + edges[1:])
    target = 0.5 * counts.max()

    # crossings
    above = counts >= target
    flips = np.flatnonzero(above[:-1] ^ above[1:])

    # --- NEW: flat-top / cropped fallback ---
    # If we never cross below half-maximum anywhere, use the sampled span.
    if flips.size == 0 and np.all(above):
        return float(edges[-1] - edges[0])

    if flips.size == 0:
        return -1.0

    def interp_cross(i):
        y1, y2 = counts[i], counts[i+1]
        x1, x2 = centers[i], centers[i+1]
        if y2 == y1:  # flat at threshold
            return 0.5 * (x1 + x2)
        return x1 + (target - y1) * (x2 - y1) / (y2 - y1)

    x_cross = np.array([interp_cross(i) for i in flips], dtype=float)
    if x_cross.size < 2 or not np.all(np.isfinite(x_cross)):
        return -1.0

    # around the global peak
    x_peak = centers[int(np.argmax(counts))]
    left  = x_cross[x_cross <= x_peak]
    right = x_cross[x_cross >= x_peak]

    # If one side is missing (e.g., peak at the boundary), treat as cropped window.
    if left.size == 0 or right.size == 0:
        return float(edges[-1] - edges[0])

    width = float(right[0] - left[-1])
    return width if np.isfinite(width) and width > 0 else -1.0


def calc_e2_width_from_particle_distribution(profile: np.ndarray, bins: Union[int, None] = None) -> float:
    """
    1/e² width of a 1D beam profile via histogram + linear interpolation,
    analogous to calc_fwhm_from_particle_distribution but using peak/e².

    Returns positive width (same units as `profile`) or -1.0 on failure.
    """
    x = np.asarray(profile, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return -1.0

    # Bin choice (same heuristic as FWHM helper)
    if bins is None:
        q75, q25 = np.percentile(x, [75, 25])
        iqr = q75 - q25
        if iqr > 0:
            h = 2.0 * iqr / (x.size ** (1/3))
            bins = max(2, int(np.ceil((x.max() - x.min()) / h))) if h > 0 else int(np.sqrt(x.size))
        else:
            bins = int(np.sqrt(x.size))
    bins = max(2, int(bins))

    # histogram
    counts, edges = np.histogram(x, bins=bins, density=True)
    if not np.any(np.isfinite(counts)) or counts.max() <= 0:
        return -1.0
    centers = 0.5 * (edges[:-1] + edges[1:])

    target = counts.max() / (np.e ** 2)

    # find threshold crossings with linear interpolation
    above = counts >= target
    flips = np.flatnonzero(above[:-1] ^ above[1:])

    # --- NEW: flat-top / cropped fallback ---
    if flips.size == 0 and np.all(above):
        return float(edges[-1] - edges[0])  # never below 1/e² → span

    if flips.size == 0:
        return -1.0

    def _interp(i):
        y1, y2 = counts[i], counts[i+1]
        x1, x2 = centers[i], centers[i+1]
        if y2 == y1:
            return 0.5 * (x1 + x2)
        return x1 + (target - y1) * (x2 - y1) / (y2 - y1)

    x_cross = np.array([_interp(i) for i in flips], dtype=float)
    if x_cross.size < 2 or not np.all(np.isfinite(x_cross)):
        return -1.0

    # pick crossings around the global peak
    x_peak = centers[int(np.argmax(counts))]
    left  = x_cross[x_cross <= x_peak]
    right = x_cross[x_cross >= x_peak]

    # --- NEW: boundary-peak fallback (one side missing) ---
    if left.size == 0 or right.size == 0:
        return float(edges[-1] - edges[0])

    width = float(right[0] - left[-1])
    return width if np.isfinite(width) and width > 0 else -1.0


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

def calc_moments_from_particle_distribution(profile: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Return (mean, std, skewness, kurtosis_excess) using population definitions,
    unweighted.
    """
    x = _finite(profile)
    if x.size == 0:
        return (np.nan, np.nan, np.nan, np.nan)
    mu = float(np.mean(x))
    xc = x - mu
    m2 = float(np.mean(xc**2))
    if not np.isfinite(m2) or m2 <= 0.0:
        return (mu, 0.0, np.nan, np.nan)
    sigma = np.sqrt(m2)
    m3 = float(np.mean(xc**3))
    m4 = float(np.mean(xc**4))
    skew = m3 / (sigma**3)
    kurt_excess = m4 / (sigma**4) - 3.0
    return (mu, sigma, skew, kurt_excess)


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
    
def _centroid(x):
    return calc_centroid_from_particle_distribution(x)

def _fwhm(x):
    return calc_fwhm_from_particle_distribution(x, bins=None)

def _e2(x):
    return calc_e2_width_from_particle_distribution(x, bins=None)

def _moments(x):
    return calc_moments_from_particle_distribution(x)

def _fl(df, col):
    return calc_focal_distance_from_particle_distribution(
        df[col].to_numpy(dtype=float), df[f"d{col}"].to_numpy(dtype=float)
    ) if {col, f"d{col}"}.issubset(df.columns) else np.nan

def _per_run_stats(df, col):
    arr = df[col].to_numpy(dtype=float)
    mu, std, skew, kurt = _moments(arr)
    centroid = _centroid(arr)
    fwhm = _fwhm(arr)
    e2 = _e2(arr) if col in ('X','Y') else np.nan
    return {
        "mean": mu, "centroid": centroid, "std": std,
        "fwhm": fwhm, "skewness": skew, "kurtosis": kurt,
        "e2": e2
    }
