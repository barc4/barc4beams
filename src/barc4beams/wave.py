# SPDX-License-Identifier: CECILL-2.1
# Copyright (c) 2026 ESRF - the European Synchrotron

"""
wave.py - Gaussian-equivalent wave-optics metrics estimated from rays.
"""

from __future__ import annotations

from typing import List, Sequence, Union

import numpy as np
import pandas as pd

from . import stats


RMS_TO_HALF_FWHM = np.sqrt(2.0 * np.log(2.0))


def get_wave_metrics(
    beams: Union[pd.DataFrame, Sequence[pd.DataFrame]],
    *,
    max_focal_distance: float = 1000.0,
    verbose: bool = False,
) -> dict:
    """
    Estimate Gaussian-equivalent wave-optics metrics from ray beams.

    For each transverse axis, the fitted focal distance from ``stats`` is used
    to propagate the rays to the image plane. The image-plane RMS divergence is
    then interpreted as a Gaussian-equivalent half-angle:

        theta = image-plane RMS divergence multiplied by sqrt(2*ln(2)),
        NA = sin(theta)
        gaussian_waist = wavelength / (pi * NA)
        depth_of_focus = 2 * pi * gaussian_waist**2 / wavelength

    The returned numerical values use SI units.
    """
    runs = _as_list(beams)
    if not all(isinstance(df, pd.DataFrame) for df in runs):
        raise TypeError("get_wave_metrics: all inputs must be pandas DataFrames")

    if len(runs) == 0:
        raise ValueError("get_wave_metrics: empty input")

    if not np.isfinite(max_focal_distance) or max_focal_distance <= 0:
        raise ValueError("get_wave_metrics: max_focal_distance must be > 0")

    per_run_energy = []
    per_run_wavelength = []
    per_axis = {"X": [], "Y": []}

    for df in runs:
        run_stats = stats.get_statistics(df)
        energy = run_stats["energy"]["mean"][0]
        wavelength = _mean_wavelength(df)
        per_run_energy.append(energy)
        per_run_wavelength.append(wavelength)

        for axis in ("X", "Y"):
            key_f = "fx" if axis == "X" else "fy"
            if key_f not in run_stats:
                continue

            per_axis[axis].append(
                _per_run_axis_metrics(
                    df,
                    axis,
                    focal_distance=run_stats[key_f][0],
                    wavelength=wavelength,
                    max_focal_distance=max_focal_distance,
                )
            )

    result = {
        "meta": {
            "method": "gaussian_equivalent",
            "n_repetitions": len(runs),
            "wavelength": _aggregate_values(per_run_wavelength),
            "energy": _aggregate_values(per_run_energy),
        }
    }

    for axis, axis_runs in per_axis.items():
        if axis_runs:
            result[axis] = _aggregate_axis_metrics(axis_runs)

    if verbose:
        _print_wave_summary(result)

    return result


def _as_list(beams: Union[pd.DataFrame, Sequence[pd.DataFrame]]) -> List[pd.DataFrame]:
    if isinstance(beams, pd.DataFrame):
        return [beams]
    return list(beams)


def _mean_wavelength(df: pd.DataFrame) -> float:
    if "wavelength" not in df.columns:
        raise ValueError("get_wave_metrics: beams must contain 'wavelength'")

    if "intensity" not in df.columns:
        raise ValueError("get_wave_metrics: beams must contain 'intensity'")

    tmp = df.loc[df["lost_ray_flag"] == 0] if "lost_ray_flag" in df.columns else df
    values = tmp["wavelength"].to_numpy(dtype=float)
    weights = tmp["intensity"].to_numpy(dtype=float)
    finite = np.isfinite(values) & np.isfinite(weights)
    values = values[finite]
    weights = np.clip(weights[finite], 0.0, np.inf)

    if values.size == 0:
        return np.nan

    w_sum = weights.sum()
    if w_sum <= 0 or not np.isfinite(w_sum):
        return np.nan

    return float(np.sum(weights * values) / w_sum)


def _per_run_axis_metrics(
    df: pd.DataFrame,
    axis: str,
    *,
    focal_distance: float,
    wavelength: float,
    max_focal_distance: float,
) -> dict:
    if (
        not np.isfinite(focal_distance)
        or abs(focal_distance) > max_focal_distance
        or not np.isfinite(wavelength)
        or wavelength <= 0
    ):
        return _nan_axis_metrics(is_finite_focus=False)

    propagated = df.copy()
    propagated[axis] = propagated[axis] + focal_distance * propagated[f"d{axis}"]
    image_stats = stats.get_statistics(propagated)

    dcol = f"d{axis}"
    theta = image_stats[dcol]["std"][0] * RMS_TO_HALF_FWHM
    beam_size = image_stats[axis]["std"][0]

    if not np.isfinite(theta) or theta <= 0:
        return _nan_axis_metrics(is_finite_focus=False)

    na = np.sin(theta)
    if not np.isfinite(na) or na <= 0:
        return _nan_axis_metrics(is_finite_focus=False)

    gaussian_waist = wavelength / (np.pi * na)
    depth_of_focus = 2.0 * np.pi * gaussian_waist**2 / wavelength
    gaussian_waist_diameter = 2.0 * gaussian_waist
    convolved = np.sqrt(beam_size**2 + gaussian_waist_diameter**2)
    geometric_depth_of_focus = _geometric_depth_of_focus(
        beam_size,
        image_stats[dcol]["std"][0],
        convolved,
    )

    return {
        "is_finite_focus": True,
        "theta": float(theta),
        "na": float(na),
        "gaussian_waist": float(gaussian_waist),
        "depth_of_focus": float(depth_of_focus),
        "convolved_beam_size": float(convolved),
        "geometric_depth_of_focus": float(geometric_depth_of_focus),
    }


def _nan_axis_metrics(*, is_finite_focus: bool) -> dict:
    return {
        "is_finite_focus": is_finite_focus,
        "theta": np.nan,
        "na": np.nan,
        "gaussian_waist": np.nan,
        "depth_of_focus": np.nan,
        "convolved_beam_size": np.nan,
        "geometric_depth_of_focus": np.nan,
    }


def _geometric_depth_of_focus(
    beam_size: float,
    divergence_rms: float,
    convolved_beam_size: float,
) -> float:
    if (
        not np.isfinite(beam_size)
        or not np.isfinite(divergence_rms)
        or not np.isfinite(convolved_beam_size)
        or divergence_rms <= 0
    ):
        return np.nan

    target = np.sqrt(2.0) * convolved_beam_size
    delta = target**2 - beam_size**2
    if delta <= 0 or not np.isfinite(delta):
        return np.nan

    return float(2.0 * np.sqrt(delta) / divergence_rms)


def _aggregate_axis_metrics(dicts: Sequence[dict]) -> dict:
    keys = (
        "theta",
        "na",
        "gaussian_waist",
        "depth_of_focus",
        "convolved_beam_size",
        "geometric_depth_of_focus",
    )
    out = {key: _aggregate_values([d[key] for d in dicts]) for key in keys}
    out["is_finite_focus"] = all(bool(d["is_finite_focus"]) for d in dicts)
    return out


def _aggregate_values(values: Sequence[float]) -> list[float]:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]

    if finite.size == 0:
        return [np.nan, np.nan]

    return [float(np.mean(finite)), float(np.std(finite, ddof=0))]


def _print_wave_summary(result: dict) -> None:
    n_reps = result["meta"]["n_repetitions"]

    print("\nGaussian-equivalent wave metrics")

    for axis in ("X", "Y"):
        if axis not in result:
            continue

        direction = "horizontal" if axis == "X" else "vertical"
        data = result[axis]
        na = _format_wave_with_unc(*data["na"], kind="na", n_reps=n_reps)
        waist = _format_wave_with_unc(
            *data["gaussian_waist"],
            kind="beam_um",
            n_reps=n_reps,
            scale=2e6,
        )
        dof = _format_wave_with_unc(
            *data["depth_of_focus"],
            kind="beam_mm",
            n_reps=n_reps,
            scale=1e3,
        )
        geometric_dof = _format_wave_with_unc(
            *data["geometric_depth_of_focus"],
            kind="beam_mm",
            n_reps=n_reps,
            scale=1e3,
        )
        convolved = _format_wave_with_unc(
            *data["convolved_beam_size"],
            kind="beam_um",
            n_reps=n_reps,
            scale=1e6,
        )

        print(f"\n------------------ {direction}-plane:")
        print(f">> NA: {na}")
        print(f">> Gaussian waist diameter: {waist}")
        print(f">> Convolved beam size: {convolved}")
        print(f">> Depth of focus: {dof}")
        print(f">> Geometric depth of focus: {geometric_dof}")

def _format_wave_with_unc(
    val: float,
    unc: float,
    *,
    kind: str,
    n_reps: int,
    scale: float = 1.0,
) -> str:
    v = val * scale
    u = unc * scale
    unit = _wave_unit(kind)

    if not np.isfinite(v):
        return f"{v:.6g}{unit}"

    if n_reps <= 1 or not np.isfinite(u) or u == 0:
        return f"{_format_wave_scalar(v, kind)}{unit}"

    if kind == "na":
        return f"{v:.3e} +- {u:.3e}{unit}"

    decimals = min(max(_significant_unc_decimals(u), 0), 3)
    return f"{v:.{decimals}f} +- {u:.{decimals}f}{unit}"


def _format_wave_scalar(val: float, kind: str) -> str:
    if not np.isfinite(val):
        return f"{val:.6g}"

    if kind == "na":
        return f"{val:.3e}"

    if kind in {"beam_um", "beam_mm"}:
        return _format_beam_scale(val)

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


def _significant_unc_decimals(unc: float) -> int:
    if unc == 0 or not np.isfinite(unc):
        return 0

    exp = int(np.floor(np.log10(abs(unc))))
    return max(-exp, 0)


def _wave_unit(kind: str) -> str:
    if kind == "beam_um":
        return " µm"
    if kind == "beam_mm":
        return " mm"
    return ""
