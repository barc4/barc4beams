# SPDX-License-Identifier: CECILL-2.1
# Copyright (c) 2026 ESRF - the European Synchrotron

"""
compare.py - compare two Beam instances numerically, ignoring row order
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .beam import Beam
from .schema import validate_beam


CompareMode = Literal["all", "alive"]


@dataclass(frozen=True)
class ColumnComparison:
    name: str
    same: bool
    max_abs_error: float
    max_rel_error: float
    mean_abs_error: float
    n_exceeding: int
    worst_index: int | None
    value_a: float | int | None
    value_b: float | int | None


@dataclass(frozen=True)
class BeamComparison:
    same: bool
    mode: CompareMode
    n_rays_a: int
    n_rays_b: int
    n_compared: int
    columns_compared: tuple[str, ...]
    column_results: dict[str, ColumnComparison]
    summary: str

    def __str__(self) -> str:
        failing = [c for c in self.column_results.values() if not c.same]

        lines = [
            f"BeamComparison: {'same' if self.same else 'different'}",
            "",
            f"Mode: {self.mode}",
            f"Rays compared: {self.n_compared}",
        ]

        if self.n_rays_a != self.n_rays_b:
            lines += [
                "",
                f"Different number of rays: A={self.n_rays_a}, B={self.n_rays_b}",
            ]

        if failing:
            lines += ["", "Failing columns:"]
            for col in failing:
                lines.append(
                    f"  {col.name:<16} "
                    f"max_abs={col.max_abs_error:.6g}  "
                    f"max_rel={col.max_rel_error:.6g}  "
                    f"n_exceeding={col.n_exceeding}"
                )

            worst = max(failing, key=lambda c: c.max_abs_error)
            lines += [
                "",
                "Worst mismatch:",
                f"  column: {worst.name}",
                f"  ray: {worst.worst_index}",
                f"  A: {worst.value_a}",
                f"  B: {worst.value_b}",
                f"  abs_error: {worst.max_abs_error:.6g}",
                f"  rel_error: {worst.max_rel_error:.6g}",
            ]

        return "\n".join(lines)


def compare_beams(
    beam_a: Beam,
    beam_b: Beam,
    *,
    mode: CompareMode = "all",
    atol: float = 1e-12,
    rtol: float = 1e-9,
    verbose: bool = False,
) -> BeamComparison:
    """
    Compare two Beam instances numerically, ignoring row order.

    Parameters
    ----------
    beam_a, beam_b : Beam
        Beam instances to compare.
    mode : {"all", "alive"}, optional
        Comparison mode. If "alive", only rays with lost_ray_flag == 0 are
        compared.
    atol : float, optional
        Absolute tolerance passed to numpy.isclose.
    rtol : float, optional
        Relative tolerance passed to numpy.isclose.
    verbose : bool, optional
        If True, print the comparison report.

    Returns
    -------
    BeamComparison
        Diagnostic comparison report.

    Raises
    ------
    TypeError
        If inputs are not Beam instances.
    ValueError
        If mode is invalid.
    """
    if not isinstance(beam_a, Beam) or not isinstance(beam_b, Beam):
        raise TypeError("compare_beams expects two Beam instances.")

    if mode not in {"all", "alive"}:
        raise ValueError("mode must be 'all' or 'alive'.")

    df_a = beam_a.df.copy()
    df_b = beam_b.df.copy()

    validate_beam(df_a)
    validate_beam(df_b)

    if mode == "alive":
        df_a = df_a[df_a["lost_ray_flag"] == 0].copy()
        df_b = df_b[df_b["lost_ray_flag"] == 0].copy()

    n_rays_a = len(df_a)
    n_rays_b = len(df_b)
    n_compared = min(n_rays_a, n_rays_b)

    columns = _comparison_columns(df_a, df_b)

    df_a = _canonical_sort(df_a, columns).reset_index(drop=True)
    df_b = _canonical_sort(df_b, columns).reset_index(drop=True)

    df_a = df_a.iloc[:n_compared]
    df_b = df_b.iloc[:n_compared]

    column_results = {
        col: _compare_column(
            col,
            df_a[col].to_numpy(),
            df_b[col].to_numpy(),
            atol=atol,
            rtol=rtol,
        )
        for col in columns
    }

    same_shape = n_rays_a == n_rays_b
    same_columns = all(res.same for res in column_results.values())
    same = same_shape and same_columns

    summary = "Beams are numerically equivalent." if same else "Beams differ."

    result = BeamComparison(
        same=same,
        mode=mode,
        n_rays_a=n_rays_a,
        n_rays_b=n_rays_b,
        n_compared=n_compared,
        columns_compared=tuple(columns),
        column_results=column_results,
        summary=summary,
    )

    if verbose:
        print(result)

    return result


def _comparison_columns(df_a: pd.DataFrame, df_b: pd.DataFrame) -> list[str]:
    preferred = [
        "energy",
        "X", "Y", "Z",
        "dX", "dY", "dZ",
        "wavelength",
        "intensity",
        "intensity_s-pol",
        "intensity_p-pol",
        "lost_ray_flag",
    ]

    return [col for col in preferred if col in df_a.columns and col in df_b.columns]


def _canonical_sort(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if not columns:
        return df.copy()

    return df.sort_values(
        by=columns,
        kind="mergesort",
        na_position="last",
    )


def _compare_column(
    name: str,
    a: np.ndarray,
    b: np.ndarray,
    *,
    atol: float,
    rtol: float,
) -> ColumnComparison:
    if name == "lost_ray_flag":
        close = a == b
        abs_err = np.abs(a.astype(float) - b.astype(float))
        rel_err = abs_err
    else:
        a = a.astype(float)
        b = b.astype(float)

        close = np.isclose(a, b, atol=atol, rtol=rtol, equal_nan=True)
        abs_err = np.abs(a - b)

        denom = np.maximum(np.abs(b), np.finfo(float).tiny)
        rel_err = abs_err / denom

    exceeding = ~close
    n_exceeding = int(np.count_nonzero(exceeding))

    if len(abs_err) == 0:
        worst_index = None
        value_a = None
        value_b = None
        max_abs = 0.0
        max_rel = 0.0
        mean_abs = 0.0
    else:
        worst_index = int(np.nanargmax(abs_err))
        value_a = a[worst_index].item()
        value_b = b[worst_index].item()
        max_abs = float(np.nanmax(abs_err))
        max_rel = float(np.nanmax(rel_err))
        mean_abs = float(np.nanmean(abs_err))

    return ColumnComparison(
        name=name,
        same=n_exceeding == 0,
        max_abs_error=max_abs,
        max_rel_error=max_rel,
        mean_abs_error=mean_abs,
        n_exceeding=n_exceeding,
        worst_index=worst_index,
        value_a=value_a,
        value_b=value_b,
    )