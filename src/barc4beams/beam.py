# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2025 Synchrotron SOLEIL

"""
beam.py — unified interface class for standardized ray-traced beams.

The Beam class is a high-level façade that wraps:
- adapters.to_standard_beam (PyOptiX / SHADOW3 / SHADOW4 → standardized beam)
- schema.validate_beam (sanity checks)
- stats.get_statistics (cached statistics)
- viz plotting functions (beam, divergence, phase space, caustic)
- io.save_beam / io.read_beam (HDF5)
- io.save_json_stats / io.read_json_stats (JSON stats)

Notes
-----
- This class does *not* propagate or modify beams — simulation is left to
  PyOptiX/SHADOW. It only standardizes, stores, analyzes, and plots.
- May wrap a single beam or multiple runs. Plotting is disabled if multiple.
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Sequence, Tuple, Union

import pandas as pd

from . import adapters, caustics, io, schema, stats, viz


class Beam:
    """Unified interface for standardized beams."""

    def __init__(self, obj: Any, code: Optional[str] = None) -> None:
        if isinstance(obj, (list, tuple)):
            self._runs = [self._standardize(o, code) for o in obj]
        else:
            self._runs = [self._standardize(obj, code)]

        for df in self._runs:
            schema.validate_beam(df)

        self._stats_cache: Optional[Dict] = None

    # ------------------------------------------------------------------
    # construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_h5(cls, path: str) -> "Beam":
        df = io.read_beam(path)
        return cls(df)

    @classmethod
    def from_df(cls, df: pd.DataFrame) -> "Beam":
        schema.validate_beam(df)
        return cls(df)

    def _standardize(self, obj: Any, code: Optional[str]) -> pd.DataFrame:
        if isinstance(obj, pd.DataFrame):
            return obj.copy()
        return adapters.to_standard_beam(obj, code=code)

    # ------------------------------------------------------------------
    # properties
    # ------------------------------------------------------------------
    @property
    def n_runs(self) -> int:
        return len(self._runs)

    @property
    def df(self) -> pd.DataFrame:
        if self.n_runs != 1:
            raise ValueError("Beam contains multiple runs; access .runs instead.")
        return self._runs[0]

    @property
    def runs(self) -> Sequence[pd.DataFrame]:
        return self._runs

    @property
    def stats(self) -> Dict:
        if self._stats_cache is None:
            self._stats_cache = stats.get_statistics(self._runs)
        return self._stats_cache

    # ------------------------------------------------------------------
    # methods
    # ------------------------------------------------------------------
    def print_stats(self, *, verbose: bool = True) -> None:
        _ = stats.get_statistics(self._runs, verbose=verbose)

    # ------------------------------------------------------------------
    # caustics
    # ------------------------------------------------------------------
    def compute_caustic(
        self,
        *,
        n_points: int = 501,
        start: float = -0.5,
        finish: float = 0.5,
    ) -> Dict:
        if self.n_runs != 1:
            raise ValueError("Caustic computation requires a single run (got multiple).")

        res = caustics.compute_caustic(
            beam=self.df,
            n_points=n_points,
            start=start,
            finish=finish,
            return_points=True,
        )

        return res

    # ------------------------------------------------------------------
    # plotting (only allowed for single run))
    # ------------------------------------------------------------------

    def plot_beam(
        self,
        *,
        mode: str = "scatter",
        aspect_ratio: bool = True,
        color: int = 1,
        x_range: Optional[Tuple[Optional(float), Optional(float)]] = None,
        y_range: Optional[Tuple[Optional(float), Optional(float)]] = None,
        bins: Optional[Union[int, Tuple[int, int]]] = None,
        bin_width: Optional[float] = None,
        bin_method: int = 0,
        dpi: int = 100,
        path: Optional[str] = None,
        showXhist: bool = True,
        showYhist: bool = True,
        envelope: bool = True,
        envelope_method: str = "edgeworth",
        apply_style: bool = True,
        k: float = 1.0,
        plot: bool = True,
    ):
        if self.n_runs != 1:
            raise ValueError("Plotting not supported for multiple runs.")
        return viz.plot_beam(
            df=self.df,
            mode=mode,
            aspect_ratio=aspect_ratio,
            color=color,
            x_range=x_range,
            y_range=y_range,
            bins=bins,
            bin_width=bin_width,
            bin_method=bin_method,
            dpi=dpi,
            path=path,
            showXhist=showXhist,
            showYhist=showYhist,
            envelope=envelope,
            envelope_method=envelope_method,
            apply_style=apply_style,
            k=k,
            plot=plot,
        )

    def plot_divergence(
        self,
        *,
        mode: str = "scatter",
        aspect_ratio: bool = False,
        color: int = 2,
        x_range: Optional[Tuple[Optional(float), Optional(float)]] = None,
        y_range: Optional[Tuple[Optional(float), Optional(float)]] = None,
        bins: Optional[Union[int, Tuple[int, int]]] = None,
        bin_width: Optional[float] = None,
        bin_method: int = 0,
        dpi: int = 100,
        path: Optional[str] = None,
        showXhist: bool = True,
        showYhist: bool = True,
        envelope: bool = True,
        envelope_method: str = "edgeworth",
        apply_style: bool = True,
        k: float = 1.0,
        plot: bool = True,
    ):
        if self.n_runs != 1:
            raise ValueError("Plotting not supported for multiple runs.")
        return viz.plot_divergence(
            df=self.df,
            mode=mode,
            aspect_ratio=aspect_ratio,
            color=color,
            x_range=x_range,
            y_range=y_range,
            bins=bins,
            bin_width=bin_width,
            bin_method=bin_method,
            dpi=dpi,
            path=path,
            showXhist=showXhist,
            showYhist=showYhist,
            envelope=envelope,
            envelope_method=envelope_method,
            apply_style=apply_style,
            k=k,
            plot=plot,
        )

    def plot_phase_space(
        self,
        *,
        direction: str = "both",
        mode: str = "scatter",
        aspect_ratio: bool = False,
        color: int = 3,
        x_range: Optional[Tuple[Optional(float), Optional(float)]] = None,
        y_range: Optional[Tuple[Optional(float), Optional(float)]] = None,
        bins: Optional[Union[int, Tuple[int, int]]] = None,
        bin_width: Optional[float] = None,
        bin_method: int = 0,
        dpi: int = 100,
        path: Optional[str] = None,
        showXhist: bool = True,
        showYhist: bool = True,
        envelope: bool = True,
        envelope_method: str = "edgeworth",
        apply_style: bool = True,
        k: float = 1.0,
        plot: bool = True,
    ):
        if self.n_runs != 1:
            raise ValueError("Plotting not supported for multiple runs.")
        return viz.plot_phase_space(
            df=self.df,
            direction=direction,
            mode=mode,
            aspect_ratio=aspect_ratio,
            color=color,
            x_range=x_range,
            y_range=y_range,
            bins=bins,
            bin_width=bin_width,
            bin_method=bin_method,
            dpi=dpi,
            path=path,
            showXhist=showXhist,
            showYhist=showYhist,
            envelope=envelope,
            envelope_method=envelope_method,
            apply_style=apply_style,
            k=k,
            plot=plot,
        )

    def plot_energy(
        self,
        *,
        bins: Optional[Union[int, Tuple[int, int]]] = None,
        bin_width: Optional[float] = None,
        bin_method: int = 0,
        dpi: int = 100,
        path: Optional[str] = None,
        apply_style: bool = True,
        k: float = 1.0,
        plot: bool = True,
    ):
        if self.n_runs != 1:
            raise ValueError("Plotting not supported for multiple runs.")
        return viz.plot_energy(
            df=self.df,
            bins=bins,
            bin_width=bin_width,
            bin_method=bin_method,
            dpi=dpi,
            path=path,
            apply_style=apply_style,
            k=k,
            plot=plot,
        )

    def plot_energy_vs_intensity(
        self,
        *,
        mode: str = "scatter",
        aspect_ratio: bool = False,
        color: Optional[int] = 3,
        x_range: Optional[Tuple[Optional(float), Optional(float)]] = None,
        y_range: Optional[Tuple[Optional(float), Optional(float)]] = None,
        bins: Optional[Union[int, Tuple[int, int]]] = None,
        bin_width: Optional[float] = None,
        bin_method: int = 0,
        dpi: int = 100,
        path: Optional[str] = None,
        showXhist: bool = True,
        showYhist: bool = True,
        envelope: bool = False,
        envelope_method: str = "edgeworth",
        apply_style: bool = True,
        k: float = 1.0,
        plot: bool = True,
    ):
        if self.n_runs != 1:
            raise ValueError("Plotting not supported for multiple runs.")
        return viz.plot_energy_vs_intensity(
            df=self.df,
            mode=mode,
            aspect_ratio=aspect_ratio,
            color=color,
            x_range=x_range,
            y_range=y_range,
            bins=bins,
            bin_width=bin_width,
            bin_method=bin_method,
            dpi=dpi,
            path=path,
            showXhist=showXhist,
            showYhist=showYhist,
            envelope=envelope,
            envelope_method=envelope_method,
            apply_style=apply_style,
            k=k,
            plot=plot,
        )

    def plot_caustic(
        self,
        *,
        which: Literal["x", "y", "both"] = "both",
        aspect_ratio: bool = False,
        color: Optional[int] = 5,
        z_range: Optional[Tuple[Optional[float], Optional[float]]] = None,
        xy_range: Optional[Tuple[Optional[float], Optional[float]]] = None,
        bins: Optional[int | Tuple[Optional[int], int]] = None,
        bin_width: Optional[float] = None,
        dpi: int = 100,
        path: Optional[str] = None,
        apply_style: bool = True,
        k: float = 1.0,
        plot: bool = True,
        top_stat: Optional[str] = None,

        n_points: int = 501,
        start: float = -0.5,
        finish: float = 0.5,
        return_points: bool = True,
    ):
        """Calculate, then plot the beam caustic"""
        ca = self.compute_caustic(
            n_points=n_points,
            start=start,
            finish=finish,
            return_points=return_points,
        )
        return viz.plot_caustic(
            caustic=ca,
            which=which,
            aspect_ratio=aspect_ratio,
            color=color,
            z_range=z_range,
            xy_range=xy_range,
            bins=bins,
            bin_width=bin_width,
            dpi=dpi,
            path=path,
            apply_style=apply_style,
            k=k,
            plot=plot,
            top_stat=top_stat,
        )

    # ------------------------------------------------------------------
    # saving
    # ------------------------------------------------------------------
    def save(self, path: str, *, meta: Optional[Dict[str, Any]] = None) -> None:
        """
        Save beam to HDF5 and stats to JSON.

        The HDF5 file will contain the beam(s).
        A sibling JSON file (same base name) will contain the statistics.
        """
        io.save_beam(self._runs[0] if self.n_runs == 1 else self._runs, path)
        base = path.rsplit(".", 1)[0]
        json_path = f"{base}.json"
        io.save_json_stats(self.stats, json_path, meta=meta)

    def save_beam(self, path: str) -> None:
        """Save the beam(s) only (HDF5)."""
        io.save_beam(self._runs[0] if self.n_runs == 1 else self._runs, path)

    def save_stats(self, path: str, *, meta: Optional[Dict[str, Any]] = None) -> None:
        """Save statistics only (JSON)."""
        io.save_json_stats(self.stats, path, meta=meta)
