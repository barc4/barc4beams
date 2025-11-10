# SPDX-License-Identifier: CECILL-2.1
# Copyright (c) 2025 Synchrotron SOLEIL

"""
beam.py — unified interface class for stadandard beams.

The Beam class is a high-level wrapper for methods for propagation,
statistics, sampling from intensity maps, and plotting.
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Sequence, Tuple, Union

import pandas as pd

from . import adapters, io, propagation, sampling, schema, stats, viz


class Beam:
    """
    High-level container for standardized photon beams.

    Wraps a validated beam DataFrame and exposes core analysis, propagation,
    and visualization methods consistent with barc4beams modules.
    """
    def __init__(self, obj: Any, code: Optional[str] = None) -> None:
        """
        Initialize a Beam instance from a validated DataFrame.
        """
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
    def from_df(cls, df: pd.DataFrame) -> "Beam":
        """
        Create a Beam directly from a standard DataFrame.
        """
        schema.validate_beam(df)
        return cls(df)
    
    @classmethod
    def from_h5(cls, path: str) -> "Beam":
        """
        Load a Beam from an HDF5 file written by io.save_beam.
        """
        df = io.read_beam(path)
        return cls(df)
    
    @classmethod
    def from_intensity(
        cls,
        *,
        far_field: dict,
        near_field: dict | None = None,
        n_rays: int,
        energy: float | None = None,
        wavelength: float | None = None,
        jitter: bool = True,
        threshold: float | None = None,
        seed: int | None = 42,
        z0: float = 0.0,
        polarization_degree: float = 1.0,
    ) -> "Beam":
        """
        Build a Beam by sampling 2D intensity maps (see sampling.beam_from_intensity).
        """
        df = sampling.beam_from_intensity(
            far_field=far_field,
            near_field=near_field,
            n_rays=n_rays,
            energy=energy,
            wavelength=wavelength,
            jitter=jitter,
            threshold=threshold,
            seed=seed,
            z0=z0,
            polarization_degree=polarization_degree,
        )
        return cls.from_df(df)

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

    # ------------------------------------------------------------------
    # stats
    # ------------------------------------------------------------------

    def stats(self, *, verbose: bool = False) -> dict:
        """
        Compute descriptive beam statistics (see stats.get_statistics).
        """
        return stats.get_statistics(self.df, verbose=verbose)

    # ------------------------------------------------------------------
    # free-space propagation
    # ------------------------------------------------------------------

    def propagate(
        self,
        z_offset: float,
        *,
        verbose: bool = False,
    ) -> "Beam":
        """
        Return a new Beam propagated through free space by `z_offset` [m].
        """
        if self.n_runs != 1:
            raise ValueError("Free-space propagation requires a single run (got multiple).")

        df2 = propagation.propagate(self.df, z_offset)
        out = Beam.from_df(df2)
        if verbose:
            stats.get_statistics(df2, verbose=True)
        return out
    
    def caustic(
        self,
        *,
        n_points: int = 501,
        start: float = -0.5,
        finish: float = 0.5,
    ) -> Dict:
        """
        Compute the free-space caustic of this Beam (see propagation.caustic).
        """
        if self.n_runs != 1:
            raise ValueError("Caustic computation requires a single run (got multiple).")

        res = propagation.caustic(
            beam=self.df,
            n_points=n_points,
            start=start,
            finish=finish,
        )
        return res

    # ------------------------------------------------------------------
    # plotting (only allowed for single run)
    # ------------------------------------------------------------------

    def plot_beam(
        self,
        *,
        mode: str = "scatter",
        aspect_ratio: bool = True,
        color: int = 1,
        x_range: Optional[Tuple[Optional[float], Optional[float]]] = None,
        y_range: Optional[Tuple[Optional[float], Optional[float]]] = None,
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
        z_offset: float = 0.0,
    ):
        """
        Plot the spatial footprint (X vs Y) with optional propagation offset.
        """
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
            z_offset=z_offset,
        )

    def plot_divergence(
        self,
        *,
        mode: str = "scatter",
        aspect_ratio: bool = False,
        color: int = 2,
        x_range: Optional[Tuple[Optional[float], Optional[float]]] = None,  
        y_range: Optional[Tuple[Optional[float], Optional[float]]] = None,  
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
        z_offset: float = 0.0,
    ):
        """
        Plot the angular distribution (dX vs dY).
        """
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
            z_offset=z_offset,
        )

    def plot_phase_space(
        self,
        *,
        direction: str = "both",
        mode: str = "scatter",
        aspect_ratio: bool = False,
        color: int = 3,
        x_range: Optional[Tuple[Optional[float], Optional[float]]] = None, 
        y_range: Optional[Tuple[Optional[float], Optional[float]]] = None, 
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
        z_offset: float = 0.0, 
    ):
        """
        Plot the phase-space diagram (X vs dX or Y vs dY).
        """
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
            z_offset=z_offset,
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
        """
        Plot the energy distribution of the beam.
        """
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
        x_range: Optional[Tuple[Optional[float], Optional[float]]] = None,
        y_range: Optional[Tuple[Optional[float], Optional[float]]] = None,
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
        """
        Plot beam intensity as a function of photon energy.
        """
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
    ):
        """
        Plot the caustic map computed from `self.caustic()`.
        """
        ca = self.caustic(
            n_points=n_points,
            start=start,
            finish=finish,
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
