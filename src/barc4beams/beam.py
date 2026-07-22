# SPDX-License-Identifier: CECILL-2.1
# Copyright (c) 2026 ESRF - the European Synchrotron

"""
beam.py — unified interface classes for standardized beams.

Beam
    Single standardized photon beam.

BeamEnsemble
    Collection of comparable beam realizations, typically repeated seeds/runs
    of the same simulation setup, used for ensemble statistics.
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Sequence, Tuple, Union

import pandas as pd

from . import adapters, io, propagation, sampling, schema, stats, viz, wave


class Beam:
    """
    High-level container for one standardized photon beam.

    Wraps a validated beam DataFrame and exposes core analysis, propagation,
    sampling, saving, and visualization methods.
    """

    def __init__(self, obj: Any, code: str | None = None) -> None:
        """
        Initialize a Beam instance from one beam-like object.

        Parameters
        ----------
        obj : Any
            Beam-like object accepted by `adapters.to_standard_beam`, or an
            already standardized pandas DataFrame.
        code : str, optional
            Source-code identifier passed to `adapters.to_standard_beam`.

        Raises
        ------
        TypeError
            If a list or tuple is passed. Use BeamEnsemble for multiple beams.
        ValueError
            If the standardized beam does not satisfy the schema.
        """
        if isinstance(obj, (list, tuple)):
            raise TypeError("Beam accepts a single beam only. Use BeamEnsemble for multiple runs.")

        self._df = self._standardize(obj, code)
        schema.validate_beam(self._df)

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
        obj = io.read_beam(path)

        if isinstance(obj, (list, tuple)):
            raise ValueError(
                "HDF5 file contains multiple beams. Use BeamEnsemble.from_h5 instead."
            )

        return cls(obj)

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
        Build a Beam by sampling 2D intensity maps.

        See `sampling.beam_from_intensity`.
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

    @classmethod
    def from_wavefront(
        cls,
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
    ) -> "Beam":
        """
        Build a Beam by sampling a spatial wavefront map.

        See `sampling.beam_from_wavefront`.
        """
        df = sampling.beam_from_wavefront(
            wavefront=wavefront,
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

    def _standardize(self, obj: Any, code: str | None) -> pd.DataFrame:
        if isinstance(obj, pd.DataFrame):
            return obj.copy()

        return adapters.to_standard_beam(obj, code=code)

    # ------------------------------------------------------------------
    # properties
    # ------------------------------------------------------------------

    @property
    def df(self) -> pd.DataFrame:
        """
        Standardized beam DataFrame.
        """
        return self._df

    # ------------------------------------------------------------------
    # stats
    # ------------------------------------------------------------------

    def stats(self, *, verbose: bool = False) -> dict:
        """
        Compute descriptive beam statistics.

        See `stats.get_statistics`.
        """
        return stats.get_statistics(self.df, verbose=verbose)

    def wave_metrics(
        self,
        *,
        max_focal_distance: float = 1000.0,
        verbose: bool = False,
    ) -> dict:
        """
        Estimate Gaussian-equivalent wave-optics metrics.

        See `wave.get_wave_metrics`.
        """
        return wave.get_wave_metrics(
            self.df,
            max_focal_distance=max_focal_distance,
            verbose=verbose,
        )

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
    ) -> dict:
        """
        Compute the free-space caustic of this Beam.

        See `propagation.caustic`.
        """
        return propagation.caustic(
            beam=self.df,
            n_points=n_points,
            start=start,
            finish=finish,
        )

    def apply_wavefront(
        self,
        *,
        wavefront: dict,
        energy: float | None = None,
        wavelength: float | None = None,
        threshold: float | None = None,
        verbose: bool = False,
    ) -> "Beam":
        """
        Returns a new Beam instance with updated intensity, slopes, and lost-ray
        flags.
        """
        df2 = sampling.apply_wavefront(
            standard_beam=self.df,
            wavefront=wavefront,
            energy=energy,
            wavelength=wavelength,
            threshold=threshold,
        )
        out = Beam.from_df(df2)

        if verbose:
            stats.get_statistics(df2, verbose=True)

        return out

    def apply_transmission_element(
        self,
        *,
        thickness: dict,
        energy: float | Sequence[float],
        n: float | complex | Sequence[float | complex] | None = None,
        delta: float | Sequence[float] | None = None,
        beta: float | Sequence[float] | None = None,
        attenuation_length: float | Sequence[float] | None = None,
        verbose: bool = False,
    ) -> "Beam":
        """
        Return a new Beam after applying a thin transmission element.

        See `sampling.apply_transmission_element`.
        """
        df2 = sampling.apply_transmission_element(
            standard_beam=self.df,
            thickness=thickness,
            energy=energy,
            n=n,
            delta=delta,
            beta=beta,
            attenuation_length=attenuation_length,
        )
        out = Beam.from_df(df2)

        if verbose:
            stats.get_statistics(df2, verbose=True)

        return out

    # ------------------------------------------------------------------
    # plotting
    # ------------------------------------------------------------------

    def plot_rays(
        self,
        *,
        color="black",
        marker=".",
        intensity_threshold: float | None = None,
        marker_size: float = 2.5,
        aspect_ratio: bool = True,
        x_range: tuple[float | None, float | None] | None = None,
        y_range: tuple[float | None, float | None] | None = None,
        z_offset: float = 0.0,
        dpi: int = 100,
        path: str | None = None,
        apply_style: bool = True,
        k: float = 1.0,
        plot: bool = True,
    ):
        """Plot alive rays above an absolute intensity threshold."""
        return viz.plot_rays(
            df=self.df,
            color=color,
            marker=marker,
            intensity_threshold=intensity_threshold,
            marker_size=marker_size,
            aspect_ratio=aspect_ratio,
            x_range=x_range,
            y_range=y_range,
            z_offset=z_offset,
            dpi=dpi,
            path=path,
            apply_style=apply_style,
            k=k,
            plot=plot,
        )

    def plot_beam(
        self,
        *,
        mode: str = "hist",
        aspect_ratio: bool = True,
        cmap="viridis",
        x_range: tuple[float | None, float | None] | None = None,
        y_range: tuple[float | None, float | None] | None = None,
        bins: Optional[Union[int, Tuple[int, int]]] = None,
        bin_width: Optional[float] = None,
        bin_method: int = 0,
        dpi: int = 100,
        path: str | None = None,
        showXhist: bool = True,
        showYhist: bool = True,
        apply_style: bool = True,
        k: float = 1.0,
        z_offset: float = 0.0,
        plot: bool = True,
    ):
        """
        Plot the spatial footprint X vs Y with optional propagation offset.
        """
        return viz.plot_beam(
            df=self.df,
            mode=mode,
            aspect_ratio=aspect_ratio,
            cmap=cmap,
            x_range=x_range,
            y_range=y_range,
            bins=bins,
            bin_width=bin_width,
            bin_method=bin_method,
            dpi=dpi,
            path=path,
            showXhist=showXhist,
            showYhist=showYhist,
            apply_style=apply_style,
            k=k,
            plot=plot,
            z_offset=z_offset,
        )

    def plot_divergence(
        self,
        *,
        mode: str = "hist",
        aspect_ratio: bool = False,
        cmap="plasma",
        x_range: tuple[float | None, float | None] | None = None,
        y_range: tuple[float | None, float | None] | None = None,
        bins: int | tuple[int, int] | None = None,
        bin_width: float | None = None,
        bin_method: int = 0,
        dpi: int = 100,
        path: str | None = None,
        showXhist: bool = True,
        showYhist: bool = True,
        apply_style: bool = True,
        k: float = 1.0,
        plot: bool = True,
    ):
        """
        Plot the angular distribution dX vs dY.
        """
        return viz.plot_divergence(
            df=self.df,
            mode=mode,
            aspect_ratio=aspect_ratio,
            cmap=cmap,
            x_range=x_range,
            y_range=y_range,
            bins=bins,
            bin_width=bin_width,
            bin_method=bin_method,
            dpi=dpi,
            path=path,
            showXhist=showXhist,
            showYhist=showYhist,
            apply_style=apply_style,
            k=k,
            plot=plot,
        )

    def plot_phase_space(
        self,
        *,
        direction: str = "both",
        mode: str = "hist",
        aspect_ratio: bool = False,
        cmap="turbo",
        x_range: tuple[float | None, float | None] | None = None,
        y_range: tuple[float | None, float | None] | None = None,
        bins: int | tuple[int, int] | None = None,
        bin_width: float | None = None,
        bin_method: int = 0,
        dpi: int = 100,
        path: str | None = None,
        showXhist: bool = True,
        showYhist: bool = True,
        apply_style: bool = True,
        k: float = 1.0,
        z_offset: float = 0.0,
        plot: bool = True,
    ):
        """
        Plot phase-space diagrams X vs dX and/or Y vs dY.
        """
        return viz.plot_phase_space(
            df=self.df,
            direction=direction,
            mode=mode,
            aspect_ratio=aspect_ratio,
            cmap=cmap,
            x_range=x_range,
            y_range=y_range,
            bins=bins,
            bin_width=bin_width,
            bin_method=bin_method,
            dpi=dpi,
            path=path,
            showXhist=showXhist,
            showYhist=showYhist,
            apply_style=apply_style,
            k=k,
            plot=plot,
            z_offset=z_offset,
        )

    def plot_energy(
        self,
        *,
        bins: int | tuple[int, int] | None = None,
        bin_width: float | None = None,
        bin_method: int = 0,
        dpi: int = 100,
        path: str | None = None,
        apply_style: bool = True,
        k: float = 1.0,
        plot: bool = True,
    ):
        """
        Plot the energy distribution of the beam.
        """
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

    def plot_intensity(
        self,
        *,
        bins: int | tuple[int, int] | None = None,
        bin_width: float | None = None,
        bin_method: int = 0,
        dpi: int = 100,
        path: str | None = None,
        apply_style: bool = True,
        k: float = 1.0,
        plot: bool = True,
    ):
        """
        Plot the intensity distribution of the beam.
        """
        return viz.plot_intensity(
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
        mode: str = "hist",
        aspect_ratio: bool = False,
        cmap="turbo",
        x_range: tuple[float | None, float | None] | None = None,
        y_range: tuple[float | None, float | None] | None = None,
        bins: int | tuple[int, int] | None = None,
        bin_width: float | None = None,
        bin_method: int = 0,
        dpi: int = 100,
        path: str | None = None,
        showXhist: bool = True,
        showYhist: bool = True,
        apply_style: bool = True,
        k: float = 1.0,
        plot: bool = True,
    ):
        """
        Plot beam intensity as a function of photon energy.
        """
        return viz.plot_energy_vs_intensity(
            df=self.df,
            mode=mode,
            aspect_ratio=aspect_ratio,
            cmap=cmap,
            x_range=x_range,
            y_range=y_range,
            bins=bins,
            bin_width=bin_width,
            bin_method=bin_method,
            dpi=dpi,
            path=path,
            showXhist=showXhist,
            showYhist=showYhist,
            apply_style=apply_style,
            k=k,
            plot=plot,
        )

    def plot_caustic(
        self,
        *,
        which: Literal["x", "y", "both"] = "both",
        aspect_ratio: bool = False,
        color: int | None = 5,
        z_range: tuple[float, float] = (-0.5, 0.5),
        xy_range: tuple[float | None, float | None] | None = None,
        bins: int | tuple[int | None, int] | None = None,
        bin_width: float | None = None,
        dpi: int = 100,
        path: str | None = None,
        apply_style: bool = True,
        k: float = 1.0,
        top_stat: str | None = None,
        n_points: int = 501,
        plot: bool = True,
    ):
        """
        Plot the caustic map computed from `self.caustic()`.
        """
        ca = self.caustic(
            n_points=n_points,
            start=z_range[0],
            finish=z_range[1],
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

    def save(self, path: str, *, meta: dict[str, Any] | None = None) -> None:
        """
        Save beam to HDF5 and stats to JSON.

        The HDF5 file contains the beam.
        A sibling JSON file with the same base name contains the statistics.
        """
        io.save_beam(self.df, path)

        base = path.rsplit(".", 1)[0]
        json_path = f"{base}.json"
        io.save_json_stats(self.stats(), json_path, meta=meta)

    def save_beam(self, path: str) -> None:
        """
        Save the beam only to HDF5.
        """
        io.save_beam(self.df, path)

    def save_stats(self, path: str, *, meta: dict[str, Any] | None = None) -> None:
        """
        Save statistics only to JSON.
        """
        io.save_json_stats(self.stats(), path, meta=meta)


class BeamEnsemble:
    """
    Container for comparable standardized beam realizations.

    This is intended for repeated runs/seeds of the same beamline setup. It owns
    ensemble-level statistics, saving, and merging. Single-beam operations such
    as plotting, propagation, and caustic calculation remain on Beam.
    """

    def __init__(self, beams: Sequence[Beam | pd.DataFrame | Any], code: str | None = None) -> None:
        """
        Initialize a BeamEnsemble from multiple beams.

        Parameters
        ----------
        beams : sequence of Beam, pandas.DataFrame, or beam-like objects
            Beam realizations to include in the ensemble.
        code : str, optional
            Source-code identifier passed to `adapters.to_standard_beam` for
            non-DataFrame, non-Beam inputs.

        Raises
        ------
        ValueError
            If no beams are provided or one beam fails schema validation.
        TypeError
            If `beams` is not a sequence of beam-like objects.
        """
        if isinstance(beams, (pd.DataFrame, Beam)):
            raise TypeError("BeamEnsemble expects a sequence of beams, not a single beam.")

        if not beams:
            raise ValueError("BeamEnsemble: no beams provided.")

        self._runs = [self._standardize(b, code) for b in beams]

        for df in self._runs:
            schema.validate_beam(df)

    # ------------------------------------------------------------------
    # construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_dfs(cls, beams: Sequence[pd.DataFrame]) -> "BeamEnsemble":
        """
        Create a BeamEnsemble directly from standard DataFrames.
        """
        return cls(beams)

    @classmethod
    def from_h5(cls, path: str) -> "BeamEnsemble":
        """
        Load a BeamEnsemble from an HDF5 file written by io.save_beam_ensemble.
        """
        beams = io.read_beam_ensemble(path)
        return cls.from_dfs(beams)

    def _standardize(self, obj: Beam | pd.DataFrame | Any, code: str | None) -> pd.DataFrame:
        if isinstance(obj, Beam):
            return obj.df.copy()

        if isinstance(obj, pd.DataFrame):
            return obj.copy()

        return adapters.to_standard_beam(obj, code=code)

    # ------------------------------------------------------------------
    # properties
    # ------------------------------------------------------------------

    @property
    def n_runs(self) -> int:
        """
        Number of beam realizations in the ensemble.
        """
        return len(self._runs)

    @property
    def runs(self) -> Sequence[pd.DataFrame]:
        """
        Standardized beam DataFrames.
        """
        return self._runs

    # ------------------------------------------------------------------
    # ensemble operations
    # ------------------------------------------------------------------

    def stats(self, *, verbose: bool = False) -> dict:
        """
        Compute ensemble statistics over all runs.

        See `stats.get_statistics`.
        """
        return stats.get_statistics(list(self._runs), verbose=verbose)

    def wave_metrics(
        self,
        *,
        max_focal_distance: float = 1000.0,
        verbose: bool = False,
    ) -> dict:
        """
        Estimate ensemble Gaussian-equivalent wave-optics metrics.

        See `wave.get_wave_metrics`.
        """
        return wave.get_wave_metrics(
            list(self._runs),
            max_focal_distance=max_focal_distance,
            verbose=verbose,
        )

    def merge(self) -> Beam:
        """
        Merge all ensemble runs into a single Beam.

        The returned object is a Beam instance.
        """
        merged = adapters.merge_standard_beams(list(self._runs))
        return Beam.from_df(merged)

    # ------------------------------------------------------------------
    # saving
    # ------------------------------------------------------------------

    def save(self, path: str, *, meta: dict[str, Any] | None = None) -> None:
        """
        Save ensemble beams to HDF5 and ensemble stats to JSON.

        The HDF5 file contains all runs.
        A sibling JSON file with the same base name contains ensemble statistics.
        """
        io.save_beam_ensemble(list(self._runs), path)

        base = path.rsplit(".", 1)[0]
        json_path = f"{base}.json"
        io.save_json_stats(self.stats(), json_path, meta=meta)

    def save_beams(self, path: str) -> None:
        """
        Save ensemble beams only to HDF5.
        """
        io.save_beam_ensemble(list(self._runs), path)

    def save_stats(self, path: str, *, meta: dict[str, Any] | None = None) -> None:
        """
        Save ensemble statistics only to JSON.
        """
        io.save_json_stats(self.stats(), path, meta=meta)
