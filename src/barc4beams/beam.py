# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2025 Synchrotron SOLEIL

"""
beam.py — unified interface class for standardized ray-traced beams.

The Beam class is a high-level façade that wraps:
- adapters.to_standard_beam (PyOptiX / SHADOW3 / SHADOW4 → standardized beam)
- schema.validate_beam (sanity checks)
- stats.get_statistics (cached statistics)
- viz plotting functions (beam, divergence, phase space)
- io.save_beam / io.read_beam (HDF5)
- io.save_json_stats / io.read_json_stats (JSON stats)

Notes
-----
- This class does *not* propagate or modify beams — simulation is left to
  PyOptiX/SHADOW. It only standardizes, stores, analyzes, and plots.
- May wrap a single beam or multiple runs. Plotting is disabled if multiple.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Dict
import pandas as pd

from . import adapters, schema, stats, viz, io


class Beam:
    """
    Unified interface for standardized beams.

    Parameters
    ----------
    obj : Any or sequence of Any
        A PyOptiX DataFrame, SHADOW3/4 beam object, or already standardized
        pandas.DataFrame. May also be a sequence of such objects for multiple
        runs.
    code : str, optional
        Backend hint passed to ``adapters.to_standard_beam``. Ignored if obj
        is already a standardized DataFrame.
    """

    def __init__(self, obj: Any, code: Optional[str] = None) -> None:
        if isinstance(obj, (list, tuple)):
            self._runs = [self._standardize(o, code) for o in obj]
        else:
            self._runs = [self._standardize(obj, code)]

        # validate all runs
        for df in self._runs:
            schema.validate_beam(df)

        self._stats_cache: Optional[Dict] = None

    # ------------------------------------------------------------------
    # construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_h5(cls, path: str) -> "Beam":
        """Load beam(s) from an HDF5 file."""
        df = io.read_beam(path)
        return cls(df)

    @classmethod
    def from_df(cls, df: pd.DataFrame) -> "Beam":
        """Wrap an already standardized DataFrame."""
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
        """Number of runs stored in this Beam."""
        return len(self._runs)

    @property
    def df(self) -> pd.DataFrame:
        """Return the single standardized DataFrame (error if multiple)."""
        if self.n_runs != 1:
            raise ValueError("Beam contains multiple runs; access .runs instead.")
        return self._runs[0]

    @property
    def runs(self) -> Sequence[pd.DataFrame]:
        """Return the list of standardized DataFrames."""
        return self._runs

    @property
    def stats(self) -> Dict:
        """Return cached statistics (aggregate if multiple runs)."""
        if self._stats_cache is None:
            self._stats_cache = stats.get_statistics(self._runs)
        return self._stats_cache

    # ------------------------------------------------------------------
    # methods
    # ------------------------------------------------------------------
    def print_stats(self, verbose: bool = True) -> None:
        """Pretty-print statistics to stdout."""
        _ = stats.get_statistics(self._runs, verbose=verbose)

    # --- plotting (only if single run) ---
    def plot_beam(self, **kwargs):
        """Plot spatial footprint (X vs Y)."""
        if self.n_runs != 1:
            raise ValueError("Plotting not supported for multiple runs.")
        return viz.plot_beam(self.df, **kwargs)

    def plot_divergence(self, **kwargs):
        """Plot angular footprint (dX vs dY)."""
        if self.n_runs != 1:
            raise ValueError("Plotting not supported for multiple runs.")
        return viz.plot_divergence(self.df, **kwargs)

    def plot_phase_space(self, **kwargs):
        """Plot phase space (X–dX / Y–dY)."""
        if self.n_runs != 1:
            raise ValueError("Plotting not supported for multiple runs.")
        return viz.plot_phase_space(self.df, **kwargs)

    # --- saving ---
    def save(self, path: str, *, meta: Optional[Dict[str, Any]] = None) -> None:
        """
        Save beam to HDF5 and stats to JSON.

        The HDF5 file will contain the beam(s).
        A sibling JSON file (same base name) will contain the statistics.
        """
        # HDF5
        io.save_beam(self._runs[0] if self.n_runs == 1 else self._runs, path)

        # JSON (append .json to base name)
        base = path.rsplit(".", 1)[0]
        json_path = f"{base}.json"
        io.save_json_stats(self.stats, json_path, meta=meta)

    def save_beam(self, path: str) -> None:
        """Save the beam(s) only (HDF5)."""
        io.save_beam(self._runs[0] if self.n_runs == 1 else self._runs, path)

    def save_stats(self, path: str, *, meta: Optional[Dict[str, Any]] = None) -> None:
        """Save statistics only (JSON)."""
        io.save_json_stats(self.stats, path, meta=meta)