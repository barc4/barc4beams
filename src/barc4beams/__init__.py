# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2025 Synchrotron SOLEIL

"""
barc4beams — analysis and plotting for ray-traced photon beams.
"""

from ._version import __version__
from .beam import Beam
from .adapters import to_standard_beam
from .stats import get_statistics, get_focal_distance
from .viz import plot_beam, plot_divergence, plot_phase_space, plot_beamline, plot_beamline_configs
from .io import save_beam, read_beam, save_json_stats, read_json_stats

__all__ = [
    "__version__",
    "Beam",
    "to_standard_beam",
    "get_statistics",
    "get_focal_distance",
    "plot_beam",
    "plot_divergence",
    "plot_phase_space",
    "plot_beamline",
    "plot_beamline_configs",
    "save_beam",
    "read_beam",
    "save_json_stats",
    "read_json_stats",
]