# SPDX-License-Identifier: CECILL-2.1
# Copyright (c) 2025 Synchrotron SOLEIL

"""
barc4beams — analysis and plotting for ray-traced photon beams.
"""

from ._version import __version__
from .adapters import merge_standard_beams, to_standard_beam
from .beam import Beam
from .caustics import compute_caustic
from .io import read_beam, read_json_stats, save_beam, save_json_stats
from .stats import calc_envelope_from_moments, get_focal_distance, get_statistics
from .viz import (
    plot,
    plot_beam,
    plot_divergence,
    plot_energy,
    plot_energy_vs_intensity,
    plot_phase_space,
    plot_caustic
)

__all__ = [
    "__version__",
    "to_standard_beam",
    "merge_standard_beams",
    "Beam",
    "compute_caustic",
    "get_statistics",
    "get_focal_distance",
    "calc_envelope_from_moments",
    "plot",
    "plot_beam",
    "plot_divergence",
    "plot_energy",
    "plot_energy_vs_intensity",
    "plot_phase_space",
    "plot_caustic",
    "save_beam",
    "read_beam",
    "save_json_stats",
    "read_json_stats",
]