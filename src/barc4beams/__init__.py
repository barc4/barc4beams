# SPDX-License-Identifier: CECILL-2.1
# Copyright (c) 2025 Synchrotron SOLEIL

"""
barc4beams — analysis and plotting for ray-traced photon beams.
"""

from ._version import __version__
from .adapters import merge_standard_beams, to_standard_beam
from .beam import Beam
from .io import read_beam, read_json_stats, save_beam, save_json_stats
from .propagation import caustic, propagate
from .sampling import beam_from_intensity
from .stats import calc_envelope_from_moments, get_focal_distance, get_statistics
from .viz import (
    plot,
    plot_beam,
    plot_caustic,
    plot_divergence,
    plot_energy,
    plot_energy_vs_intensity,
    plot_phase_space,
)

__all__ = [
    "__version__",
    # adapters
    "merge_standard_beams",
    "to_standard_beam",
    # beam
    "Beam",
    # io
    "read_beam",
    "read_json_stats",
    "save_beam",
    "save_json_stats",
    # propagation
    "caustic",
    "propagate",
    # sampling
    "beam_from_intensity",
    # stats
    "calc_envelope_from_moments",
    "get_focal_distance",
    "get_statistics",
    # viz
    "plot",
    "plot_beam",
    "plot_caustic",
    "plot_divergence",
    "plot_energy",
    "plot_energy_vs_intensity",
    "plot_phase_space",
]