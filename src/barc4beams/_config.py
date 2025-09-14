# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2025 Synchrotron SOLEIL

"""
_config.py — global defaults for barc4beams.
"""

# Plotting defaults
DEFAULT_COLOR_SCHEMES = {
    1: "Greys",
    2: "viridis",
    3: "plasma",
    4: "magma",
    5: "cividis",
}
DEFAULT_APPLY_STYLE = True
DEFAULT_FONT_SCALE = 1.0
DEFAULT_BINS_MIN = 50
DEFAULT_BINS_MAX = 300

# I/O defaults
DEFAULT_H5_CHUNKS = True

# Stats defaults
DEFAULT_HUGE_M = 1e23  # focusing at infinity threshold