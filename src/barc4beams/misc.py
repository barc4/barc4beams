# SPDX-License-Identifier: CECILL-2.1
# Copyright (c) 2025 Synchrotron SOLEIL

"""
misc.py - optical design and other auxiliary functions.
"""

from __future__ import annotations


from scipy.constants import physical_constants

PLANCK = physical_constants["Planck constant"][0]
LIGHT = physical_constants["speed of light in vacuum"][0]
CHARGE = physical_constants["atomic unit of charge"][0]

# ---------------------------------------------------------------------------
# energy/wavelength conversion
# ---------------------------------------------------------------------------

_ENERGY_UNITS = {
    "ev": 1.0,
    "mev": 1e-3,
    "kev": 1e3,
}

_LENGTH_UNITS = {
    "m": 1.0,
    "um": 1e-6,
    "µm": 1e-6,         # micro sign
    "nm": 1e-9,
    "a": 1e-10,         # angstrom
    "å": 1e-10,         # unicode angstrom
    "angstrom": 1e-10,
}

def energy_wavelength(value: float, unit: str) -> float:
    """
    Converts energy to wavelength and vice versa.
    
    Parameters:
        value (float or array-like): The value of either energy or wavelength.
        unity (str): {'eV','meV','keV','m', 'um','µm', 'nm','A','Å','Angstrom',} (case-insensitive)
        
    Returns:
        float: Converted value in meters if the input is energy, or in eV if the input is wavelength.
        
    Raises:
        ValueError: If an invalid unit is provided.
    """
    unit_norm = unit.strip().lower()
    
    if unit_norm in _ENERGY_UNITS:
        scale = _ENERGY_UNITS[unit_norm]
    elif unit_norm in _LENGTH_UNITS:
        scale = _LENGTH_UNITS[unit_norm]
    else:
        supported = sorted({* _ENERGY_UNITS.keys(), * _LENGTH_UNITS.keys()})
        raise ValueError(f"Unsupported unit {unit!r}. Supported: {supported}")

    return PLANCK * LIGHT / CHARGE / (value * scale)

    