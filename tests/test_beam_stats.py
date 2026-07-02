import numpy as np
import pandas as pd
import barc4beams as b4b
from barc4beams.stats import _format_scalar, _format_with_unc


def _make_standard_beam(
    n: int = 2000,
    sigma_x: float = 10e-6,
    sigma_y: float = 5e-6,
    sigma_dx: float = 50e-6,
    sigma_dy: float = 20e-6,
) -> b4b.Beam:
    rng = np.random.default_rng(42)
    X = rng.normal(0.0, sigma_x, n)
    Y = rng.normal(0.0, sigma_y, n)
    dX = rng.normal(0.0, sigma_dx, n)
    dY = rng.normal(0.0, sigma_dy, n)

    df = pd.DataFrame({
        "energy": np.full(n, 1000.0),        # eV, arbitrary
        "X": X,
        "Y": Y,
        "dX": dX,
        "dY": dY,
        "wavelength": np.full(n, 1e-9),      # m, arbitrary
        "intensity": np.ones(n),
        "intensity_s-pol": np.ones(n),
        "intensity_p-pol": np.ones(n),
        "lost_ray_flag": np.zeros(n),
    })
    return b4b.Beam(df)


def test_basic_stats():
    sigma_x = 10e-6
    sigma_y = 5e-6
    sigma_dx = 50e-6
    sigma_dy = 20e-6

    beam = _make_standard_beam(
        n=2000,
        sigma_x=sigma_x,
        sigma_y=sigma_y,
        sigma_dx=sigma_dx,
        sigma_dy=sigma_dy,
    )

    stats = beam.stats()

    assert abs(stats["X"]["std"][0] - sigma_x) < 2e-6
    assert abs(stats["Y"]["std"][0] - sigma_y) < 1e-6
    assert abs(stats["dX"]["std"][0] - sigma_dx) < 10e-6
    assert abs(stats["dY"]["std"][0] - sigma_dy) < 4e-6


def test_stats_ignore_lost_and_zero_intensity_rows_for_weighted_observables():
    df = pd.DataFrame(
        {
            "energy": [10.0, 20.0, 999.0, 1000.0, 1001.0],
            "X": [0.0, 2.0, 100.0, 200.0, 300.0],
            "Y": [10.0, 14.0, 100.0, 200.0, 300.0],
            "dX": [0.0, 2.0, 100.0, 200.0, 300.0],
            "dY": [10.0, 14.0, 100.0, 200.0, 300.0],
            "wavelength": [1e-9, 1e-9, 1e-9, 1e-9, 1e-9],
            "intensity": [1.0, 1.0, 0.0, 1.0, 0.0],
            "intensity_s-pol": [1.0, 1.0, 0.0, 1.0, 0.0],
            "intensity_p-pol": [1.0, 1.0, 0.0, 1.0, 0.0],
            "lost_ray_flag": [0, 0, 0, 1, 1],
            "id": ["", "", "", "", "shadow4_cleaned_lost_ray"],
        }
    )

    stats = b4b.get_statistics(df)

    assert stats["meta"]["n_rays"] == 5
    assert stats["meta"]["good_rays"] == [3.0, 0.0]
    assert stats["meta"]["transmission"] == [40.0, 0.0]
    assert stats["X"]["mean"] == [1.0, 0.0]
    assert stats["X"]["std"] == [1.0, 0.0]
    assert stats["Y"]["mean"] == [12.0, 0.0]
    assert stats["Y"]["std"] == [2.0, 0.0]
    assert stats["energy"]["mean"] == [15.0, 0.0]
    assert stats["energy"]["std"] == [5.0, 0.0]


def test_verbose_beam_scale_formatting_contract():
    assert _format_scalar(0.0904, "beam_um") == "0.090"
    assert _format_scalar(4.254, "beam_um") == "4.25"
    assert _format_scalar(42.54, "beam_um") == "42.5"
    assert _format_scalar(500.4, "beam_um") == "500"
    assert _format_scalar(3500.4, "beam_um") == "3500"
    assert _format_scalar(12500.0, "beam_um") == "1.2500e+04"


def test_verbose_focal_formatting_contract_preserves_sign():
    assert _format_scalar(9e-9, "focal") == "0"
    assert _format_scalar(-4.2e-5, "focal") == "-4.2000e-05"
    assert _format_scalar(0.0123454, "focal") == "0.012345"
    assert _format_scalar(-0.12345, "focal") == "-0.1235"
    assert _format_scalar(12.3454, "focal") == "12.345"
    assert _format_scalar(-345.674, "focal") == "-345.67"
    assert _format_scalar(3456.74, "focal") == "3456.7"
    assert _format_scalar(12345.0, "focal") == "1.2345e+04"


def test_verbose_percent_and_repetition_formatting_contract():
    assert _format_scalar(0.1234, "percent") == "0.123"
    assert _format_scalar(4.567, "percent") == "4.57"
    assert _format_scalar(45.67, "percent") == "45.7"
    assert (
        _format_with_unc(4.254e-6, 0.0, "beam_um", 1, scale=1e6)
        == "4.25 µm"
    )
    assert (
        _format_with_unc(4.254e-6, 0.034e-6, "beam_um", 3, scale=1e6)
        == "4.25 +- 0.03 µm"
    )
    assert (
        _format_with_unc(0.0904e-6, 0.004e-6, "beam_um", 3, scale=1e6)
        == "0.090 +- 0.004 µm"
    )


def test_verbose_energy_spread_formatting_contract():
    assert _format_scalar(0.1234, "energy_spread") == "0.123"
    assert _format_scalar(4.567, "energy_spread") == "4.57"
    assert _format_scalar(45.67, "energy_spread") == "45.7"
    assert _format_scalar(456.7, "energy_spread") == "457"
    assert _format_scalar(1234.5, "energy_spread") == "1234"
