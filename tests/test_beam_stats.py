import numpy as np
import pandas as pd
import barc4beams as b4b


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
