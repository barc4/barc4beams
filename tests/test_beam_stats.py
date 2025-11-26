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
    
    X = np.random.normal(0.0, sigma_x, n)
    Y = np.random.normal(0.0, sigma_y, n)
    dX = np.random.normal(0.0, sigma_dx, n)
    dY = np.random.normal(0.0, sigma_dy, n)

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