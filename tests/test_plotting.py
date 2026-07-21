import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import barc4beams as b4b


def _make_standard_beam(n: int = 500) -> b4b.Beam:
    X = np.random.normal(0.0, 5e-6, n)
    Y = np.random.normal(0.0, 5e-6, n)
    dX = np.random.normal(0.0, 1e-5, n)
    dY = np.random.normal(0.0, 1e-5, n)

    df = pd.DataFrame({
        "energy": np.full(n, 1000.0),
        "X": X,
        "Y": Y,
        "dX": dX,
        "dY": dY,
        "wavelength": np.full(n, 1e-9),
        "intensity": np.ones(n),
        "intensity_s-pol": np.ones(n),
        "intensity_p-pol": np.ones(n),
        "lost_ray_flag": np.zeros(n),
    })
    return b4b.Beam(df)


def test_plots_run():
    beam = _make_standard_beam()

    beam.plot_beam(plot=False)
    beam.plot_divergence(plot=False)
    beam.plot_phase_space(plot=False)

    plt.close("all")


def test_plot_caustic_uses_z_range(monkeypatch):
    beam = _make_standard_beam()
    captured = {}

    def fake_caustic(*, n_points, start, finish):
        captured.update(n_points=n_points, start=start, finish=finish)
        return {"caustic": True}

    def fake_plot_caustic(*, caustic, z_range, **kwargs):
        captured.update(caustic=caustic, plot_z_range=z_range)
        return "plot result"

    monkeypatch.setattr(beam, "caustic", fake_caustic)
    monkeypatch.setattr(b4b.beam.viz, "plot_caustic", fake_plot_caustic)

    result = beam.plot_caustic(z_range=(-1.25, 2.5), n_points=17, plot=False)

    assert result == "plot result"
    assert captured == {
        "n_points": 17,
        "start": -1.25,
        "finish": 2.5,
        "caustic": {"caustic": True},
        "plot_z_range": (-1.25, 2.5),
    }
