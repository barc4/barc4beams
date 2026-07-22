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

    beam.plot_rays(plot=False)
    beam.plot_beam(plot=False)
    beam.plot_divergence(plot=False)
    beam.plot_phase_space(plot=False)

    plt.close("all")


def test_plot_rays_filters_and_maps_absolute_intensity_to_alpha():
    beam = _make_standard_beam(4)
    beam.df.loc[:, "X"] = [0.0, 1e-6, 2e-6, 3e-6]
    beam.df.loc[:, "Y"] = [0.0, 1e-6, 2e-6, 3e-6]
    beam.df.loc[:, "intensity"] = [0.0, 0.2, 0.5, 1.0]
    beam.df.loc[:, "lost_ray_flag"] = [0.0, 0.0, 1.0, 0.0]

    fig, ax = beam.plot_rays(
        intensity_threshold=0.2,
        color="red",
        marker="x",
        plot=False,
    )

    collection = ax.collections[0]
    np.testing.assert_allclose(collection.get_offsets(), [[3.0, 3.0]])
    np.testing.assert_allclose(collection.get_facecolors(), [[1.0, 0.0, 0.0, 1.0]])
    assert collection.get_paths()[0].vertices.shape[0] > 1
    plt.close(fig)


def test_plot_rays_default_threshold_removes_zero_intensity():
    beam = _make_standard_beam(3)
    beam.df.loc[:, "intensity"] = [0.0, 0.5, 1.0]

    fig, ax = beam.plot_rays(plot=False)

    np.testing.assert_allclose(ax.collections[0].get_offsets()[:, 0], beam.df["X"].to_numpy()[1:] * 1e6)
    np.testing.assert_allclose(ax.collections[0].get_facecolors()[:, 3], [0.55, 1.0])
    plt.close(fig)


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
