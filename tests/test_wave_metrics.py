import numpy as np
import pandas as pd
import barc4beams as b4b


def _make_focusing_beam(
    *,
    focus_x: float = 2.0,
    focus_y: float = 4.0,
    energy: float = 1239.8419843320025,
) -> b4b.Beam:
    x = np.array([-1.0, -1.0, 1.0, 1.0]) * 1e-3
    y = np.array([-2.0, 2.0, -2.0, 2.0]) * 1e-3
    dx = -x / focus_x
    dy = -y / focus_y

    df = pd.DataFrame(
        {
            "energy": np.full(x.size, energy),
            "X": x,
            "Y": y,
            "dX": dx,
            "dY": dy,
            "wavelength": np.full(x.size, 1e-9),
            "intensity": np.ones(x.size),
            "intensity_s-pol": np.ones(x.size),
            "intensity_p-pol": np.ones(x.size),
            "lost_ray_flag": np.zeros(x.size),
        }
    )
    return b4b.Beam(df)


def test_beam_wave_metrics_uses_image_plane_divergence():
    beam = _make_focusing_beam()

    metrics = beam.wave_metrics()

    wavelength = 1e-9
    theta = 5e-4 * np.sqrt(2.0 * np.log(2.0))
    na = np.sin(theta)
    waist = wavelength / (np.pi * na)
    dof = 2.0 * np.pi * waist**2 / wavelength

    assert metrics["meta"]["method"] == "gaussian_equivalent"
    assert metrics["meta"]["wavelength"] == [wavelength, 0.0]
    assert metrics["X"]["is_finite_focus"] is True
    assert np.isclose(metrics["X"]["theta"][0], theta)
    assert np.isclose(metrics["X"]["na"][0], na)
    assert np.isclose(metrics["X"]["gaussian_waist"][0], waist)
    assert np.isclose(metrics["X"]["depth_of_focus"][0], dof)
    assert np.isclose(metrics["X"]["convolved_beam_size"][0], 2.0 * waist)
    assert np.isclose(metrics["Y"]["theta"][0], theta)


def test_wave_metrics_uses_beam_wavelength_column_not_recomputed_energy():
    beam = _make_focusing_beam(energy=9999.0)

    metrics = beam.wave_metrics()

    theta = 5e-4 * np.sqrt(2.0 * np.log(2.0))
    waist = 1e-9 / (np.pi * np.sin(theta))
    assert metrics["meta"]["wavelength"] == [1e-9, 0.0]
    assert np.isclose(metrics["X"]["gaussian_waist"][0], waist)


def test_beam_ensemble_wave_metrics_aggregates_runs():
    beam1 = _make_focusing_beam(focus_x=2.0, focus_y=4.0)
    beam2 = _make_focusing_beam(focus_x=3.0, focus_y=4.0)
    ensemble = b4b.BeamEnsemble([beam1, beam2])

    metrics = ensemble.wave_metrics()

    assert metrics["X"]["theta"][1] > 0.0
    assert metrics["Y"]["theta"][1] == 0.0


def test_wave_metrics_returns_nan_for_near_collimated_axis():
    beam = _make_focusing_beam()
    df = beam.df.copy()
    df["dX"] = 0.0
    beam = b4b.Beam(df)

    metrics = beam.wave_metrics(max_focal_distance=1000.0)

    assert metrics["X"]["is_finite_focus"] is False
    assert np.isnan(metrics["X"]["na"][0])
    assert np.isfinite(metrics["Y"]["na"][0])


def test_wave_metrics_verbose_summary_format(capsys):
    beam = _make_focusing_beam()

    beam.wave_metrics(verbose=True)

    out = capsys.readouterr().out
    assert "Gaussian-equivalent wave metrics" in out
    assert "------------------ horizontal-plane:" in out
    assert ">> NA: 5.887e-04" in out
    assert ">> Gaussian waist diameter:" in out
    assert " µm" in out
    assert ">> Depth of focus:" in out
    assert " mm" in out
    assert ">> Convolved beam size:" in out
    assert "> percentile:" not in out
    assert "> wavelength:" not in out
