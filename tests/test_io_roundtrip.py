import os
import tempfile

import numpy as np
import pandas as pd
import barc4beams as b4b


def _make_standard_beam(n: int = 100) -> b4b.Beam:
    X = np.linspace(-1e-6, 1e-6, n)
    Y = np.linspace(-2e-6, 2e-6, n)
    dX = np.zeros(n)
    dY = np.zeros(n)

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


def test_h5_roundtrip():
    beam = _make_standard_beam()

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "beam.h5")
        beam.save_beam(path)
        beam2 = b4b.Beam.from_h5(path)

    assert len(beam2.df) == len(beam.df)
    assert set(beam2.df.columns) == set(beam.df.columns)


def test_json_stats_roundtrip():
    beam = _make_standard_beam()
    stats = beam.stats()

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "stats.json")
        # use the high-level API
        beam.save_stats(path, meta={"test": True})
        record = b4b.read_json_stats(path)

    # Top-level structure
    assert "stats" in record
    assert "meta" in record
    assert record["meta"].get("test") is True

    stats_rec = record["stats"]
    assert "meta" in stats_rec
    assert "X" in stats_rec
    assert "Y" in stats_rec
    assert stats_rec["meta"]["n_rays"] > 0