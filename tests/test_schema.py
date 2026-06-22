import pandas as pd
import pytest

from barc4beams.schema import validate_beam


def _valid_beam():
    return pd.DataFrame(
        {
            "energy": [1000.0, 1000.0],
            "X": [0.0, 1.0],
            "Y": [0.0, 1.0],
            "dX": [0.0, 0.0],
            "dY": [0.0, 0.0],
            "wavelength": [1e-9, 1e-9],
            "intensity": [1.0, 0.0],
            "intensity_s-pol": [0.5, 0.0],
            "intensity_p-pol": [0.5, 0.0],
            "lost_ray_flag": [0, 1],
        }
    )


def test_validate_beam_accepts_valid_standard_beam():
    validate_beam(_valid_beam())


def test_validate_beam_rejects_missing_required_column():
    df = _valid_beam().drop(columns=["intensity"])

    with pytest.raises(ValueError, match="missing required columns"):
        validate_beam(df)


@pytest.mark.parametrize("bad_intensity", [-0.1, 1.1])
def test_validate_beam_rejects_intensity_outside_unit_interval(bad_intensity):
    df = _valid_beam()
    df.loc[0, "intensity"] = bad_intensity

    with pytest.raises(ValueError, match="within \\[0, 1\\]"):
        validate_beam(df)


def test_validate_beam_rejects_non_binary_lost_ray_flag():
    df = _valid_beam()
    df.loc[0, "lost_ray_flag"] = 2

    with pytest.raises(ValueError, match="lost_ray_flag"):
        validate_beam(df)


def test_validate_beam_rejects_lost_ray_with_nonzero_intensity():
    df = _valid_beam()
    df.loc[1, "intensity"] = 0.2

    with pytest.raises(ValueError, match="Lost rays"):
        validate_beam(df)
