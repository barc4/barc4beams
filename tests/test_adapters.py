import numpy as np
import pandas as pd

from barc4beams.adapters import to_standard_beam


class FakeShadow4Beam:
    def __init__(self, *, cleaned=True, total=5):
        self._cleaned = cleaned
        self._total = total
        self._columns = {
            1: np.array([1.0, 2.0, 3.0]),
            2: np.array([10.0, 20.0, 30.0]),
            3: np.array([4.0, 5.0, 6.0]),
            4: np.array([0.1, 0.2, 0.3]),
            5: np.array([0.4, 0.5, 0.6]),
            6: np.array([0.7, 0.8, 0.9]),
            10: np.array([1.0, 1.0, -1.0]),
            19: np.array([2.0, 4.0, 6.0]),
            23: np.array([0.8, 0.9, 0.7]),
            24: np.array([0.3, 0.4, 0.5]),
            25: np.array([0.5, 0.5, 0.2]),
            26: np.array([100.0, 200.0, 300.0]),
        }

    @property
    def Nstored(self):
        return len(self._columns[1])

    def is_cleaned(self):
        return self._cleaned

    def get_number_of_rays(self, nolost=0):
        if nolost == 0:
            return self._total
        if nolost == 1:
            return 2
        if nolost == 2:
            return 1
        return self.Nstored

    def get_columns(self, columns):
        return np.array([self._columns[col] for col in columns])


class FakeShadow3Beam:
    def __init__(self):
        self._columns = {
            1: np.array([1.0, 2.0]),
            2: np.array([10.0, 20.0]),
            3: np.array([4.0, 5.0]),
            4: np.array([0.1, 0.2]),
            5: np.array([0.4, 0.5]),
            6: np.array([0.7, 0.8]),
            10: np.array([1.0, -1.0]),
            11: np.array([100.0, 200.0]),
            19: np.array([2.0, 4.0]),
            23: np.array([1.5, 0.7]),
            24: np.array([0.3, 0.4]),
            25: np.array([0.5, 0.6]),
        }

    def getshcol(self, columns):
        return np.array([self._columns[col] for col in columns])


def test_shadow4_cleaned_lost_rays_are_appended_without_duplication():
    df = to_standard_beam(FakeShadow4Beam(), code="shadow4")

    assert len(df) == 5
    assert df["lost_ray_flag"].tolist() == [0, 0, 1, 1, 1]

    restored = df.tail(2)
    assert restored["id"].tolist() == ["shadow4_cleaned_lost_ray"] * 2
    assert np.allclose(restored[["X", "Y", "dX", "dY", "dZ"]].to_numpy(), 0.0)
    assert np.allclose(restored["Z"].to_numpy(), np.mean([10.0, 20.0, 30.0]))
    assert np.allclose(restored["energy"].to_numpy(), np.mean([100.0, 200.0, 300.0]))
    assert np.allclose(restored["wavelength"].to_numpy(), np.mean([2.0, 4.0, 6.0]) * 1e-10)
    assert np.allclose(restored[["intensity", "intensity_s-pol", "intensity_p-pol"]].to_numpy(), 0.0)


def test_shadow4_not_cleaned_does_not_append_lost_rays():
    df = to_standard_beam(FakeShadow4Beam(cleaned=False, total=5), code="shadow4")

    assert len(df) == 3
    assert "shadow4_cleaned_lost_ray" not in set(df["id"])


def test_shadow4_cleaned_without_missing_stored_rays_does_not_append():
    df = to_standard_beam(FakeShadow4Beam(cleaned=True, total=3), code="shadow4")

    assert len(df) == 3
    assert "shadow4_cleaned_lost_ray" not in set(df["id"])


def test_shadow3_flags_lost_rays_and_zeroes_their_intensity():
    df = to_standard_beam(FakeShadow3Beam(), code="shadow3")

    assert df["lost_ray_flag"].tolist() == [0, 1]
    assert df.loc[0, "intensity"] == 1.0
    assert df.loc[1, "intensity"] == 0.0
    assert df.loc[1, "intensity_s-pol"] == 0.0
    assert df.loc[1, "intensity_p-pol"] == 0.0
    assert np.allclose(df["wavelength"].to_numpy(), np.array([2.0, 4.0]) * 1e-10)


def test_pyoptix_dataframe_gets_energy_polarization_and_lost_flag():
    df = pd.DataFrame(
        {
            "X": [0.0, 1.0],
            "Y": [0.0, 1.0],
            "dX": [0.0, 0.0],
            "dY": [0.0, 0.0],
            "wavelength": [1e-9, 2e-9],
            "intensity": [0.5, 0.0],
        }
    )

    out = to_standard_beam(df, code="pyoptix")

    assert out["lost_ray_flag"].tolist() == [0, 1]
    assert np.allclose(out["intensity_s-pol"].to_numpy(), out["intensity"].to_numpy())
    assert np.allclose(out["intensity_p-pol"].to_numpy(), out["intensity"].to_numpy())
    assert np.all(np.isfinite(out["energy"]))
