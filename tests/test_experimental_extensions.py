from __future__ import annotations

import numpy as np
import pytest

from plasma_global.experimental.profile import PrescribedElectronProfile


def test_prescribed_electron_profile_interpolation_and_bounds() -> None:
    linear = PrescribedElectronProfile(
        time_s=np.array([0.0, 1.0, 2.0]),
        density_m3_by_zone={"*": np.array([1.0e15, 3.0e15, 5.0e15])},
        interpolation="linear",
        bounds="error",
    )
    previous = PrescribedElectronProfile(
        time_s=linear.time_s,
        density_m3_by_zone=linear.density_m3_by_zone,
        interpolation="previous",
    )

    assert linear.density(0.5, "source") > linear.density(0.0, "source")
    assert linear.density(1.5, "source") > linear.density(0.5, "source")
    assert previous.density(0.5, "source") == previous.density(0.0, "source")
    with pytest.raises(ValueError, match="outside"):
        linear.density(-0.1, "source")


def test_prescribed_electron_profile_reads_zone_columns(tmp_path) -> None:
    path = tmp_path / "electrons.csv"
    path.write_text(
        "time_s,source_ne,process_ne\n0,1e15,2e15\n1,2e15,4e15\n",
        encoding="utf-8",
    )
    profile = PrescribedElectronProfile.from_csv(
        path, zone_columns={"source": "source_ne", "process": "process_ne"}
    )

    values = profile.density_by_zone(0.5, ("source", "process"))
    assert np.isfinite(list(values.values())).all()
    assert values["process"] > values["source"]


def test_prescribed_profile_normalizes_string_convertible_zone_ids(tmp_path) -> None:
    direct = PrescribedElectronProfile(
        time_s=np.array([0.0, 1.0]),
        density_m3_by_zone={1: np.array([1.0, 3.0])},
    )
    assert tuple(direct.density_m3_by_zone) == ("1",)
    assert direct.density(np.float32(0.5), 1) == pytest.approx(2.0)
    assert direct.density_by_zone(0.5, (1,)) == {"1": pytest.approx(2.0)}

    path = tmp_path / "numeric-zone.csv"
    path.write_text("time_s,density\n0,1\n1,3\n", encoding="utf-8")
    loaded = PrescribedElectronProfile.from_csv(path, zone_columns={1: "density"})
    assert loaded.density(0.5, "1") == pytest.approx(2.0)


def test_prescribed_electron_profile_csv_rejects_unsorted_time(tmp_path) -> None:
    path = tmp_path / "electrons.csv"
    path.write_text(
        "time_s,electron_density_m3\n1,1e15\n0,2e15\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="strictly increasing"):
        PrescribedElectronProfile.from_csv(path)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (
            "time_s,electron_density_m3,electron_density_m3\n0,1e15,1e15\n",
            "duplicate columns",
        ),
        (
            "time_s,electron_density_m3\n0,1e15,unexpected\n",
            "extra CSV fields",
        ),
        ("time_s,electron_density_m3\n0,\n", "missing values"),
        ("time_s,electron_density_m3\n0\n", "missing values"),
    ],
)
def test_prescribed_electron_profile_rejects_ambiguous_csv_shapes(
    tmp_path, content: str, message: str
) -> None:
    path = tmp_path / "invalid-electrons.csv"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        PrescribedElectronProfile.from_csv(path)


def test_prescribed_electron_profile_rejects_unselected_columns(tmp_path) -> None:
    path = tmp_path / "electrons-with-notes.csv"
    path.write_text("time_s,source_ne,notes\n0,1e15,1\n1,2e15,2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="extra columns: notes"):
        PrescribedElectronProfile.from_csv(path, zone_columns={"source": "source_ne"})
