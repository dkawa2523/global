from __future__ import annotations

import numpy as np
import pytest

from plasma_global.experimental.accumulators import (
    ConstantSource,
    DrivenSource,
    FilmInventoryAccumulator,
    GenericState,
    GenericStateAccumulator,
    LinearRelaxation,
    ProcessContext,
    SurfaceEventRate,
)
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


def test_generic_state_processes_are_additive_finite_and_follow_driver() -> None:
    accumulator = GenericStateAccumulator(
        states=(
            GenericState("damage", owners=("wafer",), initial=0.0, lower_bound=0.0),
        ),
        processes=(
            ConstantSource("damage", rate_per_s=1.0),
            LinearRelaxation("damage", equilibrium=4.0, time_constant_s=2.0),
            DrivenSource("damage", driver="ion_flux_m2_s", coefficient=1.0e-18),
        ),
    )
    state = accumulator.initial_state()
    low = accumulator.rhs(state, ProcessContext({"ion_flux_m2_s": {"wafer": 1.0e17}}))
    high = accumulator.rhs(state, ProcessContext({"ion_flux_m2_s": {"wafer": 4.0e17}}))

    assert np.isfinite(low).all()
    assert high[0] > low[0] > 0.0
    with pytest.raises(ValueError, match="below its declared domain"):
        accumulator.rhs(np.array([-1.0]))


def test_surface_inventory_conserves_particles_and_film_follows_event_rate() -> None:
    accumulator = FilmInventoryAccumulator()
    event = SurfaceEventRate(
        event_id="deposit A",
        zone_id="plasma",
        surface_id="wafer",
        rate_m2_s=2.0e18,
        area_m2=0.04,
        site_density_m2=1.0e19,
        gas_particles_per_event={"A": -1.0},
        inventory_particles_per_event={"A": 1.0},
        film_layers_per_event=1.0,
    )
    base = accumulator.accumulate((event,))
    doubled = accumulator.accumulate(
        (
            SurfaceEventRate(
                event_id="deposit A faster",
                zone_id=event.zone_id,
                surface_id=event.surface_id,
                rate_m2_s=2.0 * event.rate_m2_s,
                area_m2=event.area_m2,
                site_density_m2=event.site_density_m2,
                gas_particles_per_event=event.gas_particles_per_event,
                inventory_particles_per_event=event.inventory_particles_per_event,
                film_layers_per_event=event.film_layers_per_event,
            ),
        )
    )

    assert base.particle_balance_s("A") == pytest.approx(0.0, abs=1.0e-12)
    gas_source = base.gas_density_source({"plasma": 0.02})
    assert gas_source["plasma", "A"] * 0.02 == pytest.approx(
        -base.inventory_particle_rate_s["wafer", "A"]
    )
    assert doubled.film_growth_m_s["wafer"] > base.film_growth_m_s["wafer"] > 0.0
    assert np.isfinite(
        [
            *base.gas_particle_rate_s.values(),
            *base.inventory_particle_rate_s.values(),
            *base.film_growth_m_s.values(),
        ]
    ).all()
