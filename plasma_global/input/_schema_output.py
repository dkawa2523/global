"""Selections and analysis artifacts declared together with a case."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from plasma_global.input._schema_base import Identifier, NonNegativeFloat, StrictModel

ArtifactName = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")]


class ProbeConfig(StrictModel):
    zone: Identifier | None = None
    surface: Identifier | None = None
    power_port: Identifier | None = None
    quantities: list[Identifier] = Field(default_factory=list)
    species: list[Identifier] | Literal["all"] = "all"

    @model_validator(mode="after")
    def one_target(self) -> Self:
        if (
            sum(item is not None for item in (self.zone, self.surface, self.power_port))
            != 1
        ):
            raise ValueError("a probe requires exactly one zone, surface or power_port")
        if self.species != "all" and (
            not self.species or len(set(self.species)) != len(self.species)
        ):
            raise ValueError("probe species must be nonempty and unique")
        if len(set(self.quantities)) != len(self.quantities):
            raise ValueError("probe quantities must be unique")
        return self


class PlotConfig(StrictModel):
    kind: Literal["series", "composition"] = "series"
    series: list[Identifier] = Field(default_factory=list)
    zone: Identifier | None = None
    species: list[Identifier] | Literal["all"] = "all"
    quantity: Literal["number_density_m3", "mole_fraction"] = "number_density_m3"
    yscale: Literal["linear", "log", "symlog"] = "linear"
    time_unit: Literal["s", "ms", "us", "ns"] = "s"
    title: str = ""
    format: Literal["png", "svg", "pdf"] = "png"
    dpi: int = Field(default=150, gt=0)
    time_range_s: list[NonNegativeFloat] | None = Field(
        default=None, min_length=2, max_length=2
    )

    @model_validator(mode="after")
    def applicable_selection(self) -> Self:
        _validate_plot_kind(self)
        if (
            self.time_range_s is not None
            and self.time_range_s[0] >= self.time_range_s[1]
        ):
            raise ValueError("plot time_range_s must be increasing")
        return self


def _validate_plot_kind(plot: PlotConfig) -> None:
    if plot.kind == "composition":
        if plot.zone is None or plot.series:
            raise ValueError("a composition plot requires zone and no series")
    elif not plot.series or plot.zone is not None or plot.species != "all":
        raise ValueError("a series plot requires series and no zone/species")
    if plot.kind != "composition" and plot.quantity != "number_density_m3":
        raise ValueError("quantity is only used by composition plots")


def _network_domains() -> list[Literal["gas", "boundary", "surface"]]:
    return ["gas", "boundary", "surface"]


class NetworkConfig(StrictModel):
    mode: Literal["mechanism", "events"] = "mechanism"
    measure: Literal["cumulative", "rate"] = "cumulative"
    time_s: NonNegativeFloat | None = None
    domains: list[Literal["gas", "boundary", "surface"]] = Field(
        default_factory=_network_domains
    )
    zones: list[Identifier] = Field(default_factory=list)
    surfaces: list[Identifier] = Field(default_factory=list)
    species: list[Identifier] = Field(default_factory=list)
    min_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    format: Literal["svg", "png", "pdf", "dot"] = "svg"

    @model_validator(mode="after")
    def applicable_measure(self) -> Self:
        if not self.domains or len(set(self.domains)) != len(self.domains):
            raise ValueError("network domains must be nonempty and unique")
        if self.mode == "mechanism" and (
            self.time_s is not None
            or self.min_fraction != 0
            or self.measure != "cumulative"
        ):
            raise ValueError("time_s, measure and min_fraction apply to event networks")
        return self


class OutputConfig(StrictModel):
    observables: list[str] = Field(default_factory=list)
    summary_series: list[str] = Field(default_factory=list)
    reaction_rates: bool = False
    probes: dict[ArtifactName, ProbeConfig] = Field(default_factory=dict)
    plots: dict[ArtifactName, PlotConfig] = Field(default_factory=dict)
    networks: dict[ArtifactName, NetworkConfig] = Field(default_factory=dict)
    csv: bool = False
    units: dict[Identifier, Identifier] = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid_series_selection(self) -> Self:
        for field_name, values in (
            ("observables", self.observables),
            ("summary_series", self.summary_series),
        ):
            if any(not value.strip() for value in values):
                raise ValueError(f"output.{field_name} must not contain empty names")
            if len(set(values)) != len(values):
                raise ValueError(f"output.{field_name} must not contain duplicates")
        return self
