from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class CaseMetadata:
    name: str
    description: str = ''
    tags: list[str] = field(default_factory=list)
    schema_version: int = 2
    kind: str = 'plasma_global_case'


@dataclass
class ChemistryFilesConfig:
    manifest: str


@dataclass
class FilesConfig:
    chamber: str
    recipe: str
    chemistry: ChemistryFilesConfig
    output_dir: str = './outputs'
    external_inputs: dict[str, Any] = field(default_factory=dict)


@dataclass
class RuntimeConfig:
    export_effective_config: bool = True
    export_resolved_paths: bool = True


@dataclass
class PhysicsConfig:
    mode: str = 'transient'
    gas_model: str = 'multi_zone_global'
    eedf_backend: str = 'maxwell'
    electrical_backend: str = 'icp'
    integrator: str = 'scipy_bdf'
    enable_gas_temperature: bool = True
    enable_surface_coverages: bool = True
    enable_wall_inventory: bool = True
    electron_density_closure: str = 'quasi_neutral'
    gas_heating_fraction: float = 0.15
    wall_relaxation_s_inv: float = 500.0


@dataclass
class NumericsConfig:
    rtol: float = 1.0e-6
    atol: float = 1.0e-14
    first_step: float | None = 1.0e-10
    max_step: float | None = 1.0e-6
    positivity: dict[str, Any] | None = None
    events: dict[str, Any] | None = None


@dataclass
class OutputFormatsConfig:
    solution_h5: bool = True
    observables_csv: bool = True
    summary_yaml: bool = True


@dataclass
class OutputPlotsConfig:
    enabled: bool = False
    format: list[str] = field(default_factory=lambda: ['png'])
    dpi: int = 150
    items: list[str] = field(default_factory=list)


@dataclass
class OutputBudgetsConfig:
    enabled: bool = False


@dataclass
class OutputsConfig:
    formats: OutputFormatsConfig = field(default_factory=OutputFormatsConfig)
    plots: OutputPlotsConfig = field(default_factory=OutputPlotsConfig)
    budgets: OutputBudgetsConfig = field(default_factory=OutputBudgetsConfig)


@dataclass
class SwarmCacheConfig:
    max_entries: int = 12
    fraction_decimals: int = 3


@dataclass
class SwarmEnergyGridConfig:
    min_eV: float = 1.0e-3
    max_eV: float = 160.0
    n: int = 360


@dataclass
class SwarmReducedFieldGridConfig:
    min: float = 0.2
    max: float = 2500.0
    n: int = 48


@dataclass
class Boltzmann2TermConfig:
    energy_grid: SwarmEnergyGridConfig = field(default_factory=SwarmEnergyGridConfig)
    reduced_field_grid_Td: SwarmReducedFieldGridConfig = field(default_factory=SwarmReducedFieldGridConfig)
    max_shape_iterations: int = 48


@dataclass
class SwarmTableConfig:
    file: str | None = None
    lookup: str | None = None
    bounds_policy: str = 'clip'
    electron_energy_mode: str | None = None
    energy_relaxation_time_s: float = 1.0e-6


@dataclass
class PrescribedElectronProfileConfig:
    file: str | None = None
    file_key: str | None = None
    zone_columns: dict[str, str] = field(default_factory=dict)
    interpolation: str = 'linear'
    hold: str = 'edge'


@dataclass
class SwarmConfig:
    model_name: str = 'table'
    closure: str = 'auto'
    mixture_key_species: list[str] = field(default_factory=list)
    cache: SwarmCacheConfig = field(default_factory=SwarmCacheConfig)
    boltzmann_2term: Boltzmann2TermConfig = field(default_factory=Boltzmann2TermConfig)
    table: SwarmTableConfig = field(default_factory=SwarmTableConfig)
    prescribed_electron_profile: PrescribedElectronProfileConfig = field(default_factory=PrescribedElectronProfileConfig)


@dataclass
class ResolvedPaths:
    source_config: str
    base_dir: str
    chamber_file: str
    recipe_file: str
    chemistry_manifest: str | None
    chemistry_dir: str | None
    output_dir: str
    external_inputs: dict[str, str | None] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunConfig:
    case: CaseMetadata
    files: FilesConfig
    runtime: RuntimeConfig
    physics: PhysicsConfig
    numerics: NumericsConfig
    outputs: OutputsConfig
    swarm: SwarmConfig = field(default_factory=SwarmConfig)

    @property
    def schema_version(self) -> int:
        return int(self.case.schema_version)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
