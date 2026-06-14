from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
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
    manifest: str | None = None
    directory: str | None = None
    species: str | None = None
    gas_reactions: str | None = None
    surface_reactions: str | None = None
    reaction_models: str | None = None
    aliases: str | None = None
    cross_sections_manifest: str | None = None


@dataclass
class FilesConfig:
    chamber: str
    recipe: str
    chemistry: ChemistryFilesConfig = field(default_factory=ChemistryFilesConfig)
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
    eedf_backend: str = 'swarm'
    electrical_backend: str = 'icp'
    integrator: str = 'scipy_bdf'
    enable_gas_temperature: bool = True
    enable_surface_coverages: bool = True
    enable_wall_inventory: bool = True
    enable_ied_proxy: bool = True
    enable_dae_fallback: bool = False
    quasi_neutrality: str = 'algebraic'
    electron_density_closure: str = 'quasi_neutral'
    gas_heating_fraction: float = 0.15
    wall_relaxation_s_inv: float = 500.0


@dataclass
class NumericsConfig:
    rtol: float = 1.0e-6
    atol: float = 1.0e-14
    first_step: float | None = 1.0e-10
    max_step: float | None = 1.0e-6
    jacobian: str = 'analytic_sparse'
    jacobian_rebuild_policy: str = 'every_step'
    positivity: dict[str, Any] | None = None
    events: dict[str, Any] | None = None


@dataclass
class OutputsConfig:
    formats: Any = None
    plots: Any = None
    save: Any = None


@dataclass
class LoggingConfig:
    level: str = 'INFO'
    write_log_file: bool = True
    log_filename: str = 'run.log'


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
    logging: LoggingConfig
    swarm: Any = None
    imports: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def schema_version(self) -> int:
        return int(self.case.schema_version)

    def to_dict(self) -> dict[str, Any]:
        def _plain(obj: Any) -> Any:
            if isinstance(obj, SimpleNamespace):
                return {k: _plain(v) for k, v in vars(obj).items()}
            if isinstance(obj, dict):
                return {k: _plain(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_plain(v) for v in obj]
            return obj

        data = asdict(self)
        data.pop('raw', None)
        return _plain(data)

    # ------------------------------------------------------------------
    # Backward-compatible views used by the rest of the code base.
    # ------------------------------------------------------------------
    @property
    def project(self) -> dict[str, Any]:
        return {
            'name': self.case.name,
            'description': self.case.description,
            'tags': list(self.case.tags),
            'schema_version': self.case.schema_version,
            'kind': self.case.kind,
        }

    @property
    def paths(self) -> SimpleNamespace:
        resolved = getattr(self, '_resolved_paths', None)
        if resolved is not None:
            return SimpleNamespace(
                chamber_file=resolved.chamber_file,
                recipe_file=resolved.recipe_file,
                chemistry_dir=resolved.chemistry_dir,
                chemistry_manifest=resolved.chemistry_manifest,
                output_dir=resolved.output_dir,
                external_inputs=resolved.external_inputs,
            )
        chemistry_dir = self.files.chemistry.directory
        if chemistry_dir is None and self.files.chemistry.manifest:
            chemistry_dir = str(Path(self.files.chemistry.manifest).parent)
        return SimpleNamespace(
            chamber_file=self.files.chamber,
            recipe_file=self.files.recipe,
            chemistry_dir=chemistry_dir,
            chemistry_manifest=self.files.chemistry.manifest,
            output_dir=self.files.output_dir,
            external_inputs=self.files.external_inputs,
        )

    @property
    def model(self) -> SimpleNamespace:
        return SimpleNamespace(
            mode=self.physics.mode,
            gas_model=self.physics.gas_model,
            eedf_backend=self.physics.eedf_backend,
            electrical_backend=self.physics.electrical_backend,
            integrator=self.physics.integrator,
            enable_gas_temperature=self.physics.enable_gas_temperature,
            enable_surface_coverages=self.physics.enable_surface_coverages,
            enable_wall_inventory=self.physics.enable_wall_inventory,
            enable_ied_proxy=self.physics.enable_ied_proxy,
            enable_dae_fallback=self.physics.enable_dae_fallback,
            quasi_neutrality=self.physics.quasi_neutrality,
            electron_density_closure=self.physics.electron_density_closure,
            gas_heating_fraction=self.physics.gas_heating_fraction,
            wall_relaxation_s_inv=self.physics.wall_relaxation_s_inv,
        )
