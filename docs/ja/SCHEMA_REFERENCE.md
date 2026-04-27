# 入力スキーマ早見表

このページは、YAML/CSV を読むときに迷いやすい必須項目、単位、default の考え方をまとめた早見表です。詳細な実装仕様ではなく、case file を確認するための参照表として使います。実際の解決後設定は `effective_case.yaml` を優先して確認してください。

## case YAML

| セクション | 必須度 | 代表 key | 単位 / 値 | 説明 |
|---|---|---|---|---|
| `include` | 任意 | `include` | path | base case を読み込む |
| `case` | 推奨 | `name`, `description`, `schema_version` | text / integer | case metadata |
| `files` | 必須 | `chamber`, `recipe`, `chemistry.manifest`, `output_dir` | path | 入力ファイルと出力先 |
| `physics` | 必須 | `eedf_backend`, `electrical_backend`, `integrator` | registry key | backend 選択 |
| `swarm` | backend 依存 | `model_name`, `closure`, `table` | registry key / path | EEDF/rate lookup 設定 |
| `numerics` | 推奨 | `rtol`, `atol`, `max_step`, `jacobian` | numeric | ODE solver 設定 |
| `outputs` | 推奨 | `formats`, `diagnostics`, `plots` | bool / list | 出力制御 |
| `runtime` | 任意 | `export_effective_config`, `export_resolved_paths` | bool | 再現性ファイルの出力 |

最小形の考え方:

```yaml
include: base_case.yaml

case:
  name: my_case

files:
  recipe: recipe_my_case.yaml
  output_dir: ../outputs/my_case
```

`include` で base が chamber、chemistry、physics、numerics を提供するなら、override case は少ない key で成立します。

## `files` の path 解決

| Key | 解決基準 | 注意 |
|---|---|---|
| `files.chamber` | case YAML のある directory | chamber YAML |
| `files.recipe` | case YAML のある directory | recipe YAML |
| `files.chemistry.manifest` | case YAML のある directory | chemistry manifest |
| `files.output_dir` | case YAML のある directory | run output |
| `files.external_inputs.*` | case YAML のある directory | profile、waveform、reference など |

run 後は `resolved_paths.yaml` に絶対パスが書かれます。

## chamber YAML

| セクション | 必須度 | 主な key | 単位 |
|---|---|---|---|
| `zones` | 必須 | `zone_id`, `volume_m3`, `pressure_Pa`, `gas_temperature_K` | `m3`, `Pa`, `K` |
| `edges` | 任意 | `from_zone`, `to_zone`, `conductance_m3_s` | `m3/s` |
| `surfaces` | surface ありなら必須 | `surface_id`, `zone_id`, `area_m2`, `temperature_K`, `models` | `m2`, `K` |
| `gas_inlets` | flow ありなら必要 | `inlet_id`, `zone_id`, `flow_sccm` | `sccm` |
| `pumps` | pressure control ありなら必要 | `pump_id`, `zone_id`, `speed_m3_s`, `target_pressure_Pa` | `m3/s`, `Pa` |
| `power_ports` | 電力入力ありなら必要 | `port_id`, `kind`, `zone_id`, `parameters` | backend 依存 |

zone の例:

```yaml
zones:
  - zone_id: source
    volume_m3: 0.010
    pressure_Pa: 8.0
    gas_temperature_K: 350.0
```

surface の例:

```yaml
surfaces:
  - surface_id: wafer
    zone_id: process
    area_m2: 0.0314
    temperature_K: 320.0
    models:
      ion_loss: bohm_edge_loss
```

## recipe YAML

| Key | 必須度 | 単位 / 値 | 説明 |
|---|---|---|---|
| `recipe_id` | 推奨 | text | recipe 名 |
| `steps` | 必須 | list | 時間順 step |
| `step_id` | 必須 | text | step 名 |
| `t_start_s`, `t_end_s` | 必須 | `s` | step の時間範囲 |
| `gas_inlets` | 任意 | species -> sccm | chamber inlet の flow override |
| `power_ports` | 任意 | backend 依存 | port command |
| `imported_inputs` | 任意 | text/path | 外部 anchor label など |

power step の例:

```yaml
steps:
  - step_id: ignition
    t_start_s: 0.0
    t_end_s: 1.0e-6
    power_ports:
      source_rf:
        mode: absorbed_power
        zone_id: source
        value_W: 300.0
        frequency_Hz: 1.356e7
```

## chemistry manifest

| Key | 必須度 | 説明 |
|---|---|---|
| `schema_version` | 推奨 | manifest schema |
| `species_file` | 必須 | species CSV |
| `gas_reactions_file` | gas reaction ありなら必須 | gas reaction CSV |
| `surface_reactions_file` | surface reaction ありなら必須 | surface reaction CSV |
| `reaction_models_file` | legacy bundle では必須 | rate / energy model YAML |
| `model_files` | split model では推奨 | `electron_impact`, `gas_rate`, `energy_loss` |
| `cross_sections_manifest` | cross-section 利用時に必須 | cross-section index |
| `aliases_file` | 任意 | species alias |

## chemistry CSV の主要列

### `species.csv`

| 列 | 必須度 | 説明 |
|---|---|---|
| `canonical_id` | 必須 | solver 内 species ID |
| `phase` | 必須 | `gas`, `surface` など |
| `charge` | 必須 | 電荷数 |
| `mass_amu` | gas species で推奨 | 質量 |
| `elements` | 推奨 | 元素組成。保存則 validation に使う |
| `aliases` | 任意 | 別名 |
| `zones`, `surfaces` | 任意 | 適用先 |

### `gas_reactions.csv` / `surface_reactions.csv`

| 列 | 必須度 | 説明 |
|---|---|---|
| `reaction_id` | 必須 | 一意 ID |
| `phase` | 必須 | `gas` / `surface` |
| `equation` | 必須 | reaction equation |
| `rate_model_key` | 必須 | rate model YAML の key |
| `energy_model_key` | 任意 | electron energy loss model |
| `zone_filter` | 任意 | 適用 zone |
| `surface_filter` | surface reaction で推奨 | 適用 surface |
| `enabled` | 推奨 | true/false |

## よくある validation error と見る場所

| 症状 | 最初に見る場所 |
|---|---|
| file not found | `resolved_paths.yaml`、`files.*` |
| unknown backend | `physics.*`、`plasma-global list-backends` |
| unknown species | `species.csv`、reaction equation |
| charge / element conservation error | reaction equation、`elements` |
| duplicate reaction | `reaction_id` |
| rate model not found | `rate_model_key` と model YAML |
| table range error | rate table metadata、observed `EoverN_*_Td` |
| mixed ion-loss modes | chamber surface `models.ion_loss` |

## よくある validation message

実行ログや validation report に次のような code が出た場合は、まず表の右端を確認してください。ここでは利用者が入力を直すための見方に絞って説明します。

| Code | Level | 意味 | 最初に見る設定 |
|---|---|---|---|
| `CHEMISTRY_PATH_MISSING` | ERROR | chemistry manifest または chemistry directory が未指定 | `files.chemistry.manifest`, `files.chemistry.directory` |
| `FILE_NOT_FOUND` | ERROR | 必須入力 path が存在しない | `resolved_paths.yaml`, `files.chamber`, `files.recipe`, `files.chemistry.*` |
| `UNKNOWN_POWER_PORT` | ERROR | recipe が chamber にない power port を参照 | `chamber.power_ports[*].port_id`, `recipe.steps[*].power_ports` |
| `DC_SERIES_CIRCUIT_CONFIG_INVALID` | ERROR | DC series backend の port 設定が不足または不正 | `physics.electrical_backend`, power port `parameters`, recipe override |
| `EXTERNAL_CIRCUIT_TABLE_CONFIG_INVALID` | ERROR | 外部 circuit table の path / columns / mapping が不正 | `files.external_inputs`, circuit table CSV, port config |
| `RF_ENVELOPE_CONFIG_INVALID` | ERROR | RF envelope port 設定が不足または不正 | RF port `parameters`, recipe override |
| `PRESCRIBED_ELECTRON_PROFILE_INVALID` | ERROR | prescribed electron profile の CSV や zone mapping が不正 | `physics.electron_density_closure`, `swarm.prescribed_electron_profile` |
| `ELECTRON_DENSITY_CLOSURE_UNRECOGNIZED` | ERROR | 未知の electron-density closure | `physics.electron_density_closure` |
| `SWARM_CLOSURE_UNRECOGNIZED` | ERROR | 未知の swarm closure | `swarm.closure` |
| `GAS_HEATING_FRACTION_RANGE` | ERROR | gas heating fraction が 0 から 1 の範囲外 | `physics.gas_heating_fraction` |
| `WALL_RELAXATION_RANGE` | ERROR | wall relaxation rate が負 | `physics.wall_relaxation_s_inv` |
| `AMBIPOLAR_LOSS_CONFIG_INVALID` | ERROR | ambipolar ion loss に必要な rate / diffusion 設定が不足 | surface `models.ion_loss` |
| `BOHM_H_FACTOR_INVALID` | ERROR | Bohm-family ion loss の `h_factor` 設定が不正 | surface `models.ion_loss`, `h_factor`, `characteristic_length_m` |
| `ION_LOSS_MODE_MIXED_IN_ZONE` | ERROR | 同一 zone で Bohm 系と ambipolar diffusion を混在 | zone 内の surface `models.ion_loss` |
| `INTEGRATOR_UNVERIFIED` | WARNING | smoke-tested default 以外の integrator | `physics.integrator` |
| `LOCAL_FIELD_CLOSURE_REDUCED_MODEL` | WARNING | local-field closure が reduced-field proxy を使う | `swarm.closure`, electrical backend field diagnostics |
| `LEGACY_SCHEMA` | WARNING | 古い run.yaml schema を読み込んでいる | `case.schema_version` |
| `SURFACE_MODELS_METADATA_ONLY` | WARNING | solver が直接消費しない surface model metadata がある | surface `models` |
| `ION_LOSS_MODEL_UNRECOGNIZED` | WARNING | 未知の ion-loss mode を Bohm-like active surface として扱う | surface `models.ion_loss` |
| `AMBIPOLAR_LOSS_MULTIPLE_SURFACES` | WARNING | 同一 zone に複数 ambipolar surface がある | zone 内の surface `models.ion_loss` |
| `RF_ENVELOPE_CALIBRATION_HINT` | WARNING | RF envelope calibration に関する注意 | RF envelope port calibration parameters |

Level が `ERROR` の場合は run 前に修正が必要です。`WARNING` は常に失敗ではありませんが、benchmark や production 解釈では「意図した近似か」を記録してください。

## 単位の命名規則

このコードでは、unit-bearing key の多くが key 名に単位を含みます。

| suffix | 単位 |
|---|---|
| `_m3` | cubic meter |
| `_m2` | square meter |
| `_m3_s` | cubic meter per second |
| `_Pa` | pascal |
| `_K` | kelvin |
| `_s` | second |
| `_Hz` | hertz |
| `_W` | watt |
| `_V` | volt |
| `_eV` | electron volt |
| `_Td` | Townsend |
| `_m2_s` | per square meter per second, flux 文脈では `m^-2 s^-1` |
