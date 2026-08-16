# Architecture

## 設計の中心

本パッケージは、反応機構・反応器・時間レシピから、体積平均した 0D / multi-zone
プラズマの粒子密度とエネルギーを積分する Global Model です。設定辞書を実行時に
参照せず、入力時に一度だけ検証・コンパイルします。

```text
case.yaml
  -> input/load.py + input/schema.py       stable schema-v3 facade
     -> input/_schema_*.py                  domain input declarations
  -> chemistry/data.py + chemistry/compile.py   public chemistry facades
     -> chemistry/_data_* + _rate_* + _compile_validation.py
  -> build.py                              composition root
     -> input/compile_*.py                  thin compiler facades
  -> core/compiled.py + models/            immutable public numerical model
     -> core/_runtime_*.py                  prepared terms and RHS assembly
  -> core/solver.py                        segmented SciPy BDF facade
     -> core/_solver_sampling.py + _solver_domain.py + _solver_result.py
  -> core/result.py                        immutable result
  -> output.py / audit.py                  stable public facades
     -> _result_*.py / _audit_*.py          formats and audit phases
```

公開 Python API は次の 3 操作だけです。

```python
from plasma_global import load_case, simulate, write_result

case = load_case("case.yaml")
result = simulate(case)  # ファイルを書かない
paths = write_result(result, "runs/case")  # result.h5 + summary.yaml
```

## 責務

| 場所 | 責務 |
|---|---|
| `input/schema.py` | 公開入力型を再exportするstable facadeと、`CaseSpec`全体の相互参照検証 |
| `input/_schema_base.py` / `_schema_{power,reactor,recipe,models,run}.py` | 共通制約と、power・reactor・recipe・物理model・実行制御ごとのstrict/frozen宣言 |
| `input/load.py` | YAML、単一 `include`、宣言元基準の相対 path 解決 |
| `chemistry/data.py` | 公開 chemistry DTO と読込み facade |
| `chemistry/_data_csv.py` / `_data_manifest.py` | CSV の厳格なdecode / manifest・path・外部表の組立て |
| `chemistry/compile.py` | 公開 compiled DTO と immutable 配列の組立て |
| `chemistry/_rate_evaluators.py` / `_compile_validation.py` | rate 数値核 / 化学量論・次数・単位の検証 |
| `input/compile_reactor.py` | 初期組成と reactor compiler の公開 facade |
| `input/_compile_reactor_components.py` | zone、wall/surface、transport の静的構築 |
| `input/compile_recipe.py` | recipe timeline と segment の公開 facade |
| `input/_compile_recipe_transport.py` / `_compile_recipe_power.py` | transport source / power command の固定化 |
| `input/migrate_v2.py` / `_migrate_v2_*.py` | 公開migration facade / power・reactor・recipe・asset変換 |
| `build.py` | `CaseSpec` から `CompiledCase` を一度だけ構築する composition root |
| `_build_electrons.py` / `_build_metadata.py` | electron closure / artifact metadata の組立て |
| `core/compiled.py` | state layout と公開 evaluation facade |
| `core/_runtime_state.py` / `_runtime_electrons.py` / `_runtime_sources.py` | state準備 / electron coupling / 物理source評価 |
| `core/_runtime_assembly.py` | RHS と ledger の唯一の組立て境界 |
| `core/solver.py` | segment BDF の orchestration facade |
| `core/_solver_sampling.py` / `_solver_domain.py` / `_solver_result.py` | sampling grid / accepted-state domain正規化 / history・metadata変換 |
| `models/kinetics.py` / `_kinetics_table.py` | 公開electron lookup / HDF5読込み・不変配列契約 |
| `models/*.py` | 公開物理型と facade。private kernel/coordinator は `_*.py` に分離 |
| `postprocess.py` | 明示選択された observable と通常診断の一回の評価 |
| `core/result.py` | solver 非依存、time-major、read-only の結果契約 |
| `output.py` | HDF5、summary、CSV、plot のstable facade |
| `_result_hdf5.py` / `_result_summary.py` / `_result_csv.py` / `_result_plot.py` | format別のI/O実装 |
| `audit.py` / `_audit_*.py` | audit facade / 分類・保存則・result検査 |
| `experimental/*.py` | 公開opt-in型とfacade。宣言compile・power kernel・film/inventory実行はprivate moduleに分離 |

`core` は `input` と `experimental` を import しません。experimental model も入力時に
標準 model と同じ小さな実行契約へ変換され、RHS には Pydantic model や YAML 辞書を
渡しません。

標準機能は、固定気体温度、準中性・電気的正性、実断面積 Maxwellian と electron-energy
closure、prescribed absorbed power、単一一価正イオンの Bohm floating wall、決定的な
boundary reaction、inlet／pump／inter-zone transport に限定します。別の設定を同じ標準契約へ
押し込まず、実行機能を保ったまま audit で `experimental` と分類します。

## コンパイルと時間積分

`simulate` は渡された `CaseSpec` を入口で再検証して snapshot を作り、呼出し側による読込み後の
入れ子データ変更を hot path へ持ち込みません。`compile_case` は chemistry、外部
rate/EEDF/power/profile table、flow、wall branch、
surface kinetics、step command を読み、配列と immutable object へ変換します。外部ファイル
の ID 解決、sort、static validation は RHS では行いません。

state は zone-major で、各 zone の heavy species density に続いて、選択した closure に
応じた electron energy と gas internal energy を置きます。その後に surface coverage と
experimental accumulator state が続きます。唯一の位置契約は `StateLayout` が所有し、
出力は `state.shape == (n_time, n_state)` の time-major です。

recipe step、square pulse、`external_table` の previous 補間、prescribed electron profile の
不連続時刻は、積分前に連続な forcing segment へ分割します。一つの `solve_ivp` 呼出しは
一つの固定 segment だけを参照するため、境界時刻を次 step と誤認しません。sampling は
積分制御と分離され、未指定時は solver accepted points と全 forcing 境界を保存します。
保存点は入力形式を増やさない固定上限 100,000 点で制限します。

## 結果と診断

`SimulationResult` が公開するのは時刻、state、label、選択済み observable、status、solver
statistics、`series(name)` です。backend、integrator、compiled system は漏らしません。

通常の `run` は fail-fast で、成功時は固定構造の `result.h5` と簡潔な `summary.yaml` だけを
生成します。両ファイルは同じ出力 directory の一時ファイルへ書き、serialize 完了後に
replace します。summary の保存則欄には保存点での charge-closure 最大残差を含めます。`audit` は
同じ case を積分し、静的な元素・電荷・site balance、particle/wall/flow と
electron/heavy energy ledger closure、
`standard` / `experimental` 分類、全入力ファイルの SHA-256 provenance を加えます。

CSV と plot は canonical artifact ではなく、`export` / `plot` の明示操作です。
matplotlib は plot 実行時だけ lazy import されます。

## 拡張の原則

標準 allow-list を広げる変更は、物理閉包と benchmark を同時に検証した場合だけ行います。
標準機能は `models/` の小さな責務として実装し、`build.py` で明示選択します。
経験式、装置固有 closure、精度検証中の状態は `experimental.*` ID と provenance を必須に
します。設定キーだけを増やす、RHS からファイルを読む、単一実装 registry や forwarding
module を作る、暗黙 fallback で別モデルへ置換する、という拡張は行いません。
