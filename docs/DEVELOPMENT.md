# 開発・保守

利用者向けの機能・物理式・設定・出力の定義は[統合マニュアル](index.md)を参照します。
ここには実装を継続して改良するための構造、検証、データ保守の手順だけを置きます。

## 実装の構造

```text
公開YAML（CaseSpec）
  → path・ID・指令・時間原点を具体化（ResolvedCase）
  → chemistry・geometry・forcing・model・output planをcompile
  → CompiledGlobalModelをBDFで積分
  → SimulationResult
  → HDF5・summary・要求されたCSV／図を新しいrunへ公開
```

ResolvedCaseは内部表現です。別の公開入力形式として説明・配布しません。
`load_case` は公開YAMLを読み、`simulate` が参照データの読込とcompile・計算を行います。
simulateは結果をディスクへ書かず、`write_result` が保存を担当します。

| 所在 | 責務 |
|---|---|
| `input/schema.py`, `input/_schema_*.py` | 公開構造と共用する型・値の定義 |
| `input/_resolve.py`, `input/_serialize.py` | 公開入力と具体化済み内部設定の対応 |
| `chemistry/` | 化学種・反応・断面積の読込、検査、速度式のcompile |
| `models/` | 電子、電源、壁、表面等の物理評価 |
| `core/` | 状態配置、輸送、モデルの合成、時間積分、結果型 |
| `build.py`, `_build_*.py` | 各機能を組み立てるcomposition root |
| `_output_plan.py`, `_output_values.py`, `postprocess.py` | 要求系列の解決、保存状態からの派生量、イベント積分 |
| `output.py`, `_result_*.py` | 保存・読込・要約・CSV・図の共通入口と実装 |
| `api.py`, `cli.py` | 操作の入口。独自の物理計算を持たない |

coreはinput・orchestrationを、chemistryはruntime modelを、modelsはcore・orchestrationを
直接参照しません。import-linterでこの境界を検査します。
新しい物理項は、必要な入力・単位・保存量・符号・適用範囲を定義して既存モデルへ接続します。
同じ意味の既定値や検査をCLIとAPIに別々に実装しません。

CSVの形状・数値・列対応は `_numeric_csv` に共通化しています。軸、反応次数、電力分配、
適用範囲の検査は各domainへ置きます。結果の派生量はRHSと同じ評価・ledgerを使い、
図を追加するためにODEの次元や適応stepを変更しません。

## 文書と例の保守

説明は次の4ファイルにまとめます。細分化した機能説明や過去の設計提案を追加しません。

| ファイル | 役割 |
|---|---|
| `README.md` | パッケージの概要、初回コマンド、マニュアルへの案内 |
| `docs/index.md` | 利用者向けマニュアル兼統合仕様書 |
| `docs/DEVELOPMENT.md` | 内部構造と保守・検証手順 |
| `docs/ROADMAP.md` | 現在地、未解決事項、次の作業 |

入力の機械可読な定義と構造確認用の例はコードから生成します。

```bash
uv sync --frozen --all-extras
uv run --frozen --all-extras python -m tools.generate_schema_docs
uv run --frozen --all-extras mkdocs build --strict
uv run --frozen --all-extras mkdocs serve
```

生成物は `docs/generated/case.schema.json` と `minimal_case.yaml` です。
生成の再現性と公開schemaへの適合を `tests/test_schema_docs.py` で確認します。
手書きの項目一覧をコードから独立した厳密schemaとして扱いません。

実行例の正本は `plasma_global/assets/examples/` で、wheelにも同梱します。
別のexamplesツリーに同じcaseを複製しません。`init` が必要な資産だけを利用者の場所へコピーします。
本文のYAMLは完全なcaseか、適用先を明記した断片かを区別します。

マニュアルの図は `argon_analysis` の `run/plots/plasma.png` と `run/networks/events.svg` を
`docs/assets/argon-plasma.png`、`argon-events.svg` として収録しています。
更新時は新しい入力・runディレクトリへ `init`／`run` して図と数値を確認し、
対応する物理条件や終端値の説明も照合します。図に合わせて入力を別管理しません。

## 検証の実行

```bash
uv run --frozen --all-extras python -m tools.quality fast
uv run --frozen --all-extras python -m tools.quality pr
uv run --frozen --all-extras python -m tools.quality nightly
```

fastは変更Pythonのformat／lint、型、通常テストです。PR gateは全体のlint・型・import構造、
通常テスト、coverage、複雑度、dead code、依存脆弱性、秘密情報、追加抑制の検査を行います。
文書は別途 `mkdocs build --strict` で確認します。
nightlyはPR gateに加えてseedを変えたproperty test、stress、mutationを行います。
Linux／WSLでは `nightly --native`、Windowsでは通常のnightly入口がWSLへ委譲します。

coverage.pyのstatement＋branch統合指標は85%以上、変更した本番コードは90%以上、
新規コードの循環的複雑度は10以下です。Ruff・Vultureの新規抑制や、skip／xfail等を
追加して失敗を隠しません。内部class配置を固定する検査や、同じ式の恒等式だけを検査するテストは増やしません。
単位、保存則、解析解、独立積分、forcing境界、入出力の復元を検査します。

`quality/baseline.json` はレビュー済みの既存Bandit・複雑度例外、`.secrets.baseline` は
detect-secretsのレビュー情報です。通常gateはこれらを書き換えません。
`python -m tools.quality baseline` は例外を精査して更新するための明示操作です。
新規・悪化したBandit／複雑度例外は拒否されます。秘密情報候補の追加は確認済み `is_secret: false` が必須で、
公開checksumには理由を記録します。未確認・実秘密を例外へ登録しません。

CIはLinuxのPython 3.11／3.12／3.14、Windowsの3.12を対象とし、別jobで依存下限を検査します。
Linux 3.12ではGraphviz本体を導入して解析例を描画します。最低構成のjobはwheelをinstallし、
`python -I` でcheckoutをimportしない実行も確認します。
ローカルでのWSL成功をWindowsでの実行成功と読み替えません。

## 比較データと校正ツール

| 資産 | 役割 |
|---|---|
| `benchmarks/references/` | 公開別コードの保存済みsummaryと回路系列 |
| `benchmarks/raw/zdplaskin_example2_eovern/` | 元出力由来のCSVと出典・生成条件 |
| `plasma_global/assets/examples/data/zdplaskin_example2/electrons.csv` | runtimeが直接読む係数CSVの正本 |

```bash
python -m tools.benchmarks.build_zdplaskin_rate_table --output new-electrons.csv
python -m tools.calibration.rf_envelope --help
```

係数builderはネットワーク接続を使わず、元gridを明示的にsortして一致を確認します。
補間や欠損補完はせず、新しいCSVへ出します。テストは再生成物と同梱物のchecksum・内容を比較します。
HDF5を入力表にする旧作成手順はありません。

RF校正ツールは測定量から係数候補とv4用の `power_model`／`power_command` 断片を出します。
測定点以外での精度認定や、自動的なcase全体の生成とは区別します。
別コード回帰の閾値と意味は[マニュアルの検証](index.md#validation)へ集約しています。

過去の収束・局所感度の数値を将来の版へ無条件に転用せず、物理や既定値を変えた場合は同じ定義で再計算します。
報告には観測量、比較時刻、誤差指標、条件差、入力・code hash、終了状態を残します。

## 配布とローカル環境の注意

wheelは新しいcheckoutまたはクリーンなsource copyから作り、install後に7つの同梱例を検査します。
古いbuildツリーを再利用すると削除したmoduleが混入し得るため、配布物の内容も確認します。
図の再生成は保存HDF5だけを使い、元caseの再計算で系列・単位・来歴を補完しません。

この作業端末ではWindowsのh5py DLLがアプリケーション制御で制限され、数値・保存の検証はWSLで実施しました。
また、以前の削除要求が自動承認審査で拒否された古い `build/`、`mutants/`、`.mypy_cache/` と
次の未参照入力HDF5が残っています。別手段で削除制限を回避しません。

- `examples/v3/chemistry/zdplaskin_example2/tables/zdplaskin_example2_eovern_rates.h5`
- `tests/fixtures/v2/chemistry_zdplaskin_example2/tables/zdplaskin_example2_eovern_rates.h5`

現行runtime・wheel・CI入力にはこれらを使いません。ローカル残置と現行仕様を混同しないでください。
