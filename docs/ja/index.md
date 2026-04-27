# Plasma Global Model 概要

このリポジトリは、低圧プラズマを 0D/global model として計算するための研究用コードです。チャンバー、ガス供給、圧力、投入電力、反応機構、電子衝突レート、表面反応を入力し、種密度、電子エネルギー、表面状態、壁 flux の時間発展を stiff ODE として解きます。

位置づけとしては、装置内の空間分布を詳細に解く PIC、CFD、2D/3D fluid code ではありません。空間分布を平均化し、反応、輸送、電力結合、壁損失、表面過程が全体の plasma balance にどう効くかを調べるための global model です。

## 計算の対象

1 回の計算では、`case YAML` が chamber、recipe、chemistry bundle、physics backend を束ねます。コードはそれらを検証し、global ODE system を組み立て、時系列の solution と diagnostics を出力します。

```mermaid
flowchart LR
  A[case YAML] --> B[chamber / recipe / chemistry]
  B --> C[model assembly]
  C --> D[global plasma ODE]
  D --> E[summary / observables / budgets]
  E --> F[benchmark / review]
```

主に扱う物理量は、gas species、charged species、electron energy、surface coverage、wall flux、absorbed power、reduced field です。反応機構を変えた時の支配反応、電力条件を変えた時の電子密度応答、壁損失や表面反応が particle balance に与える影響を確認できます。

出力は用途ごとに分かれています。`summary.yaml` は最終値と実行概要、`observables.csv` は電子密度や電力などの時系列、`solution.h5` は raw state、各種 budget file は反応・エネルギー・表面過程の寄与を確認するためのものです。

## 現在確認済みの範囲

検証は、厳密比較、反応 ODE の切り出し検証、外部 global model との sanity check、stress 条件での robustness check に分けて行っています。

| 検証 | 確認している内容 | 現在の結論 |
|---|---|---|
| ZDPlaskin example2 | Ar DC-resistor discharge の final species、E/N、電流、電力 | same-footing 条件では final state が数 % 程度で一致 |
| CRANE TwoReactionArgon | 2 反応 Ar scalar ODE、単位変換、stiff integration | `Ar`, `Ar+`, `e` の final density が小さい相対誤差で一致 |
| PyGMol Ar | pure-Ar global model との実行比較 | strict parity ではなく、rate、power、wall loss の差を診断する比較として利用 |
| robustness sweep | 圧力、電圧、chemistry family を動かした時の数値的健全性 | 選んだ stress cases では density と energy が有限・正で、E/N table coverage も確認済み |

この結果から言えるのは、限定された条件では、反応式の組み立て、単位変換、ODE 積分、簡約 DC circuit、主要 diagnostics が整合して動作しているということです。一方で、すべての圧力、ガス組成、電源方式、表面条件での予測精度が確認済みという意味ではありません。

## まだ不足している検証

現時点で不足しているのは、実行機能そのものよりも、適用範囲を支える reference data です。特に次の領域は、外部 solver、高忠実度計算、または実験データとの比較が必要です。

- ZDPlaskin base point 以外の pressure / voltage 条件。
- RF、ICP、CCP の power coupling と sheath / wall model。
- Ar 以外の mixed gas chemistry。
- surface reaction、wafer flux、IED proxy。
- cross-section data と rate table の不確かさを含む感度評価。
- process prediction として使う場合の calibration 方針。

したがって、このコードは現時点では、検証済み範囲を明示しながら global balance を調べる研究用モデルとして使うのが適切です。未知条件で絶対値の予測精度を主張するには、追加 benchmark と calibration が必要です。

## 読む順序

初めてコードを確認する場合は、次の順に読むと全体像を追いやすくなります。

| 順番 | ページ | 内容 |
|---:|---|---|
| 1 | [全体像](OVERVIEW.md) | モデルの考え方、解いている式、扱う物理量 |
| 2 | [モデルの入力と出力](MODEL_INPUT_OUTPUT.md) | 入力ファイルと出力ファイルの対応 |
| 3 | [全体ワークフロー](END_TO_END_WORKFLOW.md) | validate、run、diagnostics、benchmark の流れ |
| 4 | [外部ベンチマーク](EXTERNAL_BENCHMARKS.md) | 検証済み範囲と benchmark の読み分け |
| 5 | [モデル成熟度](MODEL_MATURITY.md) | backend ごとの成熟度と注意点 |
| 6 | [コードモジュール一覧](CODE_MODULES.md) | 実装を読むときの入口 |

## 詳細ページ

導入と使い方:

- [用語集と最小物理背景](GLOSSARY_AND_MINIMAL_PHYSICS.md)
- [全体像](OVERVIEW.md)
- [モデルの入力と出力](MODEL_INPUT_OUTPUT.md)
- [全体ワークフロー](END_TO_END_WORKFLOW.md)
- [1 ケースの読み解き](QUICKSTART_CASE_WALKTHROUGH.md)
- [入力スキーマ早見表](SCHEMA_REFERENCE.md)
- [出力の読み方](OUTPUT_READING_GUIDE.md)
- [state vector の読み方](STATE_VECTOR_GUIDE.md)
- [observables.csv 列辞書](OBSERVABLES_COLUMN_REFERENCE.md)

モデル設定と物理:

- [設定ガイド](CONFIG_GUIDE.md)
- [化学入力仕様](CHEMISTRY_INPUT_SPEC.md)
- [物理モデルと近似](PHYSICS_MODELS_AND_APPROXIMATIONS.md)
- [物理式・記号・実装対応表](PHYSICS_EQUATION_REFERENCE.md)
- [表面・ウェハ・IED ガイド](SURFACE_WAFER_IED_GUIDE.md)
- [回路結合](CIRCUIT_COUPLING.md)
- [RF envelope 較正](RF_ENVELOPE_CALIBRATION.md)
- [数値解法とソルバー](NUMERICS_AND_SOLVERS.md)
- [モデル成熟度](MODEL_MATURITY.md)
- [入出力仕様](IO_SPEC.md)

コード構造と検証:

- [コードモジュール一覧](CODE_MODULES.md)
- [アーキテクチャ](ARCHITECTURE.md)
- [core 分割リファクタリング](CORE_SPLIT_REFACTOR.md)
- [開発者ガイド](DEVELOPER_GUIDE.md)
- [外部ベンチマーク](EXTERNAL_BENCHMARKS.md)
- [ベンチマーク判定基準](BENCHMARK_ACCEPTANCE_GUIDE.md)
- [ZDPlaskin 比較](ZDPLASKIN_COMPARISON.md)
- [CRANE 比較](CRANE_COMPARISON.md)
- [Ar LXCat 基準ケース](ARGON_LXCAT_CASE.md)
- [Ar 実行比較](ARGON_EXECUTABLE_COMPARISON.md)
- [ロバスト性ベンチマーク](ROBUSTNESS_BENCHMARKS.md)
