# Plasma Global Model

低圧プラズマの反応機構、装置、運転条件から、体積平均の組成とエネルギーの時間変化を計算する
Pythonパッケージです。反応、電子エネルギー、壁・表面、電源、流入出の収支を結合します。
公開入力はYAML、外部数値表はCSVです。Python 3.11以上を対象とします。

**[モデリングマニュアル兼統合仕様書](docs/index.md)** に、適用範囲、初回実行、モデルの作り方、
区分ごとの入力変数の意味・単位・既定値、用途別のモデル選択、HDF5の保存項目・読込方法、
実データと計算図の読み方をまとめています。
初回は[環境準備と実行](docs/index.md#quickstart)、自分の装置へ展開する場合は
[用途別の手順](docs/index.md#task-guides)と[条件比較](docs/index.md#compare-cases)から進めてください。
CLIの引数と終了コードは[操作一覧](docs/index.md#commands)で確認できます。

globalモデルの基礎から学ぶ場合は、[低圧プラズマの体積平均モデル](docs/THEORY.md)を参照してください。
技術的背景と用途、他のモデルとの違い、機能構成と計算フローから支配方程式を導き、
実装の近似、参考文献、19ケースの再計算結果を説明しています。
[完成版PDF](docs/assets/global-model-theory.pdf)も収録しています。

## 1. リポジトリと実行環境を用意する

以下は本資料を収録するブランチの取得例です。GitHubからZIPを取得する場合も、表示中のブランチを選びます。

```bash
git clone --branch agent/modeling-foundation-and-manual https://github.com/dkawa2523/global.git
cd global
python -m venv .venv
```

| 環境 | 仮想環境の有効化 |
|---|---|
| Windows PowerShell | `.\.venv\Scripts\Activate.ps1` |
| Linux／macOS | `. .venv/bin/activate` |

以下のコマンドは、取得したリポジトリのルートで実行します。時系列図・ベンチマーク図に必要なMatplotlibも導入します。

```bash
python -m pip install ".[plot]"
plasma-global init --list
```

一覧には用途別の7例と数値検証用の17ケース、計24ケースが表示されます。
`plasma-global` が見つからない場合は、同じ環境の `python -m plasma_global.cli` へ読み替えます。
PowerShellで有効化できない場合は実行ポリシーを変更せず、
`.\.venv\Scripts\python.exe -m pip install ".[plot]"`、
`.\.venv\Scripts\python.exe -m plasma_global.cli init --list` のように直接実行できます。

## 2. 設定ファイルと必要データの場所

入力の正本は [plasma_global/examples/](plasma_global/examples/) に集約しています。
各ケースは装置・初期状態・運転・物理モデル・出力を記述したcase YAMLであり、共有の機構・数値表を相対パスで参照します。

| 所在 | 内容・利用方法 |
|---|---|
| [cases/](plasma_global/examples/cases/) | 用途別の7つのcase YAML |
| [cases/benchmarks/](plasma_global/examples/cases/benchmarks/) | 解析解、許容差、保存間隔、入力感度の検証用17 YAML |
| [chemistry/](plasma_global/examples/chemistry/) | 化学種・反応機構のYAMLと、電子衝突断面積CSV |
| [data/zdplaskin_example2/electrons.csv](plasma_global/examples/data/zdplaskin_example2/electrons.csv) | ZDPlaskin由来例で使う電子反応・輸送係数のCSV |
| [benchmarks/references/](benchmarks/references/) | 解析結果と比較する保存参照。通常の単独計算の入力とは区別 |
| [benchmarks/raw/](benchmarks/raw/) | 参照データ・電子係数の再生成元。通常の計算で別途配置する必要はない |
| [benchmarks/theory/results.json](benchmarks/theory/results.json) | 文書に掲載した比較結果と実行条件・ハッシュの記録 |
| [tools/benchmarks/theory.py](tools/benchmarks/theory.py) | 19計算、係数積分の比較、誤差集計、10図の生成を行う実行器 |

編集を始めるときは、次節の `init` でcaseと依存データを一緒に取り出します。
外部から入力を追加ダウンロードする必要はありません。Python依存のインストール後は、同梱ベンチマークをネットワークなしで計算できます。
YAML内の `file` は記述元のYAMLを基準に解決するため、取り出したディレクトリ一式を保持してください。

## 3. 最初の計算を実行し、結果から図を作る

基本例 `argon_maxwellian` は、Arの電子衝突・電子エネルギー・壁損失と、電源停止後の応答を計算します。

```bash
plasma-global init argon_maxwellian --output work/argon
plasma-global validate work/argon/case.yaml
plasma-global outputs work/argon/case.yaml
plasma-global run work/argon/case.yaml --output runs/argon-001
plasma-global plot runs/argon-001/result.h5 "electron_density_m3[plasma]" "electron_temperature_eV[plasma]" --output figures/argon-001
```

`validate` は入力とモデルの組立て、`outputs` は利用可能な系列名・単位・保存対象を確認します。
`run` が時間積分を行い、`plot` は保存済みの結果から図を作ります。再描画では時間積分を繰り返しません。
角括弧・コンマを含む系列名は引用符で囲みます。

```text
work/argon/
  case.yaml                               編集の入口
  chemistry/argon_lxcat/chemistry.yaml      化学種・反応・断面積の対応
  chemistry/argon_lxcat/cross_sections.csv  電子衝突断面積
runs/argon-001/
  summary.yaml                            終了状態・終端時刻・指定した最終値
  result.h5                               状態・保存系列・実効入力・来歴
  result.csv                              保存時刻ごとの数値表
  series.yaml                             CSVの列名・単位
figures/argon-001/                         plotが出力したPNG
```

まず `summary.yaml` の `status.success: true`、`status.code: completed`、`time.end_s: 0.002` を確認します。
この例は201保存時刻、終端電子密度は約4.42×10¹³ m⁻³です。図のパスはCLIにも表示されます。
`init`、`run`、`plot` の出力先は未作成のディレクトリを指定します。再実行時は `-002` 等へ変えてください。

## 4. ベンチマークを実行する

### 一つの問題を動かす

最小の例は、一次反応の減衰を解析解と比較する `benchmarks/analytic-decay` です。
反応機構はcase内に含まれるため、この例に外部CSVはありません。

```bash
plasma-global init benchmarks/analytic-decay --output work/decay
plasma-global validate work/decay/case.yaml
plasma-global run work/decay/case.yaml --output runs/decay-001
plasma-global plot runs/decay-001/result.h5 "n[plasma,Ar_star]" --output figures/decay-001
```

この単独実行は時系列、CSV、選択した系列の図を作ります。解析解との差、収束比較、参照値を重ねた図は、次の一式実行で生成します。

| 比較する問題 | `init` に渡す名前・入力 | 必要な共有データ |
|---|---|---|
| 一次減衰 | [`benchmarks/analytic-decay`](plasma_global/examples/cases/benchmarks/analytic-decay.yaml) | 反応をcase内に記述 |
| 異体積間の移送・粒子保存 | [`benchmarks/analytic-transport`](plasma_global/examples/cases/benchmarks/analytic-transport.yaml) | 種と移送条件をcase内に記述 |
| Arの許容差収束 | `benchmarks/argon-rtol-1e-6`、`benchmarks/argon-rtol-1e-7`、`benchmarks/argon-rtol-1e-8` | `chemistry/argon_lxcat/` のYAML・断面積CSV |
| 電力の±10%感度 | `benchmarks/argon-power-minus10`、`benchmarks/argon-power-plus10` | 同上 |
| 初期圧力の±10%感度 | `benchmarks/argon-pressure-minus10`、`benchmarks/argon-pressure-plus10` | 同上 |
| 壁面積の±10%感度 | `benchmarks/argon-area-minus10`、`benchmarks/argon-area-plus10` | 同上 |
| 保存間隔とイベント積分 | `benchmarks/argon-sampling-40us`、`benchmarks/argon-sampling-20us`、`benchmarks/argon-sampling-10us`、`benchmarks/argon-sampling-5us`、`benchmarks/argon-sampling-2.5us`、`benchmarks/argon-sampling-1.25us` | 同上 |
| CRANE由来の2反応Ar | [`crane_two_reaction_argon`](plasma_global/examples/cases/crane_two_reaction_argon.yaml) | `chemistry/crane_two_reaction_argon/chemistry.yaml` |
| ZDPlaskin由来の化学・DC回路 | [`zdplaskin_example2`](plasma_global/examples/cases/zdplaskin_example2.yaml) | `chemistry/zdplaskin_example2/chemistry.yaml`、`data/zdplaskin_example2/electrons.csv` |

実際に渡せる名前は `init --list` で確認できます。17の検証用YAMLとCRANE・ZDPlaskinの2例が、Theoryの19 runに対応します。
Maxwellian係数積分の試験は実行器が断面積CSVから別途行い、19 runの数には含めません。

### Theory掲載の比較と10図をまとめて再計算する

実行器と保存参照を含むリポジトリのルートで実行します。Matplotlib以外の描画依存やGraphviz本体は、この操作には不要です。

```bash
python -m tools.benchmarks.theory --output runs/theory-001
```

```text
runs/theory-001/
  results.json                  各問題の条件、比較値、誤差、来歴
  figures/theory-maxwell.png     Maxwellian係数の積分比較
  figures/theory-decay.png       一次減衰と解析解
  figures/theory-transport.png   移送と粒子保存
  figures/theory-crane.png       CRANE保存終端値との比較
  figures/theory-dc-species.png  DC例の組成と保存参照
  figures/theory-dc-circuit.png  DC回路の電圧・電流・換算電場
  figures/theory-transient.png   Arの過渡応答
  figures/theory-tolerances.png  ODE許容差による波形差
  figures/theory-sampling.png    保存間隔とイベント積分の誤差
  figures/theory-sensitivity.png 電力・圧力・壁面積の局所感度
  runs/<case名>/result.h5        19計算それぞれの保存結果
  runs/<case名>/summary.yaml     19計算それぞれの終了状態
```

`results.json` の `case_yaml` はリポジトリを基準とする入力パス、`result_h5` は `runs/theory-001/` を基準とする結果パスです。
一式の再実行には別の出力先を指定します。保存結果の系列を描き直す場合は、例えば次を実行します。

```bash
plasma-global plot runs/theory-001/runs/argon-rtol-1e-6/result.h5 "electron_density_m3[plasma]" --output figures/theory-argon-001
```

単独ケースはインストール済みパッケージから実行できます。一式の比較には、上記のリポジトリ内ツールと保存参照も必要です。
問題設定、誤差の定義、図の読み方、保存参照による比較の制限は[Theory第12〜14節](docs/THEORY.md#verification)に記しています。

## 5. 組成図と反応ネットワークも出力する

`argon_analysis` は、基本Ar例と同じ物理条件に、probe・CSV・組成図・Graphviz反応図を設定した例です。
Python側の依存を追加し、[Graphviz本体](https://graphviz.org/download/)を導入して `dot -V` が起動することを確認します。

```bash
python -m pip install ".[plot,graph]"
dot -V
plasma-global init argon_analysis --output work/analysis
plasma-global run work/analysis/case.yaml --output runs/analysis-001
```

`runs/analysis-001/plots/` に密度・分率・電子状態のPNG、`networks/` に機構とイベントのSVGが生成されます。
保存済みの設定から図を作り直す場合は、次を実行します。

```bash
plasma-global plot runs/analysis-001/result.h5 --output figures/analysis-redraw-001
```

出力量・probe・図の設定は取り出した `case.yaml` の `output` にまとめてあります。
系列の追加、反応図の生成、出力HDF5の解析方法は[マニュアル第11〜14節](docs/index.md#outputs)を参照してください。

## Pythonから使う

```python
from plasma_global import load_case, simulate, write_result

case = load_case("work/argon/case.yaml")
result = simulate(case)
paths = write_result(result, "runs/argon-python-001")
```

simulateは参照データの読込・モデルの組み立て・計算を行い、結果ファイルは作成しません。
write_resultが要求した出力を新しいrunディレクトリへ保存します。

zone内の空間分布、電磁場、流体場、計測器応答は解きません。接続したzoneは区画モデルで、
1次元の移流拡散方程式を自動的に離散化する機能はありません。
status.successは計算の終了状態、standard／experimentalは機能構成の分類です。
装置の予測精度は対象条件で別途評価します。

[開発・保守](docs/DEVELOPMENT.md)には内部構造と検証手順、
[計画と残る課題](docs/ROADMAP.md)には現在地と次の作業をまとめています。
