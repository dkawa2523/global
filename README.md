# Plasma Global Model

case入力はschema v5、化学機構はschema v4、保存結果はHDF5 format v4です。旧版の互換読込や移行コマンドは提供しません。
電子係数はMaxwell分布から評価するか、生成条件を明示した外部CSVを使います。
Swarmは電子係数、外部回路は電力・電圧の指令時系列を供給し、Globalは粒子・電荷・エネルギー収支を計算します。

低圧プラズマの反応機構、装置、運転条件から、体積平均の組成とエネルギーの時間変化を計算する
Pythonパッケージです。反応、電子エネルギー、壁・表面、電源、流入出の収支を結合します。
設定はYAMLとPythonの同じ公開データ型 `CaseSpec` で扱い、外部数値表はCSVで渡します。Python 3.11以上を対象とします。

**[モデリングマニュアル兼統合仕様書](docs/index.md)** に、適用範囲、初回実行、モデルの作り方、
区分ごとの入力変数の意味・単位・既定値、用途別のモデル選択、HDF5の保存項目・読込方法、
実データと計算図の読み方をまとめています。
初回は[環境準備と実行](docs/index.md#quickstart)、自分の装置へ展開する場合は
[用途別の手順](docs/index.md#task-guides)と[条件比較](docs/index.md#compare-cases)から進めてください。
CLIの引数と終了コードは[操作一覧](docs/index.md#commands)で確認できます。

globalモデルの基礎から学ぶ場合は、[低圧プラズマの体積平均モデル](docs/THEORY.md)を参照してください。
技術的背景と用途、他のモデルとの違い、機能構成と計算フローから支配方程式を導き、
実装の近似、参考文献、20ケースの再計算と後期残光の測定比較を説明しています。
[完成版PDF](docs/assets/global-model-theory.pdf)も収録しています。

## 1. リポジトリと実行環境を用意する

リポジトリを取得し、仮想環境を作ります。特定の版を使う場合は、取得後に対象のブランチまたはタグへ切り替えてください。

```bash
git clone https://github.com/dkawa2523/global.git
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

一覧には用途別の例と数値検証用のケースが表示されます。
`plasma-global` が見つからない場合は、同じ環境の `python -m plasma_global.cli` へ読み替えます。
PowerShellで有効化できない場合は実行ポリシーを変更せず、
`.\.venv\Scripts\python.exe -m pip install ".[plot]"`、
`.\.venv\Scripts\python.exe -m plasma_global.cli init --list` のように直接実行できます。

## 2. 設定ファイルと必要データの場所

入力の正本は [plasma_global/examples/](plasma_global/examples/) に集約しています。
各ケースは装置・初期状態・運転・物理モデル・出力を記述したcase YAMLです。化学機構をcase内に記述するか、共有の機構・数値表を相対パスで参照します。

| 所在 | 内容・利用方法 |
|---|---|
| [cases/](plasma_global/examples/cases/) | 用途別のcase YAML |
| [cases/benchmarks/](plasma_global/examples/cases/benchmarks/) | 解析解、許容差、保存間隔、入力感度、後期残光の比較用YAML |
| [chemistry/](plasma_global/examples/chemistry/) | 化学種・反応機構のYAMLと、電子衝突断面積CSV |
| [data/](plasma_global/examples/data/) | ZDPlaskin由来例とSwarm由来例で使う電子反応・輸送係数のCSV |
| [benchmarks/references/](benchmarks/references/) | 解析結果と比較する保存参照。通常の単独計算の入力とは区別 |
| [DC比較の入力来歴](benchmarks/references/zdplaskin_example2_source.yaml) | 元のFortran・参照波形・断面積と、独立に生成した電子係数表の対応 |
| [Ar後期残光の測定参照](benchmarks/references/beckers_argon_afterglow.yaml) | 15 Paの公開測定図から読み取った8時刻の密度、観測の定義、移動度の出典 |
| [benchmarks/theory/results.json](benchmarks/theory/results.json) | 文書に掲載した比較結果と実行条件・ハッシュの記録 |
| [tools/benchmarks/theory.py](tools/benchmarks/theory.py) | 20計算、係数積分の比較、誤差集計、11図の生成を行う実行器 |

モデルの用途、設定場所、必須項目・既定値、適用条件、対応する例は、入力を作る前に確認できます。

```bash
plasma-global models
plasma-global models power
plasma-global models power experimental.ccp
```

Pythonでは `from plasma_global import list_models` とし、`list_models("power", "experimental.ccp")` で同じ説明を辞書として取得できます。
対応例にはそのモデルを実際に使う例と、設定を書き換えるための雛形があり、一覧の詳細で区別します。

| 小さく始める目的 | `init` に渡す名前 | 確認する量 |
|---|---|---|
| 気体加熱・流入・排気・壁への熱緩和 | `gas_heating_flow` | 気体温度、総粒子密度、エネルギー収支 |
| 吸着・脱離と表面への蓄積 | `surface_inventory` | 被覆率、膜厚、粒子在庫、気相と表面を含む原子数 |
| 電子密度の外部指定 | `prescribed_density` | 入力した密度、回路応答、電荷残差。入力波形は合成データ |

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
plasma-global inspect runs/argon-001/result.h5
plasma-global audit runs/argon-001/result.h5
plasma-global plot runs/argon-001/result.h5 "electron_density_m3[plasma]" "electron_temperature_eV[plasma]" --output figures/argon-001
```

`validate` は入力とモデルの組立て、`outputs` は利用可能な系列名・単位・保存対象を確認します。
`run` が時間積分を行います。`inspect` は保存した終了状態・系列・単位・来歴を表示し、`audit` は保存結果の整合を検査します。
`plot` も保存結果から図を作るので、これらの操作で時間積分を繰り返しません。`audit` の合格は装置に対する予測精度の証明ではありません。
角括弧・コンマを含む系列名は引用符で囲みます。

```text
work/argon/
  case.yaml                               編集の入口
  chemistry/argon_lxcat/chemistry.yaml      化学種・反応・断面積の対応
  chemistry/argon_lxcat/cross_sections.csv  電子衝突断面積
runs/argon-001/
  summary.yaml                            終了状態・終端時刻・指定した最終値
  result.h5                               状態・保存系列・実効入力・来歴
  artifacts.yaml                          CSV・図など依頼した出力の成否
  csv/result.csv                          保存時刻ごとの数値表（output.csv: true）
  csv/series.yaml                         CSVの列名・単位
figures/argon-001/                         plotが出力したPNG
```

まず `summary.yaml` の `status.success: true`、`status.code: completed`、`time.end_s: 0.002` を確認します。
この例は0〜2 msを10 µs間隔で保存します。図のパスはCLIにも表示されます。
`init`、`run`、`plot` の出力先は未作成のディレクトリを指定します。再実行時は `-002` 等へ変えてください。

`output.profile` の既定値は `basic` です。電子密度・電子温度・平均電子エネルギー・気体温度・吸収電力・E/N・電荷残差・圧力・中性粒子密度・重粒子密度の10量から、全運転区間で定義できる系列を保存します。
選択モデルで定義されない系列は省略理由を表示します。例えば、吸収電力の指定だけではE/Nは求まりません。
自動選択が不要なら `output.profile: minimal` とし、保存したい派生系列を明示します。積分状態はどちらでも保存されます。
`outputs` で利用可能性と保存対象を確認でき、明示要求した未定義系列は計算開始前にエラーになります。
気体内部エネルギーを時間積分する設定では、壁・表面との粒子交換を含む8項で収支を照合します。
各項の符号・加算範囲と保存後の確認方法は[出力設定](docs/index.md#outputs)を参照してください。

数値結果の `result.h5` と `summary.yaml` を先に保存し、その後に依頼したCSV・図・反応ネットワークを生成します。
各依頼の成否と生成ファイルは `artifacts.yaml` に残るため、数値計算の成功と出力処理の成功を別々に確認できます。

| `run` の終了コード | 意味 |
|---|---|
| `0` | 数値計算と依頼した出力が成功 |
| `1` | 数値積分が失敗。取得できた部分結果は保存 |
| `2` | 入力・モデル準備・数値結果の保存などでエラー |
| `3` | 数値計算は成功し、その後のCSV・描画などに失敗 |

電子係数を外部計算から取り込む例は `swarm_argon_dc` です。係数CSVも取り出されます。

```bash
plasma-global init swarm_argon_dc --output work/swarm-argon
plasma-global run work/swarm-argon/case.yaml --output runs/swarm-argon-001
plasma-global plot runs/swarm-argon-001/result.h5 "probe[circuit,current_A]" --output figures/swarm-argon-001
```

取り込んだ係数とDC応答の確認方法、外部回路と分光への接続は[開発・保守](docs/DEVELOPMENT.md#external-models)を参照してください。

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
| ZDPlasKin由来の化学・DC回路 | [`zdplaskin_example2`](plasma_global/examples/cases/zdplaskin_example2.yaml) | `chemistry/zdplaskin_example2/chemistry.yaml`、断面積から生成した `data/zdplaskin_example2/electrons.csv` |
| Ar 15 Paの後期残光と測定 | [`benchmarks/argon-afterglow`](plasma_global/examples/cases/benchmarks/argon-afterglow.yaml) | 単独計算はcaseのみ。比較には `benchmarks/references/beckers_argon_afterglow.yaml` |

実際に渡せる名前は `init --list` で確認できます。18の検証用YAMLとCRANE・ZDPlasKinの2例が、Theoryの20 runに対応します。
Maxwellian係数積分の試験は実行器が断面積CSVから別途行います。
DC比較は参照波形の303時刻、Ar後期残光は測定図の8時刻で比較します。 [DC参照の再生成](docs/DEVELOPMENT.md#dc-benchmark-data) · [Ar後期残光の設定と比較結果](docs/THEORY.md#argon-afterglow-benchmark)

### Theory掲載の比較と図をまとめて再計算する

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
  figures/theory-argon-afterglow.png  Ar後期残光の測定と規格化密度
  runs/<case名>/result.h5        各計算の保存結果
  runs/<case名>/summary.yaml     各計算の終了状態
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

`runs/analysis-001/plots/<設定名>/` 以下に密度・分率・電子状態のPNG、
`networks/<設定名>/` 以下に機構とイベントのSVGが生成されます。依頼ごとの成否は `artifacts.yaml` で確認します。
保存済みの設定から図を作り直す場合は、次を実行します。

```bash
plasma-global plot runs/analysis-001/result.h5 --output figures/analysis-redraw-001
```

出力量・probe・図の設定は取り出した `case.yaml` の `output` にまとめてあります。
系列の追加、反応図の生成、出力HDF5の解析方法は[マニュアル第11〜14節](docs/index.md#outputs)を参照してください。

## Pythonで設定を編集し、準備・計算・保存する

YAML読込とPythonで作る設定は、同じ `CaseSpec` から同じ計算経路を使います。
次の例は、取り出したArケースの電力を変え、準備した入力から計算して保存結果を読み戻します。

```python
from plasma_global import (
    case_from_mapping,
    compile_case,
    load_case,
    simulate,
    write_result,
)
from plasma_global.output import read_result_h5

original = load_case("work/argon/case.yaml")
data = original.to_dict()
data["run"]["steps"][0]["commands"]["power_ports"]["source"]["absorbed_power_W"] = 0.08
case = case_from_mapping(data, base_directory=original.base_directory)

prepared = compile_case(case)  # 参照データを読み、モデル・保存対象・来歴を確定
result = simulate(prepared)
paths = write_result(result, "runs/argon-python-001")
saved = read_result_h5(paths.result_h5)
print(saved.status.code, saved.time_s[-1])
print(paths.artifacts.status)
```

`case_from_mapping` は相対パスの基準となる `base_directory` が必須です。
`case.to_yaml()` で公開形式へ書き戻せます。別の場所へ保存する場合は `relative_to=保存先ディレクトリ` を指定します。
`compile_case` は `PreparedCase` を返し、`simulate(case)` はこの準備も含めて実行します。
`simulate(prepared)` は準備時に読み込んだ入力を使うため、同じ準備結果を再利用できます。
計算とファイル保存は分かれており、`simulate` 自体は結果ファイルを作りません。

数値積分に失敗すると `IntegrationError` が送出されます。例外の `partial_result` がある場合は、到達した時刻までの結果を保存できます。

```python
from plasma_global.errors import IntegrationError

try:
    result = simulate(prepared)
except IntegrationError as error:
    if error.partial_result is not None:
        write_result(error.partial_result, "runs/argon-python-failed-001")
    raise
```

`write_result` の戻り値 `ResultPaths` には数値結果のパスと `artifacts` があり、`paths.artifacts.success`、
`paths.artifacts.requests` で依頼ごとの出力成否を確認できます。後からの検査・図の生成には保存したHDF5を使います。

zone内の空間分布、電磁場、流体場、計測器応答は解きません。接続したzoneは区画モデルで、
1次元の移流拡散方程式を自動的に離散化する機能はありません。
`status.success` は数値計算の終了状態です。選択モデルの近似、係数表の生成条件・適用範囲、
省略した物理項を確認し、装置に対する予測精度は対象条件で別途評価してください。

[開発・保守](docs/DEVELOPMENT.md)には内部構造と検証手順、
[計画と残る課題](docs/ROADMAP.md)には現在地と次の作業をまとめています。
