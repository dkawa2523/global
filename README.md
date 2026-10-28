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

## 最初の計算

リポジトリのルートでインストールし、入力データを含む例を新しい場所へ取り出します。

```bash
python -m pip install .
plasma-global init --list
plasma-global init argon_maxwellian --output work/argon
plasma-global validate work/argon/case.yaml
plasma-global run work/argon/case.yaml --output runs/argon-001
```

この基本例は `result.h5`、`summary.yaml`、解析用の `result.csv` と列定義の `series.yaml` を出力します。
YAMLのoutputに名前付きprobe、組成の時間変化、
Graphvizによる反応図を追加できます。図付きの例は `argon_analysis` です。
Matplotlibには `python -m pip install ".[plot]"`、反応図も使う場合は
`python -m pip install ".[plot,graph]"` とGraphviz本体のdotが必要です。
導入・再描画の手順は[マニュアル](docs/index.md#quickstart)を参照してください。

7つの用途別例と17の検証用ケースには、実行に必要なYAML／CSVを同梱しています。
正本は [plasma_global/examples/](plasma_global/examples/) に集約し、インストール後も
`init --list` で選び、`init benchmarks/analytic-decay --output work/decay` のように取り出せます。
各ケースの用途とベンチマーク一式の再計算は[マニュアル](docs/index.md#examples)で説明します。

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
