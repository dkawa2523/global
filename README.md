# Plasma Global Model

低圧プラズマの反応機構、装置、運転条件から、体積平均の組成とエネルギーの時間変化を計算する
Pythonパッケージです。完全混合の1区画、または区画を接続したmulti-zoneモデルを扱います。
公開入力はYAML、外部数値表はCSVです。Python 3.11以上を対象とします。

**[モデリングマニュアル兼統合仕様書](docs/index.md)** に、適用範囲、初回実行、モデルの作り方、
入力、物理式、出力、実際の計算図、結果の読み方をまとめています。

globalモデルの基礎から学ぶ場合は、[理論と数値検証](docs/THEORY.md)を参照してください。
保存則からの導出、実装の近似、参考文献、19ケースの再計算結果を論文形式で説明しています。
[完成版PDF](docs/assets/global-model-theory.pdf)も収録しています。

## 最初の計算

リポジトリのルートでインストールし、入力データを含む例を新しい場所へ取り出します。

```bash
python -m pip install .
plasma-global init argon_maxwellian --output work/argon
plasma-global validate work/argon/case.yaml
plasma-global run work/argon/case.yaml --output runs/argon-001
```

結果は `result.h5` と `summary.yaml` です。YAMLのoutputにCSV、名前付きprobe、組成の時間変化、
Graphvizによる反応図を追加できます。図付きの例は `argon_analysis` です。
Matplotlibには `python -m pip install ".[plot]"`、反応図も使う場合は
`python -m pip install ".[plot,graph]"` とGraphviz本体のdotが必要です。
導入・再描画の手順は[マニュアル](docs/index.md#quickstart)を参照してください。

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
