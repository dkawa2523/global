# Plasma Global Model

反応機構、装置、レシピを入力として、低圧プラズマの体積平均状態を計算する
0D／multi-zone global model です。空間分布を解く流体・PIC・電磁界ソルバーでは
ありません。定量利用には対象装置での検証が必要です。

Python 3.11 以上を対象とします。

```bash
python -m pip install -e ".[dev]"
```

公開 Python API は、読み込み・計算・保存の3操作だけです。

```python
from plasma_global import load_case, simulate, write_result

case = load_case("case.yaml")
result = simulate(case)
paths = write_result(result, "runs/case")
```

`write_result` は常に `result.h5` と `summary.yaml` を作成します。

通常利用の標準範囲は、固定気体温度、準中性・電気的正性、実断面積を使う
Maxwellian electron-energy closure、prescribed absorbed power、単一一価正イオンの
Bohm floating wall、生成物が一意な boundary reaction、inlet／pump／zone transport です。
`plasma-global audit` は、この allow-list 内を `standard`、それ以外を `experimental` と
分類します。experimental 機能も明示指定すれば実行できますが、装置予測精度は主張しません。

CLI は次の7コマンドです。

```bash
plasma-global validate case.yaml
plasma-global run case.yaml --output runs/case
plasma-global audit case.yaml
plasma-global migrate-v2 old-case.yaml --output migrated-case.yaml
plasma-global models
plasma-global export runs/case/result.h5 --csv result.csv
plasma-global plot runs/case/result.h5 "n[plasma,Ar_plus]"
```

`examples/v3/cases/argon_lxcat.yaml`、`rf_envelope_calibration.yaml`、`smoke.yaml` は
それぞれ trend／calibration／stress fixture であり、production benchmark ではありません。

詳細は [Architecture](docs/ARCHITECTURE.md)、[Schema](docs/SCHEMA.md)、
[Physics and Numerics](docs/PHYSICS_NUMERICS.md)、
[Migration](docs/MIGRATION.md)、[Benchmarks](docs/BENCHMARKS.md) を参照してください。
