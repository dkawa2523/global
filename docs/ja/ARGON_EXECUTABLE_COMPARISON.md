# Ar 実行比較

このページは、local Ar LXCat case と PyGMol compact Ar model の実行比較を説明します。PyGMol は外部で実行できる 0D plasma global-model package であり、ここでは strict reference ではなく sanity check として使います。

## 結論

raw PyGMol comparison は local model と大きく異なる仮定を含みます。したがって、raw mismatch はそのまま solver error ではありません。

重要なのは、electron-impact rates、absorbed power、wall loss を順に揃えたときに差の説明力が上がることです。この decomposition により、差が numerical implementation から来ているのか、model closure から来ているのかを切り分けます。

## 比較の設計

local case は two-zone global model です。一方、PyGMol comparison は equivalent cylinder に写像した one-zone model です。gas、feed、pressure、neutral temperature、power timing は揃えますが、zone structure、electron-impact rate fit、wall loss、ion flux observable は完全には揃いません。

このため、比較の目的は「同じ答えを出すこと」ではなく、次の分解を確認することです。

$$
\Delta_{\mathrm{raw}}
\approx
\Delta_{\mathrm{rates}}
+ \Delta_{\mathrm{power}}
+ \Delta_{\mathrm{wall}}
+ \Delta_{\mathrm{geometry}}
+ \Delta_{\mathrm{residual}}
$$

この式は厳密な線形分解ではありません。レビュー上の意味は、raw mismatch を単一原因に押し込まず、支配的な仮定差を順に潰して読むことです。

## 結果図

![PyGMol same-footing decomposition](assets/images/benchmarks/external_benchmark_pygmol_decomposition.png)

図の各 stage は、raw PyGMol から same-footing comparison へ進む過程を示します。

| Stage | 揃えるもの | 読み方 |
|---|---|---|
| raw PyGMol | PyGMol compact Ar model をそのまま使用 | rate、power、wall loss、geometry の差を含む |
| same rates | local LXCat/two-term rates | electron-impact kinetics の寄与を見る |
| same rates + power | local absorbed-power waveform | power deposition 定義の寄与を見る |
| same rates + power + wall loss | local global ion wall-loss coefficient | particle balance closure の寄与を見る |

local/PyGMol ratio が 1 に近づくほど、その scalar は近づきます。最後に残る差は、one-zone と two-zone の違い、observable 定義の差、残った transport closure の差として扱います。

![External benchmark overview](assets/images/benchmarks/external_benchmark_overview.png)

overview 図では、PyGMol は ZDPlaskin や CRANE と同じ validation panel ではありません。baseline と same-footing final の差を見て、model-form difference がどこまで説明できるかを確認します。

## 実行

```powershell
py -m pip install ".[compare]"
py tools\external_benchmarks\pygmol_argon.py --rerun-local
py tools\external_benchmarks\pygmol_same_footing.py --rerun-local
```

主な出力:

- `examples/outputs/argon_lxcat_icp_baseline/comparison_pygmol_summary.csv`
- `examples/outputs/argon_lxcat_icp_baseline/comparison_pygmol_solution.csv`
- `examples/outputs/argon_lxcat_icp_baseline/comparison_pygmol_metadata.yaml`
- `examples/outputs/argon_lxcat_icp_baseline/comparison_pygmol_same_footing_decomposition.csv`

## 読むべき量

powered phase では electron density の桁と trend を見ます。afterglow では mean electron energy が低下することを確認します。ion flux は electron density と定性的に連動しているかを見ます。

afterglow の差が大きい場合、最初に確認するのは wall loss、transport geometry、electron cooling closure です。PyGMol の rate を production chemistry bundle にコピーして一致を作るのは、物理モデルの改善ではなく benchmark への過適合になります。

## 参考リンク

| 対象 | 参考 |
|---|---|
| PyGMol | [PyPI: pygmol](https://pypi.org/project/pygmol/) |
| Ar LXCat baseline | [Argon LXCat production baseline](ARGON_LXCAT_CASE.md) |
| Ar cross-section data | [Zenodo DOI 10.5281/zenodo.8192503](https://doi.org/10.5281/zenodo.8192503) |
