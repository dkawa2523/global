# ZDPlaskin 比較

このページは、ZDPlaskin `example2` と local implementation の比較結果を説明します。対象は Ar DC-resistor discharge であり、評価軸は final species composition、E/N、電流、電力です。

## 結論

同じ practical footing に揃えた条件では、local run は ZDPlaskin の保存済み final state を数 % 程度で再現しています。電気回路量もよく一致しており、この isolated case では reaction balance、circuit coupling、単位処理は整合していると判断できます。

この結果は、Ar ICP、RF discharge、混合ガス、surface kinetics まで一般化できるものではありません。あくまで ZDPlaskin example2 に対応する DC Ar benchmark の結論です。

## 比較条件

ZDPlasKin は zero-dimensional plasma kinetics solver で、electron transport / rate coefficient の計算に BOLSIG+ を利用できます。この環境では ZDPlaskin executable を直接再実行せず、保存済み `example2` output を reference として使います。

local 側では、保存済み ZDPlaskin output から E/N-rate table を作り、同じ Ar chemistry、geometry、source voltage、ballast resistance に合わせます。これにより、local solver の ODE integration と circuit backend が final state を再現できるかを見ます。

主な条件は次の通りです。

| 項目 | 値 |
|---|---|
| gas | Ar |
| species | `e`, `Ar`, `Ar*`, `Ar+`, `Ar2+` |
| pressure | 100 torr |
| gas temperature | 300 K |
| gap | 0.004 m |
| source voltage | 1000 V |
| ballast resistance | 100 kOhm |
| electron-impact rates | ZDPlaskin 保存 output 由来の E/N-rate table |

## 結果

| Quantity | Local | ZDPlaskin | Local/ZDPlaskin |
|---|---:|---:|---:|
| electron density | `3.220e17 m-3` | `3.170e17 m-3` | `1.016` |
| `Ar*` density | `2.057e17 m-3` | `2.014e17 m-3` | `1.021` |
| `Ar+` density | `2.139e15 m-3` | `2.079e15 m-3` | `1.029` |
| `Ar2+` density | `3.198e17 m-3` | `3.150e17 m-3` | `1.015` |
| reduced field | `2.574 Td` | `2.561 Td` | `1.005` |
| gap voltage | `33.15 V` | `33.22 V` | `0.998` |
| current | `9.669 mA` | `9.668 mA` | `1.000` |
| absorbed power | `0.3205 W` | `0.3212 W` | `0.998` |

species density は最大でも約 3 %、E/N は約 0.5 % の差です。電流と吸収電力も reference と整合しています。したがって、この benchmark では final operating state は一致していると読めます。

## 図の読み方

![ZDPlaskin overview panel](assets/images/benchmarks/external_benchmark_overview.png)

overview 図の左上 panel が ZDPlaskin 比較です。縦軸は次の相対偏差です。

$$
100
\left|
\frac{q_{\mathrm{local}}}{q_{\mathrm{ZDPlaskin}}} - 1
\right|
$$

見るべき点は、species と E/N が同時に小さい偏差に収まっていることです。`Ar+` はこの panel で最も大きな偏差ですが、それでも約 2.85 % です。`E/N current` が小さいことは、local run が実際に使った electrical backend の field が reference と整合していることを示します。

transient peak electron density は完全には一致しません。local peak は約 `3.22e17 m-3`、保存済み ZDPlaskin peak は約 `3.57e17 m-3` です。この benchmark は final state parity を目的としているため、time-history identity は合格条件にしていません。

## 実行

```powershell
py -m plasma_global.cli run examples\configs\case_zdplaskin_example2.yaml
py tools\external_benchmarks\run_external_benchmarks.py --only zdplaskin
```

関連 file:

- `examples/configs/case_zdplaskin_example2.yaml`
- `examples/chemistry_zdplaskin_example2`
- `examples/outputs/zdplaskin_example2_surrogate`
- `tools/external_benchmarks/zdplaskin_example2.py`

## 限界

この比較は、ZDPlaskin/BOLSIG+ live call と local two-term solver が同一であることを示すものではありません。保存済み ZDPlaskin output に合わせた E/N-rate behavior を入力として使い、local ODE と circuit coupling を検証しています。

また、wall model の一般妥当性、RF case、混合ガス chemistry、surface reaction、process prediction はこの比較の外側です。

## 参考リンク

| 対象 | 参考 |
|---|---|
| ZDPlasKin 公式 | [Zero-Dimensional Plasma Kinetics solver](https://www.zdplaskin.laplace.univ-tlse.fr/) |
| ZDPlasKin example repository | [Hemadityamalla/ZDPlaskin](https://github.com/Hemadityamalla/ZDPlaskin) |
