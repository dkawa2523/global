# CRANE 比較

このページは、CRANE `tutorials/TwoReactionArgon` と local scalar chemistry ODE の比較を説明します。これは full discharge benchmark ではなく、反応式、単位変換、stiff ODE integration を切り出した検証です。

## 結論

local result は CRANE tutorial の final density とよく一致しています。`e` と `Ar+` の relative error は約 `2.4e-6`、`Ar` は約 `2.0e-12` です。したがって、この小さな反応ネットワークについては、stoichiometry、cm-to-SI conversion、charged species balance、ODE integration が整合していると読めます。

この結果は、electron Boltzmann kinetics、wall loss、surface chemistry、gas heating、electrical closure の検証ではありません。

## 比較対象

CRANE は MOOSE framework を基盤にした open-source plasma chemistry software です。この repo では CRANE 全体の discharge solver ではなく、公開 tutorial `TwoReactionArgon` の保存済み結果を reference として使います。

benchmark の条件は単純です。`e`, `Ar`, `Ar+` の 3 species に対し、ionization と three-body recombination の 2 反応だけを解きます。CRANE tutorial の density と rate coefficient は cm 系で与えられるため、local case では SI に変換して解きます。

## 条件と単位変換

| 項目 | CRANE tutorial | local SI |
|---|---:|---:|
| `n_Ar(0)` | `2.5e19 cm^-3` | `2.5e25 m^-3` |
| `n_e(0)` | `1 cm^-3` | `1.0e6 m^-3` |
| `n_Ar+(0)` | `1 cm^-3` | `1.0e6 m^-3` |
| ionization `k` | `2.1736169000623e-12 cm3/s` | `2.1736169000623e-18 m3/s` |
| recombination `k` | `1.0e-25 cm6/s` | `1.0e-37 m6/s` |
| final time | `7.5e-7 s` | `7.5e-7 s` |

二体反応では `cm3/s` から `m3/s` へ `10^-6` を掛け、三体反応では `cm6/s` から `m6/s` へ `10^-12` を掛けます。

$$
k_{\mathrm{SI}} = k_{\mathrm{cgs}} \times 10^{-6}
\quad(\mathrm{cm^3/s} \to \mathrm{m^3/s})
$$

$$
k_{\mathrm{SI}} = k_{\mathrm{cgs}} \times 10^{-12}
\quad(\mathrm{cm^6/s} \to \mathrm{m^6/s})
$$

## 図の読み方

![CRANE overview panel](assets/images/benchmarks/external_benchmark_overview.png)

overview 図の右上 panel が CRANE benchmark です。log scale の final-state relative error なので、棒が低いほど reference に近いことを意味します。

`e` と `Ar+` が同じ error になるのは、この benchmark では電子密度と ion density が直接対応するためです。`Ar` の error がさらに小さいことは、大きな neutral reservoir の微小変化が conservation と整合して扱われていることを示します。

## 実行

```powershell
py -m plasma_global.cli run examples\configs\case_crane_two_reaction_argon.yaml
py tools\external_benchmarks\crane_two_reaction_argon.py
```

主な出力:

- `examples/outputs/crane_two_reaction_argon/summary.yaml`
- `examples/outputs/crane_two_reaction_argon/observables.csv`
- `examples/outputs/crane_two_reaction_argon/comparison_crane_two_reaction_argon.yaml`

## 限界

この benchmark は、constant-rate chemistry ODE として読むべきです。CRANE tutorial は electron energy equation、Boltzmann kinetics、wall loss、surface kinetics、gas heating、external circuit を含みません。したがって、ここでの一致を discharge model 全体の精度としては扱いません。

## 参考リンク

| 対象 | 参考 |
|---|---|
| CRANE documentation | [Welcome to CRANE](https://crane-plasma-chemistry.readthedocs.io/en/latest/) |
| CRANE paper | [Keniley and Curreli, CRANE, arXiv:1905.10004](https://arxiv.org/abs/1905.10004) |
| MOOSE framework | [MOOSE](https://mooseframework.inl.gov/) |
