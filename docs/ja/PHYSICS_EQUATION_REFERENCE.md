# 物理式・記号・実装対応表

このページは、主要な物理式、記号の意味、単位、実装モジュールを対応付けるための早見表です。詳細な近似の説明は [物理モデルと近似](PHYSICS_MODELS_AND_APPROXIMATIONS.md) を参照してください。

## species balance

代表形:

$$
\frac{dn_s}{dt}
  = P_s^{\mathrm{chem}}
  - L_s^{\mathrm{chem}}
  + T_s^{\mathrm{flow}}
  + T_s^{\mathrm{zone}}
  + F_s^{\mathrm{surface}}
  + W_s^{\mathrm{wall}}
$$

| 記号 | 単位 | 意味 | 主な実装 |
|---|---|---|---|
| $n_s$ | `m^-3` | species density | `physics/gas_phase_core.py` |
| $P_s^{\mathrm{chem}}$ | `m^-3 s^-1` | gas reaction production | `GasPhaseCore` |
| $L_s^{\mathrm{chem}}$ | `m^-3 s^-1` | gas reaction loss | `GasPhaseCore` |
| $T_s^{\mathrm{flow}}$ | `m^-3 s^-1` | inlet / pump contribution | `GasPhaseCore` |
| $T_s^{\mathrm{zone}}$ | `m^-3 s^-1` | inter-zone transport | `GasPhaseCore` |
| $F_s^{\mathrm{surface}}$ | `m^-3 s^-1` | surface reaction gas flux | `SurfaceCore` |
| $W_s^{\mathrm{wall}}$ | `m^-3 s^-1` | reduced wall loss | `SurfaceCore`, `surface_models.py` |

## electron density closure

quasi-neutral mode:

$$
n_e = \max\left(\sum_i z_i n_i,\ n_{\min}\right)
$$

| 記号 | 単位 | 意味 | 主な実装 |
|---|---|---|---|
| $n_e$ | `m^-3` | electron density | `physics/gas_phase_core.py` |
| $z_i$ | dimensionless | ion charge number | chemistry species metadata |
| $n_i$ | `m^-3` | charged gas species density | state vector |
| $n_{\min}$ | `m^-3` | density floor | numerics positivity config |

prescribed profile mode では、$n_e(t)$ は CSV profile から読み、ion densities は通常通り ODE state として残ります。

## electron energy

代表形:

$$
\frac{dW_e}{dt}
  = P_{\mathrm{abs}}
  - L_{\mathrm{inelastic}}
  - L_{\mathrm{wall}}
  - L_{\mathrm{transport}}
$$

平均電子エネルギー:

$$
\langle\varepsilon\rangle
  = \frac{W_e}{n_e}
$$

| 記号 | 単位 | 意味 | 主な実装 |
|---|---|---|---|
| $W_e$ | `J m^-3` | electron energy density | `numerics/system.py`, `GasPhaseCore` |
| $P_{\mathrm{abs}}$ | `W m^-3` または zone power converted to density | absorbed power density | `ElectricalCouplingAdapter`, electrical backends |
| $L_{\mathrm{inelastic}}$ | `W m^-3` | electron-impact loss | chemistry energy models |
| $\langle\varepsilon\rangle$ | `eV` | mean electron energy | EEDF/rate lookup |

## electron-impact rate lookup

mean-energy closure:

$$
k_j = k_j(\langle\varepsilon\rangle)
$$

local-field closure:

$$
k_j = k_j(E/N)
$$

| 記号 | 単位 | 意味 | 主な実装 |
|---|---|---|---|
| $k_j$ | reaction order dependent | reaction rate coefficient | `eedf/table.py`, `eedf/boltzmann_2term.py` |
| $E/N$ | `Td` | reduced electric field | electrical backend metadata |
| $\langle\varepsilon\rangle$ | `eV` | mean electron energy | electron energy state |

## DC series circuit

$$
G_p = \frac{e n_e \mu_e A}{d}
$$

$$
R_p = \frac{1}{G_p},\qquad
I = \frac{V_s}{R_b + R_p}
$$

$$
V_{\mathrm{gap}} = I R_p,\qquad
P_{\mathrm{abs}} = f_{\mathrm{abs}} V_{\mathrm{gap}} I
$$

| 記号 | 単位 | 意味 | 主な実装 |
|---|---|---|---|
| $G_p$ | `S` | plasma conductance | `electrical/dc_series.py` |
| $R_p$ | `Ohm` | plasma resistance | `electrical/dc_series.py` |
| $\mu_e$ | `m^2 V^-1 s^-1` | electron mobility | electrical config / estimate |
| $A$ | `m^2` | electrode area | chamber / port config |
| $d$ | `m` | gap length | port config |
| $V_s$ | `V` | source voltage | recipe / port config |
| $R_b$ | `Ohm` | ballast resistance | port config |
| $f_{\mathrm{abs}}$ | dimensionless | power absorption fraction | port config |

## reduced field

$$
E/N = \frac{|V_{\mathrm{gap}}|/d}{N}
$$

$$
N = \frac{p}{k_B T_g}
$$

| 記号 | 単位 | 意味 | 主な実装 |
|---|---|---|---|
| $E/N$ | `Td` | reduced electric field | electrical backend metadata |
| $N$ | `m^-3` | neutral gas density | reactor/gas state |
| $p$ | `Pa` | pressure | chamber / recipe |
| $T_g$ | `K` | gas temperature | chamber / gas-temperature state |

## Bohm-family ion wall loss

$$
\Gamma_i
  = h\,0.61\,n_i
    \sqrt{\frac{\varepsilon_e}{m_i}}
$$

| 記号 | 単位 | 意味 | 主な実装 |
|---|---|---|---|
| $\Gamma_i$ | `m^-2 s^-1` | ion flux to wall | `reactor/surface_models.py`, `SurfaceCore` |
| $h$ | dimensionless | edge-to-center factor | surface model config |
| $n_i$ | `m^-3` | ion density | gas state |
| $\varepsilon_e$ | `J` or converted mean energy | characteristic electron energy | electron energy state |
| $m_i$ | `kg` | ion mass | species metadata |

## ambipolar diffusion loss

$$
\frac{dn_i}{dt} = -k_{\mathrm{loss}} n_i
$$

または diffusion coefficient と length から概算します。

$$
k_{\mathrm{loss}} \sim \frac{D}{L^2}
$$

| 記号 | 単位 | 意味 | 主な実装 |
|---|---|---|---|
| $k_{\mathrm{loss}}$ | `s^-1` | first-order ion loss rate | `reactor/surface_models.py` |
| $D$ | `m^2 s^-1` | diffusion coefficient | surface model config |
| $L$ | `m` | diffusion length | surface model config |

## surface coverage

$$
0 \le \theta_j \le 1,\qquad
\sum_j \theta_j + \theta_{\mathrm{free}} = 1
$$

| 記号 | 単位 | 意味 | 主な実装 |
|---|---|---|---|
| $\theta_j$ | dimensionless | surface coverage fraction | `physics/surface_core.py` |
| $\theta_{\mathrm{free}}$ | dimensionless | free-site fraction | `SurfaceCore` |

## どの式をどこで見るか

| 知りたいこと | 実装 | 関連 docs |
|---|---|---|
| gas reaction source | `physics/gas_phase_core.py` | [化学入力仕様](CHEMISTRY_INPUT_SPEC.md) |
| surface reaction source | `physics/surface_core.py` | [化学入力仕様](CHEMISTRY_INPUT_SPEC.md) |
| power/EEDF coupling | `electrical/coupling_adapter.py` | [回路結合](CIRCUIT_COUPLING.md) |
| electrical backend | `electrical/*.py` | [回路結合](CIRCUIT_COUPLING.md) |
| EEDF/rate backend | `eedf/*.py` | [物理モデルと近似](PHYSICS_MODELS_AND_APPROXIMATIONS.md) |
| ODE/Jacobian | `numerics/system.py`, `numerics/scipy_backend.py` | [数値解法とソルバー](NUMERICS_AND_SOLVERS.md) |
