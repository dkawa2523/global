# observables.csv 列辞書

`observables.csv` は、run 中の代表的な物理量・工学量・warning を時系列で並べた CSV です。列は case、chamber、surface、electrical backend、diagnostic 設定によって増減します。このページでは、代表的な命名規則と読み方を整理します。

実際の列一覧は、対象 run の `observables.csv` header を優先してください。ここにない列が出る場合でも、多くは下の命名 pattern のどれかに当てはまります。

## 最初に見る列

| 列 | 単位 | 意味 | 読み方 |
|---|---|---|---|
| `time_s` | `s` | 時刻 | simulation 範囲と保存点を確認する |
| `step_id` | text | recipe step | power on/off や recipe 切替と対応させる |
| `electron_density_m3` | `m^-3` | 代表 electron density | 放電の立ち上がり・維持・消滅を見る |
| `mean_electron_energy_eV` | `eV` | 代表 mean electron energy | electron-impact rate が動く代表軸として見る |
| `total_absorbed_power_W` | `W` | plasma に吸収された総 power | delivered power ではなく吸収 power として読む |
| `self_bias_V` | `V` | 代表 self-bias | bias / CCP 系の proxy |
| `plasma_potential_V` | `V` | 代表 plasma potential | sheath / ion energy proxy の背景量 |

## 列名 pattern

| Pattern | 例 | 意味 |
|---|---|---|
| `port_<port_id>_<metric>` | `port_source_rf_absorbed_power_W` | power port 別 diagnostics |
| `<quantity>_<zone_id>_<unit>` | `ne_source_m3` | zone 別 diagnostics |
| `<quantity>_<surface_id>_<unit>` | `ion_flux_wafer_m2_s` | 表面・ウェハ別 diagnostics |
| `incident_flux_<surface_id>_<species>_m2_s` | `incident_flux_wafer_CF2_m2_s` | surface に入射する species flux |
| `net_flux_<surface_id>_<species>_m2_s` | `net_flux_wall_Ar_m2_s` | surface reaction 後の net flux |
| `warning_<condition>_<target>` | `warning_debye_ratio_process` | 0/1 warning flag |

## power port columns

power port 列は次の形で出ます。

```text
port_<port_id>_<metric>
```

| Metric | 単位 | 意味 |
|---|---|---|
| `absorbed_power_W` | `W` | plasma に入った power |
| `delivered_power_W` | `W` | generator / recipe から渡した power |
| `frequency_Hz` | `Hz` | RF 周波数 |
| `source_voltage_V` | `V` | source voltage |
| `gap_voltage_V` | `V` | gap にかかった voltage |
| `voltage_rms_V`, `rf_voltage_rms_V`, `coil_voltage_rms_V` | `V` | RMS voltage |
| `current_A`, `current_rms_A`, `rf_current_rms_A`, `coil_current_rms_A` | `A` | current |
| `coupling_efficiency` | dimensionless | delivered から absorbed への効率 |
| `reduced_field_Td`, `effective_field_Td` | `Td` | reduced/effective field |
| `plasma_resistance_ohm`, `plasma_resistance_Ohm` | `Ohm` | plasma resistance |
| `plasma_conductance_S` | `S` | plasma conductance |
| `electric_field_V_m` | `V m^-1` | gap 電場 |

確認の目安:

| 状況 | 見る列 |
|---|---|
| recipe の power と結果が合わない | `port_*_delivered_power_W`, `port_*_absorbed_power_W`, `port_*_coupling_efficiency` |
| DC series benchmark を読む | `port_*_gap_voltage_V`, `port_*_current_A`, `port_*_reduced_field_Td` |
| RF envelope を読む | `port_*_voltage_rms_V`, `port_*_current_rms_A`, `port_*_effective_impedance_ohm` |

## zone columns

zone 別列は、`source`、`process`、`plasma` など chamber の zone ID を含みます。

| 列の例 | 単位 | 意味 |
|---|---|---|
| `ne_source_m3` | `m^-3` | source zone の electron density |
| `mean_energy_process_eV` | `eV` | process zone の mean electron energy |
| `pabs_source_W` | `W` | source zone の absorbed power |
| `Tg_process_K` | `K` | gas temperature |
| `pressure_plasma_Pa` | `Pa` | pressure |
| `total_density_plasma_m3` | `m^-3` | gas density total |
| `electronegativity_process` | dimensionless | negative ion / electron density の目安 |
| `residence_time_source_s` | `s` | gas residence time |
| `debye_length_process_m` | `m` | Debye length |
| `EoverN_plasma_Td` | `Td` | reduced electric field |

zone 列は、multi-zone case でどの領域が支配的かを見るために使います。global 列だけでは zone 間の偏りは分かりません。

## ion-loss diagnostics

ion wall loss の設定や有効面積を確認する列です。

| 列 pattern | 意味 |
|---|---|
| `ion_loss_family_<zone>` | `bohm`、`ambipolar_diffusion` など |
| `ion_loss_area_<zone>_m2` | ion loss に使う面積 |
| `ion_loss_h_factor_<zone>` | Bohm-family の edge-to-center factor |
| `ion_loss_characteristic_length_<zone>_m` | wall loss の characteristic length |
| `ambipolar_loss_rate_<zone>_s` | ambipolar diffusion の first-order loss rate |

electron density が高すぎる/低すぎる場合は、reaction budget だけでなく ion-loss diagnostics も確認します。

## warning columns

warning は 0/1 の flag として出ます。`summary.yaml` では `warning_counts` に合計回数が出ます。

| Pattern | 意味 | 次に見る列 |
|---|---|---|
| `warning_high_electronegativity_<zone>` | electronegativity が高い | charged species density、negative ion chemistry |
| `warning_ion_ion_afterglow_<zone>` | afterglow で ion-ion condition が強い | power off step、charged species |
| `warning_pressure_deviation_<zone>` | pressure が target からずれる | pump / inlet / total density |
| `warning_debye_ratio_<zone>` | Debye length が大きい | `debye_length_*_m`, density, characteristic length |
| `warning_site_overfill_<surface>` | surface site が過充填 | `site_fill_*`, surface reaction |
| `warning_site_depletion_<surface>` | free site が枯渇 | `free_site_fraction_*`, surface reaction |

warning は失敗そのものではありません。どの step のどの時間帯で立つかを確認して、物理解釈に使える run か判断します。

## 表面・ウェハ列

surface がある case では、wall や wafer の diagnostics が出ます。
入力から出力までの関係は [表面・ウェハ・IED ガイド](SURFACE_WAFER_IED_GUIDE.md) も参照してください。

| 列の例 | 単位 | 意味 |
|---|---|---|
| `ion_flux_wafer_m2_s` | `m^-2 s^-1` | wafer への total ion flux |
| `mean_ion_energy_wafer_eV` | `eV` | ion energy proxy |
| `ied_width_wafer_eV` | `eV` | IED width proxy |
| `ied_collisionality_wafer` | dimensionless | IED collisionality proxy |
| `angle_spread_wafer_deg` | `deg` | ion angle spread proxy |
| `sheath_voltage_wafer_V` | `V` | sheath voltage proxy |
| `sheath_thickness_wafer_m` | `m` | sheath thickness proxy |
| `site_fill_wafer` | dimensionless | occupied + filled site fraction |
| `occupied_site_fraction_wafer` | dimensionless | occupied site fraction |
| `free_site_fraction_wafer` | dimensionless | free site fraction |
| `film_wafer_m` | `m` | film thickness |

## species-resolved surface flux

species 名を含む列は、surface へ届く粒子種や surface reaction 後の net flux を示します。

| Pattern | 意味 |
|---|---|
| `ion_flux_<surface>_<ion>_m2_s` | ion species 別 flux |
| `mean_ion_energy_<surface>_<ion>_eV` | ion species 別 energy proxy |
| `charge_fraction_<surface>_<ion>` | ion species の charge fraction |
| `incident_flux_<surface>_<species>_m2_s` | neutral/radical species の incident flux |
| `net_flux_<surface>_<species>_m2_s` | surface reaction 後の species net flux |
| `net_total_flux_<surface>_m2_s` | surface 全体の net flux |

## radical / atom flux ratio

etch/deposition の傾向を見るための補助列です。

| 列 | 意味 |
|---|---|
| `radical_incident_flux_<surface>_m2_s` | radical incident flux |
| `halogen_atom_flux_<surface>_m2_s` | F, Cl など halogen atom flux |
| `carbon_atom_flux_<surface>_m2_s` | carbon atom flux |
| `oxygen_atom_flux_<surface>_m2_s` | oxygen atom flux |
| `radical_to_ion_flux_ratio_<surface>` | radical / ion flux ratio |
| `halogen_to_C_radical_flux_ratio_<surface>` | halogen / carbon radical flux ratio |
| `F_to_C_radical_flux_ratio_<surface>` | F / C radical flux ratio |
| `etch_deposition_balance_<surface>` | etch/deposition tendency indicator |

これらは process interpretation 用の指標であり、単独で etch rate や膜質を保証するものではありません。

## 列がない場合

| 列がない場合 | 可能性 |
|---|---|
| `port_*` がない | electrical backend が port details を出していない |
| zone 別列が少ない | single-zone case、または diagnostic が限定的 |
| wafer 列がない | chamber に `wafer` surface がない |
| `film_*` がない | film state が無効、または surface model がない |
| species flux 列がない | surface chemistry / radical diagnostics がない |
| `warning_*` が少ない | warning condition がない、または該当 diagnostics が未生成 |
