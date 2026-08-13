# Schema v3 reference

## 基本契約

runtime が受理するのは `schema_version: 3` だけです。`CaseSpec` は Pydantic v2 の strict・
frozen model で、未知キー、暗黙の文字列→数値変換、空 ID、重複 ID、未解決参照を拒否します。
旧キーや別名は runtime に存在せず、v2 は `migrate-v2` で一方向変換します。

case の最上位は次の 8 個の必須キーと、任意の `experimental` です。

| キー | 内容 |
|---|---|
| `schema_version` | literal `3` |
| `case` | name と description |
| `chemistry` | canonical chemistry manifest |
| `reactor` | zones、edges、surfaces、inlets、pumps、power ports |
| `recipe` | `start_time_s` と順序付き `duration_s` steps |
| `models` | electron/EEDF、electron density、gas energy、surface kinetics |
| `solver` | BDF tolerance、任意 step control、sampling |
| `output` | 保存する observables と summary final quantities |
| `experimental` | film／wall inventory／extension state の明示 opt-in（任意） |

型から生成した [最小 case](generated/minimal_case.yaml) と
[JSON Schema](generated/case.schema.json) は次で更新します。

```bash
python -m tools.generate_schema_docs
```

実行可能な最小 bundle は `tests/fixtures/v3_minimal/`、実用例は `examples/` にあります。

schema が受理する機能全てが標準精度範囲という意味ではありません。audit の `standard` は、
固定気体温度、準中性・電気的正性、実断面積 Maxwellian と electron-energy closure、
prescribed absorbed power、単一一価正イオンの Bohm floating wall、決定的な boundary reaction、
inlet／pump／inter-zone transport の allow-list に限ります。それ以外は実行を維持しますが
`experimental` と分類します。

## `include` と path

各 YAML は一つの `include: relative/base.yaml` を持てます。親を先に読み、mapping は再帰
merge、list と scalar は子の値で全置換します。cycle、list/mapping 形式の `include` は
エラーです。`includes`、環境変数展開、`~` 展開、`file_key` はありません。

相対 path は、merge 後の case 位置ではなく、その path を記述した YAML の directory を
基準に絶対 path へ解決します。したがって base と child を別 directory に置いても宣言元を
失いません。

## Reactor と初期状態

各 zone は `zone_id`、`volume_m3`、`pressure_Pa`、`gas_temperature_K` と初期組成を明示します。
初期組成から得る理想気体圧力は指定圧力と整合しなければなりません。electron-energy closure
では `initial_mean_energy_eV` が必須、local-field closure では禁止です。quasineutral の初期
electron density は明示した荷電種 seed の電荷和で一意に決まり、prescribed profile は外部表に
electron density を持ちます。最初の inlet や全正イオンへの runtime の暗黙 seed はありません。

初期組成は排他的な二方式です。全種を `initial_densities_m3` で与えるか、中性種だけを
`initial_mole_fractions`（総和 1）で与え、荷電種を `initial_seed_densities_m3` で加えます。
後者は compile 時に `pressure_Pa/(k_B T_g)` から具体密度へ一度だけ変換されます。どちらの
方式でも各 zone に正イオン seed が必要で、未知種、neutral seed、荷電種の mole fraction、
圧力と総密度の不整合は fail-fast です。

surface は area、temperature、site density、初期吸着 coverage、wall transport を持ちます。
free-site は ODE state ではなく吸着 coverage から代数計算します。wall 中性化・再結合生成物は
chemistry の `boundary_reactions.csv` に明記し、活動する正イオンの branch probability 合計は
1 でなければなりません。

## Recipe と model ID

recipe は `start_time_s` と連続した steps からなり、各 step は正の `duration_s` と inlet /
surface / power-port command を持ちます。command の ID と `kind` は reactor の宣言と一致する
必要があります。`continuous` と `square_pulse` の全 switching time、previous table/profile の
knots は BDF segment 境界になります。

次の表は runtime が受理する選択肢であり、標準 allow-list ではありません。

| 選択箇所 | `kind` |
|---|---|
| electrons | `maxwellian`, `table`, `experimental.approximate_two_term` |
| electron closure | `electron_energy`, `local_field` |
| electron density | `quasineutral`, `experimental.prescribed_profile` |
| gas energy | `fixed` (既定), `evolved` |
| power port | `prescribed_power`, `dc_series`, `external_table`, `experimental.rf_envelope`, `experimental.ccp`, `experimental.icp` |
| wall transport | `bohm`, `prescribed_frequency`, `ambipolar`, `off` |
| waveform | `continuous`, `square_pulse` |

`prescribed_power` は吸収電力だけを与え、E/N、self-bias、plasma potential を生成しません。
`experimental.rf_envelope` と `experimental.ccp` の `absorbed_power_W` も plasma へ実際に
吸収される実電力です。RF は `coupling_efficiency` から source-side real powerを逆算します。
CCP の lossless sheath は実電力を消費しないため、real source powerは吸収電力と等しく、
reactive成分を含む (V_{rms}I_{rms}) は別observable `apparent_power_VA` に記録します。
`external_table` と prescribed profile は strict numeric CSV を読み、重複 header、余剰列、
欠損 cell、非有限値を compile 時に拒否します。bounds は既定で error です。経験モデルは必ず
`experimental.*` を明示し、audit が `experimental` と分類して provenance を残します。

## Electron / gas closure

`electron_energy` と `local_field` は排他的です。`electron_energy` は electron energy density を
積分し、`maxwellian` または mean-energy table と組み合わせます。`local_field` は E/N から
rate、transport、mean energy を代数評価し、electron-energy state を持ちません。

`mean_energy_eV` と `electron_temperature_eV` は別 observable です。Maxwellian 等価温度は
`2/3 * mean_energy_eV` であり、Bohm/sheath には後者を渡します。

`experimental.approximate_two_term` は各reduced-field grid点でpower-balance rootがbracketされ、
指定反復内に収束した場合だけtableを生成します。rootがないendpoint値の代用、field boundsのclip、
cross-section最終energyより先への外挿は行いません。末端energy binの確率質量が `1e-3` を超える
分布も、energy gridが未解像として拒否します。既定field gridは `1..100 Td` であり、より高いfieldを
使う場合は断面積supportと `energy_grid.max_eV` を物理的に検証して明示指定します。
正のfieldだけを持つprepared tableでportがoffの
場合はcold boundary（mean energy/rateは0）を使い、明示0-field nodeを持つ一般tableではそのnodeを
そのまま評価します。

`gas_energy: fixed` は zone temperature を固定します。`evolved` は experimental 分類で、
重粒子内部エネルギーを state とし、gas species 全ての `cv_over_kb`、反応・弾性加熱、flow、
wall heat exchange から温度を導出します。inlet、pump、directed edge のflowは内部エネルギー
ではなく理想気体エンタルピー \(h_s=(c_{v,s}+k_B)T\) を運びます。

Bohm wall の数値 `h_factor` は \(\Gamma_i=h_i n_i\sqrt{eT_e/m_i}\) の \(h_i\) そのものです。
`auto` は現在の中性重粒子総密度、`ion_neutral_cross_section_m2`、characteristic lengthから
各 RHS 評価時に更新されます。密度、断面積、長さの単位はそれぞれ `m^-3`, `m^2`, `m` で、
初期圧力から固定した係数ではありません。

## Solver と output

solver method は `BDF` 固定です。`rtol` と scalar `atol` を指定します。`atol` は物理単位の
state に直接適用せず、初期状態から作る固定 scale で state を内部無次元化した後に適用します。
同じ scale から component ごとの物理 domain tolerance を導出します。`first_step_s` と
`max_step_s` は任意で、未指定値を強制しません。`sample_interval_s` と `save_at_s` は排他的で、
どちらもなければ accepted points を保存します。forcing 境界はどの sampling mode でも必ず
保存します。accepted points を含む保存点総数には固定上限 100,000 点があり、超過は設定エラーです。

`experimental.stop_when_quasi_steady` は互換性のため名称を維持した診断設定です。
`relative_rhs_norm_s_inv` と `min_time_s` を指定すると、時間不変 segment で局所的な
quasi-steady 候補を記録します。event は非終端で、積分の打切りや定数外挿は行わず、
定常 root solver の代用にはなりません。

`output.observables` は保存する派生系列、`output.summary_series` は summary に最終値を出す
state/選択 observable です。未知名は compile 時に拒否します。

## Canonical chemistry

```yaml
schema_version: 3
species: species.csv
gas_reactions: gas_reactions.csv
rate_models: rate_models.yaml
boundary_reactions: boundary_reactions.csv   # optional
surface_reactions: surface_reactions.csv     # optional
cross_sections: cross_sections.yaml          # optional
provenance: {}                               # optional
experimental: {}                             # optional
```

runtime rate ID は `electron_impact`, `arrhenius`, `constant`, `first_order`, `tabulated_1d`,
`sticking`, `ion_assisted`, `desorption`, `langmuir_hinshelwood` です。温度 power law は
`experimental.electron_temperature_power_law` と明示します。各 kind の必須 parameter と未知
parameter は厳格に検証されます。

cross-section manifest は `kind`, `target`, `threshold_eV`, `file` と、電子energy移送を表す
`electron_energy_transfer_eV` または互換入力 `energy_loss_eV` のどちらか一方を必須とします。
前者は正値が電子加熱、負値が電子冷却です。後者は従来どおり非負の電子energy損失で、内部で
符号付きtransferへ変換されます。両方を同時指定できません。
gas reaction CSVでも同じ2 fieldを任意overrideとして使え、未指定時は参照cross sectionの
transferを継承します。
現行result artifactは互換性のためledger名 `reaction_energy_loss_J_m3_s` を維持します。この値は
符号付きtransferの負値なので、superelastic反応による電子加熱では負になります。
curve CSV は exact header `energy_eV,sigma_m2`、有限・非負断面積、厳密昇順 energy を要求します。
sort、負値 clip、非有限行の黙殺はしません。LXCat / BOLSIG / ZDPlaskin の変換は
`tools/importers/` で offline に行います。

`excitation`, `dissociation`, `ionization` の正の `threshold_eV` は反応onsetです。それ未満に
非zero断面積を指定してはいけません。threshold nodeを省略したcurveは積分用の非公開gridへ
`(threshold_eV, 0)` を補います。`momentum_transfer` と `attachment` のthresholdは同じ
zero-onset契約を持たず、実データのsupportをそのまま使います。

surface thermal rateは、desorptionを
`ν exp(-Ea/kTs) Ns θ [m^-2 s^-1]`、Langmuir-Hinshelwood二体反応を
`A exp(-Ea/kTs) Ns^2 θa θb [m^-2 s^-1]` と評価します。前者のreactant総次数は1、
後者は2で、gas reactantやfree-site reactantを受理しません。

Maxwellian rate の事前 table は、CSV 最終energyより上を外挿しません。未解像の
energy-weighted Maxwellian tailが `1e-6` 以下となるmean-energy範囲だけを生成し、その範囲外は
`ModelDomainError` です。高いelectron energyを扱う場合は、rateをclipするのではなく断面積CSVを
十分高いenergyまで延長する必要があります。
低energy側は積分時だけ0 eVまで補います。正のonset thresholdを持つ非弾性curveでは、未収録の
threshold nodeを0として補い、それ未満の区間も0とします。threshold契約を持たないcurveが
有限の先頭値を明示している場合は、その先頭値を0 eVまで保持します。source CSVやruntimeの
断面積補間は書き換えません。

## 固定 `result.h5` layout

```text
/
├── time_s
├── state/
│   ├── values                 # (n_time, n_state), time-major
│   └── labels                 # UTF-8, n_state
├── observables/
│   └── <selected-name>        # scalar または (n_time,)
├── metadata/
│   ├── effective_case_yaml
│   ├── model_ids              # YAML text
│   └── provenance             # YAML text; source 情報と通常実行診断
└── solver/
    ├── success
    ├── status_code
    ├── status_message
    └── statistics_yaml
```

未知 group/dataset は reader が拒否します。`summary.yaml` は status、期間、solver 統計、model
IDs、指定 final 値、最大保存則残差だけを含みます。CSV は `plasma-global export` による明示的な
派生物です。
