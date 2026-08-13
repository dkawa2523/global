# Migration from schema v2

## 一方向変換

runtime は schema v2、旧 API、旧キー、別名を受理しません。旧 case は一度だけ変換し、出力された
v3 bundle を review して以後の source of truth にします。

```bash
plasma-global migrate-v2 old/case.yaml --output migrated/case.yaml
plasma-global validate migrated/case.yaml
```

CLI は case YAML、canonical chemistry directory、`*.migration.yaml` report を同じ出力 bundle に
作ります。Python の `migrate_v2(path)` は `MigrationResult(case, report)` を返します。report は
変換上の warning と、v3 に意味を持たない `unused_keys` を列挙します。これらを runtime の
互換分岐として残しません。

## 変換内容

migrator は v2 の split case/chamber/recipe を include 展開して一つの v3 case へまとめ、次を
具体値にします。

| v2 | schema v3 |
|---|---|
| 最初の inlet から推定した組成 | 各 zone の明示初期組成 |
| 全正イオンへの暗黙 seed | 各 species の明示初期 density |
| direct absorbed power | `prescribed_power` |
| voltage + series resistance | `dc_series` |
| saved circuit/power series | `external_table` |
| Maxwellian | `electrons: maxwellian` + `electron_energy` |
| mean-energy table | `electrons: table`, `lookup: mean_energy` |
| E/N table | `electrons: table`, `lookup: local_field` |
| prescribed electron series | `experimental.prescribed_profile` |
| RF/CCP/ICP | 対応する明示 `experimental.*` port |
| film/inventory/generic state | `experimental` セクションの opt-in |
| quasi-steady observation（旧 event stop） | `experimental.stop_when_quasi_steady` |

power/electron/wall/solver mode が曖昧なら別モデルへ置換せず `MigrationError` にします。v3 で
無作用な `physics.mode`, `gas_model`, `Zone.role`, `imported_inputs`、cache/diagnostic/alias field は
unused report に残して削除します。

変換は旧機能を v3 で実行可能にする作業であり、標準機能への認定ではありません。audit の
`standard` allow-list は固定気体温度、準中性・電気的正性、実断面積 Maxwellian と
electron-energy closure、prescribed absorbed power、単一一価正イオンの Bohm floating wall、
決定論的 boundary reaction、inlet／pump／inter-zone transport に限定されます。DC、table／
local-field、evolved gas、surface kinetics、複数イオン・負イオン、RF/CCP/ICP、拡張状態は
変換後も `experimental` 分類です。

## Chemistry 変換

旧 chemistry の変換器は runtime 外の `tools/importers` にあります。通常は `migrate-v2` が
bundle 作成時に呼びますが、単独でも実行できます。

```bash
python -m tools.importers.chemistry_v2 \
  old/chemistry_manifest.yaml migrated/chemistry
```

生成物は `chemistry.yaml`、species/gas/boundary/surface reaction CSV、rate-model YAML、必要な
cross-section/table assets です。旧 rate kind や単位は canonical SI の明示値へ変換されます。
runtime は LXCat/BOLSIG/ZDPlaskin raw format、prefactor/exponent surrogate、curve の自動 sort/
clip を読みません。

v2 species に熱容量がない場合だけ、一方向変換境界は旧理想気体仮定を `cv_over_kb` として
具体化します（単原子 `1.5`、二原子 `2.5`、3原子以上 `3.0`）。同様に欠落していた正イオン
seed は `1e13 m^-3` へ具体化します。どちらも migration warning と canonical CSV/YAML に残り、
v3 runtime に推定や fallback はありません。装置条件に応じて必ず review・置換してください。

wall neutralization の生成物を一意に決められない場合、converter は推測せず失敗します。
review した mapping を与えて再実行します。

```yaml
# boundary-products.yaml
Ar_plus: Ar
CF3_plus: CF3
```

```bash
python -m tools.importers.chemistry_v2 \
  old/chemistry_manifest.yaml migrated/chemistry \
  --boundary-products boundary-products.yaml
```

一つの旧断面積ファイルに複数 segment があり自動選択できない場合も、cross-section ID ごとの
zero-based segment mapping を importer に明示します。元素・電荷・mass、reaction energy loss、
boundary branch、evolved gas で使う `cv_over_kb` は変換 report と canonical files を必ず確認して
ください。

## Bohm `h_factor` の runtime 契約変更

数値の `h_factor` は、現在の runtime では Bohm 音速へ直接掛ける完全な係数です。同じ状態に
対する式は、旧 runtime と現在の runtime で次のように異なります。

```text
旧: Gamma_i = 0.61 * h_factor * n_i * sqrt(e * T_e / m_i)
現: Gamma_i =        h_factor * n_i * sqrt(e * T_e / m_i)
```

したがって同じ数値を残すと、瞬時の ion wall flux は旧実装の `1 / 0.61 = 1.63934...` 倍、
約 63.9% 増えます。旧 flux を維持するには `h_factor_new = 0.61 * h_factor_old` としてください。
これは同一状態での境界 flux の比較であり、反応と power balance を含む時間発展結果が一律に
63.9% 変化するという意味ではありません。必ず対象 case を再実行して比較してください。

`h_factor: auto` は現在の中性粒子密度を使う動的 closure であるため、この定数変換は適用できません。
新しい artifact は `provenance.wall_transport_closure.bohm_h_factor` に
`version: direct-multiplier-v2` と surface ごとの `numeric` / `auto` mode を保存します。
`model_ids.wall_transport` の既存形式は変わりません。この provenance がない旧 artifact は契約を
単独では特定できないため、作成時の runtime version を保持するか、review 済み設定で再実行して
ください。

## Review checklist

1. report の全 warning と unused key を確認し、意図した削除か記録する。
2. 各 zone の pressure、temperature、初期組成、electron/ion seed、initial electron energy を確認する。
3. power port と command の `kind`、setpoint、pulse boundary、external table bounds を確認する。
4. boundary reaction の incident ion、生成物、branch probability 合計 1 を確認する。
5. `validate` の後、`audit` を実行し、`standard` / `experimental` 分類、静的 reaction balance、
   動的 charge/energy ledger residual を確認する。
6. `run` → HDF5 roundtrip で `effective_case_yaml` と source provenance を確認し、`audit` の全入力
   SHA-256 一覧を migration report と一緒に保存する。
7. v2 と v3 の主要 species、energy、voltage/current/E/N を比較する。ただし物理的に誤った旧数値との
   一致を受入条件にしない。

変換後に schema が受理しても、experimental model の精度保証や装置固有校正が追加されるわけでは
ありません。benchmark provenance と対象装置での validation を別途維持してください。
