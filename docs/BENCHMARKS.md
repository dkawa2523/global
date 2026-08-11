# Benchmarks and Validation

## 合否の考え方

このモデルでは「以前の実装と同じ数値」より、単位、balance、closure、適用範囲が物理的に
妥当であることを優先します。比較対象と式が同じ場合だけ数値一致を要求し、wall、power、
EEDF、輸送の仮定が異なる比較をparityとは呼びません。

検証は次の順で積み上げます。

| 層 | 主な検査 | 例 |
|---|---|---|
| schema/data | strict型、path、ID、単位、provenance | generated JSON Schema、canonical chemistry loader |
| component | 解析式、有限性、符号、単調性 | electron closure、rate、wall flux、power port |
| balance | 同じeventから作るsource/lossの相殺 | reaction stoichiometry、zone transport、wall ledger |
| integration | stiff solver、segment境界、state/result配置 | canonical minimal v3 fixture |
| external | 公開caseまたは実験との比較 | CRANE、ZDPlaskin-derived資料、argon global-model literature |

audit の `standard` は、固定気体温度、準中性・電気的正性、実断面積 Maxwellian と
electron-energy closure、prescribed absorbed power、単一一価正イオンの Bohm floating wall、
決定論的 boundary reaction、inlet／pump／inter-zone transport の allow-list です。test が通る
experimental component を standard と読み替えません。

## 現在の自動検証

`tests/fixtures/v3_minimal/` は、v3 case、canonical chemistry、Bohm wall、prescribed power、
electron-energy closureを実際にload/compile/simulateする最小fixtureです。end-to-end testは
少なくとも次を確認します。

- chemistryとboundary reactionが一度だけコンパイルされること
- BDF結果がrecipeの開始・終了時刻を含み、time-major stateが有限であること
- square-pulseの全edgeでsegmentが分かれ、on/off forcingが混ざらないこと
- table kineticsとexternal power dataをコンパイル中に重複読込しないこと
- local-field tableとDC-series powerがelectron closureへ接続されること
- effective case、model IDs、chemistry provenanceが結果へ残ること
- scalar `atol` を内部 state scale に適用し、主要結果が tolerance 半減で安定すること
- constant／threshold断面積の解析rateと一致し、Maxwellian tail不足を拒否すること
- malformed CSV、未知 discriminator、変更済み `CaseSpec`、100,000点を超えるsamplingを拒否すること
- result serialize失敗時に新旧のHDF5／summaryを半端に公開しないこと
- auditが単一一価正イオンの標準caseだけを`standard`と分類すること

component testsでは、quasineutrality、mean-energy/temperature変換、mass-action、directed
transportの体積積分保存、wall incident fluxとboundary returnの単一ledger、result HDF5の
固定構造、auditの明示残差を検査します。

gas energy、surface kinetics、wall transport、prescribed electron profile、experimental
accumulatorについても、状態配置からRHS、solver resultまでの統合試験があります。ただし、
統合試験は装置ごとの定量精度を保証するものではありません。

## External reference cases

repositoryには次の比較資料があります。ただし、用途を区別します。

- `crane_two_reaction_argon`: `benchmarks/references/` に保存した公開CRANE出力との
  scalar chemistry比較。最終Ar+密度の相対誤差を `5e-4` 以下とする。公開tutorial由来の
  固定electron-rate係数を使うため、実断面積Maxwellianを要求するstandard判定ではなく
  experimental cross-code regressionとして扱う。
- `zdplaskin_example2`: 保存済み出力から作ったE/N-rate tableとDC-series設定を使う
  cross-code regression。最終species密度とpeak electron密度は相対誤差 `10%` 以下、
  最終plasma voltage/current/E/Nは `5%` 以下とする。元のZDPlaskin実行全体との
  同一性を意味しない。
- `argon_lxcat`: provenance付きcross sectionとmulti-zone argon case。装置固有係数を
  校正せず、EEDF・RF/CCP coupling の qualitative trend を調べる experimental fixture。

`examples/v3/cases/rf_envelope_calibration.yaml` は RF envelope の calibration fixture、
`smoke.yaml` は experimental ICP/CCP と拡張状態を短時間だけ通す stress fixture です。
Argon LXCat、RF envelope、Smoke のいずれも production benchmark や装置予測精度の根拠では
ありません。`zdplaskin_example2` も audit では experimental に分類する cross-code regression
です。

上記の閾値は、公開された別コードの保存出力に対する回帰条件です。同じ入力を独立測定した
実験検証ではなく、装置の妥当性確認や予測精度の主張には使いません。旧runtimeを呼ぶtestや
出力もv3の合否根拠にしません。

ZDPlaskin由来の資産は3層に分けます。

1. `benchmarks/references/`: 保存済みsummaryとcircuit series。
2. `benchmarks/raw/zdplaskin_example2_eovern/`: source出力から抽出したstrict CSVとmetadata。
3. `examples/v3/chemistry/zdplaskin_example2/tables/`: v3 runtimeが読むcanonical HDF5。

canonical HDF5は `py -m tools.benchmarks.build_zdplaskin_rate_table` でnetwork accessなしに
再生成でき、testはSHA-256を固定値および再生成物の両方と照合します。未参照のBOLSIGDB
dumpや丸めたrun outputを入力へ混ぜません。

比較対象の前提は [Global Model review](https://doi.org/10.1002/ppap.201600138) と
[Kemaneci et al. のwall-recombinationを含むglobal model](https://doi.org/10.1088/0963-0252/23/4/045002)
を手掛かりに明記します。圧力、体積/面積、gas温度、入力power、反応集合、wall return、
electron closureが一致しない結果を1つの相対誤差だけで評価しません。

## v2 migration baseline

[`benchmarks/v2_baseline.yaml`](../benchmarks/v2_baseline.yaml) は、v2からv3へ移行した際に
変化した量と理由を追跡するための固定比較artifactです。v2の代表3 caseについて、実行環境、
solver統計、最終値を記録しています。これは物理的な正解、v3の数値一致条件、性能の合否閾値
ではありません。closureやwall ledgerなどの物理モデルを是正したv3結果は、この値と一致しない
ことが正常です。artifactの自動testはschema、ケース集合、値の有限性だけを検証し、v3結果との
parityを要求しません。

## 保存則の測り方

内部zone transportのspecies residualは、density和ではなくparticle数で

\[
r_s(t)=\sum_z V_z S^{edge}_{z,s}(t)
\]

を評価します。closed chemistryでは元素ごとにstoichiometric residualを検査します。
wallではincident ion loss、branch return、charge transfer、electron energy lossを
`WallFluxRecord` と `ZoneTermLedger` から集計します。

`audit_result` は何を保存量とみなすかを推測しません。case/modelがobservableとして残した
residual名とtoleranceを明示して初めて合否判定します。報告値は各系列の最大絶対残差です。
絶対・相対・規格化残差のどれを使ったかをbenchmark定義に書いてください。

## Experimental の検証水準

experimental componentの基本契約は次です。

- 全出力が有限で、定義domain外を明示的に拒否する。
- event accumulatorがparticle/inventory balanceを保つ。
- power、field、density、frequencyを変えたとき、モデル式が要求するtrendを示す。
- pulse境界で不連続が意図した時刻にだけ現れる。

これはquantitative accuracyの証明ではありません。ApproximateTwoTermEEDF、RF/CCP/ICP
envelope、prescribed profile、film/inventoryを装置予測へ使うには、別のBoltzmann/
EM/circuit/surface solverまたは実験値との装置別比較が必要です。

## 再現可能な報告

benchmark runごとに次を保存します。

1. source caseとchemistry provenance。
2. `result.h5` と `summary.yaml`。
3. code revision、Python/SciPy version、使用した外部tableのchecksum。
4. 比較series、残差定義、単位、tolerance、referenceの出典。
5. solver statusと `nfev`, `njev`, `nlu`。

canonical artifactは固定layoutのHDF5です。CSVやplotはそこから再生成し、丸められたCSVを
次のreference stateとして使いません。SciPy BDFの仕様は
[`solve_ivp`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html)
を参照してください。

## Production 精度を未主張の検証範囲

- prescribed electron density: charge-balance interpretationを明示した装置別benchmark。
- approximate two-term EEDF: 公開Boltzmann solverに対するmixture別rate/transport比較。
- wall variants: prescribed-frequency/ambipolarとBohmの適用領域別case。
- gas/surface energy: calorimetryまたは独立solverとの装置別比較。
