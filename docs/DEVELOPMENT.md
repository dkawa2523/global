# 開発・保守

このページは、次の開発者が変更箇所を見つけ、公開入口から影響を確かめるための手引きです。
利用者が指定できる量と保存形式は[モデリングマニュアル](index.md)、
物理式と比較結果は[Theory](THEORY.md)、着手の判断と未達事項は[ロードマップ](ROADMAP.md)を参照してください。

<a id="foundation-design"></a>

## 変更する場所と確認する契約

公開宣言 `CaseSpec` は、YAMLとPythonで同じ物理条件を表します。準備時に参照と指令を解決し、
読んだ入力の識別情報と実効宣言をモデル・出力計画とともに `PreparedCase` へ固定します。
これにより、入力を読む操作、時間積分、結果の保存と解析をそれぞれ再実行できます。
解析時は保存された系列の対象・単位・定義可能区間を引き継ぎます。
公開操作と保存物の契約は[マニュアルの操作一覧](index.md#commands)を参照してください。

| 変更の種類 | 責務と入口 | 確認する内容 |
|---|---|---|
| ケースと参照データ | `input/` が公開型・パス・ID・指令・時刻を扱い、`build.py` が入力と各モデルを組み合わせる | YAMLとPythonで同じ宣言になり、準備後は元ファイルに依存せず同じ条件で実行できること |
| 化学種・反応・速度式 | `chemistry/` が機構や断面積を読み、速度則の引数・依存と化学量論を検査する | 反応次数と係数の単位、元素・電荷・表面サイト、反応率と符号 |
| 電子、壁、表面、気体熱、電源 | `models/` と対応する `experimental/` のモデルで物理項を評価する | 交換量の正負、面積・体積の換算、適用域、ゼロ・停止・切替時の値 |
| 状態、輸送、時間積分 | `core/` が状態配置と依存、移送、収支の合成、適応BDFと受理検査を担う | 区画をまたぐ粒子・エネルギー収支、forcing境界、補間点、失敗前の有効区間 |
| 保存・解析 | 出力計画と `postprocess.py` が系列選択・派生量を決め、`output.py` が数値結果を公開して成果物ごとの成否を記録する | 所属・単位・定義可能区間、未保存と未定義、途中結果の読込、元入力を移した後の再利用 |

coreからinputや組立て層への依存、chemistryから実行モデルへの依存、modelsからcoreや組立て層への依存は
import-linterで検査します。CSVの列や行の数値検査は共通化し、速度式の次数、係数表の軸、
電力分配のように物理的意味を持つ条件は、それを評価するモデルに置きます。

新しい壁則や速度式では、先にどの状態を参照し、どの収支にどの単位で寄与するかを決めます。
壁の入射束から体積あたりの損失へ換算するときは、面積／体積の係数を一度だけ適用します。
状態変数を増やす場合は、収支に加えて状態配置、依存の疎構造、受理時に検査する範囲を更新します。
膜厚と壁蓄積は名前を持つイベント積算状態です。蓄積が壁反応へ作用する機能には、
その依存を反応の物理モデルに接続する設計が必要です。
派生量を追加するときはRHSの評価や収支内訳を再利用し、保存後の計算だけで済む量は積分状態に加えません。

## 外部モデルと受け渡す量

<a id="external-models"></a>

Swarmからは生成条件付きの電子係数表、外部回路からは電力・電圧の指令時系列を受け取ります。
OESCRには保存した状態を渡し、視線や計測器の設定は観測モデル側で与えます。
入出力を照合した履歴は[外部モデルとの接続試験](#external-model-evidence)を参照してください。

### 電子係数表

電子係数表には、ガス組成、温度、場の条件、使用した断面積と生成方法を記録します。
Swarmの収束点から、E/N、平均エネルギー、フラックス換算移動度 μN をそれぞれ
`reduced_field_Td`、`mean_energy_eV`、`reduced_mobility_1_V_m_s` に対応付けます。
反応列 `k:<rate_id>` には標的1粒子あたりの係数を入れます。反応率では実行時の電子密度と標的密度を掛けます。
軸の範囲、0 Tdの扱い、停止条件は[公開仕様](index.md#electron-table-settings)で確認できます。
[Swarmの係数表利用手順](https://github.com/dkawa2523/swarm/blob/27ada948f9845859fc3b7fc3bbb7ac96b07d5b5c/docs/coefficient_lookup.md)

`argon_lxcat` と `rf_envelope_calibration` が使う表の生成条件は
`plasma_global/examples/data/swarm_argon_rf/provenance.yaml` にあります。Ar 100%、350 K、8 Pa、DC無磁場で求めた
1〜200 Tdの82点を収録し、電離1・励起30・運動量移行1の各過程を識別しています。
元のAr断面積ファイルでELASTICに付く 1.36×10⁻⁵ は電子と標的の質量比 m/M です。
同梱表の運動量移行断面積は `threshold_eV: 0` とし、元の値と意味は来歴に残しています。
RF・CCP・ICPでこの表を使う場合の前提は、固定組成・温度でのDC局所電場係数を適用する近似です。
[断面積の公開データ](https://doi.org/10.5281/zenodo.8192503)

Swarmを利用できる環境で係数を再生成する場合は、作業先と出力先に未作成のパスを選びます。

```bash
python -m tools.benchmarks.prepare_argon_coefficients --swarm-root PATH --work-dir tmp/swarm-new --output NEWDATA
```

### 回路指令

PCDなどの回路出力を使うときは、測定点と電力の定義を先にそろえます。素子損失や蓄積を含む装置では、
プラズマへの実電力と電子・重粒子への配分を決めてから `columns` を結び付けます。
時間は秒へ換算し、時刻範囲と補間条件を[電源CSV仕様](index.md#power-table-settings)で確認します。
装置側の状態への反作用が観測量に効く場合は、状態・時間原点・交換エネルギーの責務を設計してから連成します。

### 保存状態からの分光

OESCRとの接続では必要な電子密度、気体温度、平均電子エネルギーを計算前に保存対象に選びます。
一様ArでMaxwell分布を仮定する例では、同じ領域と時刻について、中性圧力を `n_Ar * k_B * T_g`、
電子温度を平均電子エネルギーの2/3として渡します。時刻と系列の取得は[保存結果の読込例](index.md#results)を参照してください。

OESCRではこれらを `gas_mixture` の中性圧力・気体温度、`plasma_state.ne_shells_m3`、
`eedf.kind: te_maxwell` と `eedf.zones[0].te_eV` に対応付けます。非Maxwell分布を渡す場合は分布の形と規格化を指定し、
元の分布と受取側で確率積分と励起係数を照合します。
[OESCRの入出力仕様](https://github.com/dkawa2523/CRmodel/blob/ff716d06415d65ab9cad9b14cc034935fe76426d/docs/io_spec.md)

<a id="benchmark-reproduction"></a>

## 運用例と検証の再実行

### Theoryの比較一式

比較器と保存参照があるリポジトリのルートで実行します。通常のケース単独実行は
[マニュアルの例題案内](index.md#examples)から選べます。出力先には未作成のディレクトリを指定します。

```bash
python -m pip install ".[plot]"
python -m tools.benchmarks.theory --output runs/theory-check
```

実行器は同梱20ケースと独立求積を評価し、出力先に比較値と条件を含む `results.json`、11図、
各runのHDF5と要約を作ります。DCと後期残光は比較対象の時刻を指定して保存します。
公開時の `benchmarks/theory/results.json` は本文の表示値・図に対応する記録です。
JSONの `result_h5` はその実行出力を起点としたパスです。HDF5で実効条件と実読込入力の識別情報、
JSONで外部参照・比較器を含む再現条件を確認できます。
数値を更新する場合は、比較時刻、観測量、誤差の規格化と丸め、入力・実装の一致を先に確認します。
[数値結果と解釈](THEORY.md#benchmarks)

<a id="dc-benchmark-data"></a>

### DC比較のデータを再生成する

| 資産 | 役割 |
|---|---|
| `benchmarks/references/zdplaskin_example2_source.yaml` | 元入力、Fortran、断面積、保存波形の出典と固定リビジョン |
| `benchmarks/references/zdplaskin_example2_history.csv` | 元の保存波形303行をSIへ換算した比較参照 |
| `benchmarks/references/zdplaskin_example2_bolsig.in` | 独立した電子係数計算の条件。固定Ar背景でArとAr*を標的に含む |
| `plasma_global/examples/data/zdplaskin_example2/electrons.csv` | Globalのケースが読むE/N 1.5〜85 Tdの電子係数とフラックス換算移動度 |

元の波形を再計算する場合は、出典の固定リビジョンにある `example2/` をLinux／WSLの作業コピーに取り出し、
保存出力を別途保管してから実行します。
[ZDPlasKin 2.0aの固定ソース](https://github.com/Hemadityamalla/ZDPlaskin/tree/48744203fe93368d344315645e9617ed351cadd3/example2)

```bash
gfortran -O2 -ffree-line-length-none -fallow-argument-mismatch -c dvode_f90_m.F90 master_module.F90 zdplaskin_m.F90 main.f90
gfortran -o reference-dc dvode_f90_m.o master_module.o zdplaskin_m.o main.o ./bolsig_x86_64.so
printf '\n\n' | ./reference-dc
```

独立係数には、同じ断面積 `bolsigdb.dat` と同梱の入力を使います。BOLSIG+ 07/2024の作業先に両方を置き、
`./bolsigminus zdplaskin_example2_bolsig.in` で `independent.dat` を作ります。
Globalのリポジトリから次を実行すると、波形と係数をそれぞれCSVへ変換できます。
`SOURCE.dat` は元の保存出力か、上で外部コードから再計算した出力です。
[BOLSIG+の公開配布元](https://www.bolsig.laplace.univ-tlse.fr/download.html)

```bash
python -m tools.benchmarks.prepare_zdplaskin_reference --zdplaskin-output SOURCE.dat --bolsig-output independent.dat --output work/dc-reference
```

<a id="argon-afterglow-data"></a>

### Ar後期残光の測定参照をたどる

`benchmarks/references/beckers_argon_afterglow.yaml` に、出典PDFのハッシュ、図5の8点の読取値、
時刻原点と外部のイオン移動度を記録しています。読取では横軸の0・2000 μsと縦軸の10¹⁵・10¹⁴ m⁻³で座標を校正し、
各時刻に近い3列で黒い測定線の高さの中点を換算しました。図中の赤い近似線は読取から除外しました。
±0.016 decadeは線幅などを反映した読取幅です。
傾きの求め方、移動度の報告幅と読取幅の扱い、測定量と体積平均の対応はTheoryで説明します。
[Beckersらの公開稿](https://pure.tue.nl/ws/portalfiles/portal/25507508/beckther2016.pdf) · [移動度の出典](https://doi.org/10.1088/0022-3700/10/18/036) · [後期残光の比較方法](THEORY.md#argon-afterglow-benchmark)

<a id="rf-calibration"></a>

### RF校正の入出力

`tools.calibration.rf_envelope` は同じ基準点の測定量から係数候補と `power_model`／`power_command` の断片を生成します。
ここでは仮想測定点を使って実電力の換算と校正断片の適用を確かめます。

```bash
python -m tools.calibration.rf_envelope --role bias --frequency-Hz 2e6 --commanded-power-W 100 --absorbed-power-W 80 --voltage-rms-V 200 --current-rms-A 2 --dc-self-bias-V=-40 --write runs/rf-calibration.yaml
```

この入力の効率は吸収実電力／供給実電力で0.8、有効インピーダンス係数は V_rms²／供給実電力で400 Ωです。
電圧／電流の100 Ω、皮相電力400 VA、力率0.25は測定点の診断値です。RMS電流と位相を独立に適合する機能はありません。
複数の供給電力指定は同じnet実電力を表す必要があります。出力された断片は対象の電源ポートと運転区間に入れ、
指定した測定点の指令、吸収電力、電圧、自己バイアスを出力から照合します。
物理式は[TheoryのRF包絡](THEORY.md#rf-envelope-formula)を参照してください。

## 公開文書と配布物の確認

```bash
uv sync --frozen --all-extras
uv run --frozen --all-extras python -m tools.generate_schema_docs
uv run --frozen --all-extras python -m mkdocs build --strict --site-dir runs/docs-check
```

`docs/generated/case.schema.json` と `minimal_case.yaml` は公開宣言から生成します。
`tests/test_schema_docs.py` は生成物の再現性と公開スキーマへの適合を検査します。
本文で完全なcaseを示す場合は読み込んで実行し、設定断片では適用先のcaseと置き換える範囲を明記します。
同梱例は `plasma_global/examples/` を正本とし、`init` が必要な資産をまとめて取り出します。

<a id="manual-reproduction"></a>

### マニュアル図とTheory PDF

`python -m tools.render_manual_examples --output runs/manual-check` は、`argon_analysis` と `zdplaskin_example2` を
計算・保存し、HDF5再読込との一致を検査して `values.json` を出力します。plot・graph依存とGraphviz本体が必要です。
このコマンドは出力先とは別に実行元の `docs/assets/` にマニュアルの6図をコピーします。
確認だけを行うときはリポジトリの作業コピーで実行します。
構成図とフロー図を修正したときは、同じ場所のDOTを正本として `dot -Tpng -Gdpi=200 INPUT.dot -o OUTPUT.png` で更新します。

TheoryのPDFは、同じMarkdownから `[docs,plot]` の依存と日本語フォントを用いて生成します。
日本語のRegularとBoldを指定し、新しいファイルを出力先に選びます。

```bash
python -m pip install ".[docs,plot]"
python -m tools.render_theory --output output/pdf/global-model-theory-check.pdf --font PATH/Regular.ttf --bold-font PATH/Bold.ttf
```

インラインのdollar-backtick記法と `math` コードブロックをWeb・PDF共通の数式原稿にします。
Web版の式とリンク、PDFの数式・しおり・リンク・図表を確認し、Popplerで全ページを描画して日本語、表、改ページを見ます。
確認後のPDFを `docs/assets/global-model-theory.pdf` に反映します。
[GitHubの数式記法](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/writing-mathematical-expressions)

### 実装と配布の品質検査

```bash
uv run --frozen --all-extras python -m tools.quality fast
uv run --frozen --all-extras python -m tools.quality pr
uv run --frozen --all-extras python -m tools.quality nightly
```

`fast` は変更したPythonの整形とlintをチェックし、リポジトリ全体の型と通常テストを検査します。
`pr` は全体のlint、型、import境界、通常テスト、
coverage、複雑度、依存脆弱性、秘密情報と抑制追加を検査します。
`nightly` はseedを変えたproperty test、stress、独立したソースコピーでのmutationも実行します。
Windowsの通常のnightly入口はWSLに委譲します。Linux／WSLでは `nightly --native` を使えます。
レビュー済み例外は `quality/baseline.json`、秘密情報の確認履歴は `.secrets.baseline` にあり、通常gateから独立して更新します。

配布の確認はクリーンなソースからwheelを作り、checkout外の環境で `python -I` を使います。
import元の位置、同梱入力の `init`、計算・保存・HDF5再読込まで通し、過去のbuildディレクトリによる混入を避けます。
現行CIの対象と依存下限は `.github/workflows/quality.yml` を参照してください。

<a id="change-record"></a>

## 作業記録

### 文書と公開動作の照合（2026-11-24）

現在のcheckoutで、Windows／Python 3.12.13からCLIの全10操作とPythonの準備・保存を確認しました。
元の入力を移した後も、保存HDF5から検査、CSV、時系列図と反応図を生成できました。
用途別の編集例は設定断片を指定されたcaseに適用し、切替時刻の値と保存後の再読込を照合しました。

| 今回の確認 | 記録 |
|---|---|
| CLI・API、元入力分離後の読込・出力、入力と出力のエラー | `.quality-reports/documentation-restructure/cli-api-20261123T235400Z/summary.json` と同じ場所の `checks.jsonl` |
| 用途別の編集例A〜Cと保存の往復 | `.quality-reports/documentation-restructure/manual-guides-20261123T235452Z/summary.json` |
| マニュアルのコマンドとPython例 | `.quality-reports/manual-workflows-htly3em1/`。初回実行、図、保存結果の再利用、単独ベンチマーク |
| 0.5.0 wheelとRF校正 | `.quality-reports/documentation-restructure/wheel-and-rf-20261123T235839Z/summary.json`。隔離した配布物から4例を実行し、仮想測定点の校正CLIを確認 |
| Theoryの20ケースと独立求積 | `.quality-reports/documentation-20261124/theory/results.json`。所要時間以外のJSONの値と11図が版管理済みの記録に一致 |
| マニュアル図の隔離再生成 | `.quality-reports/documentation-20261124/manual/values.json`。Ar 201時刻とDC 151時刻のHDF5往復および公開6図の一致を確認 |
| 通常テスト、サイト、PDF | `.quality-reports/documentation-20261124/`。通常777件、MkDocs strict、PDF全38ページを確認 |

この文書確認で再実行した外部比較は、同梱入力と保存参照を使うGlobal側の経路です。
外部コード・計測データの由来は下記の記録と版管理した参照を引き継ぎます。
装置条件での独立評価は[ロードマップ](ROADMAP.md#validation-plan)に残ります。
Windowsの初回通常テストではTcl/Tkのファイル不足で描画テスト1件が止まりました。
ファイル描画用の `MPLBACKEND=Agg` を指定して通常テストを再実行しました。nightly専用1件は通常テストの対象外です。

`benchmarks/analytic-decay` の単一zone IDを `plasma/test` に変えると、現行入力型は受理し、
`validate` と `outputs` は終了コード0になります。通常の `run` は積分の後、系列
`electron_density_m3[plasma/test]` がHDF5レイアウトで安全な名前にならず、終了コード2で結果公開前に停止します。
入力の識別子制約と結果名の制約を計算開始前に整合させる課題です。運用上の回避方法をマニュアルに記載します。

### 公開経路の統一（2026-11-22、fcb7924）

公開宣言を `CaseSpec` に集約し、準備済み入力の再利用、数値結果の先行公開、成果物ごとの成否、
保存結果だけでの再利用を整備しました。旧版reader・移行経路、内部の非Maxwell係数生成、汎用拡張状態宣言を廃止しました。
互換撤去以前のケース件数・旧形式での結果は履歴と当時のログで識別できます。

| 当時の確認範囲 | 記録 |
|---|---|
| 配布wheelからの全29例、CLIと元入力分離後の再利用、部分結果 | `.quality-reports/push-preflight/final-wheel-lf/receipt.json` |
| 通常778件と品質検査、Windows／WSLでの確認範囲 | `.quality-reports/push-preflight/quality-lf/`。Windowsで起動できなかったPyreflyとimport-linterはWSLで実行 |
| 独立レビューと修正後の検証 | `.quality-reports/final-review-20261122/`。表面触媒と壁還流の元素収支、反応・表面IDの衝突、粒子交換の熱収支、保存系列の所属を確認 |
| 比較実行と保存結果 | `benchmarks/theory/results.json` と `.quality-reports/final-review-20261122/theory-reviewed/` |
| DCの追加許容差比較 | `.quality-reports/final-review-20261122/dc-tight-reviewed/comparison.json`。初期値を含む304点で7量を比較。改行正規化前のsource hashと最終公開版との対応は `.quality-reports/push-preflight/final-wheel-lf/lf-numeric-equivalence.json` |
| 旧入力v4／結果v3での実行と互換経路の撤去 | 同コミットの改訂前記録、ローカルの `.quality-reports/current-removal/` |
| 外部ZDPlasKin/BOLSIG+の再生成 | 当時の外部実行では元の波形303行の全列を再現し、変換後の波形と係数のCSVを同梱物と照合。条件と固定ソースは[DCデータ](#dc-benchmark-data)を参照 |

`.quality-reports/` は作業環境の実行記録です。公開checkoutで確認できる比較値と条件は、
版管理した参照YAML・CSVと `benchmarks/theory/results.json` に収録しています。
旧記録の詳細は公開経路を統一したコミット時点からもたどれます。
[2026-11-22時点の開発記録](https://github.com/dkawa2523/global/blob/fcb79242569703517cddfc63a7017dbcb135a8a9/docs/DEVELOPMENT.md#change-record)

<a id="external-model-evidence"></a>

### 外部モデルとの接続試験

Swarm `27ada94`、PCD `d3910e0`、OESCR `f7bf408` を起点とするローカル版を使いました。
SwarmとPCDは作業ツリーの変更を含みます。ここでの比較は指定した入出力の接続確認です。

- SwarmのAr 100%、300 K、13.3 Pa、DC無磁場の計算から40・50・60 Tdの係数を使った
  `swarm_argon_dc` では、初期45 Tdで平均エネルギー5.50326 eV、電流35.7413 mA、励起速度2.73226×10²² m⁻³ s⁻¹を確認しました。
  0 Vの定常指令区間では電流と励起速度がゼロでした。
- PCDの検査では、10 Vパルス、50 Ω直列抵抗、50 Ω負荷で得た5 V・0.1 A・0.5 Wの負荷条件を用い、
  417点の指令とHDF5の吸収電力を照合しました。壁・反応損失なしの条件で、入力1.0334750 μJ、
  電子エネルギー増分1.0334754 μJでした。 [PCDの過渡電力評価](https://github.com/dkawa2523/circuit_opt/blob/d3910e0221ef8aff1e37f30e70e3b9a3fec23fc2/pcd/analysis/transient.py)
- OESCRに一様Arの一時刻を渡す例では、2台の分光器の計2052ビンで、手入力と保存値から変換した強度データが
  相対許容差10⁻¹²で一致しました。 [OESCRの入出力仕様](https://github.com/dkawa2523/CRmodel/blob/ff716d06415d65ab9cad9b14cc034935fe76426d/docs/io_spec.md)
