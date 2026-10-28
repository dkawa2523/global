# 開発・保守

利用者向けの機能・物理式・設定・出力の定義は[統合マニュアル](index.md)を参照します。
ここには実装を継続して改良するための構造、検証、データ保守の手順だけを置きます。

## 実装の構造

```text
公開YAML（CaseSpec）
  → path・ID・指令・時間原点を具体化（ResolvedCase）
  → chemistry・geometry・forcing・model・output planをcompile
  → CompiledGlobalModelをBDFで積分
  → SimulationResult
  → HDF5・summary・要求されたCSV／図を新しいrunへ公開
```

ResolvedCaseは内部表現です。別の公開入力形式として説明・配布しません。
`load_case` は公開YAMLを読み、`simulate` が参照データの読込とcompile・計算を行います。
simulateは結果をディスクへ書かず、`write_result` が保存を担当します。

| 所在 | 責務 |
|---|---|
| `input/schema.py`, `input/_schema_*.py` | 公開構造と共用する型・値の定義 |
| `input/_resolve.py`, `input/_serialize.py` | 公開入力と具体化済み内部設定の対応 |
| `chemistry/` | 化学種・反応・断面積の読込、検査、速度式のcompile |
| `models/` | 電子、電源、壁、表面等の物理評価 |
| `core/` | 状態配置、輸送、モデルの合成、時間積分、結果型 |
| `build.py`, `_build_*.py` | 各機能を組み立てるcomposition root |
| `_output_plan.py`, `_output_values.py`, `postprocess.py` | 要求系列の解決、保存状態からの派生量、イベント積分 |
| `output.py`, `_result_*.py` | 保存・読込・要約・CSV・図の共通入口と実装 |
| `api.py`, `cli.py` | 操作の入口。独自の物理計算を持たない |

coreはinput・orchestrationを、chemistryはruntime modelを、modelsはcore・orchestrationを
直接参照しません。import-linterでこの境界を検査します。
新しい物理項は、必要な入力・単位・保存量・符号・適用範囲を定義して既存モデルへ接続します。
同じ意味の既定値や検査をCLIとAPIに別々に実装しません。

CSVの形状・数値・列対応は `_numeric_csv` に共通化しています。軸、反応次数、電力分配、
適用範囲の検査は各domainへ置きます。結果の派生量はRHSと同じ評価・ledgerを使い、
図を追加するためにODEの次元や適応stepを変更しません。

## 文書と例の保守

説明は次の5ファイルにまとめます。操作と理論の役割を分け、細分化した機能説明や過去の設計提案を追加しません。

| ファイル | 役割 |
|---|---|
| `README.md` | パッケージの概要、初回コマンド、マニュアルへの案内 |
| `docs/index.md` | 利用者向けマニュアル兼統合仕様書 |
| `docs/THEORY.md` | 技術的背景、モデル比較、機能構成、方程式、数値検証。PDFの原本 |
| `docs/DEVELOPMENT.md` | 内部構造と保守・検証手順 |
| `docs/ROADMAP.md` | 現在地、未解決事項、次の作業 |

入力の機械可読な定義と構造確認用の例はコードから生成します。

```bash
uv sync --frozen --all-extras
uv run --frozen --all-extras python -m tools.generate_schema_docs
uv run --frozen --all-extras mkdocs build --strict
uv run --frozen --all-extras mkdocs serve
```

生成物は `docs/generated/case.schema.json` と `minimal_case.yaml` です。
生成の再現性と公開schemaへの適合を `tests/test_schema_docs.py` で確認します。
手書きの項目一覧をコードから独立した厳密schemaとして扱いません。

実行例の正本は `plasma_global/examples/` で、wheelにも同梱します。
別のexamplesツリーに同じcaseを複製しません。`init` が必要な資産だけを利用者の場所へコピーします。
`cases/` の7例と `cases/benchmarks/` の17ケースを `init --list` で発見できるようにし、
同梱する外部ファイルは `chemistry/` と `data/` へ置きます。ケースの追加は同じ公開schemaのYAMLで行います。
本文のYAMLは完全なcaseか、適用先を明記した断片かを区別します。

マニュアルの図は `argon_analysis` と `zdplaskin_example2` の実計算から作ります。
`python -m tools.render_manual_examples --output runs/manual-new` は、入力一式の取り出し、
計算・保存、HDF5再読込での数値照合、6図の更新と `values.json` への保存値抜粋を行います。
plot・graph依存とGraphviz本体が必要です。DC例へ追加するのは出力選択だけで、物理条件は同梱caseを使います。
図・データの説明は統合マニュアルの第3・13・14節、変数表は各入力区分の節へ置きます。
既定値や対応変数を変えた場合は公開schema・resolve・実行モデルと表を照合し、本文の例も実行します。
新しい説明ファイルや独立した物理入力を図のために追加しません。

### 利用者マニュアルの確認基準

2026-10-27の見直しでは、初見のエンジニアが「導入→用途の選択→入力の編集→実行→結果の判断」まで
たどれるかを基準にしました。変数表の網羅性だけでなく、作業のつながりを確認します。

| 観点 | 見つかった不足・偏り | 現行文書への反映 |
|---|---|---|
| 初回利用 | 仮想環境、実行Python、成功時に見る値の説明が不足 | 第2節に環境準備、終了状態・終端時刻・生成物の確認、最初の再描画を追加 |
| 用途別の編集 | 選択表はあるが、同梱例の何を変更するかが薄い | 第4節に一定通電、パルス、流通・移送、DC、表面・気体加熱・RFの作業手順を追加 |
| 読む順序 | 序盤の検証ケース一覧が、利用開始の導線を長くしていた | 第2節は用途別7例と配置案内。詳細な数値検証ケースは第15節へ移動 |
| 入出力の意味 | 終了・定常、初期圧力・圧力制御、行番号・保存時刻の取り違えが起きうる | 操作箇所に判断方法を追加。計算前の系列選択と保存後の再解析を区別 |
| 外部データと再利用 | 新しいガスの準備順序、条件比較、再開の対応範囲が不足 | データ準備、共有bundleでの条件比較、保存資産、未実装の再開・自動sweepを明記 |
| 操作と問題解決 | CLI全体の参照と環境・読込段階の対応が不足 | 操作一覧、終了コード、症状からの対応、再現に必要な報告情報を追加 |
| 詳細と適用範囲 | 入出力表・図は十分だが、拡張機能のデモを予測用の機構と誤認しうる | 変数表を維持し、各用途の必要データと判断可能な範囲を説明 |

構成の比較には次の公式資料を用いました。用途や対象方程式が違うため、ページ数や機能数の一致は基準にしません。

| 公式資料 | 参照した構成 | 本書での対応 |
|---|---|---|
| [Cantera 3.2 User Guide](https://www.cantera.org/stable/userguide/index.html) | 入門、目的別手順、リファレンス、例題への案内 | 最初に実行する第2〜4節と、必要時に参照する第5〜14節を区別 |
| [OpenFOAM v13 User Guide](https://doc.cfd.direct/openfoam/user-guide-v13/contents) | 例題で設定・実行・後処理を通し、入出力と数値設定も参照できる構成 | 各用途の編集箇所から確認量・評価へつなぎ、数値設定と保存間隔を別に説明 |
| [COMSOL 6.4 Global Model Coupled with the Two-Term Boltzmann Equation](https://doc.comsol.com/6.4/doc/com.comsol.help.models.plasma.boltzmann_global_model_argon/boltzmann_global_model_argon.html) | 目的、モデル定義、結果・考察、構築手順 | 第1・3・4節を理論文書へ接続し、機構・電子係数・回路と出力の役割を対応付け |

文章の再編集時は、公開schemaと実行モデルへの照合、編集手順を適用したcaseの計算、CLI操作、
本文のAPI例、リンクと図、通常幅と狭い画面の表示を確認します。初見の第三者による操作試験や
対象装置での独立検証まで実施したとは扱いません。機能追加時にも説明先を増やさず、この役割分担を保ちます。

理論文書の根拠は `python -m tools.benchmarks.theory --output .quality-reports/theory-new` で
再計算できます。plot依存を使い、新しい出力先へ数値JSON、10図、19 runの結果を作ります。
実行器は上記のcase YAMLを直接読み、Python内部で装置条件を書き換えません。
採用した数値を `benchmarks/theory/results.json`、図を `docs/assets/theory-*.png` に記録し、
本文の丸め値と照合します。過去の計算時間を再実行時の計算時間として再利用しません。
JSONは各runのcase path、開始・終端時刻、ソルバー統計、比較値を持ち、入力・参照・実行器の
全ファイル一覧と集約checksumを一度だけ記録します。ファイルごとのchecksumと実効caseは各runのHDF5に保存します。

計算と誤差集計は `tools/benchmarks/theory.py`、描画は `_theory_figures.py` に置き、
両者を結果JSONの入力hashへ含めます。図は保存状態と明示した参照値から作り、参照のない全波形を補いません。

### Theoryの構成・説明粒度の確認

Theoryは、初見のモデリング技術者が目的・解く量・仮定・数値根拠・適用限界をたどれる技術資料とします。
CRANE論文の方程式と適用例、LoKI-GM公開稿の物理機能と数値フロー、Turnerらのbenchmark論文の条件・誤差の開示を比較対象にしました。
出典はTheoryの[参考文献9](THEORY.md#ref-9)・[15](THEORY.md#ref-15)・[16](THEORY.md#ref-16)に記します。
モデルや検証対象が違うため、機能数やページ数の同一性は求めません。

| 観点 | 見つかった課題 | 反映した内容 |
|---|---|---|
| 問題設定 | 解析例と別コード比較の条件・省略した物理が追いにくい | 第13節の8問題群ごとに入力値、期間、数値設定、比較対象を説明 |
| 図と数値 | 一次反応・移送・CRANE・DC・許容差の図が不足 | 19 runを再計算し、10図と表を対応付け。参照が終端だけの場合は点で表示 |
| 読む順序 | 導出・計算構成・再計算の入口を見つけにくい | 要旨後の案内と、第12節の検証対象一覧、付録Cを接続 |
| 機能の偏り | Ar・DCの数値一致を全機能の検証へ広げる余地がある | 第14.4節に電子・壁・輸送・電源・表面・熱・解析の根拠と残る比較を整理 |
| 総括と将来開発 | 結論だけでは本コードの到達点と次の開発の順序を把握しにくい | 第15節で機能・用途・検証の結論を総括。第16節で開発の目的・順序・完了条件を示し、進捗はROADMAPへ集約 |
| 参照の意味 | 初期条件差、逆算係数、保存格子差が図から分からない | DCの初期電子密度差とデータ依存を明記。全波形誤差と終端比較を区別 |
| 再現性・表示 | 図と本文・PDF・数値記録の更新がずれる可能性 | 描画元の数値を照合し、同じMarkdownからPDFを再生成して図・表・式・リンクを確認 |

この確認は文書と計算の対応のレビューです。独立した第三者の操作試験、他コードとの全面比較、装置精度の認定は含みません。

機能構成図と計算フロー図は、手書きのDOTを正本にします。図中のラベルは機能の説明であり、
実装クラスや設定ファイルを表すものではありません。Graphvizで再描画します。

```bash
dot -Tpng -Gdpi=200 docs/assets/model-architecture.dot -o docs/assets/model-architecture.png
dot -Tpng -Gdpi=200 docs/assets/model-flow.dot -o docs/assets/model-flow.png
```

PDFは別の原稿を維持せず、同じMarkdownから作ります。docsとplot依存、および日本語TrueType fontが必要です。

```bash
python -m pip install ".[docs,plot]"
python -m tools.render_theory --output output/pdf/global-model-theory-new.pdf --font /path/to/Japanese.ttf
```

`--bold-font` に太字fontも指定できます。出力先は新しいファイルを使います。
数式番号・図・表・参考文献リンクとPDFのしおりを生成し、Popplerで全ページを画像へ描画して確認します。
独立数式はMarkdown中でも1行のLaTeXとして書きます。長い式は数学的な定義の境目で分け、
無関係な位置での自動改行や文字の過度な縮小を避けます。PDFではSTIXの数式を使い、
式が90%未満への縮小を要する場合は原稿を見直します。表と参考文献も項目の途中での改ページを避けます。
Web版はArithmatexで数式を保ち、MathJax 3.2.2をCDNから読みます。CDNに接続しない閲覧にはPDFを使えます。
生成中のPDFは `output/pdf/`、確認用画像は `tmp/pdfs/` に置きます。
確認済みの完成版だけを `docs/assets/global-model-theory.pdf` に収録し、原稿・結果データとともに版管理します。

## 検証の実行

```bash
uv run --frozen --all-extras python -m tools.quality fast
uv run --frozen --all-extras python -m tools.quality pr
uv run --frozen --all-extras python -m tools.quality nightly
```

fastは変更Pythonのformat／lint、型、通常テストです。PR gateは全体のlint・型・import構造、
通常テスト、coverage、複雑度、dead code、依存脆弱性、秘密情報、追加抑制の検査を行います。
文書は別途 `mkdocs build --strict` で確認します。
nightlyはPR gateに加えてseedを変えたproperty test、stress、mutationを行います。
Linux／WSLでは `nightly --native`、Windowsでは通常のnightly入口がWSLへ委譲します。

coverage.pyのstatement＋branch統合指標は85%以上、変更した本番コードは90%以上、
新規コードの循環的複雑度は10以下です。Ruff・Vultureの新規抑制や、skip／xfail等を
追加して失敗を隠しません。内部class配置を固定する検査や、同じ式の恒等式だけを検査するテストは増やしません。
単位、保存則、解析解、独立積分、forcing境界、入出力の復元を検査します。

`quality/baseline.json` はレビュー済みの既存Bandit・複雑度例外、`.secrets.baseline` は
detect-secretsのレビュー情報です。通常gateはこれらを書き換えません。
`python -m tools.quality baseline` は例外を精査して更新するための明示操作です。
新規・悪化したBandit／複雑度例外は拒否されます。秘密情報候補の追加は確認済み `is_secret: false` が必須で、
公開checksumには理由を記録します。未確認・実秘密を例外へ登録しません。

CIはLinuxのPython 3.11／3.12／3.14、Windowsの3.12を対象とし、別jobで依存下限を検査します。
Linux 3.12ではGraphviz本体を導入して解析例を描画します。最低構成のjobはwheelをinstallし、
`python -I` でcheckoutをimportしない実行も確認します。
ローカルでのWSL成功をWindowsでの実行成功と読み替えません。

## 比較データと校正ツール

| 資産 | 役割 |
|---|---|
| `benchmarks/references/` | 公開別コードの保存済みsummaryと回路系列 |
| `benchmarks/raw/zdplaskin_example2_eovern/` | 元出力由来のCSVと出典・生成条件 |
| `plasma_global/examples/data/zdplaskin_example2/electrons.csv` | runtimeが直接読む係数CSVの正本 |

```bash
python -m tools.benchmarks.build_zdplaskin_rate_table --output new-electrons.csv
python -m tools.calibration.rf_envelope --help
```

係数builderはネットワーク接続を使わず、元gridを明示的にsortして一致を確認します。
補間や欠損補完はせず、新しいCSVへ出します。テストは再生成物と同梱物のchecksum・内容を比較します。

RF校正ツールは測定量から係数候補とv4用の `power_model`／`power_command` 断片を出します。
測定点以外での精度認定や、自動的なcase全体の生成とは区別します。
別コード回帰の閾値と意味は[マニュアルの検証](index.md#validation)へ集約しています。

過去の収束・局所感度の数値を将来の版へ無条件に転用せず、物理や既定値を変えた場合は同じ定義で再計算します。
報告には観測量、比較時刻、誤差指標、条件差、入力・code hash、終了状態を残します。

## 配布の確認

wheelは新しいcheckoutまたはクリーンなsource copyから作り、install後に24の同梱ケースを検査します。
古いbuildツリーを再利用すると削除したmoduleが混入し得るため、配布物の内容も確認します。
checkout外で `python -I` を使い、import元がインストール先であることを確認します。
各caseを `init` して計算・保存し、入力参照がコピー先のYAML／CSVだけで閉じることを検査します。
図の再生成は保存HDF5だけを使い、元caseの再計算で系列・単位・来歴を補完しません。
