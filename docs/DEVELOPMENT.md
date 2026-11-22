# 開発・保守

利用者向けの機能・物理式・設定・出力の定義は[統合マニュアル](index.md)を参照します。
ここには実装を継続して改良するための構造、検証、データ保守の手順だけを置きます。

## 実装の構造

```text
公開YAML／Python辞書（CaseSpec）
  → 実読込bytes・path・ID・指令・時間原点・物理をcompile
  → 出力catalog・選択を結合しPreparedCaseへ固定
  → CompiledGlobalModelをBDFで積分
  → SimulationResult
  → HDF5・summaryを新しいrunへ公開
  → 要求されたCSV／図を生成しartifacts.yamlへ成否を記録
```

`load_case` と `case_from_mapping` は同じ公開宣言を返します。
`compile_case` は宣言をdetachして検証し、読込済み入力と実効宣言・モデル・出力を `PreparedCase` に固定します。
`simulate` は宣言または準備済みケースを受けます。準備済み実行では入力を再読込しません。
simulateは結果をディスクへ書かず、`write_result` が保存を担当します。

| 所在 | 責務 |
|---|---|
| `input/schema.py`, `input/_schema_*.py` | 公開構造と共用する型・値の定義 |
| `input/_paths.py`, `input/_validation.py` | 宣言の参照解決・再配置と公開設定間の条件 |
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

<a id="foundation-design"></a>

## 基盤の到達点と改良方針

### ゴール

対象は、低圧プラズマの区画平均の粒子・電荷・エネルギー収支を、明示した初期条件・壁・入出流・電源のもとで解くことです。利用者は一つのケースYAMLと必要な物理データで問題を定義し、選んだ時刻と物理量をHDF5から再解析できます。基盤の改良は、係数源の交換、モデルの追加・廃止、機構の編集が、その機能に関係する設定・物理項・結果の変更で済むことを目標とします。物理的に結合する項は、状態への依存と収支への寄与をコードでたどれるようにします。

Globalが受け持つのは、電子係数を使う衝突反応、電力・電場を受け取るプラズマ応答、および粒子・エネルギー収支への寄与です。電子輸送・EEDFの計算はSwarm、装置側の回路解析・設計はPCD、発光線・視線積分・装置応答はOESCRが受け持ちます。単独のケースは現行の `load_case` → `simulate` → `write_result` で再現できる形を保ちます。

### モデルを変更する場所

| 変更 | 実装の接続箇所 | 確認する量 |
|---|---|---|
| 化学種・既存の速度則で表せる反応 | 化学機構のYAML／CSV。読込時にIDを解決し、未指定の既知種の初期密度を0にする | 初期状態、反応率、元素・電荷の収支。機構から種を外す場合は初期値、境界、出力からの参照も確認 |
| 新しい速度則・壁境界・入出流 | `chemistry/` の速度評価器、`models/` の壁・表面評価、`core/transport.py` の移送 | 面積・体積による率の換算、粒子・電荷・エネルギー。新しい状態や区画間依存が生じる場合は状態配置とJacobianの疎構造 |
| 電源・回路 | `input/_compile_power.py` がモデルを組み、`input/_compile_recipe_power.py` が区間の指令を決める。各電源が評価結果、出力量の単位、停止時に定義する値を持つ | 指令、電力収支、切替時の値、選択系列の可否 |
| 電子係数・分光 | 電子係数は既存のCSV、状態の受け渡しはHDF5の系列を使う | 軸、単位、標的種、生成条件、保存時刻・区画の対応 |

壁モデルは、`WallEvaluation` に入射束、体積あたりの粒子増減、電子エネルギー損失を返します。面積対体積比の換算は壁評価で一度行います。面の条件を変えたときは、単一区画で壁入射と粒子数の変化をまず照合します。物理的な依存を変えない境界の追加や、保存結果から得る観測量の追加には、時間積分器への設定分岐を増やす必要がありません。

### 現行経路と互換方針

公開case入力はschema v5、化学機構はschema v4、結果の保存・読込はHDF5 format v4に統一します。旧版のparser、別名、
暗黙の単位・係数の補完を残さず、形式変更時は利用箇所・例題・参照・説明・検査を一緒に更新します。
電子エネルギー移送はYAML・CSV・実行時配列とも `electron_energy_transfer_eV` を使い、
負を電子冷却、正を電子加熱とします。物理の既定値と入力誤りの扱いを各層に重ねて実装しません。

電力とE/Nは `RecipeSegment.port_commands`、各power port、`PowerCoordinator` を経由します。
local-fieldの電子状態はzoneごとの `TabulatedElectronKinetics` で評価します。
Maxwellianの断面積積分はGlobal内に残し、非Maxwell電子分布と輸送係数の生成はSwarm側で行います。
Global内に係数生成用の格子やcacheを設けず、生成条件を持つCSVを計算開始前に読み込みます。

膜厚と壁蓄積は、名前付きの `experimental.film` と `experimental.wall_inventory` で有効化します。
その状態は宣言した表面イベントを積算し、物理収支へ反作用させる変更は対応するモデルへ接続します。
結果の単位はモデルと出力カタログが定め、HDF5からの再解析では保存された単位を読みます。

<a id="change-record"></a>

## 作業記録


### 利用目的からの全体再構成（2026-11-22）

YAMLとPythonを同じ公開宣言へ統一し、機能選択→自分の設定→計算→保存結果の解釈・再利用を確認します。
旧仕様の互換は保たず、実装変更に合わせて公開例・説明・検証を更新します。

| 領域・独立経路 | 調査・実行 | 確認した対象 |
|---|---|---|
| CLI・Python・例題 | 済 | 共通CaseSpec、PreparedCase再利用、モデルcatalog、設定編集 |
| 入力・参照・来歴 | 済 | 実読込bytes/hash/用途、実効宣言、rebase/init移動、CWD非依存 |
| 化学種・反応・速度式 | 済 | YAML/CSV/直接構築の同一契約、有限の速度式定義と依存 |
| 電子・係数・気体熱 | 済 | closure/係数源、停止・0密度、EOS/壁熱交換/エンタルピー |
| 電源・回路・時系列 | 済 | 型付き寄与とtopology、CCP実効入力、ICP分配、step原点 |
| 壁・表面・膜・蓄積 | 済 | 有効/無効、site制約、表面原子数、膜厚と粒子在庫の区別 |
| 輸送・状態・数値積分 | 済 | 5状態群、受理stepと補間の検査、部分結果、Jacobian、再評価可能性 |
| 保存・CSV・図・監査 | 済 | 数値先行公開、個別artifact成否、catalog、元入力なしの再利用 |
| 比較・校正・文書生成 | 済 | 独立参照の保持、20 runと追加条件、校正断片の指定点再現、抜粋値・図表・PDFの照合 |
| テスト・品質・配布 | 済 | 公開動作/科学性へ移設、mutation隔離、wheel全例、独立レビュー |

各行を調査・実行済みです。実行範囲と未確認の条件は以下に記録し、部分的な成功や担当報告だけで全体の整合を認定しません。

追加・削減時の責務は次のように整理しました。

| 変更 | 変更するもの | 維持する境界 |
|---|---|---|
| 同じ区分へ壁則・速度式・電源を追加 | 物理評価、有限の設定/入力adapter、依存・出力定義、例と検査 | 同じ戻り値で表せる変更にsolverの種別分岐を足さない |
| 新しい状態を持つ機能を追加 | モデル、compiled state block、収支への寄与・依存、入出力接続 | 試行継続と受理検査、物理kernelとorchestrationを区別 |
| 膜/蓄積などを削減 | 設定・組立て・状態/寄与・出力・例・専用検査 | 別名やgeneric extensionへ移して旧役割を残さない |

保存結果だけのinspect/audit/plot/export/networkは元入力を再コンパイルしません。
`IntegrationError.partial_result`は違反前の有効な区間までを含み、checkpoint再開機能ではありません。
数値計算の成否とCSV・図の成否は別です。`artifacts.yaml`はそのrunの要求ごとの生成記録です。

継ぎ足しの主因は、公開宣言と内部解決型の二重管理、電源ごとの戻り値への後付け判定、
同じ保存則・定義可否を別経路で再構成する処理でした。公開宣言と一度だけ読む入力storeを組立てへ集約し、
電源は型付きの領域別寄与、状態は既存5群のblock、収支は物理項のledgerとして接続します。
速度式の引数と依存は有限のモデル定義へまとめ、未知の拡張を受け入れるgeneric state/runtime宣言は撤去しました。
気体熱容量と温度の対応をEOSへ、状態の許容領域と受理検査をdomainへまとめています。

互換用のResolvedCase・serializer・audit_case・全体分類、旧版reader、移行専用fixtureは残していません。
基本量は定義できる範囲で標準保存し、未定義・未保存・失敗を区別するcatalogを保存します。
`init`、通常計算、入力を削除した後の保存結果利用まで、同じ公開経路の動作で確認します。
内部class一覧の完全一致を固定する検査や役目を終えた契約を整理し、単位・保存則・境界・失敗時の動作を維持しました。
mutationは新しい一時source copyで行い、旧作業コピーの変異結果を再利用しません。

### 今回の独立レビューと修正

| 観点 | 必須指摘と対応 | 再確認 |
|---|---|---|
| 利用者 | 描画失敗で数値結果が失われる設計、分かりにくい初回出力、保存設定とprobeの混同、古いパス・公開入口の説明 | 数値先行公開、個別artifact成否、basic/minimal、例と文書の通常入口を確認 |
| 実装者 | snapshotが元入力を再読込、fallback参照やnull fileの扱い、時刻端点の丸め、manifest失敗の報告漏れ、完了/失敗時刻の不足 | 入力変更/削除後のPreparedCase、非ゼロ開始時刻、実際の欠落と1ULPを区別、失敗注入とHDF5再読込 |
| 科学的妥当性 | EOS・domain・状態/ledgerの重複、ICP診断の単位登録欠落、local-fieldでは反作用しない下流加熱の受理、RF校正で実電力と皮相量を混同 | 熱・粒子・site・energy収支、解析解、ICP連続指標とon/off、下流分配とclosure条件、校正CLI断片の測定点再現 |

独立レビューは担当実装の完了報告と別に、公開入口から結果までを横断して行いました。
通常739件の成功後にも、独立した入力と保存結果の突合せで次の4系統を発見しました。
テスト成功を全体の整合の証明とは扱わず、修正して公開入口から再実行しました。

| 最終レビューの指摘 | 修正 | 独立した再確認 |
|---|---|---|
| ion-assistedの非触媒捕獲と壁還流で元素を二重計上 | 入射イオンを両辺に係数1で残す触媒形を共通化学契約で要求し、負のイオン増減だけを消す処理を撤去 | YAML・CSV・Mapping・直接compileの5入口で捕獲を計算前拒否。触媒形の気相＋吸着相Ar在庫のdriftは0 |
| 許容ID内の `@`／`:` が内部結合文字と衝突 | 表面イベントを `(reaction_id, surface_id)` で受け渡し、networkのIDは一意な公開系列名を使う | 4表面・4反応の衝突条件でCLI/API・CSV・HDF5・film/inventory・networkの個別選別を確認 |
| 壁・表面粒子交換の重粒子energyが保存内訳から欠落 | `wall_species_power_W_m3` と `surface_species_power_W_m3` を単位・catalog・保存・説明へ接続 | 保存1001点の公開8項和と実RHSの差は0。保存系列の積分と内部energy差の相対差は1.14×10⁻⁸ |
| 体積当たりの表面反応加熱をsurface所属と誤分類 | 系列名のprefix推測をやめ、物理量・状態・イベントの所属を出力計画へ明示的に渡す | 全29例のcatalogとgeometry、明示保存後のHDF5/inspect、最終22結果の全2484行を照合 |

再レビューの総合評価は92/100です。利用性24/25、責務・拡張性23/25、収支・数値実装22/25、検証・再利用23/25としました。
今回確認した必須指摘はすべて解消しています。モデル別の実行例拡充や装置条件での定量比較の幅は、今後の改善余地です。

### 今回の実行確認

最終packageは112 Pythonファイル、40同梱データファイルです。source fingerprintの先頭は `81ec8515d604`。
最終sourceから直接作成したwheelの全29例実行と保存・再利用を
`.quality-reports/push-preflight/final-wheel-lf/receipt.json` に記録しました。
古いbuildディレクトリを流用せず、最終wheelの全package bytesを現在のファイルと照合しています。

| 実行・照合 | 結果と範囲 |
|---|---|
| 通常入口・配布 | 最終wheelを独立環境へinstallし全29例を数値計算。30出力、CSV数値往復25件、CLI 10操作、元入力削除後5操作を確認。39保存HDF5のsourceが一致（意図した数値失敗の部分結果2件を含む） |
| 解析解・物理性 | CSTR、断熱充填、移送保存、Bohm減衰、熱/流通の解析解、表面原子数・site・膜/蓄積、電子/重粒子energy交換、 forcing境界と部分結果を確認 |
| ベンチマーク | 最終sourceで20 runを完了し、11図と `benchmarks/theory/results.json` を再生成。Maxwellianは解析解・独立quadrature、CRANE/DCは同じ初期条件・参照時刻、後期残光は測定図と比較 |
| マニュアル例 | Ar 201時刻・3状態・117派生系列、DC 151時刻・4状態を再計算。6図と抜粋値、単位、HDF5往復を照合。機能構成・計算フロー2図もDOTから再生成 |
| 来歴の照合 | 最終20 runとマニュアル2 runの全22 HDF5でsource/lock、59入力checksum（31実ファイル）、実効宣言、geometry、catalog/event測度2484行を照合し、不一致0。32入力の集約hashも独立に再計算 |
| DC数値誤差 | 参照303時刻と初期値の同一304点でrtol・atolを各1/3、最大stepを1/2にして再計算。標準波形の最大値で規格化した差は7比較量で最大6.1897×10⁻⁶ |
| RF校正・保存結果の再利用 | 校正CLIの断片を公開CaseSpecへ適用し、電圧制御・吸収電力制御で200 V、供給100 W、吸収80 W、自己バイアス−40 Vを再現。文書のnetwork 2例を保存HDF5だけで再描画し、明示flag以外のdomain・閾値等の保持を確認 |
| 最終PR gate | 通常778件成功（nightly専用1件は対象外）。statement＋branch 89.7675%、差分93%。Ruff・Pyrefly・4 import境界・複雑度・dead code・依存脆弱性・秘密情報・抑制追加の検査が成功 |
| 数値の追加確認・Windows | 改行正規化前の同一内容でpropertyは3 seed各6件、stress 1件、fresh snapshotのrates.py mutation 21/21件を検出。Git公開版へのLF正規化でPython AST・入力値の不変と対象7ファイルのhash対応を照合。通常PR入口はLF版でも再実行し、nightly追加実行と区別。Windows通常全体778件も成功 |
| 文書 | schemaの再現性と公開入力適合、本文DC表示値の実結果との対応を検査。Theory PDFは165数式・13図・全41ページを描画し、日本語・数式・図表・改ページを目視確認 |

係数CSV・参照・実験記録は保持し、最終レビューの再現・修正後の実行記録は `.quality-reports/final-review-20261122/`、
それ以前のログ・保存結果・中間検証は `.quality-reports/foundation-restructure/` に置きました。
役目を終えた15個のbytecode、前段階のPDF 1個と確認画像45個は整理し、現PDFと最終確認画像を保持しています。
最終PRは `.quality-reports/push-preflight/quality-lf/`、追加の数値検査は `.quality-reports/final-review-20261122/quality/summary.json` に記録しています。
Git公開前に既存の `eol=lf` 規則へ改行を統一し、staged sourceと実行sourceの112ファイルがbyte単位で一致することを確認しました。
実装・利用・科学的性質の必須指摘は修正し、影響する確認を再実行しました。

未確認は利用予定装置の独立測定による予測精度、および今回実行環境外のOS/Python版です。
装置検証の再開には、評価量・条件・許容差と校正に使わない測定データが必要です。
WindowsでOSの実行制御により起動できないPyrefly・import-linterはWSLで実行しました。
Windows nativeでの再開条件は、その2実行ファイルを許可した環境です。WSL成功をWindows native成功とは扱いません。


### 直前の互換経路撤去（再構成前の記録）

以下は今回の全体再構成より前の作業記録です。入力v4・結果v3、26例・570検査等はその時点の実行結果で、
現行仕様や今回の検証件数ではありません。現在の仕様は上記の現行経路と今回の検証記録を参照します。

#### 収支・出力・入力の整理

- 電源の系列名と単位は、初期状態の評価から全区間の指令に基づく判定へ変更しました。`outputs` と保存処理は同じ出力計画を使います。RF停止中の電圧・電力をゼロで保存し、CCPの停止中のシース厚さなど定義しない系列を要求すると、計算前に該当する区間を示します。外部表の切替でE/Nの供給元が変わる場合も、実際に使う区間の表で判定します。
- 外部表は、プラズマ部と表の測定点の電圧を系列名で区別します。電力を電圧と電流から得る場合は、単位倍率と補間を適用した値で計算します。DCの開回路ではギャップ電圧が電源電圧に等しく、電流と吸収電力はゼロです。DCの0 V指令でE/Nは0 Tdとなり、係数表に0 Td行があればその値を使います。 [電源CSVと系列の設定](index.md#power-table-settings)
- 新しい種を機構へ追加したとき、初期値の省略をゼロへ具体化します。表面反応の熱はYAMLとCSVで同じ符号を使います。実行時化学配列は符号付きの電子エネルギー移行量に統一し、損失だけを表す内部配列を削除しました。電源指令の未使用分岐、前版のツール設定を固定する独自検査とその専用テストも削除しました。
- 両極性拡散長の暗黙の代入を削除しました。15 Paの円筒残光例では、以前の体積／面積の値を使うと、円筒の最低次モードに対して減衰時定数が16.2%増えます。拡散長は形状と境界条件に対応する入力として明示します。

#### 互換・重複経路の撤去

- v3入力変換器と専用の例題・検査を削除し、入力はv4、結果HDF5はformat v3に統一しました。
- 正値損失の `energy_loss_eV` と符号付き移送へのfallbackを削除し、同梱機構と断面積を符号付きの単一表現へ更新しました。
- `experimental.approximate_two_term`、その生成格子・cacheと専用実装を削除しました。`argon_lxcat` と `rf_envelope_calibration` は独立に生成したSwarmのCSVを使います。
- `chemistry.experimental` の汎用状態と `experimental.extensions` を削除し、膜厚・壁蓄積のイベント積算を維持しました。
- `RecipeSegment` の直接電力・E/N指定とcallableによるE/Nからの電子エネルギー評価を削除し、power portと電子係数表へ経路を統一しました。
- 入力の `output.units` と、旧結果で不明な単位を引き継ぐ補完経路を削除しました。単独の `SimulationResult` が持つ系列単位のメタデータ契約は維持します。

削除した入口だけを維持する検査を除去し、CSTR、断熱充填、Bohm減衰、反応の保存則、
電子・気体のエネルギー収支、区間境界の電力積分は現行のportと係数表で確認します。

#### 当時の経路での再実行

通常の `tools.benchmarks.theory` で20 runと係数積分比較を実行し、全runの完了、
11図と `results.json` を確認しました。正本は `benchmarks/theory/results.json` です。
全runのHDF5に記録した入力・参照のchecksumを現在のファイルと照合し、
入力一覧の集約hash、Python実装の内容hashとも一致しました。
実行環境はWindows 11、Python 3.12.13で、単発の実測時間をその実行値へ更新しました。
物理比較の本文表示値は、丸めた桁で同じであることを確認して保持しています。
DC例ではrtol・atolを各1/3、最大step幅を1/2にした追加計算も行い、
標準設定の波形最大値で規格化した変化が全比較量で6.2×10⁻⁶以下であることを確認しました。

`tools.render_manual_examples` はWSLの通常入口で再実行し、6図を更新しました。
機能構成と計算フローの2図も、現在のDOT定義から再生成しました。
Arの201時刻・3状態・112派生系列と、DCの151時刻・4状態について、
HDF5再読込、抜粋値、系列単位を確認しました。比較元のCSV・測定参照YAMLは変更していません。

Maxwellianの断面積積分、独立電子計算の係数CSV、power portによる指令、
膜・壁の名前付きイベント積算は、それぞれ現在の物理的役割を持つため維持します。
保存結果の未知の量や単位を入力から補完する経路、旧形式だけを維持する入口・検査は戻しません。

#### 当時の最終確認（2026-11-22）

残存と誤削除を別々に点検し、公開入口、設定、動的な組み立て、配布wheelまで照合しました。
RFの0指令、外部表の時刻端点の丸め、inline chemistryの相対パス、
平均energy軸から得るE/Nの保存について、実際の計算経路で確認する回帰検査を追加しました。
表面状態の制約は汎用driverから切り離し、実際の吸着・脱離による途中時刻の逸脱を検出します。
独立監査で見つかったAr弾性衝突のm/Mと閾値の混同も修正し、Swarm係数を再生成しました。

| 確認 | 結果 |
|---|---|
| 最終PR gate（WSL/Linux、Python 3.12） | 570件成功、通常対象外のnightly 1件を除外。statement＋branch coverage 89.30%、変更行coverage 94% |
| lint・型・層構造・保守性・依存 | Ruff、Pyrefly、4つのimport境界、複雑度、dead code、秘密情報の精査、追加抑制、依存脆弱性の検査が成功 |
| nightlyの追加確認 | stress 1件、property testはseed 0・1・2で各6件成功。新しいLinux作業コピーでmutation 21/21件を検出 |
| 配布物からの利用 | 新しいwheelを独立環境にinstallし、checkout外の`python -I`で全26例を計算・保存・再読込。現行CLIの9操作も成功 |
| 保存内容 | 配布例の状態111列、派生量1662系列、解析ファイル31件の値・単位を確認。wheel内の105 Pythonファイルと36データファイルが現ソースと一致 |
| 科学的比較と説明 | Theoryの20 run、追加のDC許容差比較、マニュアル例を再実行。解析解・独立係数積分・外部計算・測定参照を区別して比較 |
| 文書 | schemaを再生成し、MkDocs strict buildが成功。Theory PDFの全41ページを描画し、日本語、数式、図表、改ページを目視確認 |

Windowsでは通常検査570件と変更後の対象検査も成功しています。
Windowsの実行制御で起動できない型・import検査はWSLで実行し、Windowsでの成功とは読み替えていません。
Windows側で同じ検査を再開する条件は、Pyreflyとimport-linterの実行が許可された環境を用意することです。
生成済みのmutation作業コピー`mutants/`は、旧実装・fixture・bytecodeを含む378ファイルと空の33ディレクトリを削除しました。
削除済みsourceに対応する旧bytecode 150ファイルも、package・tools・testsのcacheから除去しました。
既に内容を撤去した旧`examples/v3`の空階層12ディレクトリも整理しました。
旧Theoryを本文まで確認し、`output/pdf/`の旧版・途中版21 PDFと`tmp/pdfs/`の旧下書き・確認画像155ファイルを削除しました。
公開PDFと一致する完成版1 PDF、最終確認の41ページ画像と4枚の一覧画像は保持しました。
今回の検証ログ、保存HDF5、独立計算の元出力は`.quality-reports/current-removal/`に残し、
参照データ、実験記録、実行環境は撤去対象へ広げていません。

<a id="external-models"></a>

### 外部モデルとの接続確認

Swarm `27ada94`、PCD `d3910e0`、OESCR `f7bf408` を起点とするローカル版で実施しました。SwarmとPCDは作業ツリーの変更を含みます。同梱のSwarm用CSVはその試行のデータです。以下の比較は入出力の接続を確かめるもので、装置の予測精度は別途実験と比較します。 [Swarm](https://github.com/dkawa2523/swarm) · [PCD](https://github.com/dkawa2523/circuit_opt) · [OESCR](https://github.com/dkawa2523/CRmodel)

#### Swarmから電子係数を供給する

`plasma-global init swarm_argon_dc --output work/swarm-argon` で係数CSVとケースを取り出せます。Ar 100%、300 K、13.3 Pa、DC無磁場のtwo-term計算から40・50・60 Tdを取り込みました。ケースは励起1チャネルを持ち、初期45 Tdのとき平均電子エネルギーは5.50326 eV、電流は35.7413 mA、励起速度は2.73226×10²² m⁻³ s⁻¹でした。Swarmの45 Td補間値と、そこから算出した電流・反応速度に一致し、0 V区間で電流・反応速度がゼロになることを確認しました。

自分の表を作るときは、Swarmの公開版の `build-tables`、または今回使用した作業版の `export-coefficients` と `electron_swarm.io.load_coefficients` で、同じガス条件の連続した収束点を選びます。E/N、平均電子エネルギー、フラックス移動度の換算値 `μN` を `reduced_field_Td`、`mean_energy_eV`、`reduced_mobility_1_V_m_s` に対応付けます。反応の `rate_id` に対して `k:<rate_id>` に標的1粒子あたりの係数を入れます。反応項で電子密度と標的密度を掛けるため、混合比による事前の重み付けは不要です。未収束点を挟む区間や、DCとRFで異なる係数定義は入力条件を見直します。 [Swarmの公開係数利用手順](https://github.com/dkawa2523/swarm/blob/27ada948f9845859fc3b7fc3bbb7ac96b07d5b5c/docs/coefficient_lookup.md) · [BOLSIG+の係数定義](https://www.bolsig.laplace.univ-tlse.fr/manual.html)

表は固定組成の近似です。0 Tdを覆わない表では現在の冷たい電子の境界近似を使います。組成・気体温度や停止後の反応が関心量を左右する場合は、適用範囲を評価してから表を増やすか、別の電子エネルギー近似を選びます。

`argon_lxcat` と `rf_envelope_calibration` 用の表は、
`plasma_global/examples/data/swarm_argon_rf/electrons.csv` と同じ場所の `provenance.yaml` に収録します。
Swarmの `two_term`／`native_sg` を使い、Ar 100%、350 K、8 Pa、DC無磁場で、
1〜200 Tdの82点について収束を確認しました。電離1・励起30・運動量移行1の32チャネルを含み、
フラックス移動度からμNを得ます。断面積は同梱の `argon_lxcat/cross_sections.csv` と
[Zenodo 8192503](https://doi.org/10.5281/zenodo.8192503) に対応し、元データのライセンスとchecksumを記録します。

再生成はSwarmを利用できる環境で、リポジトリにある次の実行器を使います。出力先と作業先は未作成の場所にします。

```bash
python -m tools.benchmarks.prepare_argon_coefficients --swarm-root PATH --work-dir tmp/swarm-new --output NEWDATA
```

生成した表は両zoneで共有します。処理部の340 Kに合わせた再生成や、RF周波数に応じた分布計算は行いません。
RF／ICP／CCPにはDC local-field係数を使う近似として適用範囲を記録し、装置の校正とは区別します。
Globalの例題実行はCSVの読込だけで完結し、Swarm本体を実行時依存にしません。
原断面積のELASTIC過程に記された1.36×10⁻⁵は電子と標的の質量比m/Mで、反応閾値ではありません。
運動量移行断面積の `threshold_eV` は0とし、元のm/Mと定義をmetadataに残しています。
この表は1 Tdから始まるため、0 Tdでのcold/off境界と最初の移動度を使う方針をprovenanceに明記します。
0 Tdを実際に収録した表では、その行にある平均energy・係数を停止時にも使います。

#### 回路から電力と電圧を受け取る

PCDの `sim-run` が保存した過渡波形を、[電源CSV](index.md#power-table-settings)の `columns` で結び付けます。検証では10 Vのパルスを50 Ωの直列抵抗と50 Ωの負荷に与えました。負荷端で5 V、0.1 A、0.5 Wとなり、417点の吸収電力をGlobalに渡して、同じ時刻のHDF5系列と一致させました。壁・反応損失のない確認ケースで、負荷の区間入力は1.0334750 μJ、電子エネルギーの増加は1.0334754 μJでした。PCDにはパルスの時間を秒の数値で指定しました。 [PCDの電力評価](https://github.com/dkawa2523/circuit_opt/blob/d3910e0221ef8aff1e37f30e70e3b9a3fec23fc2/pcd/analysis/transient.py)

装置の負荷側に蓄積や素子損失があるときは、プラズマの吸収分と電子・気体の内訳を定めます。動的な双方向連携を追加する判断は、プラズマ応答による回路状態の変化が観測量に効くときです。その場合は、状態・時刻・エネルギーの所有者を決め、適応BDFのRHSが同じ状態で再試行できるようにします。 [SciPy BDF](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.BDF.html)

#### 保存状態からOESCRで発光を求める

`output.observables` に電子密度、気体温度、平均電子エネルギーを選び、保存状態を読み戻します。均一ArとMaxwell分布を仮定した1点の例では、次の入力を既存のOESCRケースに適用します。

```python
from plasma_global.output import read_result_h5

saved = read_result_h5("runs/global/result.h5")
i = 0
n_ar = saved.series("n[plasma,Ar]")[i]
n_e = saved.series("electron_density_m3[plasma]")[i]
tg = saved.series("gas_temperature_K[plasma]")[i]
te = (2.0 / 3.0) * saved.series("mean_energy_eV[plasma]")[i]
```

OESCRでは `gas_mixture` の中性圧力を `n_ar * kB * tg`、気体温度を `tg`、`plasma_state.ne_shells_m3` を `[n_e]`、`eedf.kind` を `te_maxwell`、`eedf.zones[0].te_eV` を `te` とします。観測の幾何・視線・分光器はOESCR側で設定します。今回、Arの一様状態を入力した2台の分光器について、手入力と保存値の受け渡しで2052ビンの強度が相対許容差10⁻¹²で一致しました。 [OESCRの入出力仕様](https://github.com/dkawa2523/CRmodel/blob/ff716d06415d65ab9cad9b14cc034935fe76426d/docs/io_spec.md)

非Maxwell分布ではEEDFの形と規格化が必要です。Swarmのセル幅積分とOESCRの点列積分で確率と励起係数を照合します。励起準位がプラズマの収支に影響する場合は、その緩和時間を運転時間と比較し、既存の化学種と反応で状態を表せるかを先に調べます。有効反応にまとめる場合は、各過程の粒子・エネルギー移行を一度ずつ計上します。 [励起準位の緩和と準定常近似](https://doi.org/10.1103/PhysRevE.49.5565)

### 次に実装する条件

DCの初期状態と壁損失を元入力に揃え、独立に計算した電子係数による波形を参照時刻で比較しました。
固定組成・固定温度の表からのずれが観測量を左右するときは、Swarm側で条件を振った係数を作り、
対象用途の感度を先に評価します。膜・壁の蓄積を反応へ反作用させる必要が生じた場合も、
物理項と依存関係を定めたモデルとして追加します。 [比較データの来歴](#dc-benchmark-data) · [計画と残る課題](ROADMAP.md)

## 文書と入力仕様の保守

説明は次の5ファイルにまとめます。リポジトリ直下の `AGENTS.md` には、実装時に共通する短い指針を置きます。

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
uv run --frozen --all-extras python -m mkdocs build --strict
uv run --frozen --all-extras python -m mkdocs serve
```

生成物は `docs/generated/case.schema.json` と `minimal_case.yaml` です。
生成の再現性と公開schemaへの適合を `tests/test_schema_docs.py` で確認します。
手書きの項目一覧をコードから独立した厳密schemaとして扱いません。

実行例の正本は `plasma_global/examples/` で、wheelにも同梱します。
別のexamplesツリーに同じcaseを複製しません。`init` が必要な資産だけを利用者の場所へコピーします。
`cases/` と `cases/benchmarks/` のケースを `init --list` で発見できるようにし、
同梱する外部ファイルは `chemistry/` と `data/` へ置きます。ケースの追加は同じ公開schemaのYAMLで行います。
本文のYAMLは完全なcaseか、適用先を明記した断片かを区別します。

マニュアルの図は `argon_analysis` と `zdplaskin_example2` の実計算から作ります。
`python -m tools.render_manual_examples --output runs/manual-new` は、入力一式の取り出し、
計算・保存、HDF5再読込での数値照合、6図の更新と `values.json` への保存値抜粋を行います。
plot・graph依存とGraphviz本体が必要です。DC例へ追加するのは出力選択だけで、物理条件は同梱caseを使います。
図・データの説明は統合マニュアルの第3・13・14節、変数表は各入力区分の節へ置きます。
既定値や対応変数を変えた場合は公開schema・compile・実行モデルと表を照合し、本文の例も実行します。
新しい説明ファイルや独立した物理入力を図のために追加しません。

### 利用者マニュアルの確認基準

2026-10-27の見直しでは、初見のエンジニアが「導入→用途の選択→入力の編集→実行→結果の判断」まで
たどれるかを基準にしました。変数表の網羅性だけでなく、作業のつながりを確認します。

| 観点 | 見つかった不足・偏り | 現行文書への反映 |
|---|---|---|
| 初回利用 | 仮想環境、実行Python、成功時に見る値の説明が不足 | 第2節に環境準備、終了状態・終端時刻・生成物の確認、最初の再描画を追加 |
| 用途別の編集 | 選択表はあるが、同梱例の何を変更するかが薄い | 第4節に一定通電、パルス、流通・移送、DC、表面・気体加熱・RFの作業手順を追加 |
| 読む順序 | 序盤の検証ケース一覧が、利用開始の導線を長くしていた | 第2節は用途別の例と配置案内。詳細な数値検証ケースは第15節へ移動 |
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
本文のAPI例、リンクと図、通常幅と狭い画面の表示を確認します。機能追加時にもこの役割分担を保ちます。

理論文書の根拠は `python -m tools.benchmarks.theory --output .quality-reports/theory-new` で
再計算できます。plot依存を使い、新しい出力先へ数値JSON、11図、20 runの結果を作ります。
実行器はcase YAMLの装置条件を使います。DCとAr後期残光の比較計算は、保存時刻を指定した参照時刻に合わせます。
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
| 問題設定 | 解析例と別コード比較の条件・省略した物理が追いにくい | 第13節の各問題群で入力値、期間、数値設定、比較対象を説明 |
| 図と数値 | 一次反応・移送・CRANE・DC・許容差の図が不足 | 20 runを再計算し、11図と表を対応付け。参照が終端だけの場合は点で表示 |
| 読む順序 | 導出・計算構成・再計算の入口を見つけにくい | 要旨後の案内と、第12節の検証対象一覧、付録Bを接続 |
| 機能の偏り | Ar・DCの数値一致を全機能の検証へ広げる余地がある | 第14.4節に電子・壁・輸送・電源・表面・熱・解析の根拠と残る比較を整理 |
| 総括と将来開発 | 結論だけでは本コードの到達点と次の開発の順序を把握しにくい | 第15節で機能・用途・検証の結論を総括。第16節で開発の目的・順序・完了条件を示し、進捗はROADMAPへ集約 |
| GitHubでの表示 | 数式のMarkdown解釈で空白指定や本文中の14式が失われ、マニュアルの表4か所でIDの山括弧がHTMLタグになる | 保護した数式記法と論理的な行分けを採用。設定名をコード表記へ修正し、GitHubの変換API・MathJax・PDFで照合 |
| 参照の意味 | 初期条件差、逆算係数、保存格子差が図から分からない | DCの初期状態を元入力に揃え、係数を独立に生成。参照時刻での波形差と終端差を表示 |
| 実測比較 | 空洞共振法と体積平均では密度の定義が異なる | Ar後期残光の空間分布が一定とみなせる時間帯で、規格化密度の傾きとその差を比較 |
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
独立数式はGitHubが扱える `math` コードブロック、本文中はdollar-backtick記法で保護します。
記法は[GitHubの公式説明](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/writing-mathematical-expressions)に従います。
複数の定義を一つの式番号で示す場合は `gathered` を使い、行末を `\\[3pt]` とします。
行末の裸の二重backslashをMarkdownの改行として解釈させず、式の定義の境目で整形します。
`tools/doc_math.py` がこの記法をWeb・PDF共通の読取へ接続し、数式自体の内容は変更しません。
GitHubのMarkdown変換後に式数・数式の内容・表・画像・リンクを照合し、Web版だけの確認で済ませません。
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
文書は別途 `python -m mkdocs build --strict` で確認します。
nightlyはPR gateに加えてseedを変えたproperty test、stress、mutationを行います。
Linux／WSLでは `nightly --native`、Windowsでは通常のnightly入口がWSLへ委譲します。

coverage.pyのstatement＋branch統合指標は85%以上、変更した本番コードは90%以上、
新規コードの循環的複雑度は10以下です。Ruff・Vultureの新規抑制や、skip／xfail等を
追加して失敗を隠しません。内部class配置を固定する検査や、同じ式の恒等式だけを検査するテストは増やしません。
単位、保存則、解析解、独立積分、forcing境界、入出力の復元を検査します。

`quality/baseline.json` はレビュー済みの既存Bandit・複雑度例外、`.secrets.baseline` は
detect-secretsのレビュー情報です。通常gateはこれらを書き換えません。
`python -m tools.quality baseline` は例外を精査して更新するための明示操作です。
Banditのレビュー済み例外はrule・severity・messageごとの件数を増やせず、実検出は登録した位置に限ります。
関数やファイルの移動だけで予算を増やしません。新規・悪化した複雑度例外は拒否されます。秘密情報候補の追加は確認済み `is_secret: false` が必須で、
公開checksumには理由を記録します。未確認・実秘密を例外へ登録しません。

CIはLinuxのPython 3.11／3.12／3.14、Windowsの3.12を対象とし、別jobで依存下限を検査します。
Linux 3.12ではGraphviz本体を導入して解析例を描画します。最低構成のjobはwheelをinstallし、
`python -I` でcheckoutをimportしない実行も確認します。
ローカルでのWSL成功をWindowsでの実行成功と読み替えません。

<a id="dc-benchmark-data"></a>

## DC比較のデータと再生成

| 資産 | 役割 |
|---|---|
| `benchmarks/references/zdplaskin_example2_source.yaml` | 元入力、Fortran、断面積、比較波形の出典と固定リビジョン |
| `benchmarks/references/zdplaskin_example2_history.csv` | 比較用の種密度・回路量。元データの303行をSIへ換算 |
| `benchmarks/references/zdplaskin_example2_bolsig.in` | BOLSIG+の計算条件。Arを背景とし、ArとAr*を係数の標的に含める |
| `plasma_global/examples/data/zdplaskin_example2/electrons.csv` | 配布ケースが読むE/N 1.5〜85 Tdの電子係数・フラックス換算移動度 |

元のZDPlasKin 2.0aを再計算するには、出典YAMLのリビジョンを取得し、Linux／WSLの作業コピー内の `example2/` で実行します。
コミット済み出力は別途保管しておくと再計算値と照合できます。

```bash
gfortran -O2 -ffree-line-length-none -fallow-argument-mismatch -c dvode_f90_m.F90 master_module.F90 zdplaskin_m.F90 main.f90
gfortran -o reference-dc dvode_f90_m.o master_module.o zdplaskin_m.o main.o ./bolsig_x86_64.so
printf '\n\n' | ./reference-dc
```

同じ断面積 `bolsigdb.dat` と同梱の `zdplaskin_example2_bolsig.in` をBOLSIG+ 07/2024の実行用ディレクトリへ置き、
`./bolsigminus zdplaskin_example2_bolsig.in` を実行して `independent.dat` を得ます。
次をGlobalのリポジトリから実行します。 `SOURCE.dat` には元の保存出力か、その再計算ファイルを指定します。 [ZDPlasKinの元入力とソース](https://github.com/Hemadityamalla/ZDPlaskin/tree/48744203fe93368d344315645e9617ed351cadd3/example2) · [BOLSIG+ 07/2024](https://www.bolsig.laplace.univ-tlse.fr/download.html)

```bash
python -m tools.benchmarks.prepare_zdplaskin_reference --zdplaskin-output SOURCE.dat --bolsig-output independent.dat --output work/dc-reference
```

元の保存出力から取り出した密度・電圧・電流などを `zdplaskin_example2_history.csv` に、BOLSIG+の量別出力を `electrons.csv` に変換します。
係数は同じグリッドの収束済み行を使います。今回の再計算では元の保存波形303行の全列を再現し、変換した2つのCSVも同梱物と一致しました。
以前の係数は保存出力の反応進行速度などから逆算していました。源流が同じ回帰で得ていた一致は、その版の履歴で確認できます。

<a id="argon-afterglow-data"></a>

## Ar後期残光の測定参照

単独ケースは `plasma-global init benchmarks/argon-afterglow --output work/afterglow` で取り出します。
壁に達したイオンを中性Arに戻し、円筒の最低次モードから得た拡散長と、独立測定の移動度から計算した拡散係数をYAMLで指定します。
単独実行では20 μs間隔、比較実行では測定図の8時刻でHDF5に保存します。 [入力と方程式](THEORY.md#argon-afterglow-benchmark)

`benchmarks/references/beckers_argon_afterglow.yaml` に、出典PDFのハッシュと測定図の読取値を収録します。
著者公開稿の図5の画像で、横軸の0・2000 μsと縦軸の10¹⁵・10¹⁴ m⁻³で座標を校正しました。
各時刻に最も近い列とその両隣で、黒い測定線が占める高さの中点を対数密度へ換算しています。
赤い指数近似線は読取の対象から外し、画像の線幅と軸の校正を含む読取幅として±0.016 decadeを設定しました。 [Beckersらの図5](https://pure.tue.nl/ws/portalfiles/portal/25507508/beckther2016.pdf)

比較器は同じ時刻で保存したHDF5を再読込し、両系列の対数密度を自由切片付きの直線で近似します。
読取の影響は、各点を指定幅内で傾きが最も急・緩やかになる方向に振った場合の範囲です。
移動度の報告幅は別に伝播させます。原著の生波形、測定の誤差構造、実験時の実測温度が得られれば、統計的な不確かさの評価へ進めます。 [移動度の独立測定](https://doi.org/10.1088/0022-3700/10/18/036)

## RF校正

```bash
python -m tools.calibration.rf_envelope --help
```

RF校正ツールは測定量から係数候補とv5用の `power_model`／`power_command` 断片を出します。
測定点以外での精度認定や、自動的なcase全体の生成とは区別します。
次は係数同定を確認するための仮想測定点で、実験データや独立予測の検証結果ではありません。

```bash
python -m tools.calibration.rf_envelope --role bias --frequency-Hz 2e6 --commanded-power-W 100 --absorbed-power-W 80 --voltage-rms-V 200 --current-rms-A 2 --dc-self-bias-V=-40 --write runs/rf-calibration.yaml
```

供給の実電力は同じ基準点のnet値を用います。`delivered_power_W`、順方向−反射、
`commanded_power_W` を複数指定する場合は、同じnet実電力を表す必要があります。
効率は吸収実電力／供給実電力、`effective_impedance_ohm` はV_rms²／供給実電力です。
上例では効率0.8、有効抵抗係数400 Ωとなり、モデルは200 V、吸収80 W、自己バイアス−40 Vを再現します。
測定したV/Iの100 Ω、V×Iの400 VA、力率0.25は別の診断値で、RMS電流と位相をモデルへfitするものではありません。
V_rms／I_rmsは位相を含まないインピーダンスの大きさで、実電力換算係数の代用にはしません。
電圧制御の断片に必要な実電力が不明な場合や、非有限値・負電力・整合しない基準値では理由を示して拒否します。
別コード回帰の閾値と意味は[マニュアルの検証](index.md#validation)へ集約しています。

過去の収束・局所感度の数値を将来の版へ無条件に転用せず、物理や既定値を変えた場合は同じ定義で再計算します。
報告には観測量、比較時刻、誤差指標、条件差、入力・code hash、終了状態を残します。

## 配布の確認

wheelは新しいcheckoutまたはクリーンなsource copyから作り、install後に同梱ケースを検査します。
古いbuildツリーを再利用すると削除したmoduleが混入し得るため、配布物の内容も確認します。
checkout外で `python -I` を使い、import元がインストール先であることを確認します。
各caseを `init` して計算・保存し、入力参照がコピー先のYAML／CSVだけで閉じることを検査します。
図の再生成は保存HDF5だけを使い、元caseの再計算で系列・単位・来歴を補完しません。
