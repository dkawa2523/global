# 外部ベンチマーク報告（日本語論文調ドラフト）

生成日時 (UTC): `2026-04-25T11:23:48.098173+00:00`

## 1. 目的と読み方

本報告の目的は、`plasma_global` が何を状態量として解き、どの物理を closure として外から与え、どの benchmark が何を実際に検証しているのかを、第三者が物理的に追跡できる形で整理することである。ここで重視しているのは、単に「合った / 合わない」を列挙することではなく、**どの理論差がどの結果差を生んでいるか**を分離して説明することである。

本コードは reactor-averaged の 0D / multi-zone global model であり、gas-phase species、電子エネルギー、必要に応じて gas temperature、surface coverage、wall inventory、film thickness を状態量として扱う。一方で、電子速度分布関数そのもの、sheath の時間発展、空間分布をもつ輸送場、RF 周期内の高速電子加熱は直接解いていない。したがって、本コードの妥当性は、**global model としてどの closure を採用し、その closure が benchmark に対してどこまで整合的か**という観点で読む必要がある。

また、本報告に含まれる benchmark は性格が異なる。ZDPlaskin と CRANE は比較的強い same-footing benchmark、PyGMol は closure 差の診断 benchmark、LoKI O2 DC glow は digitized 文献 benchmark、robustness suite は boundedness と workflow usability の補強証拠である。これらを同列に「全部 validation」とは呼ばず、**strict validation / diagnostic comparison / digitized support / applicability evidence** を分けて扱う。

## 2. コード構成と処理の流れ

### 2.1 モジュール構成

| モジュール | 主な責務 | 物理・数値上の意味 |
| --- | --- | --- |
| `config` | YAML 読み込み、path 解決、入力検証 | 何を解くかを定義する。物理モデルの矛盾はここで早めに落とす。 |
| `chemistry` | species / reaction / cross section / mechanism bundle | 反応式・断面積・速度モデルの入口を集約する。 |
| `reactor` | zone / surface / edge / inlet / power port | 0D / multi-zone 幾何と流路・表面を保持する。 |
| `eedf` | Maxwell / rate-table / internal two-term Boltzmann | 電子衝突 rate と輸送係数の closure を返す。 |
| `electrical` | direct power / dc series / external table / rf envelope / ccp / icp | 吸収電力と reduced field を backend として供給する。 |
| `physics` | gas-phase core / surface core / prescribed electrons | 反応項、壁損失、表面反応、準中性 closure を評価する。 |
| `numerics` | state layout / global system / analytic Jacobian / BDF | stiff ODE を安定に解く。 |
| `observables` と `diagnostics` | CSV / YAML / budget / provenance | 結果を物理解釈可能な形に整える。 |
| `workflows` | load -> build -> run -> export | 全体の実行順序を一つの入口にまとめる。 |

### 2.2 実行時の処理フロー

| 段階 | 処理名 | 内容 |
| --- | --- | --- |
| 1 | 入力読込 | `run.yaml` から chamber, recipe, chemistry manifest を解決する。 |
| 2 | 検証 | 設定・反応機構・電気 backend・壁損失 family を検証し、二重計上や欠損入力を防ぐ。 |
| 3 | 状態配置 | zone ごとの gas densities, electron energy, optional gas temperature, surface states の添字を確定する。 |
| 4 | backend 構築 | EEDF backend と electrical backend を registry から生成する。 |
| 5 | 大域系組立て | `GlobalPlasmaSystem` が gas, surface, electrical, observables を束ねて RHS / Jacobian を提供する。 |
| 6 | 時間積分 | recipe step ごとに SciPy BDF を走らせ、各 segment 後に admissible state へ投影する。 |
| 7 | 出力整形 | observables, summary, budgets, plots を書き出す。 |

この構成の要点は、`GlobalPlasmaSystem` が solver-facing な安定した API を持ちつつ、実際の物理は `GasPhaseCore`、`SurfaceCore`、`ElectricalCouplingAdapter`、`ObservablesAdapter` に分離されていることである。これにより、反応機構、壁損失、電子 kinetics、回路 backend を個別に変更しても、大域 ODE の入口は大きく壊れない。

### 2.3 設計カテゴリごとの利点

| 設計カテゴリ | 現在の設計 | 直接の利点 | 第三者にとっての理解上の利点 |
| --- | --- | --- | --- |
| 責務分離 | `physics`, `electrical`, `eedf`, `numerics`, `workflows` を分離 | 反応、電力、電子 kinetics、数値積分の変更が相互に絡みすぎない。 | 「何が物理差で何が数値差か」を切り分けやすい。 |
| registry ベースの backend 交換 | EEDF / electrical / integrator を registry から構築 | 既存 solver 本体を崩さずに新 backend を追加しやすい。 | 将来の circuit model や EEDF model を局所変更で差し込める。 |
| solver-facing API の安定化 | `GlobalPlasmaSystem` が `initial_state`, `rhs`, `jacobian`, `compute_observables` を提供 | 数値 solver 側のインターフェースが安定する。 | 物理モジュールを変更しても時間積分器や出力系への影響を最小化できる。 |
| 状態配置の独立化 | `StateLayout` が gas / energy / surface state の添字を集中管理 | 状態量が増えても添字の不整合を起こしにくい。 | 新しい状態変数の追加が局所化する。 |
| 診断と主 solver の分離 | budget, provenance, observables, benchmark dashboard を主 solver から分離 | production solve の経路を複雑化しにくい。 | 診断の追加や削除が計算本体を壊しにくい。 |
| benchmark ツールの外部化 | `tools/external_benchmarks` で比較ロジックを保持 | 比較用の調整が本体物理を汚染しにくい。 | PyGMol や LoKI のような外部比較を、本体設計と切り分けて育てられる。 |

ここで重要なのは、設計上の利点が単なる保守性にとどまらないことである。低圧プラズマ global model では、結果差が chemistry 差、wall-loss 差、power coupling 差、EEDF 差のどこから来るのかを追跡できないと、比較自体が不毛になりやすい。本コードの構成は、その差分分解を可能にする方向へ寄っている。

### 2.4 将来拡張時の入口と利点

| 拡張カテゴリ | 主な追加場所 | 設計上の利点 |
| --- | --- | --- |
| 新しい EEDF / rate backend | registry に backend を追加し、`EEDFResult` 相当の rate・transport を返す。 | global ODE 本体を変えずに electron-impact closure を差し替えられる。 |
| 新しい電気モデル | `electrical` backend と `PowerRequest/PowerResult` を実装する。 | DC, pulsed-DC, RF envelope, external table と同じ接続点で増設できる。 |
| 新しい wall-loss / surface model | `surface_models` と `SurfaceCore` 側に family を追加する。 | Bohm / ambipolar と同じ validation 境界で、新 closure を二重計上なく導入できる。 |
| 新しい状態変数 | `StateLayout` と core の初期化・投影・observables を拡張する。 | 添字管理が一箇所なので、状態追加時の破壊範囲を読める。 |
| 新しい診断・可視化 | `diagnostics`, `observables`, `plotters` に追加する。 | 主 solver の RHS/Jacobian を汚さずに説明力だけ増やせる。 |
| 新しい benchmark | `tools/external_benchmarks` に追加し、dashboard から集約する。 | 本体 physics を benchmark 用に曲げずに比較体系を広げられる。 |

### 2.5 拡張時に破綻しにくい理由

| ガード | 意味 |
| --- | --- |
| 入力検証が先に走る | 未設定パラメータや incompatible family を run 前に落とせる。 |
| 物理と数値の境界が明確 | closure 追加時に、どこまでが物理変更でどこからが solver 変更かを説明しやすい。 |
| benchmark を本体外に置いている | 比較のための特別処理が production solver に混ざりにくい。 |
| diagnostics が独立している | 新機能を入れたときの budget / observables を後付けしやすい。 |

言い換えると、本コードは「何でも一つの巨大 solver に押し込む」構成ではなく、**closure と workflow を差し替えながら本体の整合性を保つ**構成である。これは低圧プラズマのようにモデル差が大きい分野で特に有利であり、将来 RF、外部回路、state-resolved chemistry、surface calibration を伸ばすときにも、変更範囲を説明しやすい。

## 3. 支配方程式と closure

### 3.1 種密度方程式

本コードの gas-phase state は、各 zone の species 密度 `n_k` に対して概念的に次式で進む。

```text
d n_k / dt = Σ_r ν_k,r R_r + S_k,flow + S_k,surface - L_k,wall
R_r = k_r(Te, E/N, Tg, θ, ...) Π_j n_j^α_j
```

ここで `R_r` は反応 `r` の体積反応速度、`ν_k,r` は species `k` の化学量論係数、`S_k,flow` は流入出、`S_k,surface` は表面反応との結合、`L_k,wall` は global wall loss を表す。重要なのは、速度係数 `k_r` が単なる定数ではなく、EEDF backend から返される電子衝突 rate、あるいは Arrhenius / table model を通して **電子エネルギーや reduced field に依存している**点である。

### 3.2 電子密度 closure と電子エネルギー方程式

production 設定では電子密度 `n_e` は通常、独立 ODE ではなく準中性条件からの代数量である。概念的には

```text
n_e ≈ Σ_i z_i n_i - Σ_a |z_a| n_a
```

として与えられる。これは global model において、イオン生成と損失の結果として電子密度が決まるという立場に対応している。benchmark 用には `prescribed_profile` も用意されているが、それは one-way 連成または固定 `n_e(t)` での rate 比較のための機能であり、自己無撞着な放電解ではない。

電子エネルギー密度 `w_e` は概念的に次で進む。

```text
d w_e / dt = P_abs / V - Σ_r (Δε_r R_r) - Q_wall - Q_flow
```

ここで `P_abs` は electrical backend が返す吸収電力、`Δε_r` は反応ごとの電子エネルギー損失、`Q_wall` と `Q_flow` は壁面・流れに伴うエネルギー損失である。本コードは EEDF 全分布を ODE 状態としては持たず、**電子エネルギー状態から rate と輸送係数を問い合わせる**設計である。したがって、`local_field` closure と `mean_energy` closure は理論的に同じではない。

### 3.3 壁損失 closure

正イオンの主要 wall loss は Bohm-family closure であり、現在の既定は `bohm_edge_loss` である。概念式は

```text
Γ_i = h_factor * 0.61 * n_i * sqrt(ε_e / m_i)
(d n_i / dt)_wall = - (A_eff / V) Γ_i
```

である。ここで `h_factor` は edge-to-center 的な補正を担う。別 family として `ambipolar_diffusion` もあるが、これは

```text
d n_i / dt = -k_loss n_i
```

という一次 volumetric sink として扱われ、**Bohm と ambipolar を同時加算しない**。この制約は、同じ wall sink を二重計上しないための物理的・設計的なガードである。

### 3.4 電気 backend

DC / pulsed-DC の `dc_series_circuit` backend は、プラズマを導電性負荷として表した reduced circuit である。

```text
G_plasma = e * n_e * μ_e * A / g
R_plasma = 1 / G_plasma
I = V_source / (R_ballast + R_plasma)
V_gap = I R_plasma
P_abs = η_abs V_gap I
E/N = |V_gap / g| / N
```

これは sheath capacitance、matching network、RLC ringing、SPICE DAE を解くものではない。しかし、低圧 global model において「回路から見た gap voltage と absorbed power を self-consistent に決める」には有効であり、ZDPlaskin parity のような DC benchmark では十分に意味がある。一方、`external_circuit_table` は one-way loose coupling、`rf_envelope` は RF 周期平均の recipe-level backend であって、いずれも full circuit solve ではない。

### 3.5 本コードが自己無撞着に解いているもの / いないもの

| 項目 | 扱い | 意味 |
| --- | --- | --- |
| gas species | ODE 状態 | 反応、流れ、壁損失、表面反応で進む |
| electron energy | ODE 状態 | 吸収電力と衝突損失を介して進む |
| electron density | 代数 closure（既定） | 準中性条件から決まる |
| electron-impact rate | EEDF backend 由来 | Maxwell / rate-table / two-term backend で置換可能 |
| gap voltage / absorbed power | electrical backend 由来 | direct power / dc circuit / external table / rf envelope など |
| spatial profile | 直接は解かない | profile は global closure や external benchmark で間接評価する |
| sheath dynamics / IEDF | proxy のみ | 高忠実度な sheath solve は未実装 |

## 4. ベンチマーク体系と理論差の整理

| ベンチマーク | 参照側の理論像 | 本コードとの理論差 | この比較で実際に読めること |
| --- | --- | --- | --- |
| ZDPlaskin example2 | 0D Ar 放電 + 回路 parity。実効的には species, absorbed power, circuit-derived E/N を同じ土俵に乗せる。 | local 側は内部 `dc_series_circuit` を使い、電子衝突は committed output から逆算した external E/N-rate table を使う。live BOLSIG+ 実行ではない。 | 現在の DC Ar parity case に限れば、power・E/N・最終組成の self-consistency を見る強い benchmark になる。 |
| CRANE TwoReactionArgon | 定数反応速度の 2 反応 ODE。E/N は rate 選択用の与条件で、動的な electron energy や回路は持たない。 | 本コード側の full plasma model のうち、比較できるのは gas-phase scalar chemistry と SI 変換だけ。 | reaction assembly、単位変換、stiff ODE 積分の健全性を切り出して検証できる。 |
| PyGMol | 別実装の executable global model。自然状態では rate・power・wall loss の closure が揃っていない。 | 本コードと同一方程式ではない。したがって baseline 差は精度差ではなく closure 差を多く含む。 | same-rate -> same-power -> same-wall-loss の順に揃えると、どの closure が差を支配しているかを分解できる。 |
| LoKI O2 DC glow | 公開論文の 0D と 1D の O2 DC glow を digitize した benchmark。pressure sweep と radial profile を使う。 | official raw output ではなく digitized reference。local solver の executable parity ではない。 | 低圧 O2 の primary discharge trend と profile shape の plausibility を評価するのに向く。 |

### Coverage map

![Benchmark coverage matrix](external_benchmark_coverage_matrix.png)

この coverage map では、`S` は強い benchmark 軸、`D` は診断的に有用な軸、`0` はその benchmark で本質的には見ていない軸を表す。重要なのは、CRANE や LoKI を ZDPlaskin と同じ「厳密 validation」として読まないことである。

## 5. 定量結果の要約

実行コード parity では次の mismatch 指標を使う。

```text
M_q = 100 × | q_local / q_ref - 1 |
```

したがって `M_q = 0 %` が完全一致であり、数 % 程度であれば同一 closure / 同一入力条件の benchmark ではかなり良い一致と読める。LoKI については raw executable ratio ではなく、digitized 曲線から再構成した相対差と論文 Fig.11 の差分 envelope を用いて解釈する。

| ベンチマーク | 参照型 | 主指標 | 差 [%] | 解釈 |
| --- | --- | --- | --- | --- |
| ZDPlaskin example2 | 厳密な実行コード parity | max exact-match deviation over species and E/N | 2.9% | strong same-footing agreement for the current DC Ar parity case |
| CRANE TwoReactionArgon | 厳密な scalar chemistry parity | max final-state relative error | 0.000% | reaction assembly, units, and stiff ODE integration are effectively exact |
| PyGMol Ar baseline | 広義の実行可能 global model 比較 | production-step electron-density mismatch | 39.8% | reasonable order agreement, but still a meaningful model-closure gap |
| PyGMol same-footing final stage | 診断分解ベンチマーク | production-step electron-density mismatch | 28.1% | most of the PyGMol gap is explained by power and wall-loss closure differences |
| LoKI O2 DC glow | 文献 digitize 参照ベンチマーク | max primary-discharge difference over Tg/Tnw/E/N | 6.6% | primary discharge quantities are close, but species complexity must be read separately |

### 統合グラフ

![Integrated overview](external_benchmark_overview.png)

この図では raw ratio ではなく percent-space の mismatch を使っている。これにより、ZDPlaskin / CRANE の「ほぼ一致」と、PyGMol baseline の「closure 差を含む有意な差」、LoKI の「一次放電量は近いが化学は別に読むべき」を一枚で見分けやすくしている。

### PyGMol 差分分解グラフ

![PyGMol decomposition](external_benchmark_pygmol_decomposition.png)

PyGMol については、単一の mismatch 値よりも staged decomposition の方が本質的である。ここでは概念的に

```text
Δ_total ≈ Δ_rate + Δ_power + Δ_wall + Δ_residual
```

という読み方をする。もちろん実際の系は非線形なので厳密な線形加算ではないが、same-rate -> same-power -> same-wall-loss と段階的に固定することで、どの closure が結果差の主因かを切り分けられる。

## 6. ベンチマークごとの詳細解釈

### 6.1 ZDPlaskin example2: DC Ar parity

ZDPlaskin との比較は、現在の外部セットの中で最も強い same-footing discharge benchmark である。参照側も本コード側も、0D Ar discharge と回路由来 E/N を持つため、species・absorbed power・physical E/N を同時に比較できる。

ただし完全同一ではない。local 側は内部 `dc_series_circuit` backend を使い、electron-impact rate は committed output から逆算した external E/N-rate table を読む。したがってこれは「現在の ZDPlaskin example2 と同じ土俵の parity」であって、「任意の Ar/DC ケースに一般化された validation」ではない。

指標は次の通りである。

| 量 | 比または値 | mismatch |
| --- | --- | --- |
| 最終電子密度比 `n_e(local) / n_e(ZDPlaskin)` | 1.0155 | 1.55% |
| 最終 `Ar*` 密度比 | 1.0214 | 2.14% |
| 最終 `Ar+` 密度比 | 1.0285 | 2.85% |
| 最終 `Ar2+` 密度比 | 1.0155 | 1.55% |
| 現在 run の回路場 E/N 比 | 1.0051 | 0.51% |
| 吸収電力比 `Pabs(local) / V I(ZDPlaskin)` | 0.9978 | 0.22% |

理論的には、ZDPlaskin 側と local 側で比較すべき reduced field は固定 power proxy ではなく **回路 backend が返す gap voltage 由来の E/N** である。本コードでは

```text
E/N = (V_gap / g) / N_gas
N_gas = p / (k_B T_g)
```

を用いており、この current-run E/N が ZDPlaskin 保存値と約 0.51% の差で一致している。ここから、少なくともこの parity case では、E/N の単位換算・gap・pressure・temperature の扱いが物理的に揃っていると読める。

一方、power も species も小さな mismatch に収まっているため、現在の DC Ar parity case に限れば、**回路 -> E/N -> electron-impact rate -> species inventory** の連鎖が破綻していないことを示している。具体的には最終電子密度は local `3.2196e+17` m^-3、ZDPlaskin `3.1704e+17` m^-3 であり、species 系の最大差は 2.85% である。

### 6.2 CRANE TwoReactionArgon: scalar chemistry の切り出し検証

CRANE は full discharge benchmark ではなく、**定数速度係数を持つ 2 反応 ODE** を same-footing で解く benchmark である。理論式は概念的に

```text
d n_e / dt      = + k_ion n_e n_Ar - k_rec n_e n_Ar+
d n_Ar+ / dt    = + k_ion n_e n_Ar - k_rec n_e n_Ar+
d n_Ar / dt     = - k_ion n_e n_Ar + k_rec n_e n_Ar+
```

であり、ここには電子エネルギー方程式、回路、壁損失、EEDF backend は本質的には現れない。したがって、この比較が検証しているのは **反応組立て・cm 系から SI 系への変換・stiff ODE 積分** である。

| 量 | 相対誤差 [%] | 相対誤差（無次元） |
| --- | --- | --- |
| 電子密度相対誤差 | 0.000237% | 2.366e-06 |
| `Ar+` 密度相対誤差 | 0.000237% | 2.366e-06 |
| `Ar` 密度相対誤差 | 0.000000% | 2.044e-12 |

誤差は事実上ゼロに近く、この benchmark が示すのは「本コードが full plasma model として正しい」ことではなく、**少なくとも scalar chemistry ODE の数値実装は壊れていない**ということである。この切り分けは論理的に重要で、CRANE 一致をもって EEDF や wall loss を主張してはいけない。

### 6.3 PyGMol: closure 差の診断 benchmark

PyGMol との比較は、**別実装の executable global model と closure 差を分解する**ための benchmark である。baseline の mismatch だけを見ると、本コードが悪いのか、PyGMol が悪いのか、単に closure が違うだけなのか判別できない。そこで本レポートでは、electron-impact rate、absorbed power、wall-loss coefficient を段階的に揃えている。

| 段階 | 何を揃えたか | production `n_e` mismatch | power 比 | wall-loss 比 | mean-energy 比 |
| --- | --- | --- | --- | --- | --- |
| baseline | PyGMol ネイティブ closure | 39.8% | 1.361 | 0.0535 | 1.460 |
| same local-fit | local Arrhenius fit で electron-impact rate を寄せる | 1818.4% | 1.371 | 0.0394 | 0.778 |
| same local-table | local rate table で electron-impact rate を同一化 | 1823.0% | 1.371 | 0.0399 | 0.797 |
| same-rate / recipe-power / pygmol-wall | rate のみ local に合わせる | 1823.0% | 1.371 | 0.0399 | 0.797 |
| same-rate / local-power / pygmol-wall | rate と吸収電力を local に合わせる | 1299.3% | 0.960 | 0.0388 | 0.758 |
| same-rate / local-power / local-wall | rate・吸収電力・壁損失を local に合わせる | 28.1% | 0.928 | 1.0000 | 0.833 |

この表から、same electron-impact rate だけでは差がむしろ非常に大きく残ることが分かる。`same local-table` の production `n_e` mismatch は 1823.0% であり、電子衝突 rate を揃えるだけでは discharge state は揃わない。

一方、same-rate に加えて absorbed power を揃えると差は縮み、さらに wall-loss coefficient まで local 側に合わせると mismatch は 28.1% まで低下する。これは、PyGMol と本コードの差の主因が electron-impact rate そのものではなく、**power closure と wall-loss closure** にあることを示す。

物理的には、低圧 global model の電子密度は概念的に

```text
production ~ ionization source / global charged-particle loss
```

で決まる。ionization source は electron-impact rate と absorbed power に敏感であり、loss は wall-loss coefficient に強く支配される。したがって PyGMol 差分分解は、「なぜ合わないか」を物理量のレベルで示している点に価値がある。これは strict validation ではないが、**本コードが closure 差を論理的に把握できる設計である**ことの証拠になる。

### 6.4 LoKI O2 DC glow: 低圧 O2 trend / profile の plausibility

LoKI O2 DC glow は official raw output parity ではなく、公開論文図の digitization に基づく benchmark である。そのため、ここで主張できるのは「low-pressure O2 の primary-discharge trend と profile shape が大きく外れていない」ことであり、「数値値が完全に一致する」ことではない。

| 量 | 最大差 | 読み方 |
| --- | --- | --- |
| `Tg,av` 最大差 | 4.39% | 一次放電量として十分近い |
| `Tnw` 最大差 | 3.19% | 壁近傍温度の差も小さい |
| `E/N` 最大差 | 6.56% | 一次放電量としては整合的 |
| extended neutral subset proxy 最大差 | 40.52% | 重い中性化学では差が増える |
| `O3` 最大差 | 241.46% | 最も大きい outlier |
| profile shape check | True | 傾向としては 0D / 1D の差を超えて破綻していない |

`Tg,av`, `Tnw`, `E/N` の primary-discharge 量は最大でも 6.56% 程度であり、digitized benchmark としてはかなり良い。一方、`O3` は 241.46% の outlier であり、重い中性化学・壁面化学・輸送近似の弱さが残っていると読める。

LoKI 側は 0D と 1D の比較を含むため、この差は単なる reaction set 差ではなく、**transport と profile 仮定の差**を強く反映する。したがって LoKI は、本コードの一次放電量が plausibility を持つことの裏付けにはなるが、重い中性化学を量的に validation したことにはならない。

## 7. 本コードの優位性

本コードの優位性は、現時点では「常に他コードより精度が高い」といった単純な主張ではない。より正確には、次の 4 点にある。

1. **closure を分解して扱える設計**
   chemistry、EEDF、wall loss、electrical backend が分離されており、どの仮定が結果差を生んでいるかを追いやすい。
2. **strict validation と diagnostic comparison を混同しない benchmark discipline**
   ZDPlaskin / CRANE を backbone としつつ、PyGMol や LoKI を役割の異なる benchmark として位置づけている。
3. **stiff chemistry を扱う数値基盤**
   analytic sparse Jacobian と BDF により、反応数が増えても設計としては持ちこたえやすい。
4. **将来拡張の入口が整理されている**
   external circuit table、rf envelope、prescribed electron profile、surface diagnostics などが既に backend / workflow の単位で分かれている。

この優位性は、単一 benchmark の一致値よりも、**どこが一致し、どこが一致せず、その理由をどの程度説明できるか**に表れている。PyGMol 分解が成立していることは、その代表例である。

## 8. 制約と overclaim を避けるための整理

| ベンチマーク | 主な制約 | なぜ重要か |
| --- | --- | --- |
| ZDPlaskin example2 | single committed Ar discharge case | strong for same-footing DC parity, weak for generality across chemistry and operating space |
| CRANE TwoReactionArgon | scalar chemistry only | does not validate circuit closure, wall loss, transport closure, or electron-energy physics |
| PyGMol Ar baseline / same-footing | independent executable model with different native closure choices | useful for sensitivity decomposition, but not a strict accuracy proof for the local solver |
| LoKI O2 DC glow | digitized publication figures rather than official raw-output files | good for pressure trends and profile-shape credibility, but not raw-output parity |

本レポートで強く言えるのは、(i) scalar chemistry ODE は極めて正確、(ii) 現在の DC Ar parity case は strong same-footing agreement、(iii) low-pressure O2 の primary trend / profile shape は digitized benchmark に支えられている、までである。逆に、全 chemistry、全圧力、全電源波形での予測精度まで主張するのはまだ早い。

## 9. 現在 support される範囲と未検証範囲

### 9.1 本文向け short table

| 項目 | 現状の表現 | 根拠 |
| --- | --- | --- |
| Reaction assembly, SI conversion, and stiff ODE integration | validated within current benchmark scope | CRANE TwoReactionArgon |
| Current DC Ar same-footing species and physical E/N parity | validated within current benchmark scope | ZDPlaskin example2 |
| Low-pressure O2 primary-discharge pressure trends and profile shapes | supported by digitized benchmark | LoKI O2 DC glow digitized benchmark |
| Cross-code global-model agreement without closure harmonization | not yet validated | PyGMol executable comparison |
| Quantitative ozone-heavy O2 chemistry | not yet validated | LoKI O2 DC glow O3 comparison |
| Generality across pressures, power waveforms, and unrelated chemistries | not yet validated | current external set |

### 9.2 supplement 向け detailed table

| 項目 | 現状の表現 | 証拠 | コメント |
| --- | --- | --- | --- |
| Reaction assembly, SI conversion, and stiff ODE integration | validated within current benchmark scope | CRANE TwoReactionArgon | final-state relative error remains at about 2.4e-4 % or smaller |
| Current DC Ar same-footing species and physical E/N parity | validated within current benchmark scope | ZDPlaskin example2 | headline exact-match deviation remains below about 3 % for the current parity case |
| Low-pressure O2 primary-discharge pressure trends and profile shapes | supported by digitized benchmark | LoKI O2 DC glow digitized benchmark | primary Tg/Tnw/E/N difference stays near 6.6 % and profile-shape checks pass |
| Cross-code global-model agreement without closure harmonization | not yet validated | PyGMol executable comparison | PyGMol remains sensitive to power and wall-loss closure, so baseline cross-code agreement is not a proof of correctness |
| Quantitative ozone-heavy O2 chemistry | not yet validated | LoKI O2 DC glow O3 comparison | O3 remains an outlier at about 241.5 % |
| Generality across pressures, power waveforms, and unrelated chemistries | not yet validated | current external set | the present benchmark set is informative but still too narrow to claim broad process-space validation |

## 10. robustness suite による適用性の補強

robustness suite は authoritative external reference ではない。ここで示しているのは、pressure、source voltage、electrical backend、chemistry を parity 点から少し動かしたときに、本コードが有限・正の状態を保ち、workflow として実行可能かどうかである。

| 項目 | 値 | 意味 |
| --- | --- | --- |
| Strict external references retained | 2 | the strict validation backbone is unchanged |
| Added stress / applicability cases | 8 | pressure, source-voltage, electrical-backend, and chemistry perturbations |
| Stress cases passing boundedness checks | 8/8 | finite positive solver-health checks across the executed perturbations |
| Interpretation | boundedness / applicability evidence | supports practical usability and operating-envelope claims, not new strict accuracy validation or predictive-accuracy proof |

この suite は新しい精度 validation ではないが、strict benchmark の周囲でコードが運転範囲的に破綻しにくいことを補足する。strict validation と applicability evidence を混同しないため、artifact は別報告として保持している。

- [robustness stress overview](../robustness_sweep/robustness_stress_overview.png)
- [robustness rate-table coverage](../robustness_sweep/robustness_rate_table_coverage.png)
- [robustness applicability report](../robustness_sweep/robustness_applicability_report.md)

## 11. 低圧プラズマモデルとして残る理論課題と将来項目

本コードは global model の実装基盤としてはかなり整っているが、低圧プラズマの高忠実度予測モデルとしてはまだ未完である。主な将来項目は次の通りである。

1. **空間非一様性と輸送の高度化**
   現在は global model であり、power deposition profile、center-edge density ratio、局所輸送場は直接は解かない。multi-zone diffusion closure や edge-to-center model の体系化が必要である。
2. **sheath / IED / IADF の高忠実度化**
   現在の Bohm-family wall loss は leading-order sink としては有用だが、time-dependent sheath や collisional sheath を直接解かない。低圧 CCP / bias 問題ではここが支配的になる。
3. **RF 周期内電子加熱**
   `rf_envelope` は recipe-level には有効だが、stochastic heating、harmonic content、phase-resolved EEDF は表現できない。sub-RF effective heating model や校正手順の明文化が必要である。
4. **外部回路との双方向連成**
   `external_circuit_table` は one-way loose coupling である。将来的には plasma impedance を circuit 側へ返す macro-step loose coupling、さらに必要なら plasma ODE / circuit DAE の分割連成が候補となる。
5. **表面化学と壁材依存性の校正**
   O2 や fluorocarbon 系では sticking、quenching、recombination、coverage dependence の不確かさが大きい。`O3` outlier はその不足をよく示している。
6. **state-resolved chemistry と external rate workflow**
   vibrational ladder、metastable、superelastic collision を多く含む系では mean-energy closure だけでは粗い可能性がある。BOLSIG+ / LoKI 由来 table の ingestion、state-resolved chemistry の整理が今後重要である。
7. **実験比較と不確かさ評価**
   最終的な predictive value を示すには、`n_e`, `T_e`, `T_g`, ion flux, self-bias, radical density といった observable を実験と比較し、感度解析と UQ を組み合わせる必要がある。

要するに、本コードは **低圧プラズマ global model を、closure を可視化しながら開発・診断・比較するための基盤**としては強い。一方で、sheath、空間輸送、RF 周期内電子加熱、表面校正、実験比較まで含む「高忠実度な完全予測モデル」としては、まだ将来拡張の余地が大きい。
