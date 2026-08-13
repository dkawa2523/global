# Physics and numerics assumptions

## 適用範囲

Global Model は各 zone 内を体積平均し、heavy species number density とエネルギー収支を
時間発展させる reduced-order model です。sheath、skin、局所電場、流れ、表面の空間構造そのもの
は解かず、面積、体積、characteristic length、conductance、power coupling などの closure として
扱います。装置予測に使う前に対象条件での感度解析と実験検証が必要です。

標準的な粒子・electron energy・Bohm wall-loss の構成は、Kemaneci らの Global Model と
[Global Model review](https://eprints.whiterose.ac.uk/id/eprint/110507/) に沿います。これらは
モデル構成の根拠であり、この実装の係数全てを保証する reference solution ではありません。

audit が `standard` と分類するのは、次を全て満たす場合だけです。

- 固定気体温度、準中性、負イオンを含まない単一一価正イオン
- 実断面積 Maxwellian rate と electron-energy closure
- 全 power port が prescribed absorbed power
- Bohm floating wall または wall transport off
- 一意な生成物を持つ決定論的 boundary reaction
- surface kinetics、experimental rate、film／inventory／extension state を使わない

inlet、pump、inter-zone transport はこの範囲で利用できます。それ以外の実行機能は削除して
いませんが、audit では `experimental` と分類します。

## Heavy-species balance

zone \(z\)、heavy species \(s\) について、

\[
\frac{d n_{z,s}}{dt}=
\sum_r \nu_{r,s}R_{z,r}
+S^{wall}_{z,s}+S^{surface}_{z,s}+S^{transport}_{z,s}.
\]

反応速度は非負にした反応物密度だけで mass action を評価します。

\[
R_{z,r}=k_r(\mathcal C_z)
\prod_s \max(n_{z,s},0)^{\alpha_{r,s}}
n_{e,z}^{\alpha_{r,e}}.
\]

\(\mathcal C_z\) は gas temperature、electron mean energy、effective electron temperature、
E/N、table rate/transport を含みます。化学量論、reaction order、energy transfer は compile 時に
配列化し、RHS で reaction ID を解決しません。

標準の準中性 closure は floor を使わず、

\[
n_e=\sum_s z_s n_s
\]

を評価します。右辺が許容誤差を超えて負なら `ModelDomainError` です。強電気陰性 afterglow や
prescribed electron density は experimental opt-in とし、charge residual を診断に残します。

## Electron energy closure

`electron_energy` は electron energy density

\[
W_e=n_e e\langle\varepsilon\rangle
=\frac{3}{2}n_e eT_e
\]

を ODE state とします。したがって

\[
T_e[\mathrm{eV}]=\frac{2}{3}\langle\varepsilon\rangle[\mathrm{eV}].
\]

`mean_energy_eV` と `electron_temperature_eV` は別の量として保持し、Bohm velocity と sheath
には temperature を用います。現在の electron-energy ledger は

\[
\frac{dW_e}{dt}=
P^{abs}_e/V
-Q_{inelastic}
-Q_{wall,e}
-Q_{elastic}
+S^{transport}_{W_e}
\]

です。吸収電力、反応 energy loss、electron-heavy elastic transfer、wall/sheath loss、流出入は
同じ `ZoneTermLedger` の項から RHS と診断を作ります。

experimental 分類の `local_field` は electron energy state と緩和時間を持たず、port/circuit が与える E/N から
mean energy、rate coefficient、mobility を代数評価します。電気―輸送―EEDF coupling は E/N、
mobility、absorbed power の全 residual を相対許容誤差 `1e-6`、最大 12 iteration で判定し、
非収束時は時刻・step・port を含む明示エラーにします。
収束判定後は返却する最終E/Nでもう一度kineticsとportを評価するため、artifactのE/N、rate、mobility、
powerは同じiterateに対応します。正のfieldだけを持つprepared tableをoffにした場合だけcold-electron
boundaryを使い、明示0-field nodeがあるtableの物理値は上書きしません。

RF/CCPのpower commandはsourceへの皮相電力ではなくplasma吸収実電力です。RFのsource-side実電力は
吸収電力をcoupling efficiencyで割って求めます。lossless-sheath CCPでは実電力は (I_{rms}^2R)、
(V_{rms}I_{rms}) は `apparent_power_VA` でありelectron-energy ledgerへ加えません。

## Heavy-particle internal energy

`gas_energy: fixed` が標準です。experimental 分類の `evolved` では temperature ではなく

\[
U_g=k_B T_g\sum_s c_{v,s}/k_B\; n_s
\]

を積分し、species ごとの `cv_over_kb` から \(T_g\) を代数的に戻します。全 gas species に
heat capacity の明示値が必要です。RHS は gas-directed port power、gas/surface reaction heating、
electron elastic heating、inlet/outlet/inter-zone energy transport、wall temperature への熱交換を
同じ heavy-energy ledger で合計します。

state は内部エネルギーのままですが、開いた control volume を横切る重粒子流はエンタルピーを
運びます。inlet species の流入エネルギーは
\(\dot N_s(c_{v,s}/k_B+1)k_BT_{in}\)、pump と directed edge の流出エネルギー密度は

\[
H_g=U_g+p=U_g+k_BT_g\sum_s n_s
\]

です。したがって、定容・断熱容器の充填と排気を第一法則どおり扱います。evolved gas と
transport を組み合わせる場合、transport は同じ species 順の `cv_over_kb` を必須とし、内部
エネルギーをそのまま移流する fallback はありません。

## Wall と surface

Bohm wall の正イオン incident flux は一度だけ評価します。

\[
\Gamma_i=h_i n_i\sqrt{\frac{eT_e}{m_i}},\qquad
L_i=\Gamma_i A/V.
\]

数値の `h_factor` はこの式の \(h_i\) 全体であり、隠れた `0.61` 係数は追加しません。
`h_factor: auto` は初期状態から定数を作る設定ではなく、各 RHS 評価時の中性重粒子総密度
\(n_n\) を使って

\[
h_i=\operatorname{clamp}\!\left[
\frac{0.86}{\sqrt{3+L n_n\sigma_{in}/2}},h_{min},h_{max}
\right]
\]

を再評価します。`characteristic_length_m` 未指定時はそのsurfaceの \(V/A\) を使用します。
ここで \(n_n\) は \(\mathrm{m^{-3}}\)、\(L\) は m、\(\sigma_{in}\) は \(\mathrm{m^2}\) です。
衝突なし極限は \(0.86/\sqrt{3}\) です。

同じ flux を gas particle source、`boundary_reactions.csv` の生成物 branch、surface reaction、
electron wall loss へ分配します。活動する incident ion ごとに branch probability 合計 1 を要求し、
生成物を元素名から推測しません。electron wall energy loss は wall/sheath model で
\(2z_iT_e+\epsilon_{sheath}\) を使用します。標準域は、負イオンを含まない単一の一価正イオン、
Maxwellian/electron-energy closure、固定気体温度の floating Bohm wall です。明示 sheath energy が
ない場合は

\[
\epsilon_{sheath}=\frac{T_e}{2}\ln\!\left(\frac{m_i}{2\pi m_e}\right)
\]

を内部計算します。多イオン、負イオン、local-field、動的気体温度、経験 wall transport は実行機能を
維持しますが、audit では `experimental` かつ `production_qualified: false` と分類します。

surface coverage は吸着種だけを state とし、free-site fraction は
`1 - sum(occupied coverage)` から machine precision で計算します。独立 free-site ODE、coverage
renormalization、区間後 projection はありません。

## Transport

inlet は segment 固定の particle/enthalpy source、pump は粒子数に対する zone の first-order
loss、directed edge は source zone から `C*n` 個/s を取り target zone へ同数/s 加えるよう
compile されます。重粒子energyは前節の \(H_g\)、electron energyはそのenergy densityを同じ
粒子流に載せます。
inter-zone だけなら \(\sum_z V_z n_{z,s}\) と、対応する electron/heavy energy の体積積分が
保存されます。

## Domain contract

component-wise RHS clip や solver 後 projection は行いません。BDF は初期状態から固定した
component scale で無次元 state を積分し、公開 scalar `solver.atol` はこの無次元 state に適用します。
物理 domain tolerance と charge の丸め誤差判定は同じ scale から導出します。

- `state < lower_bound - 10 * domain_atol` は積分失敗。
- その範囲内の微小負密度は rate の反応物としてだけ 0 を使う。
- charge closure、wall、transport、RHS は生の trial state を受け取る。
- accepted result を作る時だけ、丸め誤差範囲の負値を 0 にする。
- table bounds は既定で error。`clip` / `hold` は明示 opt-in の lookup policy。

非有限 state、負 electron density、負 internal energy、coupling 非収束を silent fallback で継続
しません。

## BDF と forcing boundary

stiff reaction network は SciPy の
[`solve_ivp(method="BDF")`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html)
で積分します。化学量論の reactant dependency、charged-species closure、zone-local energy、
surface block、inter-zone transport から conservative な `jac_sparsity` を作り solver へ渡します。

recipe step、pulse edge、previous table/profile knot の全不連続点で BDF を分割します。各 solve は
固定済み segment だけを参照します。`first_step` と `max_step` は未指定時に渡しません。
sampling は積分誤差制御に使わず、accepted points または `sample_interval_s` / `save_at_s` の
指定点と全 forcing boundary を保存します。保存点は 100,000 点を固定上限とし、入力で上限を
変更する契約は設けません。

`experimental.stop_when_quasi_steady` は互換性のため旧名を保っていますが、時間不変 segment の
scaled RHS norm と、その時点の RHS から見積もる残区間変化を監視する**非終端の診断 event**です。
現在の RHS だけでは自己触媒反応などの将来の増幅を保証できないため、積分の途中停止や残区間への
state の定数外挿は行いません。`quasi_steady_events` は条件候補を観測した segment 数であり、
定常 root の証明ではありません。audit はこの設定を持つ case を引き続き `experimental` と分類します。

## Audit の数値診断

`plasma-global audit` は compile だけでなく同じ case を積分します。静的には全 gas/boundary/
surface reaction の normalized element、charge、site residual を確認します。動的には simulation
status、保存点の normalized charge-closure residual、存在する electron/heavy energy state の
「RHS と ledger 各項の和」の normalized residual を報告します。

これらは実装内部の収支整合を検出する診断です。solver tolerance 依存性、benchmark、実験一致に
代わる精度検証ではありません。

audit は `classification: standard | experimental` と `production_qualified` も返します。
上記 allow-list から一つでも外れる table/local-field、DC/external/experimental power、evolved gas、
surface kinetics、電気陰性または非単一一価イオン、experimental rate/拡張状態は `experimental` です。
`production_qualified` はstandard分類でERROR/WARNINGがない場合だけtrueであり、装置別の実験検証済みを
意味しません。

## Experimental model の位置付け

`experimental.approximate_two_term` は断面積と混合比から傾向を得る近似で、検証済み
Boltzmann solver の代替ではありません。`experimental.rf_envelope` / `ccp` / `icp`、prescribed
electron profile、film、wall inventory、generic extension state も、有限性、保存則、単調傾向を
基本契約とし、production 精度を主張しません。すべて composition root に接続済みですが、
使用には明示 model ID が必要です。
approximate two-term preparationはpower-balance residualに加え、最終energy binの確率質量を
`1e-3`以下に要求します。既定field gridは `1..100 Td` とし、高fieldではenergy-domain convergenceを
満たすようcross-section supportとenergy上限をcaseごとに検証します。
