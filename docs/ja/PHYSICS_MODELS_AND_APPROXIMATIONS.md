# 物理モデルと近似

このコードは、低圧プラズマを 0D/global model として扱います。空間分布を直接解くのではなく、各 zone を well-mixed volume とみなし、粒子数、電子エネルギー、表面状態の時間発展を保存則として積分します。

このページでは、実装で使っている物理モデルと近似を、保存則から順に説明します。記号、単位、実装 module の対応だけを確認したい場合は [物理式・記号・実装対応表](PHYSICS_EQUATION_REFERENCE.md) を参照してください。

## 1. Global model と空間平均化

global model では、zone 内の密度を空間平均値として扱います。species $s$ の zone 平均密度を $n_s$ とすると、基本形は次の balance です。

$$
\frac{dn_s}{dt}
= S_s^{\mathrm{gas}}
+ S_s^{\mathrm{surface}}
+ S_s^{\mathrm{flow}}
+ S_s^{\mathrm{zone}}
+ S_s^{\mathrm{wall}} .
$$

ここで $S_s^{\mathrm{gas}}$ は気相反応、$S_s^{\mathrm{surface}}$ は表面反応から gas phase へ戻る寄与、$S_s^{\mathrm{flow}}$ は inlet / pump、$S_s^{\mathrm{zone}}$ は zone 間輸送、$S_s^{\mathrm{wall}}$ は wall loss です。

気相反応 $r$ の寄与は、stoichiometric coefficient $\nu_{sr}$ と reaction progress rate $R_r$ を用いて

$$
S_s^{\mathrm{gas}}
= \sum_r \nu_{sr} R_r .
$$

二体反応なら $R_r = k_r n_a n_b$、一次損失なら $R_r = k_r n_a$ のように、反応次数に応じて単位が変わります。したがって、このコードでは reaction CSV と rate model の対応、rate coefficient の単位、species metadata の整合性が重要です。

この近似で見られるのは、装置平均としての粒子収支と時定数です。半径方向分布、sheath 内構造、局所的な ionization peak、wafer 面内分布は直接解きません。

## 2. 電子密度 closure

標準的な production 設定では、電子密度は独立した ODE state ではなく、charged species の準中性条件から求めます。

$$
\sum_s z_s n_s \simeq 0 .
$$

電子を $z_e=-1$ とし、electron density floor $n_{\min}$ を入れると、実装上は概念的に次の形になります。

$$
n_e
= \max \left(
\sum_{i\ne e} z_i n_i,\ n_{\min}
\right) .
$$

正イオンだけの chemistry では、この式は正イオン電荷密度の和から電子密度を決めることに相当します。負イオンを含む electronegative chemistry では、負イオンが charge balance に入るため、closure の妥当性を case ごとに確認する必要があります。

`prescribed_profile` mode では、CSV で与えた $n_e(t)$ を EEDF と電気 backend に渡します。この mode は benchmark や one-way coupling には有用ですが、自己無撞着な discharge 解ではありません。

## 3. 電子衝突レートと EEDF

電子衝突 reaction の rate coefficient は、本来は cross section $\sigma_j(\varepsilon)$ と electron energy distribution function $f(\varepsilon)$ から計算されます。

$$
k_j
= \int_0^\infty
\sigma_j(\varepsilon)\,
v(\varepsilon)\,
f(\varepsilon)\,
d\varepsilon .
$$

このコードの main ODE は $f(\varepsilon)$ そのものを state として持ちません。EEDF / swarm backend が、平均電子エネルギー $\langle\varepsilon\rangle$ または reduced field $E/N$ を入力として、rate coefficient と transport coefficient を返します。

$$
k_j = k_j(\langle\varepsilon\rangle)
\quad \text{or} \quad
k_j = k_j(E/N) .
$$

この設計は、global model としては標準的ですが、rate closure の選び方が結果に直接効きます。特に、mean-energy lookup と local-field lookup は物理的な意味が違います。電子エネルギーを ODE で解いている case で、別途 $E/N$ lookup を強く使う場合は、electron energy balance と rate closure が矛盾していないかを確認してください。

主な EEDF / rate backend の位置づけは次の通りです。

| backend | 位置づけ |
|---|---|
| `maxwell` | smoke test や reduced study 用の解析的 closure |
| `rate_table` | 外部 solver や事前計算 table を使う production 寄りの closure |
| `boltzmann_2term` | two-term-like reduced closure。BOLSIG+ の完全な代替ではない |
| `swarm` | selected swarm model に依存する experimental backend |

cross-section から rate coefficient を得る方法の背景には、BOLSIG+ のような two-term Boltzmann solver があります。この repo の internal backend はその考え方に近い reduced closure ですが、reference solver と同一視しないでください。

## 4. 電子エネルギー balance

電子エネルギー密度を $W_e$ とすると、代表的な balance は次の形です。

$$
\frac{dW_e}{dt}
= P_{\mathrm{abs}}
- L_{\mathrm{inelastic}}
- L_{\mathrm{elastic}}
- L_{\mathrm{wall}}
- L_{\mathrm{transport}} .
$$

$P_{\mathrm{abs}}$ は電子系へ入る吸収電力密度、$L_{\mathrm{inelastic}}$ は excitation、ionization、dissociation などの非弾性損失、$L_{\mathrm{elastic}}$ は gas heating に結びつく弾性損失、$L_{\mathrm{wall}}$ は charged particle loss に伴う electron energy loss、$L_{\mathrm{transport}}$ は zone 間輸送や flow に伴う項です。

平均電子エネルギーは

$$
\langle\varepsilon\rangle
= \frac{W_e}{n_e}
$$

として計算されます。実装では単位変換を通じて eV 表記の diagnostics と rate lookup に渡されます。

この式は、電子集団を 1 つの平均エネルギーで代表させる近似です。非 Maxwellian tail、二温度 EEDF、RF 周期内の EEDF 変調を直接解くものではありません。電子衝突反応が tail に強く依存する条件では、rate table や外部 Boltzmann solver との比較が重要になります。

## 5. 壁損失と Bohm-family flux

positive-ion wall loss は、Bohm-family の global flux closure として扱います。代表形は

$$
\Gamma_i
= h\,\alpha\,n_i u_B ,
$$

$$
u_B
= \sqrt{\frac{k_B T_e}{m_i}}
\simeq
\sqrt{\frac{e\langle\varepsilon\rangle}{m_i}} .
$$

$\Gamma_i$ は wall へ向かう ion flux、$h$ は edge-to-center factor、$\alpha$ は sheath-edge 近似に由来する係数、$u_B$ は Bohm speed です。実装上の `bohm_edge_loss` では、既定で $h=1$ を使い、必要に応じて `h_factor` を明示します。`bohm_global_loss` では pressure、temperature、characteristic length、ion-neutral cross section から edge-to-center estimate を行います。

この closure は、charged particle inventory の leading-order sink を与えるものです。空間分解 sheath、collisional sheath、multi-ion Bohm criterion、time-resolved ion transit を解いているわけではありません。

ambipolar diffusion loss は別の first-order volumetric loss として扱います。

$$
\frac{dn_i}{dt}
= -k_{\mathrm{loss}} n_i,
\qquad
k_{\mathrm{loss}} \sim \frac{D_a}{L^2}.
$$

同一 zone で Bohm 系 loss と ambipolar diffusion loss を同じ wall sink として重ねると二重計上になります。このため、validation では mixed ion-loss mode を拒否します。

## 6. 表面反応と coverage

surface species は gas species と別の state として扱います。coverage $\theta_j$ は surface site の占有率であり、基本制約は

$$
0 \le \theta_j \le 1,
\qquad
\sum_j \theta_j + \theta_{\mathrm{free}} = 1 .
$$

surface reaction は、gas species の消費・生成と surface coverage の変化を同時に起こします。面積 $A$ の surface で flux $\Gamma_s$ が生じる場合、zone 体積 $V$ に対する volumetric source は概念的に

$$
S_s^{\mathrm{surface}}
\sim
\frac{A}{V}\Gamma_s
$$

として gas balance に入ります。

sticking、thermal desorption、Langmuir-Hinshelwood 型 recombination、ion-assisted reaction などを扱えますが、surface temperature、site density、coverage dependence、ion flux proxy の設定に結果が強く依存します。wafer 上の 2D 分布や microscopic feature profile はこの model の外側です。

surface、wafer、IED proxy の出力を読む場合は [表面・ウェハ・IED ガイド](SURFACE_WAFER_IED_GUIDE.md) を参照してください。

## 7. 電力結合と回路近似

電気モデルは chemistry と分離した reduced-order backend として実装されています。global ODE は、各 step の条件から得られる $P_{\mathrm{abs}}$、$E/N$、voltage、current などを受け取り、electron energy balance と rate lookup に使います。

`direct_power` は、指定された吸収電力をそのまま電子系に与えます。回路や電磁場を解かないため、chemistry debug や controlled study に向いています。

`dc_series_circuit` は、voltage source、ballast resistor、conductive plasma load の簡約モデルです。代表的には

$$
G_p
= \frac{e n_e \mu_e A}{d},
\qquad
R_p = \frac{1}{G_p},
$$

$$
I
= \frac{V_s}{R_b + R_p},
\qquad
V_{\mathrm{gap}} = I R_p,
$$

$$
P_{\mathrm{abs}}
= f_{\mathrm{abs}} V_{\mathrm{gap}} I .
$$

ここで $A$ は電極面積、$d$ は gap、$\mu_e$ は electron mobility、$R_b$ は ballast resistance、$f_{\mathrm{abs}}$ は吸収率です。この backend は reduced DC / pulsed-DC model であり、sheath capacitance、RLC ringing、matching network、full SPICE DAE は解きません。

`rf_envelope` は RF 周期を直接分解せず、cycle-averaged power や RMS voltage の envelope として扱います。RF sheath motion、harmonics、ion transit phase effect を評価する model ではありません。

## 8. このページの式で判断してよいこと

このページの式から判断できるのは、どの保存則と closure が使われているか、どの量が入力依存か、どこに近似が入るかです。精度そのものは、chemistry data、cross-section source、rate table、wall loss coefficient、surface model、電力 coupling の calibration に依存します。

レビュー時には、次の順番で確認すると誤解が少なくなります。

1. その case がどの EEDF / rate backend を使っているか。
2. electron density が quasi-neutral closure か prescribed profile か。
3. ion wall loss が Bohm 系か ambipolar diffusion か。
4. 電力入力が direct power か circuit / RF envelope backend か。
5. surface reaction が gas balance と coverage balance の両方にどう入るか。
6. benchmark が見ている物理軸と、見ていない物理軸は何か。

## 参考文献

| 参考 | 関連する内容 |
|---|---|
| Lieberman and Lichtenberg, *Principles of Plasma Discharges and Materials Processing* | 低温プラズマ、材料プロセス、sheath、Bohm flux、particle / power balance の標準的背景 |
| [Chabert and Braithwaite, *Physics of Radio-Frequency Plasmas*](https://www.cambridge.org/core/books/physics-of-radiofrequency-plasmas/07AF7D2152B32335BFD5F96ADDC7CE83) | RF plasma、bounded plasma、RF sheath、global model 的な考え方 |
| [Hagelaar and Pitchford, 2005, Plasma Sources Sci. Technol.](https://doi.org/10.1088/0963-0252/14/4/011) | cross section から transport / rate coefficients を得る Boltzmann solver の代表文献 |
| [LoKI O2 benchmark paper](https://doi.org/10.1088/1361-6595/acbb9c) | 0D global model と 1D fluid model の benchmark 例 |
| [LXCat BOLSIG+ solver page](https://nl.lxcat.net/solvers/BolsigPlus/index.php?step=1) | BOLSIG+ / LXCat 文脈での rate / swarm data 生成 |
