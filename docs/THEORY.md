# 体積平均プラズマモデルの理論と数値検証

### 保存則から実装・近似・再現可能なベンチマークまで

技術論文 / 2026年10月26日 / 対象package 0.5.0・入力schema v4・結果format v3

対象コードは本リポジトリのcommit `76f38ca` に含まれる数値・物理実装である。
本稿は査読済み論文ではなく、実装と再計算結果に基づく技術解説である。
操作・設定は[統合マニュアル](index.md)、残る開発項目は[計画](ROADMAP.md)を参照する。

## 要旨

低温プラズマには電子、イオン、中性粒子、壁面が関与し、反応、エネルギー移送、流入出が異なる
時間尺度で進む。globalモデルは、空間分布を区画平均へ縮約し、粒子数とエネルギーの収支を
常微分方程式として解く方法である。本稿は保存則からその縮約を導き、Plasma Global Modelが使う
質量作用則、電子密度の準中性関係、Maxwellian反応係数、電子・気体エネルギー、Bohm壁損失、
表面反応、区画間輸送、電源・回路の方程式を説明する。標準構成と経験的拡張を区別し、
入力・出力の物理量と数式の対応も示す。今回19ケースを再計算し、断面積積分、一次反応、
区画輸送、CRANE・ZDPlaskin由来の保存結果、過渡Arの収束・感度・イベント積分を検証した。
数値的な再現性と、実験に対する予測精度は別の根拠を要する。特に低温afterglow、
初期組成へ固定した電子係数表、経験的RF結合、点計測と区画平均の差を適用上の限界として論じる。

**キーワード:** 体積平均、プラズマ化学、粒子収支、電子エネルギー、剛性ODE、数値検証

## 1. 背景: なぜ空間を平均するのか

プラズマは一部の粒子が電離した気体である。低温プラズマでは、軽い電子が電場からエネルギーを
受け取り、重い原子・分子・イオンへ衝突を通じて渡す。電子のエネルギーが中性気体より高くても、
気体全体が同じ高温になるとは限らない。電子密度だけでなく電子エネルギーを求める必要があるのは、
電離・励起・解離の確率が衝突エネルギーへ強く依存するためである。

空間を解像するモデルでは位置ごとの密度、電位、電場、流速、境界層を解く。一方、
「どの反応が組成を支配するか」「通電停止後に粒子がどう消失するか」「壁面積を変えると
平均密度はどう変わるか」という問いには、区画平均の収支から始める意義がある。
globalモデルの位置付けと一般的な限界はHurlbattらの総説が整理している。[[1]](#ref-1)

本実装は、固定体積のzoneと、面積を持つsurface、方向付きedge、電源portを組み合わせる。
zone内の空間分布を再構成する機能はなく、zoneを連結した場合も流体方程式の自動離散化にはならない。
得られるのは、選んだ反応・壁・輸送・電源の仮定のもとで成立する平均状態の時間発展である。

### 1.1 読むための最小限の記号

| 記号 | 単位 | 意味 |
|---|---|---|
| $n_{z,s}, n_{e,z}$ | m⁻³ | zone zの重粒子種sと電子の数密度 |
| $V_z, A_b$ | m³, m² | zone体積と、それに接するsurface bの面積 |
| $R_r, r_b$ | m⁻³ s⁻¹, m⁻² s⁻¹ | 気相反応と表面反応の進行速度 |
| $k_r, \nu_{s,r}$ | 次数に依存, 1 | 反応速度係数、生成を正とする化学量論差 |
| $T_g, \theta_e$ | K, eV | 気体温度、エネルギー単位で表した等価電子温度 |
| $\bar\varepsilon, W_e, U_g$ | eV, J m⁻³, J m⁻³ | 平均電子エネルギー、電子・気体内部エネルギー密度 |
| $\Gamma, C, S$ | m⁻² s⁻¹, m³ s⁻¹, m³ s⁻¹ | 面を通る粒子束、edge移送係数、排気速度 |
| $e, k_B, m_e, Z_s$ | C, J K⁻¹, kg, 1 | 素電荷の正値、Boltzmann定数、電子質量、重粒子の電荷数 |

重粒子は電子を除く気相の原子・分子・イオンをいう。closureは、解かない空間構造などの代わりに
平均量と既知の係数から必要な量を決める関係式である。ODEは常微分方程式、EEDFは電子エネルギー分布を指す。

## 2. 連続の式から体積平均モデルを導く

局所的な粒子の連続の式は、粒子数の時間変化、空間輸送、反応による生成消滅の釣り合いである。

$$
\frac{\partial n_s}{\partial t}+\nabla\cdot\boldsymbol{\Gamma}_s
=\sum_r\nu_{s,r}R_r. \tag{1}
$$

固定した体積Vの平均を次で定める。

$$
\bar n_s=\frac{1}{V}\int_V n_s\,dV. \tag{2}
$$

式(1)を積分して発散定理を適用すれば、厳密には次になる。法線は体積の外向きとする。

$$
\frac{d\bar n_s}{dt}
=\sum_r\nu_{s,r}\overline{R_r}
-\frac{1}{V}\oint_{\partial V}\boldsymbol{\Gamma}_s\cdot\hat{\boldsymbol{n}}\,dA. \tag{3}
$$

globalモデルは境界束を面平均のclosureで評価し、体積内の反応を平均密度で近似する。
これは平均操作そのものではなく、追加の物理近似である。例えば二体反応には一般に次の差がある。

$$
\overline{n_a n_b}=\bar n_a\bar n_b+
\overline{(n_a-\bar n_a)(n_b-\bar n_b)}. \tag{4}
$$

本モデルは右辺最後の空間相関を独立に計算しない。従って、平均密度を正しく与えても、
局所的に反応物が重なる場所や高エネルギー電子が偏在する場合、正しい平均反応速度とは限らない。
壁との接続、表面積／体積比、電子分布の選択が重要になる理由である。

本実装で積分する重粒子の方程式は次の構造を持つ。

$$
\frac{dn_{z,s}}{dt}=\sum_r\nu_{s,r}R_{z,r}
+S^{\mathrm{wall}}_{z,s}+S^{\mathrm{surface}}_{z,s}
+S^{\mathrm{flow}}_{z,s}. \tag{5}
$$

各項の単位はm⁻³ s⁻¹である。体積を持つzone、面積を持つsurface、流出入を結ぶedgeが
それぞれ式(5)の項へ対応する。この分解により、同じ化学機構を異なる容器や運転条件へ適用できる。

## 3. 反応速度、保存量、初期状態

### 3.1 反応係数と反応進行速度を区別する

反応物の係数をα、生成物をβとすれば、$\nu_{s,r}=\beta_{s,r}-\alpha_{s,r}$ である。
質量作用則による進行速度は次である。電子は重粒子とは別に扱う。

$$
R_{z,r}=k_r(\mathcal{C}_z)n_{e,z}^{\alpha_{e,r}}
\prod_s n_{z,s}^{\alpha_{s,r}}. \tag{6}
$$

状態条件 $\mathcal{C}_z$ には電子平均エネルギー、気体温度、E/N等が入る。
反応物総次数qならkの単位はmの3(q−1)乗毎秒、Rの単位は常にm⁻³ s⁻¹である。
Rをzone体積倍した量は毎秒のイベント頻度であり、特定種の正味生成速度ではない。
種ごとの生成消滅にはνを掛ける必要がある。

例えば $e+\mathrm{Ar}\to2e+\mathrm{Ar}^+$ は、イベント1回につきArを1個消費し、Ar⁺と電子を
それぞれ正味1個作る。励起を $e+\mathrm{Ar}\to e+\mathrm{Ar}$ のenergy損失だけで表現した場合、
Ar*の粒子数は状態に存在しない。Ar*の寿命や段階電離まで予測したい場合は、種と反応を明示的に追加する。

一定係数、一次速度、Arrhenius型、1変数の表、電子衝突断面積・電子係数表を選べる。
Arrhenius型の実装は、振幅Aの基準温度を300 Kとして次を使う。

$$
k(T_g)=A\left(\frac{T_g}{300\,\mathrm{K}}\right)^\beta
\exp\left(-\frac{eE_a}{k_BT_g}\right). \tag{7}
$$

ここで活性化エネルギー $E_a$ はeVである。電子の移送energyは加熱を正、冷却を負とする。
入力表に欠けた反応係数を0や別のモデルで補う処理は行わない。

### 3.2 保存量を粒子数で考える

元素aの原子数 $b_{a,s}$ を使うと、閉じた気相反応には次の条件がある。

$$
\sum_s b_{a,s}\nu_{s,r}=0,\qquad
\sum_s Z_s\nu_{s,r}-\nu_{e,r}=0. \tag{8}
$$

内部輸送を含む全領域の元素量は $\sum_zV_z\sum_s b_{a,s}n_{z,s}$ で評価する。
zoneの密度を単純に足すと、体積の違いを無視して保存則を誤解する。
壁が電荷を受け取るboundary反応では、気相だけの電荷保存を要求するのではなく、壁を電荷の交換先とする。
表面siteを使う反応にはsite収支も必要である。

### 3.3 初期圧力は圧力制御ではない

本コードのheavy gas圧力と電子密度の準中性関係は次である。

$$
p_z=k_BT_{g,z}\sum_s n_{z,s},\qquad
n_{e,z}=\sum_s Z_s n_{z,s}. \tag{9}
$$

圧力の和にはイオンを含み、電子を含めない。初期圧力と気体温度から全heavy密度を求め、
荷電seedを除いた中性密度へ初期モル分率を配分できる。直接密度を指定する場合はp・Tとの整合を検査する。
流量、反応、排気で粒子数が変われば、運転中の圧力も変わる。

準中性は電子密度を代数的に決める近似で、Poisson方程式や独立した電子連続式の代わりである。
表面近傍の電荷分離を解かないため、sheathの局所密度や電位分布は求まらない。
負の電子密度を小さな正値へ変えて継続することはない。外部電子密度profileを使う拡張では、
式(9)の後半を使わず、電荷残差の意味を別途確認する。

## 4. 電子のエネルギー分布と速度係数

### 4.1 平均エネルギーと温度の関係

電子のエネルギー分布を $F(\varepsilon)$、エネルギーεをeVで定める。
Fは1電子を選んだときのenergy確率密度であり、次を満たす。

$$
\int_0^\infty F(\varepsilon)\,d\varepsilon=1,\qquad
\bar\varepsilon=\int_0^\infty\varepsilon F(\varepsilon)\,d\varepsilon. \tag{10}
$$

温度をKで表す場合と区別し、$\theta_e=k_BT_e/e$ をeVで表した等価温度とする。
この実装では分布が表形式でも $\theta_e=2\bar\varepsilon/3$ を出力する。
非Maxwellian分布に対して、この1個の値だけで全衝突過程を決められるという意味ではない。

Maxwellianのenergy確率密度は次である。

$$
F_M(\varepsilon;\theta_e)=\frac{2}{\sqrt{\pi}}
\frac{\sqrt{\varepsilon}}{\theta_e^{3/2}}\exp(-\varepsilon/\theta_e). \tag{11}
$$

例として平均energy 3 eVは等価温度2 eV、気体温度300 Kはenergy単位で約0.02585 eVである。
入力の `mean_energy_eV` と出力の `electron_temperature_eV` を同じ数値として解釈してはいけない。

### 4.2 断面積からの積分

衝突断面積σと電子速度 $v=\sqrt{2e\varepsilon/m_e}$ から反応係数を求める。

$$
k_r=\int_0^\infty\sigma_r(\varepsilon)
\sqrt{\frac{2e\varepsilon}{m_e}}F(\varepsilon)\,d\varepsilon. \tag{12}
$$

Maxwellianの場合は次になる。

$$
k_r(\theta_e)=\sqrt{\frac{8e}{\pi m_e}}\theta_e^{-3/2}
\int\sigma_r(\varepsilon)\varepsilon\exp(-\varepsilon/\theta_e)\,d\varepsilon. \tag{13}
$$

本実装は収録断面積を区分線形で表し、その各区間の積分を解析的に評価する。
したがって、固定した反応係数表を途中に作ることによる補間誤差はない。
ただし元断面積の測定誤差、選択したcurve、区分線形近似、分布の仮定に由来する誤差は残る。
electron swarmの係数とEEDFの関係はHagelaar・Pitchfordが説明している。[[2]](#ref-2)

1区間を $\sigma=a+b\varepsilon$ とすれば、必要なmomentは次で表せる。

$$
I_q(l,u)=\theta_e^{q+1}
\left[\Gamma(q+1,l/\theta_e)-\Gamma(q+1,u/\theta_e)\right]. \tag{14}
$$

式(13)の区間寄与は $aI_1+bI_2$ となる。実装は狭い区間で近い原始関数を引く桁落ちを避けるため、
区間始点を0へ移した不完全gamma momentと `expm1` を使う。式(14)は数学的な表現であり、
その差を素朴に計算するという実装指定ではない。

収録終端 $E_{\max}$ より上へ断面積を外挿しない。$x=E_{\max}/\theta_e$ に対し、
energy-weighted tailの比 $f_{\mathrm{tail}}=(1+x)\exp(-x)$ を1×10⁻⁶以下に制限する。
さらに最大収録断面積を終端より上へ一定延長した仮定上の寄与を、収録域の積分の1×10⁻³以下に要求する。
後者は収録不足の検出条件であって、未知の実断面積がその上限以下であると証明するものではない。
rateの物理的誤差上限や信頼区間とは呼ばない。

### 4.3 電子係数表とlocal-field近似

表形式では平均energyまたはE/Nを軸に係数を線形補間する。E/Nは電子が中性粒子の間で
どれだけ加速されるかを整理する換算電場で、1 Td = 10⁻²¹ V m²である。
電子係数は生成時の混合組成、衝突集合、分布計算に依存する。LXCatはそのような入力データを
共有する基盤だが、個々の用途での妥当性は使用者が確認する必要がある。[[3]](#ref-3)

local-fieldは電子energy密度を時間積分せず、現在のE/Nから平均energy・係数・mobilityを求める。
急速な電源切替でenergy緩和時間が無視できない場合には、その応答を再現できない。
mobilityの相似則は次である。$N$ は現在の全中性気相密度である。

$$
\mu(N)=\frac{(\mu N)_{\mathrm{table}}}{N}
=\mu_{\mathrm{ref}}\frac{N_{\mathrm{ref}}}{N}. \tag{15}
$$

絶対μの表には生成時密度を明示し、初期密度から推測しない。μNは別の物理量として保持する。
範囲外の既定動作はエラーである。明示した端値保持は計算上の選択で、適用条件の検証を置き換えない。
経験的な近似EEDFについては付録Aで、Boltzmann方程式を解く方法との差を明記する。

## 5. 電子と重粒子のエネルギー収支

### 5.1 電子energy密度を積分する理由

energy-balance方式は次の密度をODE状態として積分する。

$$
W_{e,z}=en_{e,z}\bar\varepsilon_z
=\frac{3}{2}en_{e,z}\theta_{e,z}. \tag{16}
$$

粒子数が変化してもenergy密度から平均energyを戻すため、電子数の変化による温度の変化を
明示的に別の温度式へ書き足す必要がない。粒子がゼロなのに正のenergyを与える状態は未定義として拒否する。

$$
\frac{dW_{e,z}}{dt}=\frac{P_{e,z}}{V_z}
+e\sum_r\Delta\varepsilon_{e,r}R_{z,r}
-Q_{e,\mathrm{wall},z}-Q_{eg,z}+S^E_{e,z}. \tag{17}
$$

$\Delta\varepsilon_{e,r}$ は電子への移送を正とするeV単位の値である。
電離energyを15.76 eV失う場合は負の移送となる。反応、壁、弾性衝突、輸送は同じ評価結果からRHSと出力を作る。
すべての項はW m⁻³であり、WとW m⁻³を取り違えない。

### 5.2 弾性衝突で電子と気体が交換するenergy

中性種jの運動量移行係数を $k_{m,j}$ とすれば、このコードが用いる移送は次である。

$$
Q_{eg}=3en_e\left(\theta_e-\frac{k_BT_g}{e}\right)
\sum_j\frac{m_e}{m_j}n_jk_{m,j}. \tag{18}
$$

電子が気体より高温なら正で、電子energyから引き、気体energyへ同じ値を加える。
逆に電子が低温なら符号も反転する。運動量断面積または明示係数のない中性種については
移送を省略して警告する。宣言済み係数の欠落をMaxwellian計算で補完しない。

### 5.3 気体の内部energyと開いた容器

既定は気体温度固定であり、壁や外部環境が温度を保つ近似に相当する。
気体energyを解く拡張では、各種の定積熱容量 $c_{v,s}$ を明示して次を積分する。

$$
U_{g,z}=T_{g,z}\sum_s c_{v,s}n_{z,s},\qquad
T_{g,z}=\frac{U_{g,z}}{\sum_s c_{v,s}n_{z,s}}. \tag{19}
$$

気体向けの電力、反応・表面反応の指定加熱、式(18)、壁熱交換、粒子交換、流入出を合計する。
壁熱交換は周波数νを使い、平衡内部energyへ緩和させる。

$$
Q_{g,\mathrm{wall}}=-\nu_{\mathrm{th}}
\left[U_g-T_w\sum_s c_{v,s}n_s\right]. \tag{20}
$$

固定容積でも、流れる気体は流動仕事を含むエンタルピーを運ぶ。

$$
H_g=U_g+p,\qquad
\dot E_{\mathrm{in},s}=\dot N_s(c_{v,s}+k_B)T_{\mathrm{in}}. \tag{21}
$$

気体を満たす・排気する問題で内部energyだけを移送すると、温度変化を誤る。
壁・表面反応による粒子の出入りには現在温度での内部energy $c_{v,s}T_g$ を対応させる。
モデルが与えない反応熱や材料の熱物性は自動で推定しない。

## 6. 壁損失と表面反応

### 6.1 Bohm入射束とedge／bulk比

標準的な冷イオン近似のBohm速度と、平均密度から壁へ向かう束を次で表す。

$$
u_{B,i}=\sqrt{\frac{Z_ie\theta_e}{m_i}},\qquad
\Gamma_{i,b}=h_{i,b}n_i u_{B,i}. \tag{22}
$$

数値のhはedge／bulk密度比を直接表す。別の0.61係数を暗黙に掛けない。
`h_factor: auto` は現在の中性密度から次を評価する。

$$
h_i=\operatorname{clip}\left(
\frac{0.86}{\sqrt{3+LN\sigma_{in}/2}},h_{\min},h_{\max}\right). \tag{23}
$$

特性長Lの省略時は、そのsurfaceについてV/Aを使う。これは本実装のclosureであり、
形状を解いて得た密度分布ではない。壁面積を変えると損失のA/Vだけでなく、このLとhも変わる。
壁反応や壁温がglobal収支へ与える影響は、例えばKemaneciらのClモデルでも扱われている。[[4]](#ref-4)
本実装がそのCl機構や係数を再現したという意味ではない。

### 6.2 壁での分岐とfloating sheath

入射イオン1個あたりの分岐確率を $p_{b,j}$ とすると、分岐イベントの面速度は $r_{b,j}=p_{b,j}\Gamma_i$。
活動する入射イオンについて確率和1を要求し、戻す中性粒子等を反応式で明示する。
入射イオン損失と生成物のreturnを別々の束で計算して二重計上しない。

単一一価正イオンを中心とする標準構成で、明示sheath energyがないときの電流収支は次である。

$$
\Gamma_{e,\mathrm{th}}=n_e\sqrt{\frac{e\theta_e}{2\pi m_e}},\qquad
\varepsilon_{\mathrm{sh}}=\theta_e
\ln\frac{\Gamma_{e,\mathrm{th}}}{\sum_i Z_i\Gamma_i}. \tag{24}
$$

電子の熱束が電位障壁で抑制され、イオン電流と一致する条件を使っている。
電子壁損失は、実際に計算したイオン束から次を作る。

$$
Q_{e,\mathrm{wall}}=e\sum_b\frac{A_b}{V}
\sum_i\Gamma_{i,b}\left(2Z_i\theta_e+\varepsilon_{\mathrm{sh},b}\right). \tag{25}
$$

式(24)に非負解がない場合を0 eVへ読み替えない。多価・多イオン、電気陰性、local-field、
気体温度の発展等は実行できても標準構成とは区別する。
冷イオンBohm近似は電子温度がイオン温度より十分高い状況を前提とする。
標準という機能分類だけでは、この条件が全時刻で成立したことにならない。

### 6.3 被覆率と表面反応

表面種jの粒子数をsite密度 $N_{\mathrm{site}}$ で規格化した状態をθとする。
種jが占有するsite数をaとすれば、free siteは代数量である。

$$
\theta_* =1-\sum_j a_j\theta_j,\qquad
\frac{d\theta_j}{dt}=\frac{1}{N_{\mathrm{site}}}
\sum_r\nu^{\mathrm{surf}}_{j,r}r_r. \tag{26}
$$

面速度rはm⁻² s⁻¹であり、気相への寄与にはA/Vを掛ける。
中性粒子の付着は熱入射束と付着確率を用いる。

$$
r_{\mathrm{stick}}=s\frac{n_g}{4}
\sqrt{\frac{8k_BT_g}{\pi m_g}}\,g(\boldsymbol{\theta}). \tag{27}
$$

イオン支援反応は、入射束にenergy依存yieldと被覆率因子を掛ける。

$$
r_{\mathrm{ion}}=\Gamma_iY_0
\left(\frac{\varepsilon_i-E_{\mathrm{th}}}{E_{\mathrm{ref}}-E_{\mathrm{th}}}\right)^b
g(\boldsymbol{\theta}),\quad \varepsilon_i>E_{\mathrm{th}}. \tag{28}
$$

閾値以下は0である。脱離は $\nu N_{\mathrm{site}}\exp[-eE_a/(k_BT_s)]$、
Langmuir-Hinshelwood型は $A_{LH}N_{\mathrm{site}}^2\exp[-eE_a/(k_BT_s)]$ に
反応物の被覆率因子を掛ける。空siteを独立ODEとして積分しないため、site収支の冗長な自由度を作らない。
採用状態の過充填は丸め許容幅を超えれば失敗とする。

## 7. 流入、排気、zone間の移送

入口のspecies流量Qをsccmで与えると、粒子流は次になる。基準状態は101325 Pa・273.15 Kである。

$$
\dot N_{\mathrm{in},s}=4.477962\times10^{17}Q_s,
\qquad S_{\mathrm{in},s}=\frac{\dot N_{\mathrm{in},s}}{V}. \tag{29}
$$

sccmは装置内の実体積流量ではない。排気速度Sの損失は $-Sn_s/V$ である。
source aからtarget bへの方向付きedgeでは次を用いる。

$$
\left.\frac{dn_{a,s}}{dt}\right|_{ab}=-\frac{C_{ab}}{V_a}n_{a,s},\qquad
\left.\frac{dn_{b,s}}{dt}\right|_{ab}=+\frac{C_{ab}}{V_b}n_{a,s}. \tag{30}
$$

式(30)は同じ粒子数を両区画で減算・加算する。密度差や圧力差で流量を求める式ではない。
逆向きの流れは別edgeを定義しなければ存在しない。対応する電子energy密度と重粒子エンタルピーも移送する。
孤立した内部edgeでは体積積分の粒子・移送energyが相殺する。

空間順にzoneを並べる使い方は可能だが、距離、断面積、移流・拡散、境界条件、空間収束を
備えた1次元PDEソルバーとは異なる。区画分割とCの根拠は使用者が与える。

## 8. 電源と回路のモデル

### 8.1 吸収電力と直流直列回路

最も単純な電源は既知の吸収電力Pを、電子分 $f_eP$ と気体分 $f_gP$ に分ける。
$f_e+f_g=1$ を要求する。発電機の表示値や反射電力を吸収電力と同一視しない。

DCモデルは電子伝導率とbulk抵抗、直列抵抗から計算する。

$$
\sigma_p=en_e\mu_e,\qquad R_p=\frac{L}{\sigma_p A},\qquad
I=\frac{V_s}{R_b+R_p}. \tag{31}
$$

$$
V_p=IR_p,\qquad P_e=f_{\mathrm{abs}}I^2R_p,\qquad
\frac{E}{N}=\frac{|V_p|}{LN}. \tag{32}
$$

μがE/Nに依存するlocal-fieldでは回路と係数を連立させる。
ゼロ電子密度のときはIと吸収電力が0となる開回路極限を使う。
これは電子driftによるbulk伝導の近似で、電極sheathの詳細な電流電圧特性を解く回路ではない。

外部電源CSVは電子電力、総吸収電力、または電圧と電流から与えられる。
V・Iの表ではそれぞれを補間してから積を取る。積の列を先に作って補間する方法とは一般に異なる。
直接power列を併記した場合はその指定を優先する。

### 8.2 RF・CCP・ICPの拡張

RF envelopeは $P_{\mathrm{abs}}=\eta V_{\mathrm{rms}}^2/R_{\mathrm{eff}}$、
換算電場は基準値と吸収電力平方根の係数から作る経験式である。
周波数を入力しても波形1周期の電磁場を解くことにはならない。

CCPはbulk抵抗とlossless sheath容量の集中定数回路である。

$$
|Z|=\sqrt{R_p^2+X_s^2},\qquad
X_s=\frac{1}{\omega C_p}+\frac{1}{\omega C_g},\qquad
P_{\mathrm{abs}}=I_{\mathrm{rms}}^2R_p. \tag{33}
$$

$V_{\mathrm{rms}}I_{\mathrm{rms}}$ は皮相電力VAであり、energy収支に加える実電力ではない。
sheath厚さ、面積非対称からの自己bias、平均イオンenergyは簡略診断値である。
これらを標準の壁energy損失や表面反応へそのまま流用しない。

ICPは密度・圧力・周波数に依存する経験的結合効率ηを使い、$P_{\mathrm{abs}}=\eta P_{\mathrm{delivered}}$ とする。
E/Hの指標は経験式の連続な切替で、Maxwell方程式からmode遷移を導いたものではない。
付録Bに実装の式と数値定数を示す。これらの拡張について独立した装置精度の認定は行っていない。

## 9. 時間積分と数値的な近似

反応は非常に速い衝突と遅い粒子・壁・流量の応答を含むため、剛性を持つことがある。
本コードはSciPyの公開BDFを使う。BDF／NDF型の陰的多段法は過去の状態と現在のRHSから
次の状態を決め、Newton反復とJacobianを使う。SciPyは次数1〜5の可変次数法である。[[5]](#ref-5)
NDFを含む手法の背景はShampine・Reicheltを参照できる。[[6]](#ref-6)

本実装は成分ごとの固定scale $s_j$ で状態を規格化する。

$$
\hat y_j=\frac{y_j}{s_j},\qquad
\frac{d\hat y_j}{dt}=\frac{f_j(t,\boldsymbol{y})}{s_j}. \tag{34}
$$

既定の相対許容差は1×10⁻⁶、無次元状態への絶対許容差は1×10⁻¹⁴である。
たとえば密度のSI値へそのまま10⁻¹⁴ m⁻³を要求する指定ではない。
局所誤差の制御は、時間全体の誤差、反応係数の誤差、物理モデルの誤差を保証しない。

電源step、pulse、previous補間の切替点で区間を分け、BDFを再開する。
Newtonの試行点は物理領域へ連続化した評価用viewを使うが、solver自身の状態をその都度書き換えない。
採用した全stepで非負性、energy、site occupancy等を確認する。
大きな領域違反を強制的に射影して成功させる処理は行わない。

保存時刻は積分stepとは別である。指定点は直前stepのdense outputから保存する。
同じ許容差で保存点だけを増やしても、ODE解が自動的に高精度になるわけではない。
状態は連続だが、電源切替時刻に保存する電力等は切替前の左側値である。
イベント積分では区間ごとの片側rateを使い、不連続をまたぐ平均を避ける。

## 10. 解析出力の数学的な意味

### 10.1 分率、圧力、正味生成

組成比は電子を除く全heavy gas種を分母とする。

$$
x_s=\frac{n_s}{\sum_j n_j},\qquad
p_s=n_sk_BT_g. \tag{35}
$$

イオンを含み、図で選んでいないheavy種も含める。選択種だけで100%へ再規格化しない。
`net_production_m3_s` は式(5)のRHS全体であり、気相反応だけの寄与ではない。
名前付きprobeは既存の平均量へ名前を付ける仕組みで、計測器の空間感度や応答時間は含めない。

### 10.2 反応イベント頻度と累積数

気相反応の頻度はV倍、表面・boundary反応はA倍する。

$$
\dot\Xi_r=V R_r\quad(\mathrm{gas}),\qquad
\dot\Xi_r=A r_r\quad(\mathrm{surface}). \tag{36}
$$

累積数の出力は保存点からの区間別台形積分である。

$$
\Xi_r(t_k)\simeq\sum_{j<k}\frac{t_{j+1}-t_j}{2}
\left[\dot\Xi_r(t_j)+\dot\Xi_r(t_{j+1})\right]. \tag{37}
$$

切替点では各区間に適した片側値へ読み替えて積分する。Ξは連続量としての推定であり、
Monte Carloで発生した整数個のイベントではない。独立ODEとして積分した状態でもないため、
保存間隔を変えた収束確認が必要である。

Graphvizの図は化学種と反応を別nodeにする有向の参加関係である。[[7]](#ref-7)
化学量論と反応数を保つ一方、原子ごとの移動先や全粒子収支を一意に示すものではない。
入口・排気・zone間輸送、宣言されていない壁吸収、壁電荷、電子closureの寄与は反応図に含めない。

## 11. 検証方法と再現条件

### 11.1 検証の階層

ここでいうverificationは、選んだ方程式を実装・積分できているかの確認である。
validationは、対象装置の観測量を所定の精度で予測できるかの確認である。
解析解や別コードとの一致は前者の根拠になるが、後者の代わりにはならない。

| 検証 | 比較対象 | 読み取れること |
|---|---|---|
| 断面積積分 | 解析式・独立quadrature | 積分実装の正確さ |
| 一次反応・2区画移送 | 独立に導いた時間解 | ODE・体積係数・粒子保存 |
| 別コード由来case | 保存済み数値出力 | 限定した条件の回帰一致 |
| 許容差・保存間隔の収束 | より細かい数値解 | 数値設定への依存 |
| 入力感度 | 明示した条件差 | その条件近傍で結果を変える因子 |
| 独立実験 | 観測領域・入力を揃えた測定 | 装置予測の妥当性。今回は未実施 |

### 11.2 実行条件と誤差の定義

19ケースはWSL/Linux x86_64、Python 3.12.3、NumPy 2.5.4、SciPy 1.18.1、h5py 3.16.0、
Pydantic 2.14.1、PyYAML 6.0.3で実行した。図はMatplotlib 3.11.2を使う。
個々のrunはHDF5に実効入力、入力checksum、コードhash、依存version、終了状態を持つ。
数値表の正本は `benchmarks/theory/results.json`、再計算入口は `tools.benchmarks.theory` である。
測定した時間は設定解決後のcompileとsimulateで、図の作成・結果保存は含まない。
同一ホストでの単発測定であり、他のソルバーに対する速度優位性は評価していない。
計算時間はホスト負荷とcache状態にも左右される参考値である。

scalarの相対差と、波形全体の規格化最大差を区別する。

$$
\delta_q=\frac{|q-q_{\mathrm{ref}}|}{|q_{\mathrm{ref}}|},\qquad
E_q=\frac{\max_t|q(t)-q_{\mathrm{ref}}(t)|}{\max_t|q_{\mathrm{ref}}(t)|}. \tag{38}
$$

Eは各時刻の相対差の最大とは異なる。参照解を細かくしただけの場合、その解自体の誤差がゼロになったと
解釈しない。以下の表は表示桁を丸めるが、再計算用JSONは丸めない。

## 12. ベンチマーク結果

### 12.1 Maxwellian速度係数

一定断面積σ₀を閾値 $E_0$ から $E_{\max}$ まで与えた場合、式(13)の解析式は次である。

$$
k=\sigma_0\sqrt{\frac{8e\theta_e}{\pi m_e}}
\left[(1+a)e^{-a}-(1+b)e^{-b}\right],\quad
a=E_0/\theta_e,\ b=E_{\max}/\theta_e. \tag{39}
$$

σ₀=2×10⁻²⁰ m²、終端200 eV、閾値0／2 eV、平均energy 0.5〜10 eVの16条件を使った。
さらに、同梱Ar電離curveを独立なSciPy `quad` で積分し、平均energy 0.5〜8 eVの7条件を比べた。
断面積はZenodo収録データ由来であり、分岐した元curveの出典も入力に保持される。[[8]](#ref-8)

| 比較 | 条件数 | 最大相対差 |
|---|---:|---:|
| 一定断面積の解析式 | 16 | 1.90×10⁻¹⁶ |
| Ar電離curveの独立quadrature | 7 | 1.33×10⁻¹⁵ |

![Ar電離係数の積分比較](assets/theory-maxwell.png)

図1. 左は速度係数、右は独立quadratureとの差。収録された同じcurveとMaxwellian分布に対する
積分手段の比較であり、実際の電子分布や元断面積が正しいことを示す図ではない。

### 12.2 解析解を持つ反応と輸送

一次反応 $\mathrm{Ar}^*\to\mathrm{Ar}$、λ=25 s⁻¹では
$n_*(t)=n_*(0)\exp(-\lambda t)$ が解となる。0〜0.2 s、保存間隔0.002 sで比較した。
2区画移送は式(30)に対し、Vₐ=0.01 m³、Vᵦ=0.03 m³、C=0.02 m³/s、期間0.5 sとした。

$$
n_a(t)=n_{a0}e^{-Ct/V_a},\qquad
n_b(t)=n_{b0}+\frac{V_a}{V_b}n_{a0}(1-e^{-Ct/V_a}). \tag{40}
$$

| 比較量 | 最大誤差 |
|---|---:|
| 一次反応の各保存時刻での相対差 | 1.86×10⁻⁵ |
| 移送source密度の各時刻相対差 | 2.66×10⁻⁶ |
| 移送target密度の各時刻相対差 | 1.99×10⁻⁶ |
| 全Ar粒子数の初期値からの相対変化 | 3.64×10⁻¹⁶ |

これらは人為的に簡略化した解析検証である。使用した既定rtol 1×10⁻⁶は局所誤差の制御なので、
全区間での相対誤差が必ず1×10⁻⁶以下になるわけではない。
粒子数の保存は密度誤差とは独立の確認であり、異なる体積の扱いを検証できる。

### 12.3 CRANEの2反応Ar

CRANEはMOOSE上の化学反応ソルバーである。[[9]](#ref-9)
公開tutorialの2反応は電離と三体再結合で、定常密度の関係は $n_i=k_i/k_r$ となる。
本比較では公開保存出力の0.75 μsにおける値を用いた。[[10]](#ref-10)
初期Ar密度2.5×10²⁵ m⁻³、電子・イオンseed 1×10⁶ m⁻³、
電離係数2.1736169000623×10⁻¹⁸ m³/s、再結合係数1×10⁻³⁷ m⁶/sである。

| 量 | 本コード | 保存済みCRANE出力 |
|---|---:|---:|
| 最終Ar⁺密度 [m⁻³] | 2.17352259×10¹⁹ | 2.17352774×10¹⁹ |
| 相対差 | 2.37×10⁻⁶ | - |

既存の回帰許容差5×10⁻⁴以内であった。これは固定係数の気相化学に限る比較である。
参照tutorialの室温と本caseの300 Kは同一ではないが、密度と固定係数を揃えており、
温度依存係数や電子energyの検証には使わない。圧力は低圧Ar例と大きく異なり、
低圧の壁損失モデルを大気圧まで検証したことにもならない。

### 12.4 ZDPlaskin由来の化学・DC回路

保存済みexample2を出典とする反応係数CSVと直列回路設定を使った。[[11]](#ref-11)
条件は100 Torr、300 K、電極間0.004 m、半径0.004 m、電源1000 V、直列抵抗100 kΩであり、
終端は7.520833×10⁻⁴ sである。Ar*、Ar⁺、Ar₂⁺を扱う。

| 比較量 | 本コード | 保存参照 | 相対差 [%] |
|---|---:|---:|---:|
| 最終Ar* [m⁻³] | 2.014779×10¹⁷ | 2.014270×10¹⁷ | 0.0253 |
| 最終Ar⁺ [m⁻³] | 2.083214×10¹⁵ | 2.079332×10¹⁵ | 0.1867 |
| 最終Ar₂⁺ [m⁻³] | 3.149623×10¹⁷ | 3.149570×10¹⁷ | 0.00168 |
| 最終電子密度 [m⁻³] | 3.170455×10¹⁷ | 3.170363×10¹⁷ | 0.00290 |
| 保存点のpeak電子密度 [m⁻³] | 3.530828×10¹⁷ | 3.571898×10¹⁷ | 1.1498 |
| 最終プラズマ電圧 [V] | 33.00284 | 33.22250 | 0.6612 |
| 最終電流 [A] | 0.00966997 | 0.00966778 | 0.0227 |
| 最終E/N [Td] | 2.563261 | 2.561342 | 0.0749 |

種密度・保存peakの10%、回路量の5%という既存の回帰許容差内であった。
peakはそれぞれの保存点での最大値であり、連続時間の真の最大値とは限らない。
さらに係数表は同じ保存出力の反応rate列から逆算したデータを含む。
従って完全に独立した物理予測の比較ではなく、別コード由来データと回路・化学の接続を確認する回帰である。
元のFortran／BOLSIG+を今回独立に実行した結果ではない。密度、電圧、電流、E/Nの単位と比較時刻を明示して評価する。

### 12.5 低圧Arの過渡応答と許容差収束

体積0.01 m³、壁面積0.1 m²、初期圧力0.8283894 Pa、気体温度300 K、Ar⁺ seed 1×10¹⁴ m⁻³、
初期平均energy 3 eV、通電0〜1 msは電子吸収電力0.1 W、1〜2 msは0 Wとした。
初期中性モル分率はAr=1とし、圧力から中性密度を計算する。
同梱 `argon_maxwellian` の直接密度指定をこの方式へ置き換えたため、初期Arは厳密には
$p/(k_BT_g)-n_{i0}$ であり、単純に2×10²⁰ m⁻³と固定した入力とは微小に異なる。
気相31反応は電離1個と、励起をenergy損失として扱う30個であり、Ar*は積分しない。

![低圧Arの通電とafterglow](assets/theory-transient.png)

図2. 縦線は1 msの電源切替。通電で密度・energyが増加し、停止後に減衰する。
2 msでの密度約4.42×10¹³ m⁻³と等価温度約0.0125 eVは、この過渡例の終端値である。

| rtol / atol | 密度波形差E | 温度波形差E | RHS評価回数 | 実測時間 [s] |
|---|---:|---:|---:|---:|
| 10⁻⁶ / 10⁻¹⁴ | 1.12×10⁻⁶ | 1.41×10⁻⁶ | 774 | 1.81 |
| 10⁻⁷ / 10⁻¹⁵ | 1.91×10⁻⁷ | 2.63×10⁻⁷ | 1116 | 2.26 |
| 10⁻⁸ / 10⁻¹⁶ | 参照 | 参照 | 1624 | 2.73 |

保存間隔は全て10 μs、201点で同じである。厳しい許容差ほど参照への差が減少した。
参照は数値解であり、理論上の厳密解や実験値ではない。

### 12.6 イベント数は保存間隔で変わる

式(37)についてODE許容差を10⁻⁸／10⁻¹⁶に固定し、保存間隔だけを変えた。
全runのRHS評価回数1624、Jacobian評価6、LU分解107は同じだった。
下表の差は1.25 μsを参照とする。ゼロと表示される参照解にも積分誤差は残る。

| 保存間隔 [μs] | 保存点数 | 電離数の相対差 | 壁イベント数の相対差 |
|---|---:|---:|---:|
| 40 | 51 | 3.94×10⁻³ | 2.40×10⁻³ |
| 20 | 101 | 1.74×10⁻³ | 7.08×10⁻⁴ |
| 10 | 201 | 7.61×10⁻⁴ | 1.80×10⁻⁴ |
| 5 | 401 | 4.08×10⁻⁴ | 4.04×10⁻⁵ |
| 2.5 | 801 | 1.71×10⁻⁴ | 8.60×10⁻⁶ |
| 1.25 | 1601 | 参照 | 参照 |

![イベント積分の保存間隔依存](assets/theory-sampling.png)

図3. 保存間隔を細かくすると積分結果が変化する。粗い条件では急速なstartup応答を十分に分解できず、
この区間だけから理想的な2次収束を主張しない。
最も細かい条件の電離数は7.46775×10¹²、壁イベント数は8.02556×10¹²である。
それらの差と実際のイオン粒子数変化の残差を、大きい方のイベント数で規格化すると7.59×10⁻⁵残る。
図の線幅やイベント順位に必要な精度を決め、保存間隔の確認を行う意義が分かる。

### 12.7 初期圧力・電力・壁面積の感度

同じ過渡Arについて各入力を独立に±10%変えた。下表は2 msでの基準runからの変化率である。

| 入力変更 | 終端電子密度 [%] | 終端電子温度 [%] |
|---|---:|---:|
| 通電電力 -10% | -9.994 | 約0 |
| 通電電力 +10% | +9.994 | 約0 |
| 初期圧力 -10% | -5.789 | -5.835 |
| 初期圧力 +10% | +5.578 | +5.702 |
| 壁面積 -10% | +23.108 | +28.542 |
| 壁面積 +10% | -17.239 | -20.426 |

![入力の局所感度](assets/theory-sensitivity.png)

図4. 同じ±10%でも壁面積の影響が大きく、応答は対称でもない。
初期圧力変更時は中性密度を再計算し、荷電seedは固定した。
面積変更では式(23)の特性長V/Aも変わるため、単に粒子損失を面積比例で変えた実験ではない。
これらは任意幅の感度であり、測定誤差の分布、95%信頼区間、装置予測の誤差ではない。

## 13. 考察: どこまで有用で、何が未検証か

### 13.1 本結果から説明できる有用性

第12節の検証は、反応係数の積分、異なる体積間での粒子保存、反応・回路の接続、
時間収束をそれぞれ別の比較で調べている。単に全テストが成功したという説明より、
どの計算がどの独立な式・数値に対応するかを第三者が確認しやすい。
過渡Arでは、終端密度だけでなく電子energy、壁損失、反応数、入力感度を組み合わせて、
何を変えると結果が変わるかを探索できる。未知の壁条件や係数を優先して調べるための
モデル構築・実験設計の入口として有用である。

入力は体積・面積・反応・電源と運転条件へ分かれ、式(5)・(17)の寄与へ対応する。
1 zoneから始め、根拠のある場合に輸送区画や表面反応を増やせる。
全状態を保存し、派生量・単位・モデル・入力hashを残すため、結果の再解析と比較を継続しやすい。
これらはモデル基盤としての有用性であり、特定装置の完成済みデジタルツインという主張ではない。

### 13.2 afterglow末期とBohm近似

Ar例の終端等価電子温度0.0125 eVは、300 Kのenergy相当0.02585 eVより低い。
イオン温度を気体温度程度とみなすなら、冷イオンBohm式が仮定する電子温度優勢は成立しない。
電源停止後の電子分布、イオン温度、準中性の成立、壁電流収支を独立に確認する必要がある。
本稿の波形は実装した方程式の応答として示し、低温afterglowの装置精度の根拠には使わない。

本実装の `standard` は、固定気体温度、準中性・電気的正性、単一一価正イオン、
Maxwellian＋電子energy収支、吸収電力指定、Bohm壁等という機能構成の分類である。
電子・イオン温度比や分布の妥当性を全時刻で認定するflagではない。

### 13.3 測定との比較を成立させる条件

局所peak、線積分、体積平均は異なる量である。例えば光学計測には、発光種、励起状態、
遷移確率、消光、観測線の積分が必要であり、重粒子密度へprobe名を付けるだけでは発光強度にならない。
Langmuir probeには電流電圧応答と電子分布の仮定が必要である。
装置比較では、実効体積、壁面積、吸収電力、気体温度、観測領域の対応を先に固定する。
校正したpressure／power条件を、そのまま独立な評価条件とみなさない。

誤差の原因には、数値誤差、入力・測定の不確かさ、反応機構の不足、closureのモデル化誤差がある。
パラメータの確率分布に根拠がある場合には、例えば一次近似の共分散伝播を考えられる。

$$
\operatorname{Cov}(\boldsymbol{q})\simeq
J_p\operatorname{Cov}(\boldsymbol{p})J_p^{\mathsf{T}},\qquad
(J_p)_{ij}=\frac{\partial q_i}{\partial p_j}. \tag{41}
$$

これは不確かさを評価するための一般的な考え方で、このパッケージに自動推定・校正機能として
実装済みという意味ではない。非線形性や相関が大きければ、分布と観測を明示した別の評価が必要である。

## 14. 結論

本コードは、粒子・電子energy・必要な気体energyと表面状態の収支を、固定した区画のODEとして
共通の数値核で解く。反応、壁、輸送、電源の項と入出力の対応が明示され、
初期段階の反応機構の評価、過渡応答、条件感度、モデル拡張の基盤に利用できる。
今回の数値検証では、解析積分と解析解、輸送保存、別コード由来の限定比較、数値収束が確認された。
一方、空間相関、非Maxwellian応答、経験的電源結合、組成固定の係数、低温afterglowは、
数値一致からは解消されない限界である。今後の優先事項は、観測量を定義した独立実験との比較と、
根拠を持つ入力不確かさ・モデル化誤差の評価である。

## 付録A. 近似EEDFは何を計算しているか

`experimental.approximate_two_term` は、完全な電子Boltzmann方程式の角度展開・衝突演算子を
離散化した検証済みtwo-term solverではない。名称だけからBOLSIG+等と同じ方程式を解くと解釈しない。
式(12)へ渡す分布候補を経験的なenergy-space形状で作り、電子1個あたりの電力釣り合いで
その形状パラメータを選ぶ探索用closureである。

混合比で重み付けした運動量断面積をσₘ、非弾性損失断面積を
$L_\sigma=\sum_r x_r\Delta\varepsilon_r\sigma_r$ とする。
energyを1 eVで規格化したE、形状パラメータξ、規格化断面積M・損失Lに対して、実装の形状式は次である。

$$
\mathcal{D}_j=\frac{(\xi+0.02)\sqrt{E_j+0.03}}{1+0.35M_j},\qquad
\mathcal{B}_j=0.8+0.6M_j+\frac{1.4L_j}{\max(\xi,10^{-8})}. \tag{A1}
$$

$$
d_j=\operatorname{clip}(\mathcal{B}_j\Delta E_j/\max(\mathcal{D}_j,10^{-30}),10^{-10},90),\qquad
\widetilde F_j=\frac{\exp(-\sum_{l<j}d_l)}{\sqrt{E_j+0.02}}. \tag{A2}
$$

Mはσₘをgrid上の平均で、Lは損失断面積を正の値の平均で規格化する。
ゼロ除算を避ける最小値を使い、台形積分でFを規格化する。
0.02、0.03、0.35、0.8、0.6、1.4等は本近似の数値定数で、普遍的な物理定数ではない。

分布から運動量移行頻度、mobility、平均energy、拡散係数の候補を求める。

$$
\nu_m=N\int\sigma_m vF\,d\varepsilon,\quad
\mu=\frac{e}{m_e\nu_m},\quad
D_e=\mu\frac{2\bar\varepsilon}{3}. \tag{A3}
$$

最後の式はenergyの数値をeVとしたEinstein型関係である。
shapeを決める電力釣り合いは次である。各項は電子1個あたりWである。

$$
e\mu E_{\mathrm{field}}^2
=eN\int L_\sigma vF\,d\varepsilon
+eN\left(\int\sigma_m vF\,d\varepsilon\right)\varepsilon_{\mathrm{el}}. \tag{A4}
$$

代理弾性損失 $\varepsilon_{\mathrm{el}}$ は既定0.03 eVであり、RHSの電子・気体交換式(18)とは別の近似である。
shape範囲0.02〜220に根が挟まることを要求し、対数残差5×10⁻⁴以下を目標に反復する。
energy grid最終binの確率質量が1×10⁻³を超える解を拒否する。
電源・係数の外側反復に使う1×10⁻⁶の判定と、分布生成内部の判定を混同しない。

通常runでは初期中性組成ごとに準備した表を用い、反応で組成が変わっても再生成しない。
準備表に追加する0-field nodeは平均energy・rateを0とし、mobilityは最初の値を使う境界拡張であり、
気体との熱平衡EEDFを求めた解ではない。組成の変化、energy grid、衝突収録範囲、独立Boltzmann解との
比較を行ってから利用範囲を決める必要がある。

## 付録B. 経験的電源と蓄積状態の実装式

### B.1 RF envelope

吸収power指定ではsource側電力を $P_d=P_a/\eta$ とし、$V_{\mathrm{rms}}=\sqrt{P_dR_{\mathrm{eff}}}$。
電圧指定では $P_d=V_{\mathrm{rms}}^2/R_{\mathrm{eff}}$ と $P_a=\eta P_d$ を使う。
矩形包絡をg(t)とすると、換算電場と自己biasは次の形である。

$$
(E/N)_{\mathrm{Td}}=a_0g(t)+a_1\sqrt{P_a},\qquad
V_{\mathrm{bias}}=-f_b\sqrt{2}V_{\mathrm{rms}}. \tag{B1}
$$

bias項はbias役割のportだけに適用する。電位もoffsetと電力平方根で構成する。
これらの係数は校正入力であり、電磁場計算から決まらない。

### B.2 CCPの経験的bulkとsheath

bulk抵抗は有効長 $L_{\mathrm{eff}}=V/A_p$、$n_{16}=n_e/10^{16}$、$f_*=f/13.56\,\mathrm{MHz}$ に対し、
数値をSIで与えて次の経験式を用いる。

$$
R_p=5\frac{\max(L_{\mathrm{eff}}/\mathrm{m},10^{-6})}
{\sqrt{\max(n_{16},10^{-10})f_*}}\,\Omega. \tag{B2}
$$

sheath容量は $C_s=\epsilon_0A_s/d_s$、厚さはDebye長に電圧依存の係数を掛けて推定する。

$$
\lambda_D=\sqrt{\frac{\epsilon_0\max(\theta_e,0.03)}{e\max(n_e,10^6)}},\qquad
d_s=\max\left(4\lambda_D\sqrt{1+\frac{\max(V,1)}{\max(\theta_e,0.1)}},5\times10^{-5}\right). \tag{B3}
$$

ここでnはm⁻³、θはeV、VはV、dはmというSI数値を使う。
最小密度・最小温度・厚さはこの経験式を評価する正則化であり、独立に積分する粒子密度を修正しない。
面積比の1.35乗から自己biasを推定し、イオンenergyの診断は
$\varepsilon_i\simeq V_{s,p}/(1+N\sigma d_s)$、σ=5×10⁻²⁰ m²とする。
これらはイオンenergy分布やlossless回路に含まれないsheath実損失を解くものではない。

### B.3 ICPの連続なmode指標

密度因子d、mode指標m、滑らかなH側の重みhを次で作る。

$$
d=1-e^{-2.2n_e/10^{17}},\quad m=d\sqrt{P_d/150},\quad
x=\operatorname{clip}((m-0.45)/0.20,0,1),\quad h=x^2(3-2x). \tag{B4}
$$

pressure因子aと周波数因子b、積u、結合効率は次である。圧力pは中性密度から計算するPa値である。

$$
a=0.65+0.20\tanh\left(\frac{\max(p/10,10^{-3})-0.3}{0.6}\right),\quad
b=0.85+0.08\log_{10}(1+\max(f_*,0.1)). \tag{B5}
$$

$$
u=dab,\quad \eta_E=0.18+0.37u,\quad\eta_H=0.40+0.48u,\quad
\eta=\operatorname{clip}(d[\eta_E+h(\eta_H-\eta_E)],0,0.95). \tag{B6}
$$

`mode_H` はhであり、測定に基づくH-mode認定ではない。
換算電場は、ゼロpower・ゼロ密度で消えるactivity因子に
$25+30\sqrt{\theta_e}+0.01P_a$ を掛けた経験値である。
係数中の150はW、密度の基準はm⁻³である。装置ごとの電磁場・整合回路・吸収分布は解いていない。

### B.4 filmとwall inventory

面イベント速度r、蓄積収率y、膜の層収率ℓ、単層厚さ $d_0$ から、蓄積粒子数Bと厚さδを計算する。

$$
\frac{dB}{dt}=Ayr,\qquad
\frac{d\delta}{dt}=d_0\frac{\ell r}{N_{\mathrm{site}}}. \tag{B7}
$$

これは宣言したイベントの積算で、材料の組織、面内均一性、応力、熱伝導を予測する式ではない。
汎用extension状態も宣言した項をODEへ追加する仕組みであり、状態名だけから物理を推定しない。

## 付録C. 再計算とコードとの対応

ベンチマークはネットワーク接続を使わず、同梱入力と保存参照から再計算できる。
Python環境に本packageとplot依存を入れ、リポジトリのルートで実行する。

```bash
python -m pip install ".[plot]"
python -m tools.benchmarks.theory --output .quality-reports/theory-new
```

出力先は未作成の場所を使う。`results.json`、4つの図、19 runのHDF5・summaryを保存する。
JSONの数値と本文の表示桁を比較し、保存HDF5の実効caseで条件を確認する。
依存・codeが変わればruntime hashと測定時間も変わるため、過去の数値を新環境の検証記録へ付け直さない。

| 理論・方程式 | 主な実装 |
|---|---|
| 式(5)・(6) 粒子と反応 | `core/_runtime_sources.py`, `chemistry/compile.py` |
| 式(9)・(16) 電子状態 | `models/electrons.py` |
| 式(13)・(14) Maxwell積分 | `chemistry/_maxwell.py` |
| 式(17)・(18) 電子energy・弾性移送 | `core/_runtime_sources.py`, `models/elastic.py`, `models/gas_energy.py` |
| 式(19)〜(21) 気体energy | `models/gas_energy.py`, `core/transport.py` |
| 式(22)〜(25) 壁 | `models/_wall_runtime.py`, `models/walls.py` |
| 式(26)〜(28) 表面 | `models/_surface_runtime.py`, `models/_surface_kernels.py` |
| 式(30) 区画輸送 | `core/transport.py` |
| 式(31)・(32) DC | `models/_power_ports.py` |
| 式(34)・時間積分 | `core/solver.py`, `core/_solver_domain.py` |
| 式(36)・(37) イベント | `_output_channels.py`, `postprocess.py` |
| 付録A・B | `experimental/eedf.py`, `experimental/_power_kernels.py`, `experimental/_ccp_power_kernel.py` |

全pathの先頭には `plasma_global/` を付ける。
実効case、入力checksum、code hash、依存versionはrunのHDF5に残り、図から逆算して設定を補完しない。
本文Markdownを原本とし、PDFは同じ原本から生成する。PDF作成・日本語font指定は[開発・保守](DEVELOPMENT.md)に記す。

## 参考文献・データ出典

以下の文献はモデルの背景・手法・データ出典を支える。本実装や装置精度が文献で認定されたことを意味しない。
Web資料の確認日は2026年10月26日。式の実装上の定数・境界処理は付録Cのコードを正本とする。

<a id="ref-1"></a>

**[1]** Hurlbatt, A. et al. (2017). *Concepts, Capabilities, and Limitations of Global Models: A Review*.
Plasma Processes and Polymers 14, 1600138.
[DOI: 10.1002/ppap.201600138](https://doi.org/10.1002/ppap.201600138).

<a id="ref-2"></a>

**[2]** Hagelaar, G. J. M. and Pitchford, L. C. (2005). *Solving the Boltzmann equation to obtain electron
transport coefficients and rate coefficients for fluid models*. Plasma Sources Science and Technology 14, 722-733.
[DOI: 10.1088/0963-0252/14/4/011](https://doi.org/10.1088/0963-0252/14/4/011).

<a id="ref-3"></a>

**[3]** Pitchford, L. C. et al. (2017). *LXCat: an Open-Access, Web-Based Platform for Data Needed for
Modeling Low Temperature Plasmas*. Plasma Processes and Polymers 14, 1600098.
[DOI: 10.1002/ppap.201600098](https://doi.org/10.1002/ppap.201600098).

<a id="ref-4"></a>

**[4]** Kemaneci, E., Carbone, E., Booth, J.-P., Graef, W., van Dijk, J. and Kroesen, G. (2014).
*Global (volume-averaged) model of inductively coupled chlorine plasma: Influence of Cl wall recombination
and external heating on continuous and pulse-modulated plasmas*. Plasma Sources Science and Technology 23, 045002.
[DOI: 10.1088/0963-0252/23/4/045002](https://doi.org/10.1088/0963-0252/23/4/045002).

<a id="ref-5"></a>

**[5]** SciPy developers. *scipy.integrate.BDF*. 本数値実験の利用versionは1.18.1。
[公式API](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.BDF.html).
本文の既定許容差とstate規格化は本packageの設定で、SciPy本体の既定値とは異なる。

<a id="ref-6"></a>

**[6]** Shampine, L. F. and Reichelt, M. W. (1997). *The MATLAB ODE Suite*.
SIAM Journal on Scientific Computing 18(1), 1-22.
[DOI: 10.1137/S1064827594276424](https://doi.org/10.1137/S1064827594276424).

<a id="ref-7"></a>

**[7]** Graphviz project. *dot: hierarchical or layered drawings of directed graphs*.
[公式layout資料](https://graphviz.org/docs/layouts/dot/).

<a id="ref-8"></a>

**[8]** Schmalzried, A. (2023). *Bolsig+ format cross section of e-N2, O2, NO, Ar, O, N*, v1. Zenodo.
[DOI: 10.5281/zenodo.8192503](https://doi.org/10.5281/zenodo.8192503).
本packageのAr機構はこの収録データのAr部分を用いる。

<a id="ref-9"></a>

**[9]** Keniley, S. and Curreli, D. (2019). *CRANE: A MOOSE-based Open Source Tool for Plasma Chemistry Applications*.
[arXiv:1905.10004](https://arxiv.org/abs/1905.10004).

<a id="ref-10"></a>

**[10]** CRANE developers. *Two-Reaction Argon Plasma*, tutorial and committed output.
[公式tutorial](https://crane-plasma-chemistry.readthedocs.io/en/latest/TwoReactionArgon.html),
[source repository](https://github.com/lcpp-org/crane).
比較値の固定記録は `benchmarks/references/crane_two_reaction_argon_reference.yaml`。

<a id="ref-11"></a>

**[11]** *ZDPlaskin example2*, 保存出力を含むsource repository.
[Hemadityamalla/ZDPlaskin](https://github.com/Hemadityamalla/ZDPlaskin).
本稿が使用した資産は `benchmarks/raw/zdplaskin_example2_eovern/` と `benchmarks/references/` に保持する。
このrepository由来の保存結果との比較であり、ZDPlaskinの全版・全機能の検証を表すものではない。
