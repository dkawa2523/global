# 数値解法とソルバー

このコードが解く方程式は、反応速度、壁損失、電子エネルギー、表面反応を含む stiff ODE です。反応ごとの時定数が大きく異なり、電子衝突は速く、表面・流量・recipe step は相対的に遅いため、陽解法だけで安定に積分するのは効率が悪くなります。

現在の標準 solver は、SciPy `solve_ivp(method="BDF")` を使う implicit BDF backend です。実装上の入口は `plasma_global/numerics/scipy_backend.py`、ODE system と Jacobian は `plasma_global/numerics/system.py` が提供します。

## 解いている問題

solver から見ると、プラズマモデルは通常の初期値問題です。

$$
\frac{d\mathbf{y}}{dt}
= \mathbf{f}(t,\mathbf{y}),
\qquad
\mathbf{y}(t_0)=\mathbf{y}_0 .
$$

$\mathbf{y}$ には、zone ごとの gas species density、electron energy density、gas temperature、surface coverage、wall inventory、film thickness などが並びます。電子密度は多くの case で独立 state ではなく、準中性条件から algebraic に再構成されます。

プラズマ chemistry では、同じ state vector の中に次のような異なる時定数が混在します。

$$
\tau_r \sim \frac{1}{k_r n},
\qquad
\tau_{\mathrm{wall}} \sim \frac{V}{A u_B},
\qquad
\tau_{\mathrm{flow}} \sim \frac{V}{Q}.
$$

短い反応時定数と長い residence time が同時に現れると、ODE は stiff になります。このため、時間 step を単に細かくするより、implicit stiff solver と Jacobian を使う方が実用的です。

## BDF を使う理由

BDF は backward differentiation formula の略で、過去の解と現在時刻の未知解を使う implicit multistep method です。概念的には、時刻 $t_{n+1}$ の解 $\mathbf{y}_{n+1}$ を次の非線形方程式として求めます。

$$
\sum_{j=0}^{q} \alpha_j \mathbf{y}_{n+1-j}
= h\,\beta\,\mathbf{f}(t_{n+1},\mathbf{y}_{n+1}) .
$$

ここで $h$ は step size、$q$ は次数です。SciPy の BDF 実装は variable-order 1--5 の implicit method で、quasi-constant step scheme と NDF modification を使います。stiff problem では、explicit Runge-Kutta 法より大きな step を取れる場合があり、反応 network のような問題に向いています。

BDF の各 step では、$\mathbf{y}_{n+1}$ が右辺にも現れるため Newton iteration が必要になります。その線形化では Jacobian

$$
J(t,\mathbf{y})
= \frac{\partial \mathbf{f}}{\partial \mathbf{y}}
$$

を使い、概念的には次の線形系を解きます。

$$
\left(I - h\beta J\right)\Delta\mathbf{y}
= -\mathbf{F}(\mathbf{y}) .
$$

したがって、計算時間は RHS 評価だけでなく、Jacobian 評価と線形解法にも支配されます。

## 実装上の solver 構成

現在の integrator は次の形で SciPy に渡されます。

```python
solve_ivp(
    fun=system.rhs,
    t_span=(t0, t1),
    y0=y0,
    method="BDF",
    jac=system.jacobian,
    rtol=rtol,
    atol=atol,
    first_step=first_step,
    max_step=max_step,
)
```

default の数値設定は `RunConfig.numerics` で管理されます。

| 設定 | default | 意味 |
|---|---:|---|
| `rtol` | `1.0e-6` | relative tolerance |
| `atol` | `1.0e-14` | absolute tolerance |
| `first_step` | `1.0e-10 s` | 初期 step の目安 |
| `max_step` | `1.0e-6 s` | recipe や急峻な過渡を飛ばしすぎないための上限 |
| `jacobian` | `analytic_sparse` | RHS に対応する sparse Jacobian を使う |

SciPy の local error control は、概念的に各成分の誤差を

$$
e_i
\lesssim
\mathrm{atol}_i
+ \mathrm{rtol}\, |y_i|
$$

の範囲に抑える形で働きます。密度、coverage、electron energy のスケールが大きく違うため、`atol` を大きくしすぎると小さい成分の相対精度が落ちます。一方、`atol` を過度に小さくすると、物理的には意味の薄い floor 近傍の成分で step が細かくなりすぎます。

## Sparse Jacobian の役割

大きな reaction network では、全 state が全 state に依存するわけではありません。たとえば、ある species の反応 source は、その反応に入る species と電子エネルギーには依存しますが、無関係な surface coverage には依存しません。この構造を使うため、コードは sparse Jacobian を返します。

$$
J_{ij}
= \frac{\partial f_i}{\partial y_j}.
$$

Jacobian には、gas chemistry、transport、surface chemistry、Bohm-family ion wall loss などの主要な RHS contribution が含まれます。現在の reduced electrical coupling は、Jacobian 評価中には frozen contribution として扱います。direct power や簡約 proxy backend では実用上問題になりにくい一方、強く power-coupled な高忠実度 model では、electrical backend 側の derivative を明示的に入れる余地があります。

効率面では、sparse Jacobian は主に 2 つの意味を持ちます。

1. 有限差分で Jacobian を推定するより RHS 評価回数を抑えられる。
2. 線形系の LU 分解で dense matrix より小さいメモリと計算量にできる。

SciPy の戻り値には `nfev`, `njev`, `nlu` が含まれます。このコードではそれらを solver diagnostics として保存します。

| 指標 | 意味 | 読み方 |
|---|---|---|
| `nfev` | RHS 評価回数 | chemistry / coupling 評価の総負荷 |
| `njev` | Jacobian 評価回数 | stiffness や nonlinear coupling の強さ |
| `nlu` | LU 分解回数 | implicit step の線形解法コスト |

`nlu` が大きい場合、単に出力点数が多いのではなく、Newton iteration や step rejection が増えている可能性があります。rate table の不連続、recipe の急峻な切り替え、過度に厳しい tolerance、floor 近傍の stiff loss が原因になりやすいです。

## recipe step と時間刻み

runner は recipe step ごとに integration segment を切ります。これは、pressure、flow、power command が step boundary で変わるためです。segment 内では BDF が内部 step を自動選択し、`t_eval` が指定されていれば、必要な時刻に補間された解を返します。

`max_step` は、物理的な精度だけでなく diagnostics の解釈にも効きます。大きすぎると短い pulse や急峻な power change をまたいでしまい、小さすぎると不要に遅くなります。pulse recipe では、最短 pulse width より十分小さい `max_step` を設定するのが安全です。

## state projection と positivity

積分後、state は admissible region に戻されます。これは物理モデルではなく、数値的な保護です。

密度と電子エネルギーには floor を置きます。

$$
n_s \leftarrow \max(n_s,n_{\min}),
\qquad
W_e \leftarrow \max(W_e,W_{\min}) .
$$

surface coverage は、free site を含む site balance を回復します。

$$
0 \le \theta_j \le 1,
\qquad
\sum_j \theta_j + \theta_{\mathrm{free}} = 1 .
$$

projection は、負密度や coverage sum の小さな数値誤差を抑えるための処理です。頻繁に大きな projection が必要な case は、solver tolerance の問題ではなく、reaction rate、surface model、wall loss、initial condition の不整合を疑うべきです。

## Jacobian check

RHS や Jacobian を変更した場合は、finite-difference checker で analytic Jacobian を確認します。

```powershell
py -m plasma_global.cli check-jacobian examples\configs\case_smoke.yaml --advance-s 1e-7 --top 8
```

有限差分は

$$
\frac{\partial f_i}{\partial y_j}
\approx
\frac{f_i(y_j+\delta)-f_i(y_j-\delta)}{2\delta}
$$

で評価します。coverage state は projection の影響を受けやすいため、checker では既定で `theta` prefix を skip します。これは Jacobian を無視してよいという意味ではなく、制約付き state の有限差分比較が単純ではないためです。

## 効率を悪化させやすい条件

計算が重くなる典型例は次の通りです。

| 条件 | 何が起きるか | 対応 |
|---|---|---|
| 速い反応と遅い流量が同居 | stiffness が強くなる | BDF + sparse Jacobian を維持する |
| rate table が粗い / 不連続 | Newton iteration と step rejection が増える | table grid と補間を確認する |
| `max_step` が小さすぎる | 必要以上に内部 step が増える | pulse 幅と diagnostics に合わせて緩める |
| `atol` が小さすぎる | floor 近傍の species が計算を支配する | species scale に対して tolerance を見直す |
| 電力 backend が state に強く依存 | Jacobian frozen 近似の影響が出る | backend derivative の追加を検討する |
| surface coverage が飽和近傍 | projection と stiff surface rate が効く | site density、initial coverage、rate law を確認する |

## 現在の限界

現在の solver は stiff ODE solver であり、DAE solver ではありません。quasi-neutrality は algebraic reconstruction として扱っていますが、Poisson equation や circuit DAE を同時に解くわけではありません。

また、BDF backend は現在の標準経路として検証されていますが、すべての backend derivative が analytic Jacobian に完全に入っているわけではありません。特に、高忠実度の power-coupled model、RF sheath dynamics、外部 circuit との双方向 coupling を入れる場合は、DAE formulation、mass matrix、または専用 integrator の検討が必要です。

長時間の周期運転では、毎周期をそのまま積分すると非効率になる場合があります。将来的には periodic-orbit acceleration、checkpoint / restart、event-driven step control が有用です。

## 実装上の確認先

- `plasma_global/numerics/scipy_backend.py`: SciPy BDF への接続。
- `plasma_global/numerics/system.py`: RHS、Jacobian、projection、state manifest。
- `plasma_global/numerics/jacobian_check.py`: finite-difference Jacobian checker。
- `plasma_global/numerics/state_layout.py`: state vector の並び。
- [state vector の読み方](STATE_VECTOR_GUIDE.md): 出力から state の意味を確認する手順。

## 参考文献

| 参考 | 関連する内容 |
|---|---|
| [SciPy `solve_ivp` documentation](https://docs.scipy.org/doc/scipy-1.15.3/reference/generated/scipy.integrate.solve_ivp.html) | BDF、Radau、LSODA、tolerance、Jacobian、`nfev` / `njev` / `nlu` の仕様 |
| [Hairer and Wanner, *Solving Ordinary Differential Equations II*](https://link.springer.com/book/10.1007/978-3-642-05221-7) | stiff ODE、implicit method、DAE の標準的背景 |
| [Shampine and Reichelt, 1997, *The MATLAB ODE Suite*](https://doi.org/10.1137/S1064827594276424) | BDF / NDF 系の実装と practical ODE solver design |
| [Curtis, Powell and Reid, 1974](https://doi.org/10.1093/imamat/13.1.117) | sparse Jacobian 推定と sparsity exploitation |
| [Hindmarsh, ODEPACK](https://computing.llnl.gov/sites/default/files/ODEPACK_pub2_u113855.pdf) | stiff / non-stiff ODE solver collection と LSODA 系の背景 |
