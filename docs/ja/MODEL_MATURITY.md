# モデル成熟度

このページでは、各 backend を process interpretation にどの程度使えるかを整理します。重要なのは、「コードとして呼び出せる」ことと「物理的な結論に使える」ことを分けることです。

## 基本方針

global model は、chemistry、EEDF/rate closure、電力結合、壁損失、表面反応を組み合わせて動きます。どれか 1 つの backend が未検証であれば、case 全体の予測精度もその影響を受けます。

このため、backend ごとに成熟度を付けています。成熟度は実装品質だけでなく、想定 regime、参照データ、benchmark の有無を含む判断です。

## ラベル

| Label | 意味 | 使い方 |
|---|---|---|
| `stable` | 周辺 case が検証済みなら標準設定として使える | 数値解法など、基盤として使うもの |
| `stable-debug` | smoke test や reduced study に使える | 物理主張よりも実行経路確認に向く |
| `experimental` | 実装済みだが regime limit と検証範囲の確認が必要 | benchmark や感度評価と合わせて使う |
| `placeholder` | interface shell または仮の近似 | process conclusion には使わない |

`experimental` は「使えない」という意味ではありません。入力、仮定、検証範囲を明記すれば、研究用の比較や設計探索には使えます。ただし、絶対値予測や外部説明に使う場合は追加 validation が必要です。

## 現在の backend status

| Category | Backend | Status | Notes |
|---|---|---|---|
| EEDF | `maxwell` | `stable-debug` | smoke test と mechanism debug 用の高速 closure |
| EEDF | `swarm` | `experimental` | selected swarm model と入力 data に成熟度が依存 |
| EEDF | `boltzmann_2term` | `experimental` | reduced two-term-like closure。BOLSIG+ replacement ではない |
| EEDF | `rate_table` | `experimental` | table source、grid 範囲、metadata が精度を支配 |
| Electrical | `direct_power` | `stable-debug` | 回路・電磁場を解かず、吸収電力を直接入れる |
| Electrical | `dc_series_circuit` | `experimental` | voltage source + ballast resistor + conductive plasma load の簡約 DC model |
| Electrical | `external_circuit_table` | `experimental` | 測定または外部回路計算の waveform を一方向入力 |
| Electrical | `rf_envelope` | `experimental` | RF 周期平均の power / voltage / bias envelope |
| Electrical | `ccp` | `experimental` | lumped CCP と sheath proxy。calibration が必要 |
| Electrical | `icp` | `experimental` | reduced ICP coupling proxy。装置依存の較正が必要 |
| Electron density | `quasi_neutral` | `stable-debug` | charged species から代数的に電子密度を再構成 |
| Electron density | `prescribed_profile` | `experimental` | benchmark または one-way coupling 用の外部 `ne(t)` driver |
| Integrator | `scipy_bdf` | `stable` | 標準 stiff ODE backend |

backend registry から同じ情報を確認できます。

```powershell
py -m plasma_global.cli list-backends
```

## 使い分けの目安

| 目的 | 推奨される扱い |
|---|---|
| 入力形式や実行経路の確認 | `stable-debug` backend でよい |
| reaction set の相対比較 | rate source と wall loss を固定し、差分だけを見る |
| 外部 benchmark との比較 | benchmark が揃えている軸を確認し、未検証 backend を結論に使わない |
| process condition の絶対値予測 | chemistry、rate table、電力結合、surface model の calibration が必要 |
| 新しい backend の導入 | smoke test、Jacobian check、少なくとも 1 つの reference comparison を追加する |

## 昇格条件

成熟度を上げる前に、少なくとも次を確認します。

| Check | 内容 |
|---|---|
| Interface | request / result interface が文書化されている |
| Assumptions | 仮定、適用 regime、failure regime が書かれている |
| Test | backend を通す smoke / unit test がある |
| Consistency | conservation、unit、positivity、range check がある |
| Reference | analytic limit、external solver、reference dataset のいずれかと比較済み |

成熟度ラベルは固定ではありません。reference data が増え、外部比較が揃えば上げられます。逆に、適用範囲を超えた使い方をする場合は、`stable` な backend でも case 全体としては未検証になります。
