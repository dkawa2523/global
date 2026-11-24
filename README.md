# Plasma Global Model

低圧プラズマの粒子・電荷・エネルギー収支から、区画ごとの平均組成と過渡応答を計算するPythonパッケージです。
反応機構、装置、初期状態、壁・流入出、電源を設定し、反応の寄与や運転条件への感度を調べます。
YAMLとPythonは同じ公開宣言 `CaseSpec` を使い、計算後はHDF5に保存した数値を時刻・領域・単位・来歴とともに読み出せます。

## はじめに

Python 3.11以上を用意し、同梱の `argon_maxwellian` を使って、通電中から停止後までの電子密度と等価電子温度を調べます。
インストール、入力の取り出し、計算、保存結果の確認は[初回実行](docs/index.md#quickstart)を参照してください。

| 目的 | 参照先 |
|---|---|
| 自分の条件でケースを作り、必要な量を保存する | [用途別の手順](docs/index.md#task-guides)、[入力仕様](docs/index.md#input)、[結果形式とPython API](docs/index.md#results) |
| 近似と比較根拠から適用範囲を判断する | [低圧プラズマの体積平均モデル](docs/THEORY.md)、[同じ原稿から生成したPDF](docs/assets/global-model-theory.pdf) |
| 実装を追う、例題や比較を再現する、文書を更新する | [開発・保守](docs/DEVELOPMENT.md) |
| 今後の検証条件と拡張の判断を確認する | [計画と残る課題](docs/ROADMAP.md) |

同梱入力の正本は [plasma_global/examples/](plasma_global/examples/) にあります。
検証はArを中心とし、解析解、別コードの保存参照、限定した公開測定との比較を扱います。
個別装置への適用では、観測量の定義、係数の適用条件、境界条件を揃えて評価します。
比較条件と根拠は[検証方法と再現条件](docs/THEORY.md#verification)を参照してください。
