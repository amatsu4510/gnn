# 04. STEP 3：カルマンフィルタを用いたGNN

## 1. STEP 2 からの変更点

| 変更 | STEP 2 | STEP 3 |
|---|---|---|
| 状態 | 位置・速度（`Vector2d` × 2） | 状態ベクトル x（`Vector4d`）+ 誤差共分散 P（`Matrix4d`） |
| 推定 | α-β フィルタ（固定ゲイン） | カルマンフィルタ（`KalmanFilter`、ゲインは P から計算） |
| 速度の初期化 | 2回目の観測の差分 | 初期 P の速度分散を大きくするだけ（特別扱い不要） |
| コスト | ユークリッド距離 [px] | マハラノビス距離の二乗 d² [無次元] |
| ゲート | 固定 30 px | d² ≤ χ²₂(P_G)。S が大きいトラックほど実質的に広い |
| `GNNAssociator` の入力 | 予測位置 | `PredictedMeasurement`（ẑ と S） |

`TrackManager` の処理順序・状態遷移・削除条件は STEP 2 と同じ。

## 2. Eigen の行列・ベクトルの意味と次元

| 記号 | 意味 | 次元 | Eigen の型（エイリアス） | 単位 |
|---|---|---|---|---|
| x | 状態ベクトル [px, py, vx, vy]ᵀ | 4×1 | `Vector4d`（`StateVector`） | px, px/s |
| P | 状態の誤差共分散 | 4×4 | `Matrix4d`（`StateMatrix`） | px², px²/s, px²/s² |
| F | 状態遷移行列 | 4×4 | `Matrix4d` | – |
| Q | プロセスノイズ共分散 | 4×4 | `Matrix4d` | P と同じ |
| z | 観測ベクトル [px, py]ᵀ | 2×1 | `Vector2d`（`MeasurementVector`） | px |
| H | 観測行列 | 2×4 | `Matrix<double,2,4>`（`MeasurementMatrix`） | – |
| R | 観測ノイズ共分散 | 2×2 | `Matrix2d`（`MeasurementCovariance`） | px² |
| y | イノベーション（残差）z − Hx⁻ | 2×1 | `Vector2d` | px |
| S | 残差の共分散 HP⁻Hᵀ + R | 2×2 | `Matrix2d` | px² |
| K | カルマンゲイン | 4×2 | `Matrix<double,4,2>`（`GainMatrix`） | –, 1/s |
| C | コスト行列（d²） | N×M | `MatrixXd` | 無次元 |

次元が決まっている行列は**固定サイズ型**（`Matrix4d` など）を使っている。サイズがコンパイル時に分かるので
ヒープ確保がなく、ループ展開などの最適化も効く。トラック数で大きさが変わるコスト行列だけ `MatrixXd`（動的サイズ）にした。

## 3. 数式とコードの対応

### 3.1 状態遷移行列 F とプロセスノイズ Q

$$
F = \begin{bmatrix} 1 & 0 & \Delta t & 0 \\ 0 & 1 & 0 & \Delta t \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}, \qquad
Q = q \begin{bmatrix}
\Delta t^3/3 & 0 & \Delta t^2/2 & 0 \\
0 & \Delta t^3/3 & 0 & \Delta t^2/2 \\
\Delta t^2/2 & 0 & \Delta t & 0 \\
0 & \Delta t^2/2 & 0 & \Delta t
\end{bmatrix}
$$

```cpp
// KalmanFilter::transitionMatrix
StateMatrix F = StateMatrix::Identity();
F(0, 2) = dt;
F(1, 3) = dt;

// KalmanFilter::processNoise
Q(0, 0) = Q(1, 1) = q_ * dt3 / 3.0;
Q(0, 2) = Q(2, 0) = q_ * dt2 / 2.0;
Q(1, 3) = Q(3, 1) = q_ * dt2 / 2.0;
Q(2, 2) = Q(3, 3) = q_ * dt;
```

Q は「加速度が白色雑音（スペクトル密度 q）」という連続時間モデルを Δt で離散化したもの。
**Q は Δt から毎回計算する**。資料 p.16 の `predict(Track&, double dt, const M4& Q)` は Q を固定値として外から受け取る形で、
フレーム間隔が変わると予測の不確かさが正しく増えない（テスト「KF_行列F_H_Qの構造」で Δt を2倍にすると位置分散が8倍になることを確認）。

### 3.2 予測

$$
\mathbf{x}^- = F \mathbf{x}, \qquad P^- = F P F^\top + Q
$$

```cpp
x = F * x;
P = F * P * F.transpose() + processNoise(dt);
```

Eigen では行列積は一時オブジェクトに評価されてから代入されるので、`x = F * x;` のように左辺と右辺に同じ変数があっても
正しく計算される（エイリアシングが起きない）。資料の `(F * t.x).eval()` の `.eval()` は不要だが害もない。
一方、`P = P.transpose()` のような**要素ごとの演算**では左辺が計算途中で書き換わるため `.eval()` が必要になる
（`KalmanFilter::update()` の対称化 `P = 0.5 * (P + P.transpose()).eval();` を参照）。

### 3.3 残差の共分散とマハラノビス距離

$$
\mathbf{y} = \mathbf{z} - H\mathbf{x}^-, \qquad S = H P^- H^\top + R, \qquad d^2 = \mathbf{y}^\top S^{-1} \mathbf{y}
$$

```cpp
// GNNAssociator::computeMahalanobisMatrix
const Eigen::LLT<Eigen::Matrix2d> llt(predictions[i].S);   // S = L Lᵀ（トラックごとに1回）
...
const Eigen::Vector2d y = detections[j].position - predictions[i].z;
d2(i, j) = y.dot(llt.solve(y));                             // yᵀ (S⁻¹ y)
```

- `S.inverse()` は使わず、コレスキー分解 `LLT`（S = LLᵀ）で連立方程式 S w = y を解いて w = S⁻¹y を得る。
  逆行列を作るより計算量が少なく、数値誤差も小さい。S は対称正定値なので LLT が使える。
  正定値でない（壊れた共分散）場合は `llt.info() != Eigen::Success` になるので例外を投げる。
- 分解は**トラックごとに1回**行い、すべての最明点に使い回している（N 回の分解 + N×M 回の前進・後退代入）。
- `H x⁻` は位置成分そのものなので、`predictedMeasurement()` では `x.head<2>()`（先頭2要素）で取り出している。

> 資料 p.7 では $d^2 = \mathbf{r} S^{-1} \mathbf{r}^\top$ と書かれているが、r（= 観測 − 予測）が列ベクトル（2×1）なら
> $\mathbf{r} S^{-1}$ は 2×1 と 2×2 の積となり定義できない。正しくは $\mathbf{r}^\top S^{-1} \mathbf{r}$（r を行ベクトルとする流儀なら資料の式でよいが、
> 同じ資料の p.16 以降のコードは列ベクトル `Vector2d` を使っている）。

### 3.4 カイ二乗分布に基づくゲーティング

観測モデルが正しければ、正しい観測の d² は**自由度2のカイ二乗分布**に従う。自由度2のカイ二乗分布は平均2の指数分布なので

$$
P(d^2 \le \gamma) = 1 - e^{-\gamma/2} \quad\Longrightarrow\quad \gamma = -2 \ln(1 - P_G)
$$

と閉じた式で閾値が求まる（自由度2の場合だけの性質）。

```cpp
// TrackingConfig::gateThreshold
double gateThreshold() const { return -2.0 * std::log(1.0 - gateProbability); }
```

| P_G | 0.95 | 0.99 | 0.999 | 0.9999 |
|---|---|---|---|---|
| γ | 5.99 | 9.21 | 13.82 | 18.42 |

本サンプルでは **P_G = 0.999（γ = 13.82）** を既定にした。理由は [06_parameters_and_limits.md](06_parameters_and_limits.md) の「ゲート確率」を参照。

未検出コスト・未割り当て観測コストもユークリッド距離ではなく **d² の単位**で指定する（既定は和 = γ）。
ゲートの単位（d²）とコストの単位を混ぜない点に注意する。

### 3.5 カルマンゲインと状態更新

$$
K = P^- H^\top S^{-1}, \qquad \mathbf{x} = \mathbf{x}^- + K\mathbf{y}
$$

```cpp
// KalmanFilter::update
const Eigen::LLT<MeasurementCovariance> llt(S);
const GainMatrix K = llt.solve(H * P).transpose();
const MeasurementVector y = z - H * x;
x = x + K * y;
```

$K S = P H^\top$ の両辺を転置すると $S K^\top = H P$（P, S は対称）となるので、
`llt.solve(H * P)` で $K^\top$（2×4）を求めてから転置している。ここでも逆行列は使っていない。

### 3.6 Joseph 形式による共分散の更新

$$
P = (I - KH) P^- (I - KH)^\top + K R K^\top
$$

```cpp
const StateMatrix A = StateMatrix::Identity() - K * H;
P = A * P * A.transpose() + K * R_ * K.transpose();
P = 0.5 * (P + P.transpose()).eval();   // 丸め誤差による非対称性を除去
```

よく使われる $P = (I - KH)P^-$ は、K が厳密に最適な場合にだけ成り立つ式で、丸め誤差で P が非対称になったり
負の固有値を持ったりすることがある。Joseph 形式は「正定値行列の二次形式 + 半正定値行列」の形なので、
数値的に対称・半正定値が保たれやすい。テスト「KF_カルマンゲインが定義式と一致し_Joseph形式で対称正定値を保つ」で、
200回の予測・更新後も P が対称・正定値であることを確認している。

### 3.7 新規トラックの初期化

$$
\mathbf{x}_0 = [z_x, z_y, 0, 0]^\top, \qquad
P_0 = \mathrm{diag}(\sigma_r^2, \sigma_r^2, \sigma_v^2, \sigma_v^2)
$$

```cpp
x << z.x(), z.y(), 0.0, 0.0;                     // カンマ初期化
P.topLeftCorner<2, 2>() = R_;                    // 左上 2×2 ブロック
P.bottomRightCorner<2, 2>() = Eigen::Matrix2d::Identity() * initialVelocityStd * initialVelocityStd;
```

速度の分散を大きく（σ_v = 100 px/s）しておくと、2回目の観測時にカルマンゲインが大きくなり、
STEP 2 のような「2回目だけ差分で初期化」という特別処理なしで速度が推定される。
また、生成直後のトラックは S が大きい（σ ≈ 10 px）ため、ゲートも自動的に広くなる。

## 4. 使用した Eigen の機能と使用意図

| API | 使用箇所 | 意図 |
|---|---|---|
| `Vector2d`, `Vector4d`, `Matrix2d`, `Matrix4d`, `Matrix<double,2,4>` | 状態・共分散・観測 | 固定サイズでヒープ確保なし・高速 |
| `MatrixXd` | コスト行列・拡張行列 | トラック数・観測数で大きさが変わる |
| `Matrix<bool, Dynamic, Dynamic>` | `GateMask` | ゲート判定結果を行列で保持 |
| `::Identity()`, `::Zero()`, `::Constant()` | F, H, P の初期化、拡張行列 | 意図が明確な初期化 |
| `.array() <= x`, `.matrix()` | ゲーティング | 要素ごとの比較 |
| `.norm()` | ユークリッド距離 | ベクトルの長さ |
| `.dot()` | d² = yᵀ(S⁻¹y) | 内積 |
| `Eigen::LLT` / `.solve()` / `.info()` | d², カルマンゲイン | 逆行列を作らずに連立方程式を解く・正定値性の確認 |
| `.transpose()` | Fᵀ, Kᵀ | 転置（式テンプレートでコピーなし） |
| `.head<2>()`, `.tail<2>()` | 位置・速度の取り出し | 部分ベクトル |
| `.topLeftCorner<2,2>()`, `.bottomRightCorner<2,2>()`, `.block()` | P の初期化、拡張行列の右下 | 部分行列への代入 |
| `.diagonal()`, `.cwiseSqrt()` | 位置σ・残差σの表示 | 対角成分の取り出し・要素ごとの平方根 |
| `.unaryExpr()` | ハンガリアン法の禁止要素の置き換え | 要素ごとに関数を適用 |
| `.eval()` | P の対称化 | エイリアシングの回避 |
| `<<`（カンマ初期化） | x の初期化、テストの行列 | 読みやすい初期化 |
| `SelfAdjointEigenSolver` | テストのみ | P の固有値（正定値性）の確認 |

注意：Eigen の式（`a + b` など）は**遅延評価される式テンプレート**なので、`auto r = a + b;` と書くと
`r` は計算結果ではなく「式」になり、`a` や `b` が変わると結果も変わったり、一時オブジェクトへのダングリング参照になったりする。
本サンプルでは Eigen の式を `auto` で受けず、`Eigen::Vector2d` のように型を明示している。

## 5. デモの出力例（overall シナリオ、フレーム6）

```
[予測位置] カルマンフィルタの予測 x⁻ = F x
  T1   位置 (  137.6,   111.3)  速度 (   63.5,    16.6) px/s  位置σ (   1.26,    1.26)  Confirmed  ...
  T5   位置 (  274.9,   281.9)  速度 (  -42.8,    55.9) px/s  位置σ (   1.26,    1.26)  Confirmed  ...
  T15  位置 (  151.2,   399.9)  速度 (    0.0,     0.0) px/s  位置σ (  10.05,   10.05)  Tentative  ...
[残差の標準偏差 √diag(S)]（ゲートの大きさの目安。ゲート閾値 d² ≤ 13.816）
  T1 : (   1.61,    1.61) px
  T15 : (  10.10,   10.10) px
...
[GNN対応付け]
  T1 -> Z3  コスト 2.492
  T3 -> Z2  コスト 1.412
  T15 -> Z0  コスト 0.444
[未割り当てトラック] T5 T10 T11 T12 T13 T14
```

確定トラック T1 の残差σは 1.6 px なのでゲートは半径 約 1.6×√13.8 ≈ 6 px、
生成直後の T15 は 10 px なので半径 約 38 px となり、不確かさに応じてゲートの大きさが変わっていることが分かる。
T5（目標2）はこのフレームで未検出なので未割り当てとなり、Coasting に移る。
