# 03. STEP 2：複数フレームの継続的な目標追跡

## 1. STEP 1 から追加したもの

| 追加 | 実装 |
|---|---|
| トラックID・トラック情報の保持 | `Track`（`id_`, `position_`, `velocity_`, `state_`, `history_`） |
| 位置・速度の推定 | `Track::update()`（α-β フィルタ） |
| 等速直線運動による予測 | `Track::predict()` |
| 予測位置と観測の対応付け | `TrackManager::process()` が予測位置を `GNNAssociator` に渡す |
| 仮トラックの生成 | `TrackManager::initiateTracks()` |
| 未検出トラックの予測追跡 | `Track::markMissed()`（予測位置のまま維持） |
| 確定・削除 | `TrackManager::updateTrackStates()`, `deleteLostTracks()` |
| フレームごとの処理内容の記録 | `FrameReport`（表示は `ReportPrinter`） |

`GNNAssociator` は STEP 1 と**まったく同じコード**である。基準点が「前フレームの点」から「トラックの予測位置」に
変わっただけで、対応付けの処理は変わらない（GNN は対応付けだけを担当し、予測は別の責務であることが分かる）。

## 2. 数式とコードの対応

### 2.1 等速直線運動モデルによる予測

$$
\hat{\mathbf{p}}_{k|k-1} = \mathbf{p}_{k-1} + \mathbf{v}_{k-1} \Delta t
$$

```cpp
// step2_multi_frame/Track.cpp
void Track::predict(double dt) {
    position_ += velocity_ * dt;   // Vector2d × スカラー
}
```

Δt は**フレームの時刻の差**から計算する（`TrackManager::process()`）。フレーム番号の差ではないので、
フレーム間隔が一定でなくても速度 [px/s] の意味は変わらない。

### 2.2 α-β フィルタによる位置・速度の推定

予測残差 $\mathbf{r} = \mathbf{z} - \hat{\mathbf{p}}$ を使って

$$
\mathbf{p} \leftarrow \hat{\mathbf{p}} + \alpha \mathbf{r}, \qquad
\mathbf{v} \leftarrow \mathbf{v} + \frac{\beta}{\Delta t_{\text{upd}}} \mathbf{r}
$$

```cpp
const Eigen::Vector2d residual = z - position_;
position_ += config.alpha * residual;
velocity_ += (config.beta / elapsed) * residual;
```

- $\Delta t_{\text{upd}}$（`elapsed`）は「前回**更新**してからの時間」。未検出が続いた後に再検出した場合、
  残差は複数フレーム分の誤差なので、1フレーム分の Δt で割ると速度を過大に修正してしまう。
- 生成直後のトラックは速度が分からない（0 としている）。2回目の観測で α-β の式を使うと速度が大きく遅れるため、
  2回目だけは差分 $(\mathbf{z}_2 - \mathbf{z}_1) / \Delta t$ で速度を初期化している（`velocityInitialized_`）。
- α-β フィルタは「ゲインが固定のカルマンフィルタ」と見なせる。STEP 3 ではゲインが不確かさに応じて自動で変わる。

### 2.3 トラック管理（M/N 確定・連続未検出による削除）

| 状態 | 対応あり | 対応なし |
|---|---|---|
| Tentative | 直近 N=5 回中 M=3 回対応したら Confirmed | 履歴中の未検出が N−M=2 回を超えたら削除 |
| Confirmed | Confirmed のまま | Coasting へ |
| Coasting | Confirmed へ戻る | 連続未検出が 5 回になったら削除 |

- 資料 p.15 の `Track` は対応回数 `hits` だけを持つが、「直近5回中3回」を判定するには**履歴**が必要なので
  `std::deque<bool> history_` で直近 N 回の対応有無を保持している。
- 削除は `std::remove_if` + `erase`（erase-remove イディオム）で行う。条件を満たす要素を末尾に寄せて一度に消すので、
  ループ中に `erase` するより安全で効率がよい。

## 3. デモの内容

`step2 [overall|crossing|approach|uneven]` でシナリオを選べる（既定は overall）。

| シナリオ | 内容 | 結果 |
|---|---|---|
| overall | 3目標 + フレーム5で新規出現、目標2がフレーム6,7で未検出、目標3はフレーム7の後に消失、誤検出 0〜2個/フレーム、順序シャッフル | 4目標ともIDを維持。目標2のトラックは Coasting → Confirmed、目標3のトラックはフレーム12で削除 |
| crossing | 2目標がX字に交差（±60 px/s） | IDを維持 |
| approach | 2目標が浅い角度（約3.4°）で接近・交差 | **IDが入れ替わる**（GNN の限界。[06_parameters_and_limits.md](06_parameters_and_limits.md)） |
| uneven | フレーム間隔が 0.02〜0.2 s で変化 | IDを維持、速度も正しく推定 |

## 4. STEP 2 の限界

- ゲートが固定（30 px）。Coasting が続くと予測誤差が大きくなるのに、ゲートは広がらない。
  逆に推定が安定しているときも 30 px のままなので、近くの誤検出を拾いやすい。
  実際に、overall シナリオを当初シード 42 で作ったところ、消失した目標3の Coasting トラックが近くの誤検出（距離 26 px）を拾い、
  削除されずに生き延びた。試した13個のシードのうち4個で、このような誤検出の捕捉によってテストの期待値と異なる結果になった
  （テストではこれが起きないシード 12 を使用している）。
- α, β が固定なので、「生成直後は観測を信じ、収束したら平滑化する」といった調整ができない。
- 距離が x, y 方向で同じ重みになる（速度方向に誤差が大きい、といった性質を扱えない）。

→ STEP 3 でカルマンフィルタに置き換え、不確かさ（共分散）に応じたゲイン・ゲートにする。
