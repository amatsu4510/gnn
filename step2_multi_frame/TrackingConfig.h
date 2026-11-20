#pragma once
// ============================================================================
// TrackingConfig.h（STEP 2）
//   STEP 1 のパラメータに、位置・速度推定とトラック管理のパラメータを追加。
// ============================================================================

namespace gnn {

struct TrackingConfig {
    // ---- 対応付け（STEP 1 と同じ）----
    double gateDistance = 30.0;   // ゲート半径 [px]（予測位置からの距離）
    double missCost = 15.0;       // 未検出コスト [px]
    double falseAlarmCost = 15.0; // 未割り当て観測コスト [px]

    // ---- 位置・速度推定（α-β フィルタ）----【STEP 2 で追加】
    //   予測残差 r = z − p_pred に対して
    //     p ← p_pred + α·r,   v ← v + (β / Δt)·r
    //   α, β が大きいほど観測に素早く追従し、小さいほど平滑化が強くなる。
    double alpha = 0.7;
    double beta = 0.3;

    // ---- トラック管理 ----【STEP 2 で追加】
    // 確定条件：直近 confirmWindow フレーム中 confirmHits 回以上対応したら Tentative → Confirmed
    //   （仮トラックのうちに confirmWindow − confirmHits 回を超えて未検出になったら削除）
    int confirmHits = 3;
    int confirmWindow = 5;
    // 削除条件：Confirmed / Coasting のトラックが maxCoastingFrames 回連続で未検出なら削除
    int maxCoastingFrames = 5;
};

}  // namespace gnn
