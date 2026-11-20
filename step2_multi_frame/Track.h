#pragma once
// ============================================================================
// Track.h（STEP 2）
//   複数フレームにわたる目標の状態（ID・位置・速度・追跡状態・対応履歴）を保持する。
//   位置・速度は α-β フィルタ（等速直線運動モデル）で推定する。
//
//   状態遷移の「判断」は TrackManager が行い、Track は自分の状態と履歴を
//   正しく更新することだけに責任を持つ。
// ============================================================================

#include "Detection.h"
#include "TrackingConfig.h"

#include <Eigen/Dense>

#include <deque>

namespace gnn {

enum class TrackState {
    Tentative,  // 仮トラック：生成直後。誤検出の可能性がある
    Confirmed,  // 確定トラック：十分な回数対応した
    Coasting,   // 予測追跡：確定後に未検出となり、予測だけで維持している
};

const char* toString(TrackState state);

class Track {
public:
    // 未割り当ての最明点から仮トラックを生成する（速度は未知なので 0）
    Track(int id, const Detection& detection, int historyLength);

    // 等速直線運動モデルで dt 秒後の位置を予測する：p ← p + v·Δt
    void predict(double dt);

    // 対応付けされた最明点で位置・速度を更新する（α-β フィルタ）
    void update(const Detection& detection, const TrackingConfig& config);

    // 今回のフレームで対応する最明点がなかったことを記録する
    void markMissed();

    int id() const { return id_; }
    TrackState state() const { return state_; }
    void setState(TrackState state) { state_ = state; }

    const Eigen::Vector2d& position() const { return position_; }
    const Eigen::Vector2d& velocity() const { return velocity_; }

    int consecutiveMisses() const { return consecutiveMisses_; }
    int totalHits() const { return totalHits_; }
    int hitsInWindow() const;    // 直近の履歴中の対応回数
    int missesInWindow() const;  // 直近の履歴中の未検出回数

private:
    void pushHistory(bool hit);

    int id_;
    TrackState state_ = TrackState::Tentative;

    Eigen::Vector2d position_;  // 推定位置（predict 後は予測位置）[px]
    Eigen::Vector2d velocity_;  // 推定速度 [px/s]
    bool velocityInitialized_ = false;  // 2回目の観測で速度を初期化したか

    double lastUpdateTime_;      // 最後に観測で更新した時刻 [s]

    int consecutiveMisses_ = 0;  // 連続未検出回数
    int totalHits_ = 1;          // 累計の対応回数（生成時の観測を含む）
    std::deque<bool> history_;   // 直近の対応履歴（true = 対応あり）
    int historyLength_;
};

}  // namespace gnn
