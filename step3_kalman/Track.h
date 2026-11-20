#pragma once
// ============================================================================
// Track.h（STEP 3）
//   STEP 2 の Track の位置・速度（α-β フィルタ）を、
//   カルマンフィルタの状態ベクトル x と誤差共分散行列 P に置き換えたもの。
//   予測・更新の計算は KalmanFilter に任せ、Track は状態と対応履歴を保持する。
// ============================================================================

#include "Detection.h"
#include "KalmanFilter.h"

#include <Eigen/Dense>

#include <deque>

namespace gnn {

enum class TrackState {
    Tentative,  // 仮トラック
    Confirmed,  // 確定トラック
    Coasting,   // 未検出による予測追跡
};

const char* toString(TrackState state);

class Track {
public:
    Track(int id, const Detection& detection, const KalmanFilter& kf, double initialVelocityStd,
          int historyLength);

    void predict(const KalmanFilter& kf, double dt);
    void update(const KalmanFilter& kf, const Detection& detection);
    void markMissed();

    int id() const { return id_; }
    TrackState state() const { return state_; }
    void setState(TrackState state) { state_ = state; }

    const KalmanFilter::StateVector& stateVector() const { return x_; }       // x = [px, py, vx, vy]ᵀ
    const KalmanFilter::StateMatrix& covariance() const { return P_; }   // P
    Eigen::Vector2d position() const { return x_.head<2>(); }
    Eigen::Vector2d velocity() const { return x_.tail<2>(); }

    int consecutiveMisses() const { return consecutiveMisses_; }
    int hitsInWindow() const;
    int missesInWindow() const;

private:
    void pushHistory(bool hit);

    int id_;
    TrackState state_ = TrackState::Tentative;

    KalmanFilter::StateVector x_;  // 状態ベクトル（4×1）
    KalmanFilter::StateMatrix P_;  // 誤差共分散行列（4×4）

    int consecutiveMisses_ = 0;
    std::deque<bool> history_;
    int historyLength_;
};

}  // namespace gnn
