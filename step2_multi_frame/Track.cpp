// ============================================================================
// Track.cpp（STEP 2）
// ============================================================================

#include "Track.h"

#include <algorithm>

namespace gnn {

const char* toString(TrackState state) {
    switch (state) {
        case TrackState::Tentative: return "Tentative";
        case TrackState::Confirmed: return "Confirmed";
        case TrackState::Coasting:  return "Coasting";
    }
    return "Unknown";
}

Track::Track(int id, const Detection& detection, int historyLength)
    : id_(id),
      position_(detection.position),
      velocity_(Eigen::Vector2d::Zero()),
      lastUpdateTime_(detection.timestamp),
      historyLength_(historyLength) {
    pushHistory(true);  // 生成時の観測を1回目の対応として数える
}

void Track::predict(double dt) {
    // 等速直線運動モデル
    //   x_pred = x + vx·Δt,  y_pred = y + vy·Δt
    position_ += velocity_ * dt;
}

void Track::update(const Detection& detection, const TrackingConfig& config) {
    const double elapsed = detection.timestamp - lastUpdateTime_;  // 前回の更新からの経過時間
    const Eigen::Vector2d& z = detection.position;

    if (!velocityInitialized_) {
        // 2回目の観測：前回の観測位置との差分から速度を初期化する
        //   （生成時は速度が未知のため、α-β の式を使うと速度が大きく遅れる）
        //   ※ 生成後に速度0で予測しているため、position_ は生成時の観測位置のまま
        velocity_ = (z - position_) / elapsed;
        position_ = z;
        velocityInitialized_ = true;
    } else {
        // α-β フィルタ
        //   r = z − p_pred（予測残差）
        //   p ← p_pred + α·r
        //   v ← v + (β / Δt)·r     Δt は前回の更新からの経過時間（未検出が続いた場合も正しく扱う）
        const Eigen::Vector2d residual = z - position_;
        position_ += config.alpha * residual;
        velocity_ += (config.beta / elapsed) * residual;
    }

    lastUpdateTime_ = detection.timestamp;
    consecutiveMisses_ = 0;
    ++totalHits_;
    pushHistory(true);
}

void Track::markMissed() {
    // 位置は predict() で予測済みの値をそのまま使う（予測追跡）
    ++consecutiveMisses_;
    pushHistory(false);
}

int Track::hitsInWindow() const {
    return static_cast<int>(std::count(history_.begin(), history_.end(), true));
}

int Track::missesInWindow() const {
    return static_cast<int>(history_.size()) - hitsInWindow();
}

void Track::pushHistory(bool hit) {
    history_.push_back(hit);
    while (static_cast<int>(history_.size()) > historyLength_) {
        history_.pop_front();
    }
}

}  // namespace gnn
