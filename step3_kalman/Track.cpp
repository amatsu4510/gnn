// ============================================================================
// Track.cpp（STEP 3）
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

Track::Track(int id, const Detection& detection, const KalmanFilter& kf, double initialVelocityStd,
             int historyLength)
    : id_(id), historyLength_(historyLength) {
    // STEP 2 では2回目の観測で速度を差分から初期化したが、
    // カルマンフィルタでは「速度の分散が大きい」初期状態にしておけば、
    // 2回目の更新でカルマンゲインが大きくなり、自然に速度が推定される。
    kf.initialize(detection.position, initialVelocityStd, x_, P_);
    pushHistory(true);
}

void Track::predict(const KalmanFilter& kf, double dt) {
    kf.predict(x_, P_, dt);
}

void Track::update(const KalmanFilter& kf, const Detection& detection) {
    kf.update(x_, P_, detection.position);
    // 対応回数の管理はカルマンフィルタではなくトラック側の責務
    consecutiveMisses_ = 0;
    pushHistory(true);
}

void Track::markMissed() {
    // 観測更新を行わないので、x, P は予測値のまま（P は予測のたびに大きくなる）
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
