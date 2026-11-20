// ============================================================================
// TrackManager.cpp（STEP 3）
// ============================================================================

#include "TrackManager.h"

#include <algorithm>
#include <stdexcept>

namespace gnn {

int FrameReport::trackIdOfDetection(int detectionIndex) const {
    for (const Match& m : association.matches) {
        if (m.detectionIndex == detectionIndex) {
            return predicted[m.trackIndex].id;
        }
    }
    return -1;
}

TrackManager::TrackManager(const TrackingConfig& config)
    : config_(config),
      kf_(config.processNoiseDensity, config.measurementNoiseStd),
      associator_(config) {}

FrameReport TrackManager::process(const Frame& frame) {
    FrameReport report;
    report.frame = frame;

    // ---- 経過時間 Δt（フレーム間隔は一定でなくてよい）----
    if (hasPreviousFrame_) {
        report.dt = frame.timestamp - lastTimestamp_;
        if (report.dt <= 0.0) {
            throw std::invalid_argument("TrackManager: フレームの時刻が前フレーム以下です");
        }
    }
    hasPreviousFrame_ = true;
    lastTimestamp_ = frame.timestamp;

    // 1. 予測
    predictTracks(report.dt);
    report.predicted = snapshot();

    // 2. GNN による対応付け（予測観測 ẑ = H x⁻ と残差共分散 S = H P⁻ Hᵀ + R を渡す）
    std::vector<PredictedMeasurement> predictions;
    for (const Track& t : tracks_) {
        predictions.push_back({KalmanFilter::predictedMeasurement(t.stateVector()),
                               kf_.innovationCovariance(t.covariance())});
    }
    report.association = associator_.associate(predictions, frame.detections);

    // 3. 更新 / 4. 未検出の記録
    updateMatchedTracks(report.association, frame);
    markUnmatchedTracks(report.association);

    // 5. 状態遷移と削除（tracks_ の添字が変わるので、対応付け結果を使う処理の後に行う）
    updateTrackStates();
    report.deletedIds = deleteLostTracks();

    // 6. 未割り当ての最明点から仮トラックを生成
    report.createdIds = initiateTracks(report.association, frame);

    report.updated = snapshot();
    return report;
}

void TrackManager::predictTracks(double dt) {
    if (dt <= 0.0) {
        return;
    }
    for (Track& t : tracks_) {
        t.predict(kf_, dt);
    }
}

void TrackManager::updateMatchedTracks(const AssociationResult& result, const Frame& frame) {
    for (const Match& m : result.matches) {
        tracks_[m.trackIndex].update(kf_, frame.detections[m.detectionIndex]);
    }
}

void TrackManager::markUnmatchedTracks(const AssociationResult& result) {
    for (int i : result.unassignedTracks) {
        tracks_[i].markMissed();
    }
}

void TrackManager::updateTrackStates() {
    for (Track& t : tracks_) {
        const bool hitNow = t.consecutiveMisses() == 0;
        switch (t.state()) {
            case TrackState::Tentative:
                // 直近 N 回中 M 回対応で確定
                if (t.hitsInWindow() >= config_.confirmHits) {
                    t.setState(TrackState::Confirmed);
                }
                break;
            case TrackState::Confirmed:
                if (!hitNow) {
                    t.setState(TrackState::Coasting);  // 未検出 → 予測追跡
                }
                break;
            case TrackState::Coasting:
                if (hitNow) {
                    t.setState(TrackState::Confirmed);  // 再検出 → 確定に戻る
                }
                break;
        }
    }
}

std::vector<int> TrackManager::deleteLostTracks() {
    std::vector<int> deleted;
    const int allowedTentativeMisses = config_.confirmWindow - config_.confirmHits;
    // remove_if で削除対象を末尾に寄せてから erase する（erase-remove イディオム）
    auto shouldDelete = [&](const Track& t) {
        bool lost = false;
        if (t.state() == TrackState::Tentative) {
            // N 回中 M 回の確定条件を満たせなくなった仮トラック
            lost = t.missesInWindow() > allowedTentativeMisses;
        } else {
            // 長期間未検出の確定トラック
            lost = t.consecutiveMisses() >= config_.maxCoastingFrames;
        }
        if (lost) {
            deleted.push_back(t.id());
        }
        return lost;
    };
    tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(), shouldDelete), tracks_.end());
    return deleted;
}

std::vector<int> TrackManager::initiateTracks(const AssociationResult& result, const Frame& frame) {
    std::vector<int> created;
    for (int j : result.unassignedDetections) {
        tracks_.emplace_back(nextTrackId_, frame.detections[j], kf_, config_.initialVelocityStd, config_.confirmWindow);
        created.push_back(nextTrackId_);
        ++nextTrackId_;
    }
    return created;
}

std::vector<TrackSnapshot> TrackManager::snapshot() const {
    std::vector<TrackSnapshot> snaps;
    for (const Track& t : tracks_) {
        const Eigen::Matrix2d S = kf_.innovationCovariance(t.covariance());
        TrackSnapshot snap;
        snap.id = t.id();
        snap.state = t.state();
        snap.position = t.position();
        snap.velocity = t.velocity();
        // diagonal() で対角成分を取り出し、cwiseSqrt() で要素ごとの平方根をとる
        snap.positionStd = t.covariance().diagonal().head<2>().cwiseSqrt();
        snap.residualStd = S.diagonal().cwiseSqrt();
        snap.consecutiveMisses = t.consecutiveMisses();
        snap.hitsInWindow = t.hitsInWindow();
        snaps.push_back(snap);
    }
    return snaps;
}

}  // namespace gnn
