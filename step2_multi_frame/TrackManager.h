#pragma once
// ============================================================================
// TrackManager.h（STEP 2）
//   トラックの生成・予測・更新・確定・削除を行い、1フレームの処理を統括する。
//
//   1フレームの処理順序（process）
//     1. 全トラックを予測（等速直線運動）
//     2. GNN で予測位置と最明点を対応付け
//     3. 対応ありのトラックを観測で更新
//     4. 対応なしのトラックを未検出として記録（予測位置で維持）
//     5. 状態遷移（確定・予測追跡）と削除
//     6. 未割り当ての最明点から仮トラックを生成
//
//   画面出力は行わず、処理内容を FrameReport として返す（表示は ReportPrinter が担当）。
// ============================================================================

#include "AssociationResult.h"
#include "Detection.h"
#include "GNNAssociator.h"
#include "Track.h"
#include "TrackingConfig.h"

#include <Eigen/Dense>

#include <vector>

namespace gnn {

// 表示・評価用のトラックの写し
struct TrackSnapshot {
    int id = 0;
    TrackState state = TrackState::Tentative;
    Eigen::Vector2d position = Eigen::Vector2d::Zero();
    Eigen::Vector2d velocity = Eigen::Vector2d::Zero();
    int consecutiveMisses = 0;
    int hitsInWindow = 0;
};

// 1フレーム分の処理結果
struct FrameReport {
    Frame frame;                          // 入力
    double dt = 0.0;                      // 前フレームからの経過時間 [s]（最初のフレームは 0）
    std::vector<TrackSnapshot> predicted; // 予測後のトラック（association の行と同じ順）
    AssociationResult association;        // GNN の結果
    std::vector<TrackSnapshot> updated;   // 更新・生成・削除後のトラック
    std::vector<int> createdIds;          // 新規生成されたトラックID
    std::vector<int> deletedIds;          // 削除されたトラックID

    // 今回の最明点 j に対応付けられたトラックID（なければ -1）。評価用
    int trackIdOfDetection(int detectionIndex) const;
};

class TrackManager {
public:
    explicit TrackManager(const TrackingConfig& config);

    // 1フレームを処理する。時刻が前フレーム以下の場合は std::invalid_argument
    FrameReport process(const Frame& frame);

    const std::vector<Track>& tracks() const { return tracks_; }

private:
    void predictTracks(double dt);
    void updateMatchedTracks(const AssociationResult& result, const Frame& frame);
    void markUnmatchedTracks(const AssociationResult& result);
    void updateTrackStates();
    std::vector<int> deleteLostTracks();
    std::vector<int> initiateTracks(const AssociationResult& result, const Frame& frame);

    std::vector<TrackSnapshot> snapshot() const;

    TrackingConfig config_;
    GNNAssociator associator_;
    std::vector<Track> tracks_;
    int nextTrackId_ = 1;
    bool hasPreviousFrame_ = false;
    double lastTimestamp_ = 0.0;
};

}  // namespace gnn
