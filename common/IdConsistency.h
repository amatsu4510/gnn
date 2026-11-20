#pragma once
// ============================================================================
// IdConsistency.h
//   疑似データの真値IDと、追跡結果のトラックIDの対応を記録し、
//   「目標ごとにトラックIDが維持されたか」を評価する（デモとテストの評価専用）。
//
//   - 最初と最後に対応したトラックIDが同じ → ID維持
//   - 途中で別のトラックIDに対応したフレームがある → 一時的な誤対応
//     （交差の瞬間に1フレームだけ観測が入れ替わる、など）
// ============================================================================

#include "ScenarioGenerator.h"

#include <map>
#include <set>

namespace gnn {

class IdConsistency {
public:
    struct Summary {
        int firstTrackId = -1;      // 最初に対応したトラックID
        int lastTrackId = -1;       // 最後に対応したトラックID
        std::set<int> allTrackIds;  // 対応したことのあるトラックID
        int matchedFrames = 0;      // トラックに対応したフレーム数
        int otherTrackFrames = 0;   // 最初とは別のトラックに対応したフレーム数
        bool idKept() const { return firstTrackId == lastTrackId; }
    };

    // Report は trackIdOfDetection(int) を持つ型（STEP 2 / 3 の FrameReport）
    template <class Report>
    void record(const ScenarioFrame& sf, const Report& report) {
        for (int j = 0; j < static_cast<int>(sf.truthIds.size()); ++j) {
            const int truth = sf.truthIds[j];
            const int trackId = report.trackIdOfDetection(j);
            if (truth < 0 || trackId < 0) {
                continue;  // 誤検出、または未割り当て（新規生成された観測）
            }
            Summary& s = summaries_[truth];
            if (s.firstTrackId < 0) {
                s.firstTrackId = trackId;
            }
            s.lastTrackId = trackId;
            s.allTrackIds.insert(trackId);
            ++s.matchedFrames;
            if (trackId != s.firstTrackId) {
                ++s.otherTrackFrames;
            }
        }
    }

    const std::map<int, Summary>& summaries() const { return summaries_; }
    const Summary& of(int truthId) const { return summaries_.at(truthId); }

private:
    std::map<int, Summary> summaries_;  // 真値ID → 評価結果
};

}  // namespace gnn
