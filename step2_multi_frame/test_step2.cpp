// ============================================================================
// test_step2.cpp（STEP 2：複数フレームの継続的な目標追跡 のテスト）
// ============================================================================

#include "IdConsistency.h"
#include "Printing.h"
#include "StandardScenarios.h"
#include "TestFramework.h"
#include "TrackManager.h"
#include "TrackingConfig.h"

#include <algorithm>
#include <stdexcept>
#include <vector>

using namespace gnn;

namespace {

// シナリオを最後まで処理した結果
struct RunResult {
    std::vector<FrameReport> reports;
    IdConsistency ids;  // 真値IDとトラックIDの対応
};

RunResult run(const ScenarioSpec& spec, const TrackingConfig& config = TrackingConfig()) {
    RunResult result;
    TrackManager manager(config);
    for (const ScenarioFrame& sf : generateScenario(spec)) {
        result.reports.push_back(manager.process(sf.frame));
        result.ids.record(sf, result.reports.back());
    }
    return result;
}

const TrackSnapshot* findTrack(const std::vector<TrackSnapshot>& tracks, int id) {
    for (const TrackSnapshot& t : tracks) {
        if (t.id == id) return &t;
    }
    return nullptr;
}

bool contains(const std::vector<int>& v, int x) {
    return std::find(v.begin(), v.end(), x) != v.end();
}

int countConfirmed(const std::vector<TrackSnapshot>& tracks) {
    return static_cast<int>(std::count_if(tracks.begin(), tracks.end(), [](const TrackSnapshot& t) {
        return t.state != TrackState::Tentative;
    }));
}

}  // namespace

TEST_CASE("1_2_通常移動と順序入れ替えでIDを維持") {
    const RunResult r = run(scenarios::overall());
    // 一時的な誤対応も含めて、各目標がずっと同じトラックに対応している
    for (const auto& [truth, summary] : r.ids.summaries()) {
        CHECK_EQ(static_cast<int>(summary.allTrackIds.size()), 1);
    }
    CHECK_EQ(static_cast<int>(r.ids.summaries().size()), 4);
}

TEST_CASE("位置と速度の推定") {
    ScenarioSpec spec;
    spec.timestamps = uniformTimestamps(20, 0.1);
    spec.targets = {{1, 0, 19, {100, 100}, {80, -40}, {}}};
    spec.noiseStd = 0.5;
    const RunResult r = run(spec);
    const TrackSnapshot& t = r.reports.back().updated.at(0);
    CHECK(t.state == TrackState::Confirmed);
    CHECK_NEAR(t.velocity.x(), 80.0, 10.0);
    CHECK_NEAR(t.velocity.y(), -40.0, 10.0);
    CHECK_NEAR(t.position.x(), 100.0 + 80.0 * 1.9, 3.0);
    CHECK_NEAR(t.position.y(), 100.0 - 40.0 * 1.9, 3.0);
}

TEST_CASE("4_新規目標は仮トラックから確定トラックになる") {
    const RunResult r = run(scenarios::overall());
    const int id = r.ids.of(4).firstTrackId;  // 目標4（フレーム5で出現）のトラック
    CHECK(contains(r.reports[5].createdIds, id));
    CHECK(findTrack(r.reports[5].updated, id)->state == TrackState::Tentative);
    CHECK(findTrack(r.reports[6].updated, id)->state == TrackState::Tentative);
    CHECK(findTrack(r.reports[7].updated, id)->state == TrackState::Confirmed);  // 3回目の対応で確定
}

TEST_CASE("5_一時的な未検出はCoastingで維持し再検出で同じIDに戻る") {
    const RunResult r = run(scenarios::overall());
    const int id = r.ids.of(2).firstTrackId;  // 目標2（フレーム6,7で未検出）
    CHECK(findTrack(r.reports[5].updated, id)->state == TrackState::Confirmed);
    CHECK(findTrack(r.reports[6].updated, id)->state == TrackState::Coasting);
    CHECK(findTrack(r.reports[7].updated, id)->state == TrackState::Coasting);
    CHECK(findTrack(r.reports[8].updated, id)->state == TrackState::Confirmed);
    CHECK_EQ(findTrack(r.reports[7].updated, id)->consecutiveMisses, 2);
}

TEST_CASE("6_完全な消失は規定回数の未検出後に削除") {
    const RunResult r = run(scenarios::overall());
    const int id = r.ids.of(3).firstTrackId;  // 目標3（フレーム7の後に消失）
    // フレーム8〜12で5回連続未検出 → フレーム12で削除
    CHECK(findTrack(r.reports[11].updated, id) != nullptr);
    CHECK(contains(r.reports[12].deletedIds, id));
    CHECK(findTrack(r.reports[12].updated, id) == nullptr);
}

TEST_CASE("7_ランダムな誤検出から確定トラックが生まれない") {
    const RunResult r = run(scenarios::overall());
    // 最終フレームの確定トラックは、存在している目標 1, 2, 4 の3つだけ
    CHECK_EQ(countConfirmed(r.reports.back().updated), 3);
    // 誤検出から生成された仮トラックは、確定前に削除されている
    int createdTotal = 0, deletedTotal = 0;
    for (const FrameReport& rep : r.reports) {
        createdTotal += static_cast<int>(rep.createdIds.size());
        deletedTotal += static_cast<int>(rep.deletedIds.size());
    }
    CHECK(createdTotal > 4);  // 誤検出による仮トラックが実際に生成されている
    CHECK(deletedTotal > 0);
}

TEST_CASE("8_X字の交差ではIDを維持") {
    const RunResult r = run(scenarios::crossing());
    CHECK(r.ids.of(1).idKept());
    CHECK(r.ids.of(2).idKept());
}

TEST_CASE("8_浅い角度の接近ではIDが入れ替わり得る（GNNの限界）") {
    // 交差角が小さく、観測ノイズに対して相対移動量が小さいと、位置と速度だけでは
    // 2目標を区別できない。このシナリオ（シード固定）では交差後にIDが入れ替わる。
    // 「GNNでは必ずIDを維持できるわけではない」ことを確認するためのテスト
    const RunResult r = run(scenarios::closeApproach());
    CHECK(!r.ids.of(1).idKept());
    CHECK(!r.ids.of(2).idKept());
    // トラック自体は2本のまま維持される（分裂はしない）
    CHECK_EQ(static_cast<int>(r.reports.back().updated.size()), 2);
}

TEST_CASE("9_10_最明点0件のフレームではすべて未検出") {
    TrackManager manager{TrackingConfig()};
    manager.process(makeFrame(0, 0.0, {{100, 100}, {200, 200}}));
    const FrameReport empty = manager.process(makeFrame(1, 0.1, {}));
    CHECK(empty.association.matches.empty());
    CHECK_EQ(static_cast<int>(empty.association.unassignedTracks.size()), 2);
    // 仮トラックのまま未検出 → まだ削除はされない（許容回数 5−3=2 回）
    CHECK_EQ(static_cast<int>(empty.updated.size()), 2);

    TrackManager fresh{TrackingConfig()};
    const FrameReport none = fresh.process(makeFrame(0, 0.0, {}));
    CHECK(none.updated.empty());
    CHECK(none.createdIds.empty());
}

TEST_CASE("11_すべての観測がゲート範囲外") {
    TrackManager manager{TrackingConfig()};
    for (int k = 0; k < 4; ++k) {
        manager.process(makeFrame(k, 0.1 * k, {{100.0 + 5 * k, 100}}));
    }
    // 目標が急に遠くへ移動したように見える観測（ゲート外）
    const FrameReport r = manager.process(makeFrame(4, 0.4, {{400, 400}}));
    CHECK(r.association.matches.empty());
    CHECK_EQ(static_cast<int>(r.createdIds.size()), 1);                    // 新しい仮トラック
    CHECK(findTrack(r.updated, 1)->state == TrackState::Coasting);         // 既存は予測追跡
}

TEST_CASE("12_フレーム間隔が一定ではない") {
    const RunResult r = run(scenarios::unevenInterval());
    for (const auto& [truth, summary] : r.ids.summaries()) {
        CHECK_EQ(static_cast<int>(summary.allTrackIds.size()), 1);
    }
    // 速度は px/s 単位で推定されるため、間隔が変わっても真値に近い
    const int id1 = r.ids.of(1).firstTrackId;
    const TrackSnapshot* t = findTrack(r.reports.back().updated, id1);
    CHECK(t != nullptr);
    CHECK_NEAR(t->velocity.x(), 80.0, 15.0);
    CHECK_NEAR(t->velocity.y(), 40.0, 15.0);
}

TEST_CASE("時刻が戻るフレームは例外") {
    TrackManager manager{TrackingConfig()};
    manager.process(makeFrame(0, 1.0, {}));
    bool thrown = false;
    try {
        manager.process(makeFrame(1, 1.0, {}));
    } catch (const std::invalid_argument&) {
        thrown = true;
    }
    CHECK(thrown);
}

int main() {
    setupConsole();
    return gnn::test::runAllTests();
}
