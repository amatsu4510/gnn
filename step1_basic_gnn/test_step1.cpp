// ============================================================================
// test_step1.cpp（STEP 1：基本的なGNN のテスト）
// ============================================================================

#include "GNNAssociator.h"
#include "Printing.h"
#include "ScenarioGenerator.h"
#include "TestFramework.h"
#include "TrackingConfig.h"

#include <vector>

using namespace gnn;

namespace {

std::vector<Eigen::Vector2d> positionsOf(const Frame& frame) {
    std::vector<Eigen::Vector2d> positions;
    for (const Detection& d : frame.detections) {
        positions.push_back(d.position);
    }
    return positions;
}

// 対応付けされた組が真値IDで一致しているか（前後フレームで同じ目標同士か）を数える
int countCorrectMatches(const AssociationResult& r, const ScenarioFrame& prev, const ScenarioFrame& curr) {
    int correct = 0;
    for (const Match& m : r.matches) {
        const int a = prev.truthIds[m.trackIndex];
        const int b = curr.truthIds[m.detectionIndex];
        if (a >= 0 && a == b) {
            ++correct;
        }
    }
    return correct;
}

}  // namespace

TEST_CASE("1_複数目標の通常移動") {
    ScenarioSpec spec;
    spec.timestamps = uniformTimestamps(6, 0.1);
    spec.targets = {{1, 0, 5, {100, 100}, {80, 0}, {}},
                    {2, 0, 5, {300, 300}, {0, -80}, {}},
                    {3, 0, 5, {500, 100}, {-60, 60}, {}}};
    spec.shuffle = false;
    const auto frames = generateScenario(spec);
    const GNNAssociator associator{TrackingConfig()};
    for (std::size_t k = 1; k < frames.size(); ++k) {
        const AssociationResult r = associator.associate(positionsOf(frames[k - 1].frame), frames[k].frame.detections);
        CHECK_EQ(static_cast<int>(r.matches.size()), 3);
        CHECK_EQ(countCorrectMatches(r, frames[k - 1], frames[k]), 3);
    }
}

TEST_CASE("2_検出順序がフレームごとに変わる") {
    ScenarioSpec spec;
    spec.timestamps = uniformTimestamps(8, 0.1);
    spec.targets = {{1, 0, 7, {100, 100}, {80, 0}, {}},
                    {2, 0, 7, {200, 100}, {80, 0}, {}},
                    {3, 0, 7, {300, 100}, {80, 0}, {}},
                    {4, 0, 7, {400, 100}, {80, 0}, {}}};
    spec.shuffle = true;  // 毎フレーム順序を入れ替える
    spec.seed = 99;
    const auto frames = generateScenario(spec);
    const GNNAssociator associator{TrackingConfig()};
    for (std::size_t k = 1; k < frames.size(); ++k) {
        const AssociationResult r = associator.associate(positionsOf(frames[k - 1].frame), frames[k].frame.detections);
        CHECK_EQ(countCorrectMatches(r, frames[k - 1], frames[k]), 4);
    }
}

TEST_CASE("3_貪欲法では最適にならないケース") {
    TrackingConfig config;
    config.gateDistance = 60.0;
    config.missCost = 30.0;
    config.falseAlarmCost = 30.0;
    const Frame prev = makeFrame(0, 0.0, {{40, 0}, {65, 0}});
    const Frame curr = makeFrame(1, 0.1, {{55, 0}, {10, 0}});
    const AssociationResult r = GNNAssociator(config).associate(positionsOf(prev), curr.detections);
    // 貪欲法：P0→Z0(15), P1→Z1(55) = 70 ／ GNN：P0→Z1(30), P1→Z0(10) = 40
    CHECK_EQ(static_cast<int>(r.matches.size()), 2);
    CHECK_EQ(r.matches[0].detectionIndex, 1);
    CHECK_EQ(r.matches[1].detectionIndex, 0);
    CHECK_NEAR(r.matchedCost, 40.0, 1e-9);
}

TEST_CASE("資料p8の例_合計19.84px") {
    const Frame prev = makeFrame(0, 0.0, {{100, 100}, {200, 150}, {300, 200}});
    const Frame curr = makeFrame(1, 0.1, {{105, 103}, {198, 148}, {310, 205}, {500, 300}, {110, 115}});
    const AssociationResult r = GNNAssociator(TrackingConfig()).associate(positionsOf(prev), curr.detections);
    CHECK_EQ(static_cast<int>(r.matches.size()), 3);
    CHECK_NEAR(r.matchedCost, 19.84, 0.01);
    CHECK(r.unassignedTracks.empty());
    CHECK_EQ(static_cast<int>(r.unassignedDetections.size()), 2);
    CHECK(r.gateMask(0, 4));   // P0–Z4 は 18.03 px でゲート内（ただし割り当てはされない）
    CHECK(!r.gateMask(0, 3));  // P0–Z3 はゲート外
}

TEST_CASE("4_新規目標の出現は未割り当て最明点になる") {
    const Frame prev = makeFrame(0, 0.0, {{100, 100}});
    const Frame curr = makeFrame(1, 0.1, {{105, 100}, {400, 400}});
    const AssociationResult r = GNNAssociator(TrackingConfig()).associate(positionsOf(prev), curr.detections);
    CHECK_EQ(static_cast<int>(r.matches.size()), 1);
    CHECK_EQ(static_cast<int>(r.unassignedDetections.size()), 1);
    CHECK_EQ(r.unassignedDetections[0], 1);
}

TEST_CASE("5_6_未検出や消失は未割り当てトラックになる") {
    const Frame prev = makeFrame(0, 0.0, {{100, 100}, {300, 300}});
    const Frame curr = makeFrame(1, 0.1, {{302, 301}});
    const AssociationResult r = GNNAssociator(TrackingConfig()).associate(positionsOf(prev), curr.detections);
    CHECK_EQ(static_cast<int>(r.matches.size()), 1);
    CHECK_EQ(static_cast<int>(r.unassignedTracks.size()), 1);
    CHECK_EQ(r.unassignedTracks[0], 0);
}

TEST_CASE("7_ゲート外の誤検出は割り当てられない") {
    ScenarioSpec spec;
    spec.timestamps = uniformTimestamps(10, 0.1);
    spec.targets = {{1, 0, 9, {100, 100}, {50, 30}, {}}, {2, 0, 9, {400, 300}, {-40, 20}, {}}};
    spec.maxClutterPerFrame = 3;
    spec.seed = 7;
    const auto frames = generateScenario(spec);
    const GNNAssociator associator{TrackingConfig()};
    for (std::size_t k = 1; k < frames.size(); ++k) {
        const AssociationResult r = associator.associate(positionsOf(frames[k - 1].frame), frames[k].frame.detections);
        // 目標同士の対応は必ず正しく取れている（誤検出が偶然近くに出た場合を除けば2組）
        CHECK(countCorrectMatches(r, frames[k - 1], frames[k]) >= 1);
        for (const Match& m : r.matches) {
            CHECK(m.cost <= TrackingConfig().gateDistance);
        }
    }
}

TEST_CASE("9_トラック数と最明点数が異なる") {
    const GNNAssociator associator{TrackingConfig()};
    {   // 3 → 5
        const Frame prev = makeFrame(0, 0.0, {{0, 0}, {100, 0}, {200, 0}});
        const Frame curr = makeFrame(1, 0.1, {{201, 0}, {500, 500}, {1, 0}, {101, 0}, {600, 0}});
        const AssociationResult r = associator.associate(positionsOf(prev), curr.detections);
        CHECK_EQ(static_cast<int>(r.matches.size()), 3);
        CHECK_EQ(static_cast<int>(r.unassignedDetections.size()), 2);
        CHECK_EQ(r.matches[0].detectionIndex, 2);
        CHECK_EQ(r.matches[1].detectionIndex, 3);
        CHECK_EQ(r.matches[2].detectionIndex, 0);
    }
    {   // 5 → 2
        const Frame prev = makeFrame(0, 0.0, {{0, 0}, {100, 0}, {200, 0}, {300, 0}, {400, 0}});
        const Frame curr = makeFrame(1, 0.1, {{302, 0}, {98, 0}});
        const AssociationResult r = associator.associate(positionsOf(prev), curr.detections);
        CHECK_EQ(static_cast<int>(r.matches.size()), 2);
        CHECK_EQ(static_cast<int>(r.unassignedTracks.size()), 3);
    }
}

TEST_CASE("10_トラックまたは最明点が0件") {
    const GNNAssociator associator{TrackingConfig()};
    const Frame empty = makeFrame(0, 0.0, {});
    const Frame two = makeFrame(1, 0.1, {{1, 1}, {2, 2}});

    const AssociationResult noRefs = associator.associate({}, two.detections);
    CHECK(noRefs.matches.empty());
    CHECK_EQ(static_cast<int>(noRefs.unassignedDetections.size()), 2);

    const AssociationResult noDets = associator.associate(positionsOf(two), empty.detections);
    CHECK(noDets.matches.empty());
    CHECK_EQ(static_cast<int>(noDets.unassignedTracks.size()), 2);

    const AssociationResult none = associator.associate({}, {});
    CHECK(none.matches.empty());
    CHECK_NEAR(none.totalCost(), 0.0, 0.0);
}

TEST_CASE("11_すべての観測がゲート範囲外") {
    const Frame prev = makeFrame(0, 0.0, {{100, 100}, {200, 200}});
    const Frame curr = makeFrame(1, 0.1, {{400, 100}, {100, 400}, {600, 450}});
    const AssociationResult r = GNNAssociator(TrackingConfig()).associate(positionsOf(prev), curr.detections);
    CHECK(r.matches.empty());
    CHECK_EQ(static_cast<int>(r.unassignedTracks.size()), 2);
    CHECK_EQ(static_cast<int>(r.unassignedDetections.size()), 3);
    CHECK(!r.gateMask.any());
}

TEST_CASE("ゲート境界_距離がゲート半径ちょうどはゲート内") {
    const Frame prev = makeFrame(0, 0.0, {{0, 0}});
    const Frame curr = makeFrame(1, 0.1, {{30, 0}});
    TrackingConfig config;
    config.missCost = 20.0;  // miss + fa = 35 > 30 なので割り当てた方が得
    const AssociationResult r = GNNAssociator(config).associate(positionsOf(prev), curr.detections);
    CHECK(r.gateMask(0, 0));
    CHECK_EQ(static_cast<int>(r.matches.size()), 1);
}

int main() {
    setupConsole();
    return gnn::test::runAllTests();
}
