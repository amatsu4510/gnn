// ============================================================================
// main.cpp（STEP 1：基本的なGNN）
//   連続する2フレームの最明点を GNN で対応付け、その過程をすべて表示する。
//   位置予測は行わず、前フレームの位置をそのまま基準にする。
// ============================================================================

#include "GNNAssociator.h"
#include "Printing.h"
#include "ScenarioGenerator.h"
#include "TrackingConfig.h"

#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace gnn;

namespace {

// 行ラベル（前フレームの点 P0, P1, ...）
std::vector<std::string> makeRowLabels(int count) {
    std::vector<std::string> labels;
    for (int i = 0; i < count; ++i) {
        labels.push_back("P" + std::to_string(i));
    }
    return labels;
}

std::vector<Eigen::Vector2d> positionsOf(const Frame& frame) {
    std::vector<Eigen::Vector2d> positions;
    for (const Detection& d : frame.detections) {
        positions.push_back(d.position);
    }
    return positions;
}

// 前フレーム → 今回フレーム の対応付けを1回実行して表示する
void associateAndPrint(const Frame& previous, const Frame& current, const TrackingConfig& config) {
    std::cout << "\n---------------- フレーム " << previous.frameNumber << " → " << current.frameNumber
              << " ----------------\n";
    std::cout << "（前フレーム）";
    printDetections(previous, "P");
    std::cout << "（今回）";
    printDetections(current);

    const GNNAssociator associator(config);
    const std::vector<Eigen::Vector2d> references = positionsOf(previous);
    const AssociationResult result = associator.associate(references, current.detections);

    const std::vector<std::string> labels = makeRowLabels(static_cast<int>(references.size()));
    printGateMask(result, labels);
    printCostMatrix(result, labels);
    printAssignment(result, labels, config.missCost, config.falseAlarmCost);
    std::cout << "  ※ STEP 1 では未割り当ての前フレーム点 = 消失または未検出の候補、"
                 "未割り当ての今回点 = 新規目標または誤検出の候補\n";
}

// 比較用：行の順に最も近い列を取っていく逐次NN（貪欲法）
double greedyTotalCost(const Eigen::MatrixXd& cost, std::vector<int>& rowToCol) {
    const int N = static_cast<int>(cost.rows());
    const int M = static_cast<int>(cost.cols());
    std::vector<bool> used(M, false);
    rowToCol.assign(N, -1);
    double total = 0.0;
    for (int i = 0; i < N; ++i) {
        int best = -1;
        for (int j = 0; j < M; ++j) {
            if (!used[j] && (best < 0 || cost(i, j) < cost(i, best))) {
                best = j;
            }
        }
        if (best >= 0) {
            used[best] = true;
            rowToCol[i] = best;
            total += cost(i, best);
        }
    }
    return total;
}

}  // namespace

int main() {
    setupConsole();

    // ------------------------------------------------------------------
    // デモ1：資料（p.8）のコスト行列の例。ゲート 30 px
    // ------------------------------------------------------------------
    printHeading("デモ1：資料の例（3トラック・5最明点、ゲート 30 px）");
    {
        TrackingConfig config;  // gate 30, miss 15, falseAlarm 15
        const Frame previous = makeFrame(0, 0.0, {{100, 100}, {200, 150}, {300, 200}});
        const Frame current = makeFrame(1, 0.1, {{105, 103}, {198, 148}, {310, 205}, {500, 300}, {110, 115}});
        associateAndPrint(previous, current, config);
        std::cout << "  → 期待値：P0→Z0, P1→Z1, P2→Z2, 合計 19.84 px（Z3, Z4 は未割り当て）\n";
    }

    // ------------------------------------------------------------------
    // デモ2：貪欲法では最適にならない例
    // ------------------------------------------------------------------
    printHeading("デモ2：逐次NN（貪欲法）と GNN の比較");
    {
        TrackingConfig config;
        config.gateDistance = 60.0;
        config.missCost = 30.0;
        config.falseAlarmCost = 30.0;
        const Frame previous = makeFrame(0, 0.0, {{40, 0}, {65, 0}});
        const Frame current = makeFrame(1, 0.1, {{55, 0}, {10, 0}});
        associateAndPrint(previous, current, config);

        std::vector<int> greedy;
        const double greedyCost =
            greedyTotalCost(GNNAssociator::computeDistanceMatrix(positionsOf(previous), current.detections), greedy);
        std::cout << "  逐次NN（P0 から順に最も近い点を選ぶ）: P0->Z" << greedy[0] << ", P1->Z" << greedy[1]
                  << "  合計 " << std::fixed << std::setprecision(3) << greedyCost << "\n"
                  << "  → GNN は全体の合計を最小化するため、合計 40 で貪欲法の 70 より小さい\n";
    }

    // ------------------------------------------------------------------
    // デモ3：疑似データの連続フレーム（順序入れ替え・未検出・新規目標・誤検出）
    // ------------------------------------------------------------------
    printHeading("デモ3：疑似データ（3目標 + 新規目標 + 誤検出, 検出順序はフレームごとに変化）");
    {
        ScenarioSpec spec;
        spec.timestamps = uniformTimestamps(5, 0.1);
        spec.targets = {
            {1, 0, 4, {100, 100}, {80, 20}, {}},    // 通常移動
            {2, 0, 4, {300, 250}, {-50, 60}, {2}},  // フレーム2で未検出
            {3, 0, 4, {500, 120}, {0, 90}, {}},     // 通常移動
            {4, 3, 4, {200, 400}, {60, 0}, {}},     // フレーム3で新規出現
        };
        spec.noiseStd = 1.0;
        spec.maxClutterPerFrame = 1;
        spec.seed = 2024;
        const std::vector<ScenarioFrame> frames = generateScenario(spec);

        TrackingConfig config;
        for (std::size_t k = 1; k < frames.size(); ++k) {
            associateAndPrint(frames[k - 1].frame, frames[k].frame, config);
        }
        std::cout << "\n  ※ STEP 1 は前フレームの位置だけを基準にするため、1フレーム未検出になると\n"
                     "    次のフレームで対応先が見つからない。これは STEP 2 の予測追跡で解決する。\n";
    }
    return 0;
}
