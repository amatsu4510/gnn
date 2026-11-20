// ============================================================================
// test_hungarian.cpp
//   HungarianSolver と solveGatedAssignment の単体テスト。
//   小規模な行列を全探索した結果と比較し、最適解であることを検証する。
// ============================================================================

#include "GatedAssignment.h"
#include "HungarianSolver.h"
#include "Printing.h"
#include "ScenarioGenerator.h"
#include "TestFramework.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
#include <vector>

using namespace gnn;

namespace {

const double F = HungarianSolver::kForbidden;

// ---- 全探索（割り当て問題）----
//   評価：(1) 割り当て数が多いほど良い (2) 同数ならコストが小さいほど良い
//   ※ HungarianSolver は禁止要素を巨大コストに置き換えているため、
//     「禁止要素を使う数を最小化 → コスト最小化」となり、この評価と一致する。
struct BruteForceResult {
    int count = -1;
    double cost = std::numeric_limits<double>::max();
};

void bruteForceVisit(const Eigen::MatrixXd& c, int row, std::vector<bool>& usedCol, int count, double cost,
                     BruteForceResult& best) {
    if (row == c.rows()) {
        if (count > best.count || (count == best.count && cost < best.cost - 1e-12)) {
            best.count = count;
            best.cost = cost;
        }
        return;
    }
    // この行を割り当てない
    bruteForceVisit(c, row + 1, usedCol, count, cost, best);
    // 割り当て可能な列を1つ選ぶ
    for (int j = 0; j < c.cols(); ++j) {
        if (!usedCol[j] && std::isfinite(c(row, j))) {
            usedCol[j] = true;
            bruteForceVisit(c, row + 1, usedCol, count + 1, cost + c(row, j), best);
            usedCol[j] = false;
        }
    }
}

BruteForceResult bruteForceAssignment(const Eigen::MatrixXd& c) {
    BruteForceResult best;
    std::vector<bool> used(c.cols(), false);
    bruteForceVisit(c, 0, used, 0, 0.0, best);
    return best;
}

// ---- 全探索（未割り当てを許すGNNの目的関数）----
//   Σ(割り当てた組のコスト) + 未検出数×missCost + 未割り当て観測数×falseAlarmCost を最小化
double bruteForceGnnVisit(const Eigen::MatrixXd& c, const GateMask& gate, double miss, double fa, int row,
                          std::vector<bool>& usedCol) {
    if (row == c.rows()) {
        const auto unusedCols = std::count(usedCol.begin(), usedCol.end(), false);
        return static_cast<double>(unusedCols) * fa;
    }
    double best = miss + bruteForceGnnVisit(c, gate, miss, fa, row + 1, usedCol);
    for (int j = 0; j < c.cols(); ++j) {
        if (!usedCol[j] && gate(row, j)) {
            usedCol[j] = true;
            best = std::min(best, c(row, j) + bruteForceGnnVisit(c, gate, miss, fa, row + 1, usedCol));
            usedCol[j] = false;
        }
    }
    return best;
}

// 解が「1行1列・禁止要素なし・コスト合計が正しい」かを検証する
void checkValidSolution(const Eigen::MatrixXd& c, const AssignmentSolution& s) {
    CHECK_EQ(static_cast<int>(s.rowToCol.size()), static_cast<int>(c.rows()));
    CHECK_EQ(static_cast<int>(s.colToRow.size()), static_cast<int>(c.cols()));
    std::set<int> usedCols;
    double total = 0.0;
    for (int i = 0; i < c.rows(); ++i) {
        const int j = s.rowToCol[i];
        if (j < 0) {
            continue;
        }
        CHECK(std::isfinite(c(i, j)));     // 禁止要素に割り当てていない
        CHECK(usedCols.insert(j).second);  // 同じ列を2回使っていない
        CHECK_EQ(s.colToRow[j], i);        // 逆方向の対応と一致
        total += c(i, j);
    }
    CHECK_NEAR(total, s.totalCost, 1e-9);
}

}  // namespace

// ---------------------------------------------------------------------------
// 基本ケース
// ---------------------------------------------------------------------------

TEST_CASE("資料の例_貪欲法では最適にならない") {
    // 資料 p.5：[[2,3],[1,8]] は 貪欲法 2+8=10, 最適 3+1=4
    Eigen::MatrixXd c(2, 2);
    c << 2, 3,
         1, 8;
    const AssignmentSolution s = HungarianSolver().solve(c);
    CHECK_EQ(s.rowToCol[0], 1);
    CHECK_EQ(s.rowToCol[1], 0);
    CHECK_NEAR(s.totalCost, 4.0, 1e-12);
}

TEST_CASE("貪欲法では最適にならない_3x3") {
    // 行0 が最小の 1 を取ると、残りが大きくなる例
    Eigen::MatrixXd c(3, 3);
    c << 1, 2, 9,
         2, 9, 9,
         9, 9, 3;   // 貪欲: (0,0)+(1,1)+(2,2)=1+9+3=13, 最適: (0,1)+(1,0)+(2,2)=2+2+3=7
    const AssignmentSolution s = HungarianSolver().solve(c);
    CHECK_NEAR(s.totalCost, 7.0, 1e-12);
    checkValidSolution(c, s);
}

TEST_CASE("空の入力") {
    const HungarianSolver solver;
    const AssignmentSolution s00 = solver.solve(Eigen::MatrixXd(0, 0));
    CHECK(s00.rowToCol.empty());
    CHECK(s00.colToRow.empty());

    const AssignmentSolution s03 = solver.solve(Eigen::MatrixXd(0, 3));
    CHECK(s03.rowToCol.empty());
    CHECK_EQ(static_cast<int>(s03.colToRow.size()), 3);
    CHECK_EQ(s03.colToRow[0], -1);

    const AssignmentSolution s30 = solver.solve(Eigen::MatrixXd(3, 0));
    CHECK_EQ(static_cast<int>(s30.rowToCol.size()), 3);
    CHECK_EQ(s30.rowToCol[2], -1);
    CHECK_NEAR(s30.totalCost, 0.0, 0.0);
}

TEST_CASE("矩形行列_行が少ない場合と多い場合") {
    Eigen::MatrixXd wide(2, 4);
    wide << 5, 1, 7, 3,
            2, 1, 9, 8;     // 最適: 行0→列1(1), 行1→列0(2) = 3
    const AssignmentSolution sw = HungarianSolver().solve(wide);
    CHECK_NEAR(sw.totalCost, 3.0, 1e-12);
    checkValidSolution(wide, sw);

    const Eigen::MatrixXd tall = wide.transpose();  // 4×2（内部で転置して解く）
    const AssignmentSolution st = HungarianSolver().solve(tall);
    CHECK_NEAR(st.totalCost, 3.0, 1e-12);
    CHECK_EQ(st.rowToCol[1], 0);
    CHECK_EQ(st.rowToCol[0], 1);
    CHECK_EQ(st.rowToCol[2], -1);
    checkValidSolution(tall, st);
}

TEST_CASE("禁止要素は割り当てない") {
    Eigen::MatrixXd c(3, 3);
    c << 1, F, F,
         2, F, F,   // 行0 と行1 は列0 しか使えない → どちらかは未割り当て
         F, F, 4;
    const AssignmentSolution s = HungarianSolver().solve(c);
    checkValidSolution(c, s);
    CHECK_EQ(s.rowToCol[0], 0);   // コストの小さい行0 が列0 を取る
    CHECK_EQ(s.rowToCol[1], -1);  // 行1 は禁止要素しか残らないため未割り当て
    CHECK_EQ(s.rowToCol[2], 2);
    CHECK_NEAR(s.totalCost, 5.0, 1e-12);

    Eigen::MatrixXd allForbidden = Eigen::MatrixXd::Constant(2, 3, F);
    const AssignmentSolution sf = HungarianSolver().solve(allForbidden);
    CHECK_EQ(sf.rowToCol[0], -1);
    CHECK_EQ(sf.rowToCol[1], -1);
}

TEST_CASE("NaNは例外") {
    Eigen::MatrixXd c(1, 1);
    c(0, 0) = std::numeric_limits<double>::quiet_NaN();
    bool thrown = false;
    try {
        HungarianSolver().solve(c);
    } catch (const std::invalid_argument&) {
        thrown = true;
    }
    CHECK(thrown);
}

TEST_CASE("同コストの場合も決定的な結果") {
    const Eigen::MatrixXd c = Eigen::MatrixXd::Constant(3, 3, 1.0);
    const AssignmentSolution a = HungarianSolver().solve(c);
    const AssignmentSolution b = HungarianSolver().solve(c);
    CHECK(a.rowToCol == b.rowToCol);
    checkValidSolution(c, a);
}

// ---------------------------------------------------------------------------
// 全探索との比較（ランダム行列。シード固定で再現可能）
// ---------------------------------------------------------------------------

TEST_CASE("全探索との比較_ランダム行列") {
    Rng rng(20240601);
    const HungarianSolver solver;
    int compared = 0;
    for (int trial = 0; trial < 3000; ++trial) {
        const int rows = rng.uniformInt(1, 6);
        const int cols = rng.uniformInt(1, 6);
        const double forbiddenRatio = rng.uniform(0.0, 0.6);
        Eigen::MatrixXd c(rows, cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                // 整数コストにして同コスト（タイ）も発生させる。負のコストも含める
                c(i, j) = rng.uniform01() < forbiddenRatio ? F : std::floor(rng.uniform(-5.0, 20.0));
            }
        }
        const AssignmentSolution s = solver.solve(c);
        const BruteForceResult bf = bruteForceAssignment(c);
        const int count = static_cast<int>(std::count_if(s.rowToCol.begin(), s.rowToCol.end(), [](int j) { return j >= 0; }));
        checkValidSolution(c, s);
        CHECK_EQ(count, bf.count);
        CHECK_NEAR(s.totalCost, bf.cost, 1e-9);
        ++compared;
    }
    CHECK_EQ(compared, 3000);
}

// ---------------------------------------------------------------------------
// ダミー行・列による未割り当て許容（GNNの目的関数）の検証
// ---------------------------------------------------------------------------

TEST_CASE("ダミー拡張_資料の例") {
    // 資料 p.18 の例：コスト [[2,3],[1,8]]、未検出コスト 10
    Eigen::MatrixXd c(2, 2);
    c << 2, 3,
         1, 8;
    const GateMask gate = GateMask::Constant(2, 2, true);
    const AssociationResult r = solveGatedAssignment(c, gate, 10.0, 0.0);
    CHECK_EQ(static_cast<int>(r.matches.size()), 2);
    CHECK_EQ(r.matches[0].detectionIndex, 1);
    CHECK_EQ(r.matches[1].detectionIndex, 0);
    CHECK_NEAR(r.matchedCost, 4.0, 1e-12);
}

TEST_CASE("ダミー拡張_コストが高い組より未割り当てを選ぶ") {
    // 距離 25 の組は miss+fa = 10+10 = 20 より高いので、両方未割り当ての方が良い
    Eigen::MatrixXd c(1, 1);
    c << 25;
    const AssociationResult r = solveGatedAssignment(c, GateMask::Constant(1, 1, true), 10.0, 10.0);
    CHECK(r.matches.empty());
    CHECK_EQ(static_cast<int>(r.unassignedTracks.size()), 1);
    CHECK_EQ(static_cast<int>(r.unassignedDetections.size()), 1);
    CHECK_NEAR(r.totalCost(), 20.0, 1e-12);
}

TEST_CASE("ダミー拡張_全探索との比較") {
    Rng rng(777);
    for (int trial = 0; trial < 2000; ++trial) {
        const int N = rng.uniformInt(0, 5);
        const int M = rng.uniformInt(0, 5);
        const double miss = rng.uniform(1.0, 10.0);
        const double fa = rng.uniform(0.0, 10.0);
        Eigen::MatrixXd c(N, M);
        GateMask gate(N, M);
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < M; ++j) {
                c(i, j) = rng.uniform(0.0, 20.0);
                gate(i, j) = rng.uniform01() < 0.6;
            }
        }
        const AssociationResult r = solveGatedAssignment(c, gate, miss, fa);
        std::vector<bool> used(M, false);
        const double expected = bruteForceGnnVisit(c, gate, miss, fa, 0, used);
        CHECK_NEAR(r.totalCost(), expected, 1e-9);

        // 整合性：ゲート外を割り当てない・同一観測を重複して割り当てない・件数が合う
        std::set<int> usedDetections;
        for (const Match& m : r.matches) {
            CHECK(gate(m.trackIndex, m.detectionIndex));
            CHECK(usedDetections.insert(m.detectionIndex).second);
        }
        CHECK_EQ(static_cast<int>(r.matches.size() + r.unassignedTracks.size()), N);
        CHECK_EQ(static_cast<int>(r.matches.size() + r.unassignedDetections.size()), M);
    }
}

int main() {
    setupConsole();
    return gnn::test::runAllTests();
}
