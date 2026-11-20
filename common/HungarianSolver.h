#pragma once
// ============================================================================
// HungarianSolver.h
//   最小コストの割り当て問題（線形割り当て問題）をハンガリアン法で解く。
//
//   - 行数と列数が異なる矩形行列に対応する（行数 > 列数 なら転置して解く）
//   - 0行・0列の空の行列に対応する
//   - 割り当て禁止の要素は HungarianSolver::kForbidden（+∞）で指定する
//     （内部では十分大きな有限値に置き換え、∞を演算に使わない）
//   - 1つの列が複数の行に割り当てられることはない（逆も同様）
//
//   GNNやトラックの概念には依存しない純粋なアルゴリズムクラスなので、
//   単体でテストできる。
// ============================================================================

#include <Eigen/Dense>

#include <limits>
#include <vector>

namespace gnn {

// 割り当て問題の解
struct AssignmentSolution {
    std::vector<int> rowToCol;  // 各行に割り当てられた列（未割り当て・禁止要素なら -1）
    std::vector<int> colToRow;  // 各列に割り当てられた行（未割り当てなら -1）
    double totalCost = 0.0;     // 割り当てられた（禁止でない）要素のコストの合計
};

class HungarianSolver {
public:
    // 割り当て禁止を表す値
    static constexpr double kForbidden = std::numeric_limits<double>::infinity();

    // コスト行列 cost を解く。
    //   行数 ≤ 列数 のときは全行が、行数 > 列数 のときは全列が、いずれかに割り当てられる
    //   （ただし禁止要素に割り当てざるを得なかった行・列は -1 として返す）。
    //   NaN や -∞ を含む場合は std::invalid_argument を投げる。
    //
    //   計算量：n = min(行数, 列数), m = max(行数, 列数) として O(n² m)。
    //   正方行列なら O(n³)。
    AssignmentSolution solve(const Eigen::MatrixXd& cost) const;

private:
    // 行数 ≤ 列数 の行列を解く本体。禁止要素はすでに有限値へ置き換え済みとする。
    // 戻り値は各行に割り当てられた列。
    static std::vector<int> solveRowsLeCols(const Eigen::MatrixXd& cost);
};

}  // namespace gnn
