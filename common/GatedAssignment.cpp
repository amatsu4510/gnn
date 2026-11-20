// ============================================================================
// GatedAssignment.cpp
// ============================================================================

#include "GatedAssignment.h"

#include <stdexcept>

namespace gnn {

AssociationResult solveGatedAssignment(const Eigen::MatrixXd& costMatrix,
                                       const GateMask& gateMask,
                                       double missCost,
                                       double falseAlarmCost,
                                       const HungarianSolver& solver) {
    const int N = static_cast<int>(costMatrix.rows());  // トラック数
    const int M = static_cast<int>(costMatrix.cols());  // 最明点数

    if (gateMask.rows() != N || gateMask.cols() != M) {
        throw std::invalid_argument("solveGatedAssignment: costMatrix と gateMask のサイズが一致しません");
    }

    AssociationResult result;
    result.costMatrix = costMatrix;
    result.gateMask = gateMask;

    // ---- (N+M)×(N+M) の拡張行列を作る。初期値はすべて割り当て禁止 ----
    const int size = N + M;
    Eigen::MatrixXd augmented = Eigen::MatrixXd::Constant(size, size, HungarianSolver::kForbidden);

    // 左上：ゲート内の組だけ実コストを入れる
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            if (gateMask(i, j)) {
                augmented(i, j) = costMatrix(i, j);
            }
        }
    }
    // 右上：トラック i 専用の「未検出」列
    for (int i = 0; i < N; ++i) {
        augmented(i, M + i) = missCost;
    }
    // 左下：最明点 j 専用の「未割り当て」行
    for (int j = 0; j < M; ++j) {
        augmented(N + j, j) = falseAlarmCost;
    }
    // 右下：ダミー同士はコスト 0（block で部分行列を参照して一括設定）
    if (N > 0 && M > 0) {
        augmented.block(N, M, M, N).setZero();
    }

    // ---- ハンガリアン法で最適割り当てを求める ----
    const AssignmentSolution solution = solver.solve(augmented);

    // ---- 拡張行列の解を「対応付け・未検出・未割り当て」に分解する ----
    for (int i = 0; i < N; ++i) {
        const int j = solution.rowToCol[i];
        if (j >= 0 && j < M) {
            result.matches.push_back({i, j, costMatrix(i, j)});
            result.matchedCost += costMatrix(i, j);
        } else {
            result.unassignedTracks.push_back(i);
            result.penaltyCost += missCost;
        }
    }
    for (int j = 0; j < M; ++j) {
        const int i = solution.colToRow[j];
        if (i < 0 || i >= N) {
            result.unassignedDetections.push_back(j);
            result.penaltyCost += falseAlarmCost;
        }
    }
    return result;
}

}  // namespace gnn
