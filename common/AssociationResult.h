#pragma once
// ============================================================================
// AssociationResult.h
//   GNNによる対応付けの結果を保持する。
//   「行」= トラック（STEP1では前フレームの最明点）、「列」= 今回の最明点。
// ============================================================================

#include <Eigen/Dense>

#include <vector>

namespace gnn {

// ゲート判定結果（true = ゲート内 = 割り当て候補）
using GateMask = Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>;

// 割り当てられた1組
struct Match {
    int trackIndex = -1;      // 行番号（トラックの添字。STEP1では前フレームの最明点の添字）
    int detectionIndex = -1;  // 列番号（今回の最明点の添字）
    double cost = 0.0;        // その組のコスト
};

struct AssociationResult {
    Eigen::MatrixXd costMatrix;  // N×M のコスト行列（ゲート外の値も参考として保持）
    GateMask gateMask;           // N×M のゲート判定結果

    std::vector<Match> matches;               // 対応付けされた組（trackIndex昇順）
    std::vector<int> unassignedTracks;        // 未割り当て（未検出）のトラック
    std::vector<int> unassignedDetections;    // 未割り当ての最明点

    double matchedCost = 0.0;  // 割り当てられた組のコストの合計
    double penaltyCost = 0.0;  // 未検出コスト・未割り当て観測コストの合計
    double totalCost() const { return matchedCost + penaltyCost; }  // 最小化された目的関数の値
};

}  // namespace gnn
