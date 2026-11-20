#pragma once
// ============================================================================
// GNNAssociator.h（STEP 2：STEP 1 から変更なし）
//   基準点（STEP 2 ではトラックの予測位置）と今回の最明点を GNN で対応付ける。
//     1. ユークリッド距離のコスト行列を作る
//     2. 距離の閾値でゲーティングする
//     3. ダミー行・列を加えてハンガリアン法で全体最適な割り当てを求める
// ============================================================================

#include "AssociationResult.h"
#include "Detection.h"
#include "HungarianSolver.h"
#include "TrackingConfig.h"

#include <Eigen/Dense>

#include <vector>

namespace gnn {

class GNNAssociator {
public:
    explicit GNNAssociator(const TrackingConfig& config) : config_(config) {}

    // references : 対応付けの基準となる位置（N 個）
    // detections : 今回の最明点（M 個）
    AssociationResult associate(const std::vector<Eigen::Vector2d>& references,
                                const std::vector<Detection>& detections) const;

    // N×M のユークリッド距離行列を作る
    static Eigen::MatrixXd computeDistanceMatrix(const std::vector<Eigen::Vector2d>& references,
                                                 const std::vector<Detection>& detections);

    // 距離が gateDistance 以下の組を true にする
    GateMask computeGate(const Eigen::MatrixXd& distance) const;

private:
    TrackingConfig config_;
    HungarianSolver solver_;
};

}  // namespace gnn
