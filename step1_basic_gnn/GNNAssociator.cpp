// ============================================================================
// GNNAssociator.cpp（STEP 1）
// ============================================================================

#include "GNNAssociator.h"

#include "GatedAssignment.h"

namespace gnn {

Eigen::MatrixXd GNNAssociator::computeDistanceMatrix(const std::vector<Eigen::Vector2d>& references,
                                                     const std::vector<Detection>& detections) {
    const int N = static_cast<int>(references.size());
    const int M = static_cast<int>(detections.size());
    Eigen::MatrixXd distance(N, M);  // 0×M や N×0 も作成できる
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            // d = √((xt − xz)² + (yt − yz)²) = ‖t − z‖₂
            distance(i, j) = (references[i] - detections[j].position).norm();
        }
    }
    return distance;
}

GateMask GNNAssociator::computeGate(const Eigen::MatrixXd& distance) const {
    // 配列演算：要素ごとに比較して bool 行列を得る
    return (distance.array() <= config_.gateDistance).matrix();
}

AssociationResult GNNAssociator::associate(const std::vector<Eigen::Vector2d>& references,
                                           const std::vector<Detection>& detections) const {
    const Eigen::MatrixXd distance = computeDistanceMatrix(references, detections);
    const GateMask gate = computeGate(distance);
    return solveGatedAssignment(distance, gate, config_.missCost, config_.falseAlarmCost, solver_);
}

}  // namespace gnn
