// ============================================================================
// GNNAssociator.cpp（STEP 3）
// ============================================================================

#include "GNNAssociator.h"

#include "GatedAssignment.h"

#include <stdexcept>

namespace gnn {

Eigen::MatrixXd GNNAssociator::computeMahalanobisMatrix(const std::vector<PredictedMeasurement>& predictions,
                                                        const std::vector<Detection>& detections) {
    const int N = static_cast<int>(predictions.size());
    const int M = static_cast<int>(detections.size());
    Eigen::MatrixXd d2(N, M);
    for (int i = 0; i < N; ++i) {
        // S のコレスキー分解はトラックごとに1回だけ行い、全観測で使い回す
        const Eigen::LLT<Eigen::Matrix2d> llt(predictions[i].S);
        if (llt.info() != Eigen::Success) {
            throw std::runtime_error("GNNAssociator: S が正定値ではありません");
        }
        for (int j = 0; j < M; ++j) {
            // y = z − ẑ,  d² = yᵀ S⁻¹ y
            //   S⁻¹ y は逆行列を作らずに S w = y を解いて求める
            const Eigen::Vector2d y = detections[j].position - predictions[i].z;
            d2(i, j) = y.dot(llt.solve(y));
        }
    }
    return d2;
}

GateMask GNNAssociator::computeGate(const Eigen::MatrixXd& d2) const {
    return (d2.array() <= config_.gateThreshold()).matrix();
}

AssociationResult GNNAssociator::associate(const std::vector<PredictedMeasurement>& predictions,
                                           const std::vector<Detection>& detections) const {
    const Eigen::MatrixXd d2 = computeMahalanobisMatrix(predictions, detections);
    const GateMask gate = computeGate(d2);
    return solveGatedAssignment(d2, gate, config_.missCost, config_.falseAlarmCost, solver_);
}

}  // namespace gnn
