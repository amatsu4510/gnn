#pragma once
// ============================================================================
// GNNAssociator.h（STEP 3）
//   STEP 2 のユークリッド距離を「マハラノビス距離の二乗」に置き換え、
//   ゲーティングをカイ二乗分布に基づく閾値で行う。
//   割り当て（ダミー行・列 + ハンガリアン法）は STEP 1・2 と同じ。
//
//   トラックやカルマンフィルタには依存せず、各トラックの
//   「予測観測 ẑ = H x⁻」と「残差の共分散 S」だけを受け取る。
// ============================================================================

#include "AssociationResult.h"
#include "Detection.h"
#include "HungarianSolver.h"
#include "TrackingConfig.h"

#include <Eigen/Dense>

#include <vector>

namespace gnn {

// 1トラック分の予測観測
struct PredictedMeasurement {
    Eigen::Vector2d z = Eigen::Vector2d::Zero();      // 予測観測 ẑ = H x⁻（2×1）
    Eigen::Matrix2d S = Eigen::Matrix2d::Identity();  // 残差の共分散 S = H P⁻ Hᵀ + R（2×2）
};

class GNNAssociator {
public:
    explicit GNNAssociator(const TrackingConfig& config) : config_(config) {}

    AssociationResult associate(const std::vector<PredictedMeasurement>& predictions,
                                const std::vector<Detection>& detections) const;

    // N×M のマハラノビス距離の二乗 d² = yᵀ S⁻¹ y の行列を作る
    //   S が正定値でない場合は std::runtime_error
    static Eigen::MatrixXd computeMahalanobisMatrix(const std::vector<PredictedMeasurement>& predictions,
                                                    const std::vector<Detection>& detections);

    // d² ≤ ゲート閾値（カイ二乗分布の上側点）の組を true にする
    GateMask computeGate(const Eigen::MatrixXd& d2) const;

private:
    TrackingConfig config_;
    HungarianSolver solver_;
};

}  // namespace gnn
