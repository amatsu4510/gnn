#pragma once
// ============================================================================
// KalmanFilter.h（STEP 3）
//   等速直線運動モデルの線形カルマンフィルタ。
//
//   状態ベクトル  x = [px, py, vx, vy]ᵀ   （4×1, 位置 [px], 速度 [px/s]）
//   観測ベクトル  z = [px, py]ᵀ           （2×1）
//
//   予測：x⁻ = F x,          P⁻ = F P Fᵀ + Q
//   更新：y  = z − H x⁻       （イノベーション / 残差）
//         S  = H P⁻ Hᵀ + R    （残差の共分散）
//         K  = P⁻ Hᵀ S⁻¹      （カルマンゲイン）
//         x  = x⁻ + K y
//         P  = (I − K H) P⁻ (I − K H)ᵀ + K R Kᵀ   （Joseph 形式）
//
//   このクラスは状態 (x, P) を持たない。トラックの状態は Track が持ち、
//   KalmanFilter は「計算方法」だけを提供する。対応回数などトラック管理の情報も扱わない。
// ============================================================================

#include <Eigen/Dense>

namespace gnn {

class KalmanFilter {
public:
    using StateVector = Eigen::Vector4d;                 // x : 4×1
    using StateMatrix = Eigen::Matrix4d;                 // P, F, Q : 4×4
    using MeasurementVector = Eigen::Vector2d;           // z, y : 2×1
    using MeasurementCovariance = Eigen::Matrix2d;       // R, S : 2×2
    using MeasurementMatrix = Eigen::Matrix<double, 2, 4>;  // H : 2×4
    using GainMatrix = Eigen::Matrix<double, 4, 2>;      // K : 4×2

    // processNoiseDensity : q [px²/s³]
    // measurementNoiseStd : σ_r [px]
    KalmanFilter(double processNoiseDensity, double measurementNoiseStd);

    // 状態遷移行列 F（Δt に依存）
    static StateMatrix transitionMatrix(double dt);
    // プロセスノイズ共分散行列 Q（Δt に依存）
    StateMatrix processNoise(double dt) const;
    // 観測行列 H（状態から位置成分を取り出す）
    static MeasurementMatrix observationMatrix();
    // 観測ノイズ共分散行列 R
    const MeasurementCovariance& measurementNoise() const { return R_; }

    // 状態予測：x ← F x,  P ← F P Fᵀ + Q
    void predict(StateVector& x, StateMatrix& P, double dt) const;

    // 予測観測 H x
    static MeasurementVector predictedMeasurement(const StateVector& x);
    // 残差の共分散 S = H P Hᵀ + R
    MeasurementCovariance innovationCovariance(const StateMatrix& P) const;

    // 状態更新（カルマンゲイン + Joseph 形式）
    //   S が正定値でない場合は std::runtime_error
    void update(StateVector& x, StateMatrix& P, const MeasurementVector& z) const;

    // 観測 z から新規トラックの初期状態を作る（速度 0、速度の分散は大きく）
    void initialize(const MeasurementVector& z, double initialVelocityStd,
                    StateVector& x, StateMatrix& P) const;

private:
    double q_;                    // プロセスノイズのスペクトル密度
    MeasurementCovariance R_;     // 観測ノイズ共分散
};

}  // namespace gnn
