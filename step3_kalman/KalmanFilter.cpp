// ============================================================================
// KalmanFilter.cpp（STEP 3）
// ============================================================================

#include "KalmanFilter.h"

#include <stdexcept>

namespace gnn {

KalmanFilter::KalmanFilter(double processNoiseDensity, double measurementNoiseStd)
    : q_(processNoiseDensity),
      // R = σ_r² I（x, y の観測誤差は独立で同じ大きさと仮定）
      R_(MeasurementCovariance::Identity() * measurementNoiseStd * measurementNoiseStd) {}

KalmanFilter::StateMatrix KalmanFilter::transitionMatrix(double dt) {
    //     ┌ 1 0 Δt 0  ┐
    // F = │ 0 1 0  Δt │   px' = px + vx·Δt,  vx' = vx
    //     │ 0 0 1  0  │
    //     └ 0 0 0  1  ┘
    StateMatrix F = StateMatrix::Identity();
    F(0, 2) = dt;
    F(1, 3) = dt;
    return F;
}

KalmanFilter::StateMatrix KalmanFilter::processNoise(double dt) const {
    // 連続時間の白色加速度雑音モデル（スペクトル密度 q）を Δt で離散化したもの
    //   1軸あたり  Q_axis = q · ┌ Δt³/3  Δt²/2 ┐
    //                           └ Δt²/2  Δt    ┘
    //   状態の並びが [px, py, vx, vy] なので、x軸は (0,2)、y軸は (1,3) の位置に入る。
    //   Δt に応じて大きさが変わるため、フレーム間隔が一定でなくても正しく扱える。
    const double dt2 = dt * dt;
    const double dt3 = dt2 * dt;
    StateMatrix Q = StateMatrix::Zero();
    Q(0, 0) = Q(1, 1) = q_ * dt3 / 3.0;
    Q(0, 2) = Q(2, 0) = q_ * dt2 / 2.0;
    Q(1, 3) = Q(3, 1) = q_ * dt2 / 2.0;
    Q(2, 2) = Q(3, 3) = q_ * dt;
    return Q;
}

KalmanFilter::MeasurementMatrix KalmanFilter::observationMatrix() {
    // H = ┌ 1 0 0 0 ┐   z = H x で位置成分だけを取り出す
    //     └ 0 1 0 0 ┘
    MeasurementMatrix H = MeasurementMatrix::Zero();
    H(0, 0) = 1.0;
    H(1, 1) = 1.0;
    return H;
}

void KalmanFilter::predict(StateVector& x, StateMatrix& P, double dt) const {
    const StateMatrix F = transitionMatrix(dt);
    // 行列積は一時オブジェクトに評価されてから代入されるため、x = F * x と書いても
    // エイリアシング（計算途中で x が書き換わる問題）は起きない
    x = F * x;
    P = F * P * F.transpose() + processNoise(dt);
}

KalmanFilter::MeasurementVector KalmanFilter::predictedMeasurement(const StateVector& x) {
    // H x は位置成分 [px, py] と同じなので、行列積の代わりに head<2>() で先頭2要素を取り出す
    return x.head<2>();
}

KalmanFilter::MeasurementCovariance KalmanFilter::innovationCovariance(const StateMatrix& P) const {
    const MeasurementMatrix H = observationMatrix();
    return H * P * H.transpose() + R_;
}

void KalmanFilter::update(StateVector& x, StateMatrix& P, const MeasurementVector& z) const {
    const MeasurementMatrix H = observationMatrix();
    const MeasurementCovariance S = innovationCovariance(P);

    // ---- カルマンゲイン K = P Hᵀ S⁻¹ ----
    //   S⁻¹ を直接計算せず、コレスキー分解（LLT: S = L Lᵀ）で連立方程式を解く。
    //   K Sᵀ = P Hᵀ → 両辺を転置して S Kᵀ = H P（P, S は対称）
    //   つまり Kᵀ = S \ (H P) を解けばよい。
    const Eigen::LLT<MeasurementCovariance> llt(S);
    if (llt.info() != Eigen::Success) {
        throw std::runtime_error("KalmanFilter::update: S が正定値ではありません");
    }
    const GainMatrix K = llt.solve(H * P).transpose();

    // ---- 状態の更新 ----
    const MeasurementVector y = z - H * x;  // イノベーション（残差）
    x = x + K * y;

    // ---- 誤差共分散の更新（Joseph 形式）----
    //   P = (I − K H) P (I − K H)ᵀ + K R Kᵀ
    //   単純な P = (I − K H) P よりも計算量は多いが、丸め誤差があっても
    //   P が対称・半正定値に保たれやすい。また K が最適ゲインでなくても正しい式である。
    const StateMatrix A = StateMatrix::Identity() - K * H;
    P = A * P * A.transpose() + K * R_ * K.transpose();
    // 丸め誤差によるわずかな非対称性を取り除く
    P = 0.5 * (P + P.transpose()).eval();
}

void KalmanFilter::initialize(const MeasurementVector& z, double initialVelocityStd,
                              StateVector& x, StateMatrix& P) const {
    // 位置は観測値、速度は未知なので 0
    x << z.x(), z.y(), 0.0, 0.0;
    // 位置の分散 = 観測ノイズの分散、速度の分散 = (初期速度の標準偏差)²
    P = StateMatrix::Zero();
    P.topLeftCorner<2, 2>() = R_;
    P.bottomRightCorner<2, 2>() = Eigen::Matrix2d::Identity() * initialVelocityStd * initialVelocityStd;
}

}  // namespace gnn
