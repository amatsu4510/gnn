// ============================================================================
// test_step3.cpp（STEP 3：カルマンフィルタを用いたGNN のテスト）
//   STEP 2 と同じシナリオテストに、カルマンフィルタ固有のテストを追加している。
// ============================================================================

#include "IdConsistency.h"
#include "Printing.h"
#include "StandardScenarios.h"
#include "TestFramework.h"
#include "GNNAssociator.h"
#include "KalmanFilter.h"
#include "TrackManager.h"
#include "TrackingConfig.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

using namespace gnn;

namespace {

// シナリオを最後まで処理した結果
struct RunResult {
    std::vector<FrameReport> reports;
    IdConsistency ids;  // 真値IDとトラックIDの対応
};

RunResult run(const ScenarioSpec& spec, const TrackingConfig& config = TrackingConfig()) {
    RunResult result;
    TrackManager manager(config);
    for (const ScenarioFrame& sf : generateScenario(spec)) {
        result.reports.push_back(manager.process(sf.frame));
        result.ids.record(sf, result.reports.back());
    }
    return result;
}

const TrackSnapshot* findTrack(const std::vector<TrackSnapshot>& tracks, int id) {
    for (const TrackSnapshot& t : tracks) {
        if (t.id == id) return &t;
    }
    return nullptr;
}

bool contains(const std::vector<int>& v, int x) {
    return std::find(v.begin(), v.end(), x) != v.end();
}

int countConfirmed(const std::vector<TrackSnapshot>& tracks) {
    return static_cast<int>(std::count_if(tracks.begin(), tracks.end(), [](const TrackSnapshot& t) {
        return t.state != TrackState::Tentative;
    }));
}

}  // namespace

TEST_CASE("1_2_通常移動と順序入れ替えでIDを維持") {
    const RunResult r = run(scenarios::overall());
    // 一時的な誤対応も含めて、各目標がずっと同じトラックに対応している
    for (const auto& [truth, summary] : r.ids.summaries()) {
        CHECK_EQ(static_cast<int>(summary.allTrackIds.size()), 1);
    }
    CHECK_EQ(static_cast<int>(r.ids.summaries().size()), 4);
}

TEST_CASE("位置と速度の推定") {
    ScenarioSpec spec;
    spec.timestamps = uniformTimestamps(20, 0.1);
    spec.targets = {{1, 0, 19, {100, 100}, {80, -40}, {}}};
    spec.noiseStd = 0.5;
    const RunResult r = run(spec);
    const TrackSnapshot& t = r.reports.back().updated.at(0);
    CHECK(t.state == TrackState::Confirmed);
    CHECK_NEAR(t.velocity.x(), 80.0, 10.0);
    CHECK_NEAR(t.velocity.y(), -40.0, 10.0);
    CHECK_NEAR(t.position.x(), 100.0 + 80.0 * 1.9, 3.0);
    CHECK_NEAR(t.position.y(), 100.0 - 40.0 * 1.9, 3.0);
}

TEST_CASE("4_新規目標は仮トラックから確定トラックになる") {
    const RunResult r = run(scenarios::overall());
    const int id = r.ids.of(4).firstTrackId;  // 目標4（フレーム5で出現）のトラック
    CHECK(contains(r.reports[5].createdIds, id));
    CHECK(findTrack(r.reports[5].updated, id)->state == TrackState::Tentative);
    CHECK(findTrack(r.reports[6].updated, id)->state == TrackState::Tentative);
    CHECK(findTrack(r.reports[7].updated, id)->state == TrackState::Confirmed);  // 3回目の対応で確定
}

TEST_CASE("5_一時的な未検出はCoastingで維持し再検出で同じIDに戻る") {
    const RunResult r = run(scenarios::overall());
    const int id = r.ids.of(2).firstTrackId;  // 目標2（フレーム6,7で未検出）
    CHECK(findTrack(r.reports[5].updated, id)->state == TrackState::Confirmed);
    CHECK(findTrack(r.reports[6].updated, id)->state == TrackState::Coasting);
    CHECK(findTrack(r.reports[7].updated, id)->state == TrackState::Coasting);
    CHECK(findTrack(r.reports[8].updated, id)->state == TrackState::Confirmed);
    CHECK_EQ(findTrack(r.reports[7].updated, id)->consecutiveMisses, 2);
}

TEST_CASE("6_完全な消失は規定回数の未検出後に削除") {
    const RunResult r = run(scenarios::overall());
    const int id = r.ids.of(3).firstTrackId;  // 目標3（フレーム7の後に消失）
    // フレーム8〜12で5回連続未検出 → フレーム12で削除
    CHECK(findTrack(r.reports[11].updated, id) != nullptr);
    CHECK(contains(r.reports[12].deletedIds, id));
    CHECK(findTrack(r.reports[12].updated, id) == nullptr);
}

TEST_CASE("7_ランダムな誤検出から確定トラックが生まれない") {
    const RunResult r = run(scenarios::overall());
    // 最終フレームの確定トラックは、存在している目標 1, 2, 4 の3つだけ
    CHECK_EQ(countConfirmed(r.reports.back().updated), 3);
    // 誤検出から生成された仮トラックは、確定前に削除されている
    int createdTotal = 0, deletedTotal = 0;
    for (const FrameReport& rep : r.reports) {
        createdTotal += static_cast<int>(rep.createdIds.size());
        deletedTotal += static_cast<int>(rep.deletedIds.size());
    }
    CHECK(createdTotal > 4);  // 誤検出による仮トラックが実際に生成されている
    CHECK(deletedTotal > 0);
}

TEST_CASE("8_X字の交差ではIDを維持") {
    const RunResult r = run(scenarios::crossing());
    CHECK(r.ids.of(1).idKept());
    CHECK(r.ids.of(2).idKept());
}

TEST_CASE("8_浅い角度の接近ではIDが入れ替わり得る（GNNの限界）") {
    // 交差角が小さく、観測ノイズに対して相対移動量が小さいと、位置と速度だけでは
    // 2目標を区別できない。このシナリオ（シード固定）では交差後にIDが入れ替わる。
    // 「GNNでは必ずIDを維持できるわけではない」ことを確認するためのテスト
    const RunResult r = run(scenarios::closeApproach());
    CHECK(!r.ids.of(1).idKept());
    CHECK(!r.ids.of(2).idKept());
    // トラック自体は2本のまま維持される（分裂はしない）
    CHECK_EQ(static_cast<int>(r.reports.back().updated.size()), 2);
}

TEST_CASE("9_10_最明点0件のフレームではすべて未検出") {
    TrackManager manager{TrackingConfig()};
    manager.process(makeFrame(0, 0.0, {{100, 100}, {200, 200}}));
    const FrameReport empty = manager.process(makeFrame(1, 0.1, {}));
    CHECK(empty.association.matches.empty());
    CHECK_EQ(static_cast<int>(empty.association.unassignedTracks.size()), 2);
    // 仮トラックのまま未検出 → まだ削除はされない（許容回数 5−3=2 回）
    CHECK_EQ(static_cast<int>(empty.updated.size()), 2);

    TrackManager fresh{TrackingConfig()};
    const FrameReport none = fresh.process(makeFrame(0, 0.0, {}));
    CHECK(none.updated.empty());
    CHECK(none.createdIds.empty());
}

TEST_CASE("11_すべての観測がゲート範囲外") {
    TrackManager manager{TrackingConfig()};
    for (int k = 0; k < 4; ++k) {
        manager.process(makeFrame(k, 0.1 * k, {{100.0 + 5 * k, 100}}));
    }
    // 目標が急に遠くへ移動したように見える観測（ゲート外）
    const FrameReport r = manager.process(makeFrame(4, 0.4, {{400, 400}}));
    CHECK(r.association.matches.empty());
    CHECK_EQ(static_cast<int>(r.createdIds.size()), 1);                    // 新しい仮トラック
    CHECK(findTrack(r.updated, 1)->state == TrackState::Coasting);         // 既存は予測追跡
}

TEST_CASE("12_フレーム間隔が一定ではない") {
    const RunResult r = run(scenarios::unevenInterval());
    for (const auto& [truth, summary] : r.ids.summaries()) {
        CHECK_EQ(static_cast<int>(summary.allTrackIds.size()), 1);
    }
    // 速度は px/s 単位で推定されるため、間隔が変わっても真値に近い
    const int id1 = r.ids.of(1).firstTrackId;
    const TrackSnapshot* t = findTrack(r.reports.back().updated, id1);
    CHECK(t != nullptr);
    CHECK_NEAR(t->velocity.x(), 80.0, 15.0);
    CHECK_NEAR(t->velocity.y(), 40.0, 15.0);
}

TEST_CASE("時刻が戻るフレームは例外") {
    TrackManager manager{TrackingConfig()};
    manager.process(makeFrame(0, 1.0, {}));
    bool thrown = false;
    try {
        manager.process(makeFrame(1, 1.0, {}));
    } catch (const std::invalid_argument&) {
        thrown = true;
    }
    CHECK(thrown);
}

// ---------------------------------------------------------------------------
// カルマンフィルタ固有のテスト
// ---------------------------------------------------------------------------

namespace {
const KalmanFilter kTestKf(200.0, 1.0);

bool isSymmetric(const Eigen::Matrix4d& P) {
    return (P - P.transpose()).cwiseAbs().maxCoeff() < 1e-12;
}
bool isPositiveDefinite(const Eigen::Matrix4d& P) {
    // 対称行列の固有値がすべて正なら正定値
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> es(P);
    return es.eigenvalues().minCoeff() > 0.0;
}
}  // namespace

TEST_CASE("KF_行列F_H_Qの構造") {
    const Eigen::Matrix4d F = KalmanFilter::transitionMatrix(0.5);
    CHECK_NEAR(F(0, 2), 0.5, 0.0);
    CHECK_NEAR(F(1, 3), 0.5, 0.0);
    CHECK_NEAR(F.diagonal().sum(), 4.0, 0.0);

    const Eigen::Matrix<double, 2, 4> H = KalmanFilter::observationMatrix();
    Eigen::Vector4d x(1, 2, 3, 4);
    CHECK((H * x).isApprox(Eigen::Vector2d(1, 2)));

    const Eigen::Matrix4d Q1 = kTestKf.processNoise(0.1);
    const Eigen::Matrix4d Q2 = kTestKf.processNoise(0.2);
    CHECK(isSymmetric(Q1));
    CHECK(isPositiveDefinite(Q1));
    // Δt が2倍なら 位置の分散は 8倍（Δt³）、速度の分散は 2倍（Δt）
    CHECK_NEAR(Q2(0, 0) / Q1(0, 0), 8.0, 1e-9);
    CHECK_NEAR(Q2(2, 2) / Q1(2, 2), 2.0, 1e-9);
    CHECK_NEAR(Q1(0, 1), 0.0, 0.0);  // x 軸と y 軸は独立
}

TEST_CASE("KF_予測で状態が進み共分散が増える") {
    Eigen::Vector4d x(10, 20, 100, -50);
    Eigen::Matrix4d P = Eigen::Matrix4d::Identity();
    kTestKf.predict(x, P, 0.1);
    CHECK((x - Eigen::Vector4d(20, 15, 100, -50)).norm() < 1e-12);
    CHECK(P(0, 0) > 1.0);
    CHECK(isSymmetric(P));
}

TEST_CASE("KF_カルマンゲインが定義式と一致し_Joseph形式で対称正定値を保つ") {
    Eigen::Vector4d x;
    Eigen::Matrix4d P;
    kTestKf.initialize(Eigen::Vector2d(100, 100), 100.0, x, P);
    kTestKf.predict(x, P, 0.1);

    // 検算用：逆行列を使った定義どおりの計算（実装では逆行列を使っていない）
    const Eigen::Matrix<double, 2, 4> H = KalmanFilter::observationMatrix();
    const Eigen::Matrix2d S = H * P * H.transpose() + kTestKf.measurementNoise();
    const Eigen::Matrix<double, 4, 2> Kref = P * H.transpose() * S.inverse();
    const Eigen::Vector2d z(105, 98);
    const Eigen::Vector4d xRef = x + Kref * (z - H * x);
    const Eigen::Matrix4d PRef = (Eigen::Matrix4d::Identity() - Kref * H) * P;  // 単純形式（最適ゲインなら Joseph 形式と一致）

    kTestKf.update(x, P, z);
    CHECK((x - xRef).norm() < 1e-9);
    CHECK((P - PRef).cwiseAbs().maxCoeff() < 1e-9);
    CHECK(isSymmetric(P));
    CHECK(isPositiveDefinite(P));

    // 更新を繰り返しても対称・正定値が保たれる
    for (int k = 0; k < 200; ++k) {
        kTestKf.predict(x, P, 0.1);
        kTestKf.update(x, P, Eigen::Vector2d(105 + k, 98 - k));
        CHECK(isSymmetric(P));
        CHECK(isPositiveDefinite(P));
    }
}

TEST_CASE("KF_マハラノビス距離とカイ二乗ゲート閾値") {
    PredictedMeasurement pm;
    pm.z = Eigen::Vector2d(0, 0);
    pm.S << 4, 1,
            1, 2;
    Detection d;
    d.position = Eigen::Vector2d(3, -1);
    const Eigen::MatrixXd d2 = GNNAssociator::computeMahalanobisMatrix({pm}, {d});
    const Eigen::Vector2d y = d.position - pm.z;
    CHECK_NEAR(d2(0, 0), y.dot(pm.S.inverse() * y), 1e-12);  // 逆行列による検算

    TrackingConfig config;
    config.gateProbability = 0.99;
    CHECK_NEAR(config.gateThreshold(), 9.2103, 1e-4);   // 資料の値
    config.gateProbability = 0.999;
    CHECK_NEAR(config.gateThreshold(), 13.8155, 1e-4);
}

TEST_CASE("KF_正定値でないSは例外") {
    PredictedMeasurement pm;
    pm.S << 1, 2,
            2, 1;  // 固有値 3, -1
    Detection d;
    bool thrown = false;
    try {
        GNNAssociator::computeMahalanobisMatrix({pm}, {d});
    } catch (const std::runtime_error&) {
        thrown = true;
    }
    CHECK(thrown);
}

TEST_CASE("KF_未検出が続くとSが大きくなりゲートが広がる") {
    const RunResult r = run(scenarios::overall());
    const int id = r.ids.of(2).firstTrackId;  // フレーム6,7で未検出
    const double before = findTrack(r.reports[6].predicted, id)->residualStd.x();
    const double during = findTrack(r.reports[7].predicted, id)->residualStd.x();
    const double after = findTrack(r.reports[8].updated, id)->residualStd.x();
    CHECK(during > before);  // 未検出中は予測のみで不確かさが増える
    CHECK(after < during);   // 再検出で更新されると小さくなる
}

TEST_CASE("KF_一貫性_NISの平均が自由度2に近い") {
    // 正規化イノベーション二乗（NIS）d² は、モデルが正しければ平均が観測次元 2 になる
    ScenarioSpec spec;
    spec.timestamps = uniformTimestamps(400, 0.1);
    spec.targets = {{1, 0, 399, {50, 50}, {1, 1}, {}}};  // 画像外に出ないようにゆっくり動かす
    spec.noiseStd = 1.0;
    const RunResult r = run(spec);
    double sum = 0.0;
    int count = 0;
    for (std::size_t k = 50; k < r.reports.size(); ++k) {  // 収束後を評価
        for (const Match& m : r.reports[k].association.matches) {
            sum += m.cost;
            ++count;
        }
    }
    const double meanNis = sum / count;
    CHECK_EQ(count, 350);
    // 真の運動は等速（加速度0）なので、q > 0 の分だけ S が大きめになり、平均は 2 よりやや小さくなる
    CHECK(meanNis > 1.2);
    CHECK(meanNis < 2.3);
}

int main() {
    setupConsole();
    return gnn::test::runAllTests();
}
