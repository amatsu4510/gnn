#pragma once
// ============================================================================
// StandardScenarios.h
//   STEP 2 / STEP 3 のデモとテストで共通に使う疑似データのシナリオ。
//   同じシナリオを使うことで、α-β フィルタとカルマンフィルタの結果を比較できる。
// ============================================================================

#include "ScenarioGenerator.h"

namespace gnn::scenarios {

// 総合シナリオ（通常移動・順序入れ替え・新規出現・一時的な未検出・消失・誤検出）
inline ScenarioSpec overall() {
    ScenarioSpec spec;
    spec.name = "総合シナリオ";
    spec.timestamps = uniformTimestamps(16, 0.1);
    spec.targets = {
        {1, 0, 15, {100, 100}, {60, 20}, {}},       // 通常移動
        {2, 0, 15, {300, 250}, {-40, 50}, {6, 7}},  // フレーム6,7で一時的に未検出
        {3, 0, 7, {500, 120}, {0, 70}, {}},         // フレーム7の後に完全に消失
        {4, 5, 15, {150, 400}, {70, -10}, {}},      // フレーム5で新規出現
    };
    spec.noiseStd = 1.0;
    spec.maxClutterPerFrame = 2;
    spec.seed = 12;
    return spec;
}

// 2目標の交差（X字に交差。交差時刻 t = 1.0 s で同じ位置になる）
inline ScenarioSpec crossing() {
    ScenarioSpec spec;
    spec.name = "交差シナリオ";
    spec.timestamps = uniformTimestamps(21, 0.1);
    spec.targets = {
        {1, 0, 20, {100, 100}, {100, 60}, {}},
        {2, 0, 20, {100, 220}, {100, -60}, {}},
    };
    spec.noiseStd = 1.0;
    spec.seed = 3;
    return spec;
}

// 2目標が浅い角度で接近してすれ違う（GNN の限界を示す難しいケース）
//   y 方向の間隔 9 px から ±3 px/s で近づき、t = 1.5 s で交差する（交差角 約3.4°）。
//   観測ノイズ（σ = 1 px）に対して1フレームあたりの相対移動量（0.6 px）が小さいため、
//   どちらの目標がどちらの観測かを位置と速度から区別できない。
inline ScenarioSpec closeApproach() {
    ScenarioSpec spec;
    spec.name = "浅い角度の接近シナリオ";
    spec.timestamps = uniformTimestamps(31, 0.1);
    spec.targets = {
        {1, 0, 30, {100, 195.5}, {100, 3}, {}},   // ゆっくり下へ
        {2, 0, 30, {100, 204.5}, {100, -3}, {}},  // ゆっくり上へ
    };
    spec.noiseStd = 1.0;
    spec.seed = 8;
    return spec;
}

// フレーム間隔が一定ではない
inline ScenarioSpec unevenInterval() {
    ScenarioSpec spec;
    spec.name = "不等間隔シナリオ";
    spec.timestamps = {0.0, 0.1, 0.15, 0.35, 0.40, 0.60, 0.62, 0.80, 1.00, 1.05, 1.25, 1.30};
    spec.targets = {
        {1, 0, 11, {100, 100}, {80, 40}, {}},
        {2, 0, 11, {400, 300}, {-60, -30}, {}},
    };
    spec.noiseStd = 1.0;
    spec.seed = 5;
    return spec;
}

}  // namespace gnn::scenarios
