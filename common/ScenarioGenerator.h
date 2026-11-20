#pragma once
// ============================================================================
// ScenarioGenerator.h
//   画像の代わりに「最明点の座標」を生成する疑似データ生成器。
//   等速直線運動する目標に観測ノイズ・未検出・誤検出（クラッタ）・検出順序の
//   入れ替えを加えたフレーム列を作る。
//
//   【再現性について】
//   std::normal_distribution 等の分布クラスは、出力値の計算方法が標準ライブラリの
//   実装ごとに異なる（MSVC と GCC で値が変わる）。std::mt19937 の生成する整数列だけは
//   規格で決まっているため、一様乱数・正規乱数は Rng クラスで自前に変換している。
//   これにより、シードを固定すれば Visual Studio でも GCC でも同じデータになる。
// ============================================================================

#include "Detection.h"

#include <Eigen/Dense>

#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace gnn {

// 処理系に依存しない乱数生成器
class Rng {
public:
    explicit Rng(std::uint32_t seed) : engine_(seed) {}

    double uniform01();                         // [0, 1) の一様乱数
    double uniform(double lo, double hi);       // [lo, hi) の一様乱数
    int uniformInt(int lo, int hi);             // [lo, hi] の整数一様乱数
    double normal(double mean, double stddev);  // 正規乱数（Box–Muller 法）

private:
    std::mt19937 engine_;
};

// 1目標の動き
struct TargetSpec {
    int truthId = 0;                          // 真値のID（評価用）
    int firstFrame = 0;                       // 出現するフレーム番号
    int lastFrame = 0;                        // 最後に存在するフレーム番号（この後は消失）
    Eigen::Vector2d initialPosition = Eigen::Vector2d::Zero();  // firstFrame 時点の位置 [px]
    Eigen::Vector2d velocity = Eigen::Vector2d::Zero();         // 速度 [px/s]
    std::vector<int> missedFrames;            // 未検出にするフレーム番号
    double brightness = 200.0;                // 輝度（保持するだけ）
};

// シナリオ全体の設定
struct ScenarioSpec {
    std::string name;
    std::vector<double> timestamps;  // 各フレームの取得時刻 [s]（不等間隔も可）
    std::vector<TargetSpec> targets;
    double noiseStd = 1.0;           // 観測ノイズの標準偏差 [px]
    int maxClutterPerFrame = 0;      // 1フレームあたりの誤検出数の最大値（0〜この値を一様に選ぶ）
    double imageWidth = 640.0;       // 誤検出を発生させる画像範囲 [px]
    double imageHeight = 480.0;
    bool shuffle = true;             // 検出順序をフレームごとにシャッフルするか
    std::uint32_t seed = 12345;      // 乱数シード（固定して再現性を確保）
};

// 生成された1フレーム
struct ScenarioFrame {
    Frame frame;
    std::vector<int> truthIds;  // frame.detections[k] の真値ID（誤検出は -1）。評価専用
};

// 一定間隔の時刻列を作る
std::vector<double> uniformTimestamps(int frameCount, double dt, double t0 = 0.0);

// シナリオを生成する
std::vector<ScenarioFrame> generateScenario(const ScenarioSpec& spec);

// 座標を直接指定してフレームを作る（小さな手作りテスト用）
Frame makeFrame(int frameNumber, double timestamp, const std::vector<Eigen::Vector2d>& points);

}  // namespace gnn
