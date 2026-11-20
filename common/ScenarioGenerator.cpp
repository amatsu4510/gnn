// ============================================================================
// ScenarioGenerator.cpp
// ============================================================================

#include "ScenarioGenerator.h"

#include <algorithm>
#include <cmath>
#include <utility>

namespace gnn {

namespace {
constexpr double kPi = 3.14159265358979323846;  // M_PI は MSVC で標準では定義されないため自前で定義
}

double Rng::uniform01() {
    // mt19937 の 32bit 出力を [0, 1) に変換（0.5 を足して 0 ちょうどを避ける）
    return (static_cast<double>(engine_()) + 0.5) / 4294967296.0;
}

double Rng::uniform(double lo, double hi) {
    return lo + (hi - lo) * uniform01();
}

int Rng::uniformInt(int lo, int hi) {
    const int span = hi - lo + 1;
    const int k = static_cast<int>(uniform01() * span);
    return lo + std::min(k, span - 1);
}

double Rng::normal(double mean, double stddev) {
    // Box–Muller 法（毎回2つの一様乱数から1つの正規乱数を作る単純な形）
    const double u1 = uniform01();
    const double u2 = uniform01();
    const double z = std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * kPi * u2);
    return mean + stddev * z;
}

std::vector<double> uniformTimestamps(int frameCount, double dt, double t0) {
    std::vector<double> t(frameCount);
    for (int k = 0; k < frameCount; ++k) {
        t[k] = t0 + dt * k;
    }
    return t;
}

std::vector<ScenarioFrame> generateScenario(const ScenarioSpec& spec) {
    Rng rng(spec.seed);
    std::vector<ScenarioFrame> frames;

    const int frameCount = static_cast<int>(spec.timestamps.size());
    for (int k = 0; k < frameCount; ++k) {
        const double t = spec.timestamps[k];
        ScenarioFrame sf;
        sf.frame.frameNumber = k;
        sf.frame.timestamp = t;

        // ---- 目標の観測 ----
        for (const TargetSpec& target : spec.targets) {
            if (k < target.firstFrame || k > target.lastFrame) {
                continue;  // 未出現 or 消失済み
            }
            const bool missed = std::find(target.missedFrames.begin(), target.missedFrames.end(), k) !=
                                target.missedFrames.end();
            // 未検出のフレームでも乱数を消費しておき、他の目標の値が変わらないようにする
            const double nx = rng.normal(0.0, spec.noiseStd);
            const double ny = rng.normal(0.0, spec.noiseStd);
            if (missed) {
                continue;
            }
            const double elapsed = t - spec.timestamps[target.firstFrame];
            Detection d;
            d.position = target.initialPosition + target.velocity * elapsed + Eigen::Vector2d(nx, ny);
            d.timestamp = t;
            d.brightness = target.brightness;
            sf.frame.detections.push_back(d);
            sf.truthIds.push_back(target.truthId);
        }

        // ---- ランダムな誤検出（クラッタ）----
        const int clutterCount = spec.maxClutterPerFrame > 0 ? rng.uniformInt(0, spec.maxClutterPerFrame) : 0;
        for (int c = 0; c < clutterCount; ++c) {
            Detection d;
            d.position = Eigen::Vector2d(rng.uniform(0.0, spec.imageWidth), rng.uniform(0.0, spec.imageHeight));
            d.timestamp = t;
            d.brightness = rng.uniform(50.0, 120.0);
            sf.frame.detections.push_back(d);
            sf.truthIds.push_back(-1);
        }

        // ---- 検出順序のシャッフル（Fisher–Yates）----
        if (spec.shuffle) {
            for (int i = static_cast<int>(sf.frame.detections.size()) - 1; i > 0; --i) {
                const int j = rng.uniformInt(0, i);
                std::swap(sf.frame.detections[i], sf.frame.detections[j]);
                std::swap(sf.truthIds[i], sf.truthIds[j]);
            }
        }
        frames.push_back(std::move(sf));
    }
    return frames;
}

Frame makeFrame(int frameNumber, double timestamp, const std::vector<Eigen::Vector2d>& points) {
    Frame frame;
    frame.frameNumber = frameNumber;
    frame.timestamp = timestamp;
    for (const Eigen::Vector2d& p : points) {
        Detection d;
        d.position = p;
        d.timestamp = timestamp;
        frame.detections.push_back(d);
    }
    return frame;
}

}  // namespace gnn
