#pragma once
// ============================================================================
// Detection.h
//   画像解析で抽出された「局所最明点」1点分の情報と、1フレーム分の入力をまとめる。
//   全STEPで共通に使用する入力データ型。
// ============================================================================

#include <Eigen/Dense>

#include <optional>
#include <vector>

namespace gnn {

// 最明点（観測）1点分の情報
struct Detection {
    Eigen::Vector2d position = Eigen::Vector2d::Zero();  // 画像座標 [x, y]ᵀ（ピクセル）
    double timestamp = 0.0;                              // フレームの取得時刻 [s]
    std::optional<double> brightness;                    // 輝度（任意）。保持するだけでコストには使わない
};

// 1フレーム分の入力
//   検出個数・検出順序はフレームごとに変化してよい。
struct Frame {
    int frameNumber = 0;                // フレーム番号
    double timestamp = 0.0;             // フレームの取得時刻 [s]
    std::vector<Detection> detections;  // そのフレームで抽出された最明点
};

}  // namespace gnn
