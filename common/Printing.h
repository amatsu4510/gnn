#pragma once
// ============================================================================
// Printing.h
//   コンソール出力の補助関数。アルゴリズム本体から入出力を切り離すために分けている。
// ============================================================================

#include "AssociationResult.h"
#include "Detection.h"

#include <string>
#include <vector>

namespace gnn {

// Windows のコンソールで UTF-8 の日本語を表示できるようにする（他のOSでは何もしない）
void setupConsole();

// 見出しを表示する
void printHeading(const std::string& title);

// 入力された最明点の一覧を表示する（label0, label1, ... のラベルを付ける）
void printDetections(const Frame& frame, const std::string& label = "Z");

// ゲーティング結果を表示する（O = ゲート内, - = ゲート外）
void printGateMask(const AssociationResult& result,
                   const std::vector<std::string>& rowLabels);

// コスト行列を表示する（ゲート外は値の後ろに * を付ける）
void printCostMatrix(const AssociationResult& result,
                     const std::vector<std::string>& rowLabels);

// 対応付け結果・合計コスト・未割り当てを表示する
void printAssignment(const AssociationResult& result,
                     const std::vector<std::string>& rowLabels,
                     double missCost,
                     double falseAlarmCost);

// 2次元ベクトルを "(x, y)" 形式の文字列にする
std::string formatVec(double x, double y, int precision = 1);

}  // namespace gnn
