#pragma once
// ============================================================================
// GatedAssignment.h
//   ゲーティング済みのコスト行列（N×M）に「ダミー行・ダミー列」を追加して
//   (N+M)×(N+M) の拡張行列を作り、ハンガリアン法で解く。
//   未割り当て（未検出・誤検出）を許容するGNNの割り当て部分。
//
//   拡張行列の構造（F = 割り当て禁止）
//
//                 最明点 0..M-1           未検出ダミー 0..N-1
//            ┌──────────────────────┬──────────────────────┐
//   トラック │  c_ij（ゲート外は F）│  対角 = missCost     │
//   0..N-1   │                      │  非対角 = F          │
//            ├──────────────────────┼──────────────────────┤
//   未割当   │  対角 = falseAlarm-  │                      │
//   ダミー   │         Cost         │  すべて 0            │
//   0..M-1   │  非対角 = F          │                      │
//            └──────────────────────┴──────────────────────┘
//
//   - トラック i が右上の対角に割り当てられる  → トラック i は未検出
//   - ダミー行 j が左下の対角に割り当てられる  → 最明点 j は未割り当て
//   - 右下（ダミー同士）はコスト 0 で、余った組み合わせを吸収する
//   全トラックを未検出、全最明点を未割り当てにする解が常に存在するため、
//   禁止要素が選ばれることはない。
// ============================================================================

#include "AssociationResult.h"
#include "HungarianSolver.h"

#include <Eigen/Dense>

namespace gnn {

// costMatrix : N×M のコスト行列
// gateMask   : N×M のゲート判定（false の組は割り当て禁止）
// missCost   : トラックが未検出となるコスト
// falseAlarmCost : 最明点が未割り当て（誤検出・新規目標）となるコスト
//
// 戻り値の costMatrix / gateMask には入力をそのまま格納する。
AssociationResult solveGatedAssignment(const Eigen::MatrixXd& costMatrix,
                                       const GateMask& gateMask,
                                       double missCost,
                                       double falseAlarmCost,
                                       const HungarianSolver& solver = HungarianSolver());

}  // namespace gnn
