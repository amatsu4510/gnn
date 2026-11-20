// ============================================================================
// HungarianSolver.cpp
//   ポテンシャル（双対変数）と最短増加路を用いるハンガリアン法の実装。
//   （Kuhn–Munkres 法を Jonker–Volgenant 風に O(n² m) で実装した形式）
// ============================================================================

#include "HungarianSolver.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace gnn {

AssignmentSolution HungarianSolver::solve(const Eigen::MatrixXd& cost) const {
    const int rows = static_cast<int>(cost.rows());
    const int cols = static_cast<int>(cost.cols());

    AssignmentSolution solution;
    solution.rowToCol.assign(rows, -1);
    solution.colToRow.assign(cols, -1);

    // ---- 空の入力：何も割り当てない ----
    if (rows == 0 || cols == 0) {
        return solution;
    }

    // ---- 入力チェックと「禁止コスト」の決定 ----
    //   禁止要素（+∞）を十分大きな有限値 bigCost に置き換える。
    //   bound = Σ_i max_j |c_ij|（有限要素のみ）とすると、
    //   禁止要素を使わない任意の割り当てのコストは bound 以下、
    //   禁止要素を1つでも使う割り当てのコストは bigCost - bound 以上。
    //   bigCost = 2·bound + 1 とすれば、禁止要素を使わない解が存在する限り必ずそちらが選ばれる。
    double bound = 0.0;
    for (int i = 0; i < rows; ++i) {
        double rowMax = 0.0;
        for (int j = 0; j < cols; ++j) {
            const double c = cost(i, j);
            if (std::isnan(c) || c == -std::numeric_limits<double>::infinity()) {
                throw std::invalid_argument("HungarianSolver: cost に NaN または -inf が含まれています");
            }
            if (std::isfinite(c)) {
                rowMax = std::max(rowMax, std::abs(c));
            }
        }
        bound += rowMax;
    }
    const double bigCost = 2.0 * bound + 1.0;

    // 禁止要素を有限値に置き換えた作業用行列
    //   unaryExpr でラムダを各要素に適用する。
    const Eigen::MatrixXd work =
        cost.unaryExpr([bigCost](double c) { return std::isfinite(c) ? c : bigCost; });

    // ---- 行数 > 列数 なら転置して「行数 ≤ 列数」の形にする ----
    const bool transposed = rows > cols;
    std::vector<int> assignment;
    if (transposed) {
        // work.transpose() は式テンプレート（遅延評価）なので、MatrixXd に代入して実体化する
        const Eigen::MatrixXd workT = work.transpose();
        assignment = solveRowsLeCols(workT);  // 列 → 行 の対応
    } else {
        assignment = solveRowsLeCols(work);   // 行 → 列 の対応
    }

    // ---- 結果を行・列の両方向で整理（禁止要素への割り当ては未割り当て扱い）----
    for (int k = 0; k < static_cast<int>(assignment.size()); ++k) {
        const int i = transposed ? assignment[k] : k;
        const int j = transposed ? k : assignment[k];
        if (i < 0 || j < 0) {
            continue;
        }
        if (!std::isfinite(cost(i, j))) {
            continue;  // 禁止要素：割り当てない
        }
        solution.rowToCol[i] = j;
        solution.colToRow[j] = i;
        solution.totalCost += cost(i, j);
    }
    return solution;
}

// ----------------------------------------------------------------------------
// 行数 n ≤ 列数 m の行列に対するハンガリアン法
//
//   双対問題のポテンシャル u(行), v(列) を保ち、常に
//       c(i, j) - u(i) - v(j) ≥ 0     （被約コストが非負）
//   を満たす。被約コストが 0 の辺だけで構成される最短増加路を
//   Dijkstra 法と同様の手順で探し、行を1つずつ割り当てに追加していく。
//
//   配列は 1 始まりで使い、添字 0 は「仮想の列」（探索の起点）を表す。
//     u[i]    : 行 i のポテンシャル
//     v[j]    : 列 j のポテンシャル
//     p[j]    : 列 j に割り当てられている行（0 = 未割り当て）
//     way[j]  : 増加路で列 j の直前に通った列
//     minv[j] : 現在の探索木から列 j までの最小被約コスト
//     used[j] : 列 j が探索木に含まれているか
//
//   計算量：行を1つ追加するごとに最大 n 回の反復 × 各 O(m) → 全体 O(n² m)。
// ----------------------------------------------------------------------------
std::vector<int> HungarianSolver::solveRowsLeCols(const Eigen::MatrixXd& cost) {
    const int n = static_cast<int>(cost.rows());
    const int m = static_cast<int>(cost.cols());

    // 「まだ値が入っていない」ことを表す初期値。比較にのみ使い、算術演算には使わない。
    //   ※ 内側ループで未訪問の全列の minv が有限値で上書きされるため、
    //     minv[j] -= delta が初期値に対して実行されることはない。
    //     また n ≤ m なので未訪問列が必ず残り、delta は常に有限値になる。
    const double kUnset = std::numeric_limits<double>::max();

    std::vector<double> u(n + 1, 0.0), v(m + 1, 0.0);
    std::vector<int> p(m + 1, 0), way(m + 1, 0);

    for (int i = 1; i <= n; ++i) {
        // 行 i を仮想列 0 に置いて探索を開始する
        p[0] = i;
        int j0 = 0;
        std::vector<double> minv(m + 1, kUnset);
        std::vector<bool> used(m + 1, false);

        // ---- 未割り当ての列に到達するまで最短増加路を延ばす ----
        do {
            used[j0] = true;
            const int i0 = p[j0];
            double delta = kUnset;
            int j1 = 0;
            for (int j = 1; j <= m; ++j) {
                if (used[j]) {
                    continue;
                }
                const double reduced = cost(i0 - 1, j - 1) - u[i0] - v[j];  // 被約コスト
                if (reduced < minv[j]) {
                    minv[j] = reduced;
                    way[j] = j0;
                }
                // 厳密な「<」で比較するため、同コストなら添字の小さい列が選ばれる（決定的）
                if (minv[j] < delta) {
                    delta = minv[j];
                    j1 = j;
                }
            }
            // ポテンシャルを更新し、探索木内の被約コストを 0 に保ったまま
            // 新しい列 j1 への被約コストを 0 にする
            for (int j = 0; j <= m; ++j) {
                if (used[j]) {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
        } while (p[j0] != 0);

        // ---- 増加路に沿って割り当てを入れ替える ----
        do {
            const int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0 != 0);
    }

    // p（列 → 行）を 行 → 列 に変換（0 始まりに戻す）
    std::vector<int> rowToCol(n, -1);
    for (int j = 1; j <= m; ++j) {
        if (p[j] != 0) {
            rowToCol[p[j] - 1] = j - 1;
        }
    }
    return rowToCol;
}

}  // namespace gnn
