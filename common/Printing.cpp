// ============================================================================
// Printing.cpp
// ============================================================================

#include "Printing.h"

#include <iomanip>
#include <iostream>
#include <sstream>

#ifdef _WIN32
// windows.h の min/max マクロが std::min や Eigen と衝突しないようにする
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace gnn {

void setupConsole() {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
}

void printHeading(const std::string& title) {
    std::cout << "\n==================================================================\n"
              << title << "\n"
              << "==================================================================\n";
}

std::string formatVec(double x, double y, int precision) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(precision) << "(" << std::setw(7) << x << ", " << std::setw(7) << y
       << ")";
    return os.str();
}

void printDetections(const Frame& frame, const std::string& label) {
    std::cout << "[入力] 最明点 " << frame.detections.size() << " 個 (時刻 " << std::fixed
              << std::setprecision(3) << frame.timestamp << " s)\n";
    for (std::size_t j = 0; j < frame.detections.size(); ++j) {
        const Detection& d = frame.detections[j];
        std::cout << "  " << label << j << " : " << formatVec(d.position.x(), d.position.y());
        if (d.brightness) {
            std::cout << "  輝度 " << std::setprecision(0) << *d.brightness;
        }
        std::cout << "\n";
    }
}

namespace {
// 列見出し（Z0 Z1 ...）を表示する
void printColumnHeader(int cols, int labelWidth, int cellWidth) {
    std::cout << std::string(labelWidth, ' ');
    for (int j = 0; j < cols; ++j) {
        std::cout << std::setw(cellWidth) << ("Z" + std::to_string(j));
    }
    std::cout << "\n";
}
}  // namespace

void printGateMask(const AssociationResult& result, const std::vector<std::string>& rowLabels) {
    const int N = static_cast<int>(result.gateMask.rows());
    const int M = static_cast<int>(result.gateMask.cols());
    std::cout << "[ゲーティング] O = ゲート内（候補）, - = ゲート外（割り当て禁止）\n";
    if (N == 0 || M == 0) {
        std::cout << "  （トラックまたは最明点が0件のため判定なし）\n";
        return;
    }
    printColumnHeader(M, 8, 5);
    for (int i = 0; i < N; ++i) {
        std::cout << std::setw(8) << rowLabels[i];
        for (int j = 0; j < M; ++j) {
            // 全角記号は環境によって表示幅が変わるため、半角の O / - で表示する
            std::cout << std::setw(5) << (result.gateMask(i, j) ? "O" : "-");
        }
        std::cout << "\n";
    }
}

void printCostMatrix(const AssociationResult& result, const std::vector<std::string>& rowLabels) {
    const int N = static_cast<int>(result.costMatrix.rows());
    const int M = static_cast<int>(result.costMatrix.cols());
    std::cout << "[コスト行列] " << N << "×" << M << "（* = ゲート外。割り当てには使われない）\n";
    if (N == 0 || M == 0) {
        std::cout << "  （空）\n";
        return;
    }
    printColumnHeader(M, 8, 10);
    for (int i = 0; i < N; ++i) {
        std::cout << std::setw(8) << rowLabels[i];
        for (int j = 0; j < M; ++j) {
            std::ostringstream cell;
            cell << std::fixed << std::setprecision(2) << result.costMatrix(i, j)
                 << (result.gateMask(i, j) ? " " : "*");
            std::cout << std::setw(10) << cell.str();
        }
        std::cout << "\n";
    }
}

void printAssignment(const AssociationResult& result,
                     const std::vector<std::string>& rowLabels,
                     double missCost,
                     double falseAlarmCost) {
    std::cout << "[GNN対応付け]\n";
    if (result.matches.empty()) {
        std::cout << "  （対応付けなし）\n";
    }
    for (const Match& m : result.matches) {
        std::cout << "  " << rowLabels[m.trackIndex] << " -> Z" << m.detectionIndex << "  コスト "
                  << std::fixed << std::setprecision(3) << m.cost << "\n";
    }
    std::cout << std::fixed << std::setprecision(3) << "[合計割り当てコスト] " << result.matchedCost
              << "  （未検出 " << missCost << "/件・未割り当て観測 " << falseAlarmCost
              << "/件 を含む目的関数値 " << result.totalCost() << "）\n";

    std::cout << "[未割り当てトラック] ";
    if (result.unassignedTracks.empty()) {
        std::cout << "なし";
    }
    for (int i : result.unassignedTracks) {
        std::cout << rowLabels[i] << " ";
    }
    std::cout << "\n[未割り当て最明点] ";
    if (result.unassignedDetections.empty()) {
        std::cout << "なし";
    }
    for (int j : result.unassignedDetections) {
        std::cout << "Z" << j << " ";
    }
    std::cout << "\n";
}

}  // namespace gnn
