// ============================================================================
// ReportPrinter.cpp（STEP 2）
// ============================================================================

#include "ReportPrinter.h"

#include "Printing.h"

#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace gnn {

namespace {

void printTracks(const std::vector<TrackSnapshot>& tracks) {
    if (tracks.empty()) {
        std::cout << "  （トラックなし）\n";
    }
    for (const TrackSnapshot& t : tracks) {
        std::cout << "  T" << std::left << std::setw(3) << t.id << std::right << " 位置 "
                  << formatVec(t.position.x(), t.position.y()) << "  速度 "
                  << formatVec(t.velocity.x(), t.velocity.y()) << " px/s  " << std::left << std::setw(9)
                  << toString(t.state) << std::right << "  連続未検出 " << t.consecutiveMisses
                  << "  直近対応 " << t.hitsInWindow << "\n";
    }
}

void printIds(const char* title, const std::vector<int>& ids) {
    std::cout << title;
    if (ids.empty()) {
        std::cout << "なし";
    }
    for (int id : ids) {
        std::cout << "T" << id << " ";
    }
    std::cout << "\n";
}

}  // namespace

void printFrameReport(const FrameReport& report, const TrackingConfig& config) {
    std::cout << "\n################ フレーム " << report.frame.frameNumber << "  (Δt = " << std::fixed
              << std::setprecision(3) << report.dt << " s) ################\n";
    printDetections(report.frame);

    std::cout << "[予測位置] 等速直線運動 p + v·Δt\n";
    printTracks(report.predicted);

    std::vector<std::string> labels;
    for (const TrackSnapshot& t : report.predicted) {
        labels.push_back("T" + std::to_string(t.id));
    }
    printGateMask(report.association, labels);
    printCostMatrix(report.association, labels);
    printAssignment(report.association, labels, config.missCost, config.falseAlarmCost);

    std::cout << "[更新後のトラック]\n";
    printTracks(report.updated);
    printIds("[新規生成] ", report.createdIds);
    printIds("[削除] ", report.deletedIds);
}

}  // namespace gnn
