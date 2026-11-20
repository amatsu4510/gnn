// ============================================================================
// main.cpp（STEP 3：カルマンフィルタを用いたGNN）
//   トラックIDを維持しながら、各フレームの処理内容をすべて表示する。
//   引数でシナリオを選べる： overall（既定）/ crossing / approach / uneven
// ============================================================================

#include "IdConsistency.h"
#include "Printing.h"
#include "ReportPrinter.h"
#include "StandardScenarios.h"
#include "TrackManager.h"
#include "TrackingConfig.h"

#include <iostream>
#include <string>

using namespace gnn;

namespace {

ScenarioSpec selectScenario(const std::string& name) {
    if (name == "crossing") return scenarios::crossing();
    if (name == "approach") return scenarios::closeApproach();
    if (name == "uneven") return scenarios::unevenInterval();
    return scenarios::overall();
}

}  // namespace

int main(int argc, char** argv) {
    setupConsole();

    const ScenarioSpec spec = selectScenario(argc > 1 ? argv[1] : "overall");
    printHeading("STEP 3：" + spec.name + "（カルマンフィルタ + マハラノビス距離 + GNN）");

    const TrackingConfig config;
    TrackManager manager(config);

    // 真値IDとトラックIDの対応（IDが維持されたかの確認用）
    IdConsistency ids;

    for (const ScenarioFrame& sf : generateScenario(spec)) {
        const FrameReport report = manager.process(sf.frame);
        printFrameReport(report, config);
        ids.record(sf, report);
    }

    printHeading("まとめ：真値の目標ごとのトラックID（最初と最後が同じならID維持）");
    for (const auto& [truthId, summary] : ids.summaries()) {
        std::cout << "  目標 " << truthId << " : 最初 T" << summary.firstTrackId << " / 最後 T"
                  << summary.lastTrackId << "  対応したID {";
        for (int id : summary.allTrackIds) {
            std::cout << " T" << id;
        }
        std::cout << " }  別IDに対応したフレーム " << summary.otherTrackFrames << "/" << summary.matchedFrames
                  << (summary.idKept() ? "  → ID維持" : "  → IDが入れ替わった") << "\n";
    }
    return 0;
}
