#pragma once
// ============================================================================
// ReportPrinter.h（STEP 3）
//   TrackManager が返した FrameReport をコンソールに表示する。
// ============================================================================

#include "TrackManager.h"
#include "TrackingConfig.h"

namespace gnn {

void printFrameReport(const FrameReport& report, const TrackingConfig& config);

}  // namespace gnn
