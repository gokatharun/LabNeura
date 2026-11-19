#pragma once

#include "support/eda_netlist_evaluator.h"

#include <map>
#include <string>
#include <vector>

namespace labneura::eda::test {

// One externally driven input snapshot. The first frame establishes the
// initial input levels; each later frame applies one atomic input transition.
struct SequentialStimulusFrame {
    VerificationPortValues inputs;
};

// Initial state is explicit and keyed by DFF/DFFR/DFFS instance name. Each
// supported register cell is one bit. No default value is assumed.
struct SequentialEvaluationOptions {
    std::map<std::string, std::uint8_t> initial_state;
    std::size_t max_frames = 1024;
    std::size_t max_cell_evaluations_per_frame = 100000;
};

struct SequentialTraceFrame {
    VerificationPortValues outputs;
    std::map<std::string, std::uint8_t> state;
};

struct SequentialEvaluationResult {
    NetlistVerificationStatus status = NetlistVerificationStatus::InvalidNetlist;
    std::vector<SequentialTraceFrame> trace;
    std::vector<NetlistVerificationDiagnostic> diagnostics;

    bool succeeded() const noexcept {
        return status == NetlistVerificationStatus::Succeeded;
    }
};

// Test-only bounded transition evaluation for the native lowering's DFF,
// DFFR, and DFFS cells. Each frame is evaluated against old state; all register
// updates caused by that transition are committed simultaneously. This models
// no general SystemVerilog event regions or arbitrary scheduler behavior.
SequentialEvaluationResult evaluate_sequential_netlist(
    const Module& module, const std::vector<SequentialStimulusFrame>& frames,
    const SequentialEvaluationOptions& options);

} // namespace labneura::eda::test
