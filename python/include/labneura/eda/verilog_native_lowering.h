#pragma once

#include "labneura/eda/netlist.h"
#include "labneura/eda/verilog_elaboration.h"
#include "labneura/eda/verilog_process_analysis.h"

#include <optional>
#include <string>
#include <vector>

namespace labneura::eda {

enum class VerilogNativeLoweringStatus : uint8_t {
    Succeeded,
    Unsupported,
    InvalidInput
};

struct VerilogNativeLoweringDiagnostic {
    std::string code;
    std::string message;
    VerilogSyntaxLocation location;
    std::optional<VerilogSyntaxLocation> related_location;
};

// Failure is atomic: module is present only when the whole requested top
// module was lowered and validated by the native lowering pipeline.
struct VerilogNativeLoweringResult {
    VerilogNativeLoweringStatus status =
        VerilogNativeLoweringStatus::InvalidInput;
    std::optional<Module> module;
    std::vector<VerilogNativeLoweringDiagnostic> diagnostics;

    bool succeeded() const noexcept {
        return status == VerilogNativeLoweringStatus::Succeeded &&
               module.has_value();
    }
};

// Explicit opt-in native pipeline endpoint. The supplied analysis objects
// must refer to the same syntax and semantic model. This entry point never
// reparses source and never invokes Slang.
VerilogNativeLoweringResult lower_verilog_native(
    const VerilogCompilationUnit& syntax,
    const VerilogElaborationResult& elaboration,
    const VerilogTypeAnalysisResult& type_analysis,
    const VerilogSynthesisEligibilityResult& process_analysis,
    std::string top_module = {});

} // namespace labneura::eda
