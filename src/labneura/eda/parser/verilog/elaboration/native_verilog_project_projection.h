#pragma once

#include "labneura/eda/parser/verilog/elaboration/native_verilog_project_context.h"
#include "labneura/eda/netlist.h"

#include <memory>
#include <string>
#include <vector>

namespace labneura::eda::detail {

struct NativeVerilogProjectionDiagnostic {
    std::string code;
    std::string message;
    VerilogSyntaxLocation location;
};

struct NativeVerilogProjectProjectionResult {
    Project project;
    std::vector<NativeVerilogProjectionDiagnostic> diagnostics;
    std::vector<std::string> compatibility_gaps;

    bool succeeded() const noexcept { return diagnostics.empty(); }
};

NativeVerilogProjectProjectionResult project_native_verilog(
    std::shared_ptr<NativeVerilogProjectContext> context);

} // namespace labneura::eda::detail
