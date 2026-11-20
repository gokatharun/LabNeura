#pragma once

#include "labneura/eda/verilog_callable_execution.h"
#include "labneura/eda/verilog_semantic.h"
#include "labneura/eda/verilog_type.h"

#include <cstddef>
#include <vector>

namespace labneura::eda {

enum class VerilogTaskExecutionStatus : uint8_t {
    Completed,
    Failed,
    // The supplied finite transition trace ended before an event trigger.
    // This is not a resumable continuation.
    NoMatchingEvent
};

struct VerilogTaskExecutionResult {
    bool completed = false;
    VerilogTaskExecutionStatus status = VerilogTaskExecutionStatus::Failed;
    // Logical ticks advanced by delays and consumed signal transitions.
    uint64_t elapsed_delay_ticks = 0;
    std::vector<VerilogCallableExecutionDiagnostic> diagnostics;
    size_t executed_steps = 0;
};

VerilogTaskExecutionResult execute_verilog_task_call(
    const VerilogCompilationUnit& syntax,
    const VerilogSemanticAnalysisResult& semantic,
    const VerilogTypeAnalysisResult& type_analysis,
    const VerilogExpression& call,
    VerilogConstantEnvironment& caller_environment,
    VerilogCallableExecutionOptions options = {});

VerilogTaskExecutionResult execute_verilog_task_call(
    const VerilogCompilationUnit& syntax,
    const VerilogSemanticAnalysisResult& semantic,
    const VerilogTypeAnalysisResult& type_analysis,
    const VerilogExpression& call,
    VerilogConstantEnvironment& caller_environment,
    VerilogCallableExecutionState& state);

} // namespace labneura::eda
