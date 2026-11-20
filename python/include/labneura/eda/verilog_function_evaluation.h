#pragma once

#include "labneura/eda/verilog_callable_binding.h"
#include "labneura/eda/verilog_callable_execution.h"

#include <optional>

namespace labneura::eda {

struct VerilogFunctionEvaluationResult {
    std::optional<VerilogConstantValue> value;
    std::vector<VerilogCallableExecutionDiagnostic> diagnostics;
    size_t executed_steps = 0;
};

VerilogFunctionEvaluationResult evaluate_verilog_function_call(
    const VerilogCompilationUnit& syntax,
    const VerilogSemanticAnalysisResult& semantic,
    const VerilogTypeAnalysisResult& type_analysis,
    const VerilogExpression& call,
    const VerilogConstantEnvironment& caller_environment = {},
    VerilogCallableExecutionOptions options = {});

VerilogFunctionEvaluationResult evaluate_verilog_function_call(
    const VerilogCompilationUnit& syntax,
    const VerilogSemanticAnalysisResult& semantic,
    const VerilogTypeAnalysisResult& type_analysis,
    const VerilogExpression& call,
    const VerilogConstantEnvironment& caller_environment,
    VerilogCallableExecutionState& state);

} // namespace labneura::eda
