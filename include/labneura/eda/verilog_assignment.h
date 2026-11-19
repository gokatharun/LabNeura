#pragma once

#include "labneura/eda/verilog_constant.h"
#include "labneura/eda/verilog_semantic.h"
#include "labneura/eda/verilog_type.h"

#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace labneura::eda {

// One destination storage object participating in an lvalue. bit_positions is
// indexed by the lvalue's least-significant-first bit order; null entries are
// legal out-of-range selections and are ignored on assignment.
struct VerilogResolvedAssignmentDestination {
    const VerilogSemanticSymbol* symbol = nullptr;
    std::string storage_name;
    VerilogIntegralType declared_type;
    size_t target_lsb_offset = 0;
    std::vector<std::optional<size_t>> bit_positions;
};

struct VerilogResolvedAssignmentTarget {
    VerilogSyntaxLocation location;
    size_t bit_width = 0;
    bool is_integral = false;
    bool is_signed = false;
    bool four_state = true;
    std::vector<VerilogResolvedAssignmentDestination> destinations;
};

struct VerilogAssignmentDiagnostic {
    std::string code;
    std::string message;
    VerilogSyntaxLocation location;
    std::optional<VerilogSyntaxLocation> related_location;
};

struct VerilogLvalueResolutionResult {
    std::optional<VerilogResolvedAssignmentTarget> target;
    std::vector<VerilogAssignmentDiagnostic> diagnostics;
    bool has_errors() const noexcept { return !diagnostics.empty(); }
};

using VerilogAssignmentExpressionEvaluator =
    std::function<std::optional<VerilogConstantValue>(const VerilogExpression&)>;

// Evaluation is supplied by the active interpreter frame so indices may use
// runtime locals, formals and supported function calls.
VerilogLvalueResolutionResult resolve_verilog_lvalue(
    const VerilogExpression& expression,
    const VerilogSemanticAnalysisResult& semantic,
    const VerilogTypeAnalysisResult& type_analysis,
    const VerilogAssignmentExpressionEvaluator& evaluate_expression);

struct VerilogLvalueReadResult {
    std::optional<VerilogConstantValue> value;
    std::vector<VerilogAssignmentDiagnostic> diagnostics;
};

// Reads a resolved lvalue for inout copy-in and other interpreter consumers.
// Legal out-of-range selected bits produce X in the read value.
VerilogLvalueReadResult read_verilog_lvalue(
    const VerilogResolvedAssignmentTarget& target,
    const VerilogConstantEnvironment& environment,
    size_t maximum_width = 65536);

struct VerilogAssignmentOptions {
    size_t maximum_width = 65536;
};

struct VerilogAssignmentResult {
    bool assigned = false;
    std::vector<VerilogAssignmentDiagnostic> diagnostics;
};

// Applies assignment conversion, stages every affected storage value, and
// publishes the updates only if all conversions and destination checks pass.
VerilogAssignmentResult assign_verilog_lvalue(
    const VerilogResolvedAssignmentTarget& target,
    const VerilogConstantValue& value,
    VerilogConstantEnvironment& environment,
    VerilogAssignmentOptions options = {});

} // namespace labneura::eda
