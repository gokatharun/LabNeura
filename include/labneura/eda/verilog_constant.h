#pragma once

#include "labneura/eda/verilog_syntax.h"

#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace labneura::eda {

enum class VerilogLogicBit : uint8_t { Zero, One, X, Z };

struct VerilogConstantValue {
    size_t bit_width = 0;
    bool is_signed = false;
    // Bits are stored least-significant first.
    std::vector<VerilogLogicBit> bits;
    bool unbased_unsized = false;
    VerilogSyntaxLocation location;

    bool is_fully_known() const noexcept;
    std::optional<uint64_t> to_uint64() const noexcept;
    std::optional<int64_t> to_int64() const noexcept;
    std::string to_binary_string() const;
};

struct VerilogConstantDiagnostic {
    std::string code;
    std::string message;
    VerilogSyntaxLocation location;
};

struct VerilogConstantEvaluationResult {
    std::optional<VerilogConstantValue> value;
    std::vector<VerilogConstantDiagnostic> diagnostics;

    bool has_errors() const noexcept { return !diagnostics.empty(); }
};

using VerilogConstantEnvironment =
    std::unordered_map<std::string, VerilogConstantValue>;
using VerilogConstantResolver = std::function<std::optional<VerilogConstantValue>(
    std::string_view, const VerilogSyntaxLocation&)>;

struct VerilogConstantEvaluationOptions {
    size_t maximum_width = 65536;
    size_t maximum_replication_count = 65536;
};

VerilogConstantEvaluationResult evaluate_verilog_constant(
    const VerilogExpression& expression,
    const VerilogConstantEnvironment& environment = {},
    VerilogConstantResolver resolver = {},
    VerilogConstantEvaluationOptions options = {});

} // namespace labneura::eda
