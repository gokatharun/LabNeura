#pragma once

#include "labneura/eda/verilog_constant.h"
#include "labneura/eda/verilog_semantic.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace labneura::eda {

enum class VerilogTypeCategory : uint8_t {
    Integral,
    Real,
    String,
    Enumeration,
    Structure,
    Union,
    Array,
    Unknown
};

struct VerilogPackedDimensionType {
    std::optional<int64_t> left;
    std::optional<int64_t> right;
    std::optional<size_t> width;
};

struct VerilogTypeMember {
    std::string name;
    std::shared_ptr<const struct VerilogIntegralType> type;
    std::optional<size_t> bit_offset;
    std::optional<size_t> bit_width;
    VerilogSyntaxLocation declaration_location;
};

struct VerilogTypeEnumerator {
    std::string name;
    VerilogSyntaxLocation declaration_location;
    std::optional<VerilogConstantValue> value;
};

struct VerilogIntegralType {
    VerilogTypeCategory category = VerilogTypeCategory::Unknown;
    std::optional<size_t> bit_width;
    bool is_signed = false;
    bool four_state = true;
    bool unbased_unsized = false;
    std::string declared_type;
    std::string alias_name;
    VerilogSyntaxLocation declaration_location;
    std::vector<VerilogPackedDimensionType> packed_dimensions;
    std::vector<VerilogPackedDimensionType> unpacked_dimensions;
    std::optional<size_t> unpacked_element_count;
    std::shared_ptr<const VerilogIntegralType> element_type;
    std::vector<VerilogTypeMember> members;
    std::vector<VerilogTypeEnumerator> enumerators;
    std::string identity_name;
    bool packed = false;
    bool is_aggregate_signed = false;

    bool is_known_integral() const noexcept {
        const bool integral_representation =
            category == VerilogTypeCategory::Integral ||
            category == VerilogTypeCategory::Enumeration ||
            (packed && (category == VerilogTypeCategory::Structure ||
                        category == VerilogTypeCategory::Union));
        return integral_representation && bit_width.has_value() && *bit_width != 0;
    }
};

using VerilogTypeEnvironment = std::unordered_map<std::string, VerilogIntegralType>;

enum class VerilogTypeDiagnosticSeverity : uint8_t { Warning, Error };

struct VerilogTypeDiagnostic {
    VerilogTypeDiagnosticSeverity severity = VerilogTypeDiagnosticSeverity::Error;
    std::string code;
    std::string message;
    VerilogSyntaxLocation location;
};

struct VerilogExpressionTypeResult {
    VerilogIntegralType type;
    std::vector<VerilogTypeDiagnostic> diagnostics;
};

struct VerilogTypeAnalysisResult {
    std::unordered_map<const VerilogExpression*, VerilogIntegralType> expression_types;
    // Symbol keys point into the semantic model passed to the analysis function.
    std::unordered_map<const VerilogSemanticSymbol*, VerilogIntegralType> symbol_types;
    std::unordered_map<const VerilogSemanticSymbol*, VerilogConstantValue> symbol_constants;
    std::vector<VerilogTypeDiagnostic> diagnostics;

    bool has_errors() const noexcept;
};

VerilogIntegralType infer_verilog_declared_type(
    std::string_view declared_type,
    const std::vector<VerilogExpressionRange>& packed_dimensions = {},
    const VerilogConstantEnvironment& constant_environment = {},
    const VerilogIntegralType* inferred_type = nullptr);

VerilogExpressionTypeResult infer_verilog_expression_type(
    const VerilogExpression& expression,
    const VerilogTypeEnvironment& type_environment = {},
    const VerilogConstantEnvironment& constant_environment = {});

// Resolves expression types using a prior type analysis result for symbol
// resolution. Identifier expressions are resolved against symbol types
// collected during type analysis, so declared types from parameters, ports,
// and variables are honored.
VerilogExpressionTypeResult infer_verilog_expression_type(
    const VerilogExpression& expression,
    const VerilogTypeAnalysisResult& type_analysis,
    const VerilogTypeEnvironment& type_environment = {},
    const VerilogConstantEnvironment& constant_environment = {});

VerilogTypeAnalysisResult analyze_verilog_expression_types(
    const VerilogCompilationUnit& syntax,
    const VerilogSemanticAnalysisResult& semantic);

} // namespace labneura::eda
