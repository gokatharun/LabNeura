#pragma once

#include "labneura/eda/verilog_syntax.h"

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace labneura::eda {

enum class VerilogSemanticScopeKind : uint8_t {
    CompilationUnit,
    Module,
    Interface,
    Package,
    Generate,
    Procedural,
    Callable
};

enum class VerilogSemanticSymbolKind : uint8_t {
    Module,
    Interface,
    Package,
    Parameter,
    LocalParameter,
    Type,
    EnumValue,
    Port,
    Net,
    Variable,
    ProceduralBlock,
    Instance,
    Function,
    Task,
    CallableArgument,
    Argument = CallableArgument
};

struct VerilogSemanticScope;

// Callable declaration data is owned by the semantic model. Formal symbols are
// stored in source order in VerilogSemanticCallableMetadata::formal_arguments;
// their direction, declared type, dimensions, and source maps live on each
// corresponding symbol so type analysis has one canonical copy to consult.
struct VerilogSemanticCallableMetadata {
    std::string return_type;
    std::vector<VerilogExpressionRange> return_packed_dimensions;
    std::vector<VerilogExpressionRange> return_unpacked_dimensions;
    std::vector<const struct VerilogSemanticSymbol*> formal_arguments;
};

struct VerilogSemanticSymbol {
    std::string name;
    VerilogSemanticSymbolKind kind = VerilogSemanticSymbolKind::Variable;
    VerilogSyntaxLocation declaration_location;
    VerilogSemanticScope* scope = nullptr;
    VerilogSemanticScope* member_scope = nullptr;
    const VerilogSemanticSymbol* resolved_declaration = nullptr;
    std::optional<VerilogPortDirection> argument_direction;
    std::string declared_type;
    std::vector<VerilogExpressionRange> packed_dimensions;
    std::vector<VerilogExpressionRange> unpacked_dimensions;
    std::optional<VerilogSemanticCallableMetadata> callable;
};

struct VerilogSemanticScope {
    std::string name;
    VerilogSemanticScopeKind kind = VerilogSemanticScopeKind::CompilationUnit;
    VerilogSemanticScope* parent = nullptr;
    std::vector<std::unique_ptr<VerilogSemanticScope>> children;
    std::vector<std::unique_ptr<VerilogSemanticSymbol>> symbols;
    std::unordered_map<std::string, VerilogSemanticSymbol*> symbols_by_name;
};

struct VerilogSemanticReference {
    std::string name;
    VerilogSyntaxLocation location;
    const VerilogSemanticScope* scope = nullptr;
    const VerilogSemanticSymbol* declaration = nullptr;
};

struct VerilogSemanticDiagnostic {
    std::string code;
    std::string message;
    VerilogSyntaxLocation location;
    std::optional<VerilogSyntaxLocation> related_location;
};

struct VerilogSemanticModel {
    std::unique_ptr<VerilogSemanticScope> compilation_unit_scope;
};

struct VerilogSemanticAnalysisResult {
    VerilogSemanticModel model;
    std::vector<VerilogSemanticReference> references;
    std::vector<VerilogSemanticDiagnostic> diagnostics;

    bool has_errors() const noexcept { return !diagnostics.empty(); }
};

const VerilogSemanticSymbol* lookup_verilog_symbol(
    const VerilogSemanticScope& scope, std::string_view name,
    bool include_parent_scopes = true);

VerilogSemanticAnalysisResult analyze_verilog_semantics(
    const VerilogCompilationUnit& syntax);

} // namespace labneura::eda
