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

struct VerilogSemanticReferenceIndexEntry {
    const VerilogSourceBuffer* source = nullptr;
    size_t byte_offset = 0;
    size_t reference_ordinal = 0;
};

struct VerilogSemanticReferenceIndexRange {
    using Iterator = std::vector<VerilogSemanticReferenceIndexEntry>::const_iterator;
    Iterator first;
    Iterator last;

    Iterator begin() const noexcept { return first; }
    Iterator end() const noexcept { return last; }
    bool empty() const noexcept { return first == last; }
    size_t size() const noexcept {
        return static_cast<size_t>(last - first);
    }
};

struct VerilogSemanticReferenceIndexStatistics {
    size_t reference_count = 0;
    size_t entry_count = 0;
    size_t estimated_storage_bytes = 0;
    uint64_t construction_nanoseconds = 0;
    size_t duplicate_location_count = 0;
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
    // The ordered references remain the compatibility and diagnostic oracle.
    // This compact ordinal index is built after resolution and preserves the
    // insertion order of duplicate source locations. Call rebuild after any
    // intentional external mutation of references.
    std::vector<VerilogSemanticReferenceIndexEntry> reference_index;
    VerilogSemanticReferenceIndexStatistics reference_index_statistics;

    void rebuild_reference_index();
    VerilogSemanticReferenceIndexRange references_at(
        const VerilogSyntaxLocation& location) const;

    bool has_errors() const noexcept { return !diagnostics.empty(); }
};

const VerilogSemanticSymbol* lookup_verilog_symbol(
    const VerilogSemanticScope& scope, std::string_view name,
    bool include_parent_scopes = true);

VerilogSemanticAnalysisResult analyze_verilog_semantics(
    const VerilogCompilationUnit& syntax);

} // namespace labneura::eda
