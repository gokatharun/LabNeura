#include "labneura/eda/verilog_semantic.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

using namespace labneura::eda;

void require(bool condition, std::string_view message) {
    if (!condition)
        throw std::runtime_error(std::string(message));
}

bool has_code(const VerilogSemanticAnalysisResult& result, std::string_view code) {
    return std::any_of(result.diagnostics.begin(), result.diagnostics.end(),
                       [&](const auto& diagnostic) { return diagnostic.code == code; });
}

const VerilogSemanticScope& child_scope(const VerilogSemanticScope& root,
                                        std::string_view name) {
    const auto found = std::find_if(root.children.begin(), root.children.end(),
                                    [&](const auto& scope) { return scope->name == name; });
    if (found == root.children.end())
        throw std::runtime_error("expected semantic scope was not collected");
    return **found;
}

void test_symbol_collection_and_resolution() {
    const auto syntax = parse_verilog_syntax(
        "package cfg;\n"
        "  parameter int SIZE = 8;\n"
        "  localparam int BASE = 2;\n"
        "endpackage\n"
        "module leaf #(parameter WIDTH = 4) (\n"
        "  input logic [WIDTH-1:0] data,\n"
        "  output logic valid\n"
        ");\n"
        "  logic ready;\n"
        "  assign valid = ready;\n"
        "  always_comb begin\n"
        "    ready = data[0];\n"
        "    if (valid) ready = cfg.BASE;\n"
        "  end\n"
        "endmodule\n"
        "module top(input logic a, output logic y);\n"
        "  parameter N = 4;\n"
        "  localparam M = N;\n"
        "  logic x;\n"
        "  assign x = a;\n"
        "  leaf #(.WIDTH(N)) u (.data({a}), .valid(x));\n"
        "  assign y = cfg.SIZE + $clog2(N);\n"
        "endmodule\n");
    require(!syntax.has_errors(), "semantic success fixture must parse");
    auto result = analyze_verilog_semantics(syntax.syntax);
    require(!result.has_errors(), "valid declarations and references must resolve");

    const auto& root = *result.model.compilation_unit_scope;
    const auto* top = lookup_verilog_symbol(root, "top", false);
    const auto* leaf = lookup_verilog_symbol(root, "leaf", false);
    const auto* cfg = lookup_verilog_symbol(root, "cfg", false);
    require(top && top->kind == VerilogSemanticSymbolKind::Module && leaf &&
                leaf->kind == VerilogSemanticSymbolKind::Module && cfg &&
                cfg->kind == VerilogSemanticSymbolKind::Package,
            "compilation-unit symbols were not collected");

    const auto& top_scope = child_scope(root, "top");
    const auto& leaf_scope = child_scope(root, "leaf");
    const auto& package_scope = child_scope(root, "cfg");
    const auto* local_parameter =
        lookup_verilog_symbol(top_scope, "M", false);
    require(local_parameter &&
                local_parameter->kind == VerilogSemanticSymbolKind::LocalParameter,
            "localparam kind was not preserved in the semantic table");
    require(lookup_verilog_symbol(top_scope, "WIDTH", false) == nullptr &&
                lookup_verilog_symbol(leaf_scope, "N", false) == nullptr &&
                lookup_verilog_symbol(top_scope, "N") != nullptr,
            "module scope boundaries or parent-scope lookup are incorrect");
    require(lookup_verilog_symbol(package_scope, "SIZE", false) &&
                lookup_verilog_symbol(package_scope, "BASE", false),
            "package parameter declarations are missing");

    const auto* instance = lookup_verilog_symbol(top_scope, "u", false);
    require(instance && instance->kind == VerilogSemanticSymbolKind::Instance &&
                instance->resolved_declaration == leaf,
            "module instance did not resolve to its module declaration");
    require(std::all_of(result.references.begin(), result.references.end(),
                        [](const auto& reference) {
                            return reference.declaration ||
                                   reference.name == "$clog2";
                        }),
            "a valid identifier reference was left unresolved");
}

void test_invalid_declarations_and_instance_connections() {
    const auto syntax = parse_verilog_syntax(
        "module child #(parameter P = 1) (input a);\n"
        "endmodule\n"
        "module broken(input a);\n"
        "  logic local_value;\n"
        "  logic local_value;\n"
        "  assign local_value = absent;\n"
        "  child #(.BAD(missing)) u (.ghost(absent));\n"
        "  child v();\n"
        "  child v();\n"
        "  unknown_type w();\n"
        "endmodule\n"
        "module other;\n"
        "  logic absent;\n"
        "endmodule\n");
    require(!syntax.has_errors(), "semantic error fixture must be syntactically valid");
    const auto result = analyze_verilog_semantics(syntax.syntax);
    require(has_code(result, "DuplicateDeclaration") &&
                has_code(result, "DuplicateInstanceName") &&
                has_code(result, "UnresolvedIdentifier") &&
                has_code(result, "UnresolvedModuleType") &&
                has_code(result, "InvalidParameterOverride") &&
                has_code(result, "InvalidPortConnection"),
            "expected semantic declaration, resolution, and connection diagnostics are missing");

    const auto duplicate = std::find_if(
        result.diagnostics.begin(), result.diagnostics.end(), [](const auto& diagnostic) {
            return diagnostic.code == "DuplicateDeclaration";
        });
    require(duplicate != result.diagnostics.end() && duplicate->related_location &&
                duplicate->location.location.line == 5 &&
                duplicate->related_location->location.line == 4,
            "duplicate declaration diagnostic lacks original declaration location");

    const auto unresolved = std::find_if(
        result.diagnostics.begin(), result.diagnostics.end(), [](const auto& diagnostic) {
            return diagnostic.code == "UnresolvedIdentifier" &&
                   diagnostic.message.find("'absent'") != std::string::npos;
        });
    require(unresolved != result.diagnostics.end() &&
                unresolved->location.location.line == 6 &&
                unresolved->location.source &&
                unresolved->location.source->location(
                    unresolved->location.byte_offset).line == 6,
            "unresolved identifier location or scope isolation is incorrect");
}

void test_macro_origins_and_unsupported_recovery() {
    const auto syntax = parse_verilog_syntax(
        "`define UNKNOWN missing_symbol\n"
        "module macro_user;\n"
        "  logic result;\n"
        "  assign result = `UNKNOWN;\n"
        "  always_comb begin\n"
        "    for (int i = 0; i < 2; i = i + 1) result = i;\n"
        "  end\n"
        "endmodule\n"
        "module later;\n"
        "  logic result;\n"
        "  assign result = 1'b0;\n"
        "endmodule\n");
    auto result = analyze_verilog_semantics(syntax.syntax);
    require(has_code(result, "UnresolvedIdentifier") &&
                syntax.syntax.modules.size() == 2 &&
                syntax.syntax.modules.back().name == "later",
            "macro identifier or recovery module was not analyzed");

    const auto macro_diagnostic = std::find_if(
        result.diagnostics.begin(), result.diagnostics.end(), [](const auto& diagnostic) {
            return diagnostic.code == "UnresolvedIdentifier" &&
                   diagnostic.message.find("missing_symbol") != std::string::npos;
        });
    require(macro_diagnostic != result.diagnostics.end() &&
                macro_diagnostic->location.origin_kind ==
                    VerilogPreprocessedOriginKind::MacroExpansion &&
                !macro_diagnostic->location.expansion_stack.empty() &&
                macro_diagnostic->location.expansion_stack.back().macro_name == "UNKNOWN",
            "semantic diagnostic lost its macro expansion source origin");

    const auto later_scope = std::find_if(
        result.model.compilation_unit_scope->children.begin(),
        result.model.compilation_unit_scope->children.end(),
        [](const auto& scope) { return scope->name == "later"; });
    require(later_scope != result.model.compilation_unit_scope->children.end() &&
                (*later_scope)->symbols_by_name.contains("result"),
            "semantic collection did not continue after an unsupported statement");
}

void test_qualified_package_and_scope_lookup() {
    const auto syntax = parse_verilog_syntax(
        "package constants;\n"
        "  parameter VALUE = 3;\n"
        "endpackage\n"
        "module uses_package;\n"
        "  logic [constants.VALUE-1:0] bits;\n"
        "  assign bits[0] = constants.VALUE;\n"
        "endmodule\n");
    require(!syntax.has_errors(), "qualified package fixture must parse");
    const auto result = analyze_verilog_semantics(syntax.syntax);
    require(!result.has_errors(), "qualified package member should resolve");
    require(std::any_of(result.references.begin(), result.references.end(),
                        [](const auto& reference) {
                            return reference.name == "constants.VALUE" &&
                                   reference.declaration &&
                                   reference.declaration->name == "VALUE";
                        }),
            "qualified package member reference was not recorded");
}

void test_procedural_loop_resolution_and_assignment_diagnostics() {
    const auto syntax = parse_verilog_syntax(
        "module procedural(input logic enable, output logic [7:0] q);\n"
        "  integer index;\n"
        "  initial begin\n"
        "    for (int local_index = 0; local_index < 2; local_index++)\n"
        "      q[local_index] = local_index;\n"
        "    while (enable) q = q + 1;\n"
        "    repeat (2) q = q - 1;\n"
        "    forever @(posedge enable) q = q;\n"
        "  end\n"
        "endmodule\n");
    require(!syntax.has_errors(), "valid procedural loop fixture should parse");
    const auto result = analyze_verilog_semantics(syntax.syntax);
    require(!result.has_errors(),
            "procedural loop variables and references did not resolve in their scopes");

    const auto invalid = parse_verilog_syntax(
        "module invalid_target(input logic a, output logic q);\n"
        "  always_comb begin (a + q) = missing; end\n"
        "endmodule\n");
    const auto invalid_result = analyze_verilog_semantics(invalid.syntax);
    require(has_code(invalid_result, "InvalidAssignmentTarget") &&
                has_code(invalid_result, "UnresolvedIdentifier"),
            "semantic analysis missed invalid assignment target or unresolved RHS");
    const auto target_diagnostic =
        std::find_if(invalid_result.diagnostics.begin(), invalid_result.diagnostics.end(),
                     [](const auto& diagnostic) {
                         return diagnostic.code == "InvalidAssignmentTarget";
                     });
    require(target_diagnostic != invalid_result.diagnostics.end() &&
                target_diagnostic->location.location.line == 2,
            "invalid procedural assignment target diagnostic has the wrong location");
}

} // namespace

int main() {
    try {
        test_symbol_collection_and_resolution();
        test_invalid_declarations_and_instance_connections();
        test_macro_origins_and_unsupported_recovery();
        test_qualified_package_and_scope_lookup();
        test_procedural_loop_resolution_and_assignment_diagnostics();
    }
    catch (const std::exception& error) {
        std::cerr << "EDA Verilog semantic test failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
