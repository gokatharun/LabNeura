#include "labneura/eda/verilog_elaboration.h"
#include "labneura/eda/verilog_type.h"

#include <slang/ast/Compilation.h>
#include <slang/ast/symbols/CompilationUnitSymbols.h>
#include <slang/ast/symbols/InstanceSymbols.h>
#include <slang/ast/symbols/ParameterSymbols.h>
#include <slang/ast/types/Type.h>
#include <slang/syntax/SyntaxTree.h>

#include <algorithm>
#include <functional>
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

bool has_code(const VerilogTypeAnalysisResult& result, std::string_view code) {
    return std::any_of(result.diagnostics.begin(), result.diagnostics.end(),
                       [&](const auto& diagnostic) { return diagnostic.code == code; });
}

const VerilogSemanticScope& child_scope(const VerilogSemanticScope& root,
                                        std::string_view name);

const VerilogExpression& parse_expression(std::string expression) {
    static thread_local VerilogSyntaxParseResult storage;
    storage = parse_verilog_syntax("module type_test; parameter VALUE = " + expression +
                                   "; endmodule");
    if (storage.has_errors() || storage.syntax.modules.empty())
        throw std::runtime_error("could not parse type-expression test input");
    return *storage.syntax.modules.front().parameters.front().value;
}

void expect_width(const VerilogExpression& expression, size_t width, bool is_signed,
                  bool four_state) {
    const auto result = infer_verilog_expression_type(expression);
    require(result.type.is_known_integral() && result.type.bit_width == width &&
                result.type.is_signed == is_signed &&
                result.type.four_state == four_state,
            "expression type did not match expected width, signedness, and state model");
}

void test_literal_and_operator_types() {
    expect_width(parse_expression("8'hff"), 8, false, true);
    // Hexadecimal digits e/E are base digits, not real-number exponents.
    expect_width(parse_expression("4'he"), 4, false, true);
    expect_width(parse_expression("4'he ^ 4'h8"), 4, false, true);
    expect_width(parse_expression("32'sd1"), 32, true, true);
    expect_width(parse_expression("42"), 32, true, false);
    const auto& wide_decimal = parse_expression("4294967296");
    const auto decimal_type = infer_verilog_expression_type(wide_decimal).type;
    const auto decimal_value = evaluate_verilog_constant(wide_decimal).value;
    require(decimal_value && decimal_type.bit_width == decimal_value->bit_width &&
                *decimal_type.bit_width > 32,
            "wide unsized decimal type did not preserve the evaluator's required width");
    expect_width(parse_expression("'hF"), 32, false, true);
    expect_width(parse_expression("8'hff + 16'h1"), 16, false, true);
    expect_width(parse_expression("8'shff + 16'sh1"), 16, true, true);
    expect_width(parse_expression("8'hff < 16'h1"), 1, false, true);
    expect_width(parse_expression("8'hff && 16'h1"), 1, false, true);
    expect_width(parse_expression("~8'hff"), 8, false, true);
    expect_width(parse_expression("1'b1 ? 8'hff : 4'h1"), 8, false, true);
    expect_width(parse_expression("{8'h1, 4'h2}"), 12, false, true);
    expect_width(parse_expression("{3{4'hf}}"), 12, false, true);
    expect_width(parse_expression("8'hff[3:1]"), 3, false, true);
    expect_width(parse_expression("8'hff[0]"), 1, false, true);
}

void test_declarations_assignments_and_scope_types() {
    const auto parsed = parse_verilog_syntax(
        "module typed #(parameter int WIDTH = 8) "
        "(input logic [WIDTH-1:0] data, output logic [3:0] result);\n"
        "  localparam logic [2:0] LOCAL_WIDTH = 3'd5;\n"
        "  logic signed [7:0] signed_value;\n"
        "  logic [WIDTH-1:0] copy;\n"
        "  assign result = data;\n"
        "  assign copy = data;\n"
        "  always_comb signed_value = data;\n"
        "endmodule\n");
    require(!parsed.has_errors(), "type-analysis design fixture must parse");
    const auto semantic = analyze_verilog_semantics(parsed.syntax);
    const auto result = analyze_verilog_expression_types(parsed.syntax, semantic);
    require(!result.has_errors(), "width conversions should not be hard errors");
    require(has_code(result, "AssignmentWidthMismatch"),
            "known assignment width mismatch was not diagnosed");

    const auto* module = lookup_verilog_symbol(*semantic.model.compilation_unit_scope,
                                               "typed", false);
    require(module && module->member_scope, "module semantic scope was not collected");
    const auto* data = lookup_verilog_symbol(*module->member_scope, "data", false);
    const auto* result_port = lookup_verilog_symbol(*module->member_scope, "result", false);
    const auto* signed_value =
        lookup_verilog_symbol(*module->member_scope, "signed_value", false);
    const auto* width = lookup_verilog_symbol(*module->member_scope, "WIDTH", false);
    const auto* local_width =
        lookup_verilog_symbol(*module->member_scope, "LOCAL_WIDTH", false);
    require(data && result_port && signed_value && width && local_width,
            "typed declarations were not collected");
    require(result.symbol_types.at(data).bit_width == 8 &&
                result.symbol_types.at(result_port).bit_width == 4 &&
                result.symbol_types.at(signed_value).bit_width == 8 &&
                result.symbol_types.at(signed_value).is_signed &&
                result.symbol_types.at(width).bit_width == 32 &&
                result.symbol_types.at(width).is_signed &&
                !result.symbol_types.at(width).four_state &&
                result.symbol_types.at(local_width).bit_width == 3,
            "declaration type or packed width was inferred incorrectly");
    const auto mismatch = std::find_if(
        result.diagnostics.begin(), result.diagnostics.end(), [](const auto& diagnostic) {
            return diagnostic.code == "AssignmentWidthMismatch";
        });
    require(mismatch != result.diagnostics.end() && mismatch->location.source &&
                mismatch->location.location.line > 0,
            "assignment type diagnostic lost its source location");
}

void test_instance_parameter_types_are_independent() {
    const auto parsed = parse_verilog_syntax(
        "module child #(parameter logic [7:0] P = 8'h01); endmodule\n"
        "module top;\n"
        "  child #(.P(16'h1234)) first();\n"
        "  child #(.P(8'h56)) second();\n"
        "endmodule\n");
    require(!parsed.has_errors(), "parameter type fixture must parse");
    VerilogElaborationOptions options;
    options.top_module = "top";
    const auto elaborated = elaborate_verilog_design(parsed.syntax, options);
    require(!elaborated.has_errors() && elaborated.top &&
                elaborated.top->children.size() == 2,
            "typed parameter instances should elaborate");
    const auto& first = *elaborated.top->children[0];
    const auto& second = *elaborated.top->children[1];
    require(first.parameter_types.at("P").bit_width == 8 &&
                second.parameter_types.at("P").bit_width == 8 &&
                first.parameters.at("P").bit_width == 8 &&
                second.parameters.at("P").bit_width == 8,
            "instance overrides did not preserve the declared parameter type");
    require(first.parameters.at("P").to_uint64() == 0x34 &&
                second.parameters.at("P").to_uint64() == 0x56,
            "instance-specific parameter values were shared or altered");
}

void test_supported_binary_type_against_slang() {
    constexpr std::string_view expression = "8'hff + 16'h1";
    const auto& native_expression = parse_expression(std::string(expression));
    const auto native = infer_verilog_expression_type(native_expression);

    slang::SourceManager source_manager;
    auto tree = slang::syntax::SyntaxTree::fromText(
        "module type_reference; localparam R = " + std::string(expression) +
            "; endmodule\n",
        source_manager, "type_reference.sv");
    slang::ast::Compilation compilation;
    compilation.addSyntaxTree(tree);
    require(compilation.getAllDiagnostics().empty(),
            "Slang rejected the supported type differential expression");
    const auto& root = compilation.getRoot();
    const auto top = std::find_if(root.topInstances.begin(), root.topInstances.end(),
                                  [](const auto* instance) {
                                      return instance->name == "type_reference";
                                  });
    require(top != root.topInstances.end(), "Slang did not elaborate type reference module");
    const slang::ast::ParameterSymbol* parameter = nullptr;
    for (const auto& member : (*top)->body.members())
        if (member.kind == slang::ast::SymbolKind::Parameter &&
            member.name == "R")
            parameter = &member.as<slang::ast::ParameterSymbol>();
    require(parameter, "Slang did not expose the reference parameter type");
    const auto& slang_type = parameter->getType();
    require(native.type.bit_width == slang_type.getBitWidth() &&
                native.type.is_signed == slang_type.isSigned() &&
                native.type.four_state == slang_type.isFourState(),
            "native expression width/signedness/state differs from Slang for supported arithmetic");
}

void test_unknown_user_types_remain_unknown() {
    const auto type = infer_verilog_declared_type("custom_t");
    require(type.category == VerilogTypeCategory::Unknown && !type.bit_width,
            "user-defined type should remain unknown instead of guessing a width");
    const auto integer_type = infer_verilog_declared_type("integer");
    require(integer_type.bit_width == 32 && integer_type.is_signed &&
                integer_type.four_state &&
                infer_verilog_declared_type("int").bit_width == 32 &&
                !infer_verilog_declared_type("int").four_state,
            "integer atom and integer types lost their distinct state semantics");
}

void test_generated_and_procedural_declaration_types() {
    const auto parsed = parse_verilog_syntax(
        "module typed_scopes;\n"
        "  if (1) begin : selected\n"
        "    logic [3:0] generated_value;\n"
        "  end\n"
        "  always_comb for (int i = 0; i < 2; i++) begin end\n"
        "endmodule\n");
    require(!parsed.has_errors(), "generated/procedural type fixture must parse");
    const auto semantic = analyze_verilog_semantics(parsed.syntax);
    const auto result = analyze_verilog_expression_types(parsed.syntax, semantic);

    const VerilogSemanticScope* generated_scope = nullptr;
    const VerilogSemanticSymbol* generated_value = nullptr;
    const VerilogSemanticSymbol* loop_variable = nullptr;
    std::function<void(const VerilogSemanticScope&)> find_symbols =
        [&](const VerilogSemanticScope& scope) {
            if (scope.kind == VerilogSemanticScopeKind::Generate &&
                scope.name == "selected")
                generated_scope = &scope;
            if (scope.kind == VerilogSemanticScopeKind::Procedural)
                for (const auto& symbol : scope.symbols)
                    if (symbol->name == "i")
                        loop_variable = symbol.get();
            for (const auto& symbol : scope.symbols)
                if (symbol->name == "generated_value")
                    generated_value = symbol.get();
            for (const auto& child : scope.children)
                find_symbols(*child);
        };
    find_symbols(*semantic.model.compilation_unit_scope);
    require(generated_scope && generated_value && loop_variable &&
                result.symbol_types.at(generated_value).bit_width == 4 &&
                result.symbol_types.at(loop_variable).bit_width == 32 &&
                result.symbol_types.at(loop_variable).is_signed,
            "generated or procedural declaration types were not collected");
}

void test_typedef_enum_aggregate_and_array_types() {
    const auto parsed = parse_verilog_syntax(
        "package type_pkg;\n"
        "  typedef later_t first_t;\n"
        "  typedef logic [2:0] later_t;\n"
        "  typedef logic [3:0] nibble_t;\n"
        "  typedef enum logic signed [2:0] { NEG = -1, ZERO = 0, ONE } state_t;\n"
        "  typedef enum logic [1:0] { UZERO = 0, UONE } unsigned_state_t;\n"
        "  typedef struct packed { nibble_t lo; nibble_t hi; } pair_t;\n"
        "  typedef struct packed { pair_t pair; logic flag; } nested_t;\n"
        "  typedef struct { int code; logic [1:0] flags; } record_t;\n"
        "  typedef union packed { logic [7:0] word; logic [7:0] bytes; } byte_u;\n"
        "  typedef union { int word; logic [7:0] bytes; } data_u;\n"
        "  typedef logic [3:0] memory_t [2:0];\n"
        "  typedef nibble_t matrix_t [1:0][2:0];\n"
        "  typedef struct { absent_t value; } unresolved_t;\n"
        "  typedef union packed { logic [7:0] wide; logic [3:0] narrow; } bad_union_t;\n"
        "endpackage\n"
        "package second_type_pkg;\n"
        "  typedef logic [1:0] nibble_t;\n"
        "endpackage\n"
        "module type_top;\n"
        "  typedef logic [1:0] local_t;\n"
        "  local_t local_value;\n"
        "  type_pkg::pair_t pair;\n"
        "  type_pkg::nested_t nested;\n"
        "  type_pkg::record_t record_value;\n"
        "  type_pkg::memory_t memory;\n"
        "  type_pkg::matrix_t matrix;\n"
        "  second_type_pkg::nibble_t second_nibble;\n"
        "  logic [7:0] result;\n"
        "  assign result = pair.lo;\n"
        "  assign result = nested.pair.hi;\n"
        "  assign result = memory[0];\n"
        "  assign result = matrix[0][1];\n"
        "  assign result = type_pkg::state_t::ONE;\n"
        "  assign result = pair.missing;\n"
        "  if (1) begin : generated\n"
        "    typedef logic [2:0] generated_t;\n"
        "    generated_t generated_value;\n"
        "  end\n"
        "endmodule\n");
    require(!parsed.has_errors(), "user-defined type fixture must parse");
    const auto semantic = analyze_verilog_semantics(parsed.syntax);
    const auto result = analyze_verilog_expression_types(parsed.syntax, semantic);
    require(has_code(result, "InvalidMemberAccess"),
            "invalid struct member access was not diagnosed");
    require(has_code(result, "UnknownAggregateMemberType") &&
                has_code(result, "PackedUnionWidthMismatch"),
            "unknown aggregate members or incompatible packed unions were not diagnosed");
    const auto unknown_member = std::find_if(
        result.diagnostics.begin(), result.diagnostics.end(), [](const auto& diagnostic) {
            return diagnostic.code == "UnknownAggregateMemberType";
        });
    require(unknown_member != result.diagnostics.end() &&
                unknown_member->location.source &&
                unknown_member->location.location.line > 0,
            "unknown aggregate member diagnostic lost its source location");

    const auto* package =
        lookup_verilog_symbol(*semantic.model.compilation_unit_scope, "type_pkg", false);
    const auto* module =
        lookup_verilog_symbol(*semantic.model.compilation_unit_scope, "type_top", false);
    require(package && package->member_scope && module && module->member_scope,
            "type fixture scopes were not collected");
    const auto get_type = [&](const VerilogSemanticScope& scope, std::string_view name)
        -> const VerilogIntegralType& {
        const auto* symbol = lookup_verilog_symbol(scope, name, false);
        require(symbol && result.symbol_types.contains(symbol),
                "expected a resolved user-defined type symbol");
        return result.symbol_types.at(symbol);
    };

    const auto& forward = get_type(*package->member_scope, "first_t");
    require(forward.is_known_integral() && forward.bit_width == 3 &&
                forward.alias_name == "first_t",
            "forward typedef reference did not resolve");
    const auto& state = get_type(*package->member_scope, "state_t");
    require(state.category == VerilogTypeCategory::Enumeration && state.bit_width == 3 &&
                state.is_signed && state.identity_name == "state_t" &&
                state.enumerators.size() == 3 &&
                state.declaration_location.source &&
                state.declaration_location.location.line > 0 &&
                state.enumerators[0].value->to_uint64() == 7 &&
                state.enumerators[2].value->to_uint64() == 1,
            "enum identity, signed width, or implicit value is incorrect");
    const auto& unsigned_state = get_type(*package->member_scope, "unsigned_state_t");
    require(unsigned_state.category == VerilogTypeCategory::Enumeration &&
                unsigned_state.bit_width == 2 && !unsigned_state.is_signed &&
                unsigned_state.enumerators[1].value->to_uint64() == 1,
            "unsigned enum width or implicit value is incorrect");

    const auto& pair = get_type(*package->member_scope, "pair_t");
    require(pair.category == VerilogTypeCategory::Structure && pair.packed &&
                pair.bit_width == 8 && pair.members.size() == 2 &&
                pair.members[0].bit_offset == 4 && pair.members[1].bit_offset == 0,
            "packed struct member layout is incorrect");
    const auto& nested = get_type(*package->member_scope, "nested_t");
    require(nested.bit_width == 9 && nested.members.size() == 2 &&
                nested.members[0].bit_offset == 1 &&
                nested.members[1].bit_offset == 0,
            "nested packed struct layout is incorrect");
    const auto& record = get_type(*package->member_scope, "record_t");
    require(record.category == VerilogTypeCategory::Structure && !record.packed &&
                !record.bit_width && record.members.size() == 2 &&
                record.members[0].type->bit_width == 32 &&
                record.members[1].type->bit_width == 2,
            "unpacked struct members or unknown aggregate width are incorrect");
    const auto& union_type = get_type(*package->member_scope, "byte_u");
    require(union_type.category == VerilogTypeCategory::Union &&
                union_type.bit_width == 8 && union_type.members.size() == 2 &&
                union_type.members[0].bit_offset == 0 &&
                union_type.members[1].bit_offset == 0,
            "packed union shared storage layout is incorrect");
    const auto& unpacked_union = get_type(*package->member_scope, "data_u");
    require(unpacked_union.category == VerilogTypeCategory::Union &&
                !unpacked_union.packed && !unpacked_union.bit_width &&
                unpacked_union.members.size() == 2 &&
                unpacked_union.members[0].type->bit_width == 32 &&
                unpacked_union.members[1].type->bit_width == 8,
            "unpacked union member types or unknown width are incorrect");
    const auto& memory = get_type(*package->member_scope, "memory_t");
    require(memory.category == VerilogTypeCategory::Array &&
                memory.unpacked_element_count == 3 && memory.element_type &&
                memory.element_type->bit_width == 4,
            "unpacked array element type and extent were not retained");
    const auto& matrix = get_type(*package->member_scope, "matrix_t");
    require(matrix.category == VerilogTypeCategory::Array &&
                matrix.unpacked_element_count == 2 && matrix.element_type &&
                matrix.element_type->category == VerilogTypeCategory::Array &&
                matrix.element_type->unpacked_element_count == 3 &&
                matrix.element_type->element_type &&
                matrix.element_type->element_type->bit_width == 4,
            "multidimensional array dimensions were not retained independently");

    const auto& module_scope = *module->member_scope;
    const auto* pair_symbol = lookup_verilog_symbol(module_scope, "pair", false);
    const auto* nested_symbol = lookup_verilog_symbol(module_scope, "nested", false);
    const auto* memory_symbol = lookup_verilog_symbol(module_scope, "memory", false);
    const auto* local_symbol = lookup_verilog_symbol(module_scope, "local_value", false);
    const auto* second_nibble =
        lookup_verilog_symbol(module_scope, "second_nibble", false);
    require(pair_symbol && nested_symbol && memory_symbol &&
                result.symbol_types.at(pair_symbol).bit_width == 8 &&
                result.symbol_types.at(nested_symbol).bit_width == 9 &&
                result.symbol_types.at(memory_symbol).category ==
                    VerilogTypeCategory::Array && local_symbol &&
                result.symbol_types.at(local_symbol).bit_width == 2 && second_nibble &&
                result.symbol_types.at(second_nibble).bit_width == 2,
            "package typedefs were not applied to module declarations");
    const auto& assignments = parsed.syntax.modules.front().assignments;
    require(assignments.size() == 6 &&
                result.expression_types.at(assignments[0].right.get()).bit_width == 4 &&
                result.expression_types.at(assignments[1].right.get()).bit_width == 4 &&
                result.expression_types.at(assignments[2].right.get()).bit_width == 4 &&
                result.expression_types.at(assignments[3].right.get()).bit_width == 4 &&
                result.expression_types.at(assignments[4].right.get()).category ==
                    VerilogTypeCategory::Enumeration,
            "struct member access or array indexing inferred an incorrect type");

    const auto* generated = [&]() -> const VerilogSemanticScope* {
        std::function<const VerilogSemanticScope*(const VerilogSemanticScope&)> find =
            [&](const VerilogSemanticScope& scope) -> const VerilogSemanticScope* {
            if (scope.name == "generated" &&
                scope.kind == VerilogSemanticScopeKind::Generate)
                return &scope;
            for (const auto& child : scope.children)
                if (const auto* found = find(*child))
                    return found;
            return nullptr;
        };
        return find(*semantic.model.compilation_unit_scope);
    }();
    require(generated, "generated type scope was not collected");
    const auto* generated_value =
        lookup_verilog_symbol(*generated, "generated_value", false);
    require(generated_value && result.symbol_types.at(generated_value).bit_width == 3,
            "typedef in a generate scope was not resolved");
}

void test_type_scope_isolation_and_duplicate_diagnostics() {
    const auto parsed = parse_verilog_syntax(
        "module defines_type;\n"
        "  typedef logic [1:0] local_t;\n"
        "endmodule\n"
        "module cannot_see_sibling_type;\n"
        "  local_t value;\n"
        "endmodule\n"
        "module duplicate_types;\n"
        "  typedef logic first_t;\n"
        "  typedef logic second_t;\n"
        "  typedef logic first_t;\n"
        "  typedef enum logic [1:0] { SAME = 0, SAME = 1 } duplicate_enum_t;\n"
        "endmodule\n");
    require(!parsed.has_errors(), "scope and duplicate type fixture must parse");
    const auto semantic = analyze_verilog_semantics(parsed.syntax);
    const auto types = analyze_verilog_expression_types(parsed.syntax, semantic);
    const auto duplicate = std::find_if(
        semantic.diagnostics.begin(), semantic.diagnostics.end(), [](const auto& diagnostic) {
            return diagnostic.code == "DuplicateDeclaration";
        });
    require(duplicate != semantic.diagnostics.end() && duplicate->related_location &&
                duplicate->location.location.line > duplicate->related_location->location.line,
            "duplicate type or enum declarations lost duplicate/original locations");
    require(has_code(types, "UnknownTypeReference"),
            "type defined in a sibling module leaked into another scope or was not diagnosed");

    const auto* root = semantic.model.compilation_unit_scope.get();
    const auto* defining = lookup_verilog_symbol(*root, "defines_type", false);
    const auto* sibling = lookup_verilog_symbol(*root, "cannot_see_sibling_type", false);
    require(defining && sibling && defining->member_scope && sibling->member_scope,
            "module scopes were not available for type isolation checks");
    const auto* local_symbol =
        lookup_verilog_symbol(*defining->member_scope, "local_t", false);
    const auto* leaked_symbol =
        lookup_verilog_symbol(*sibling->member_scope, "value", false);
    require(local_symbol && types.symbol_types.contains(local_symbol) &&
                leaked_symbol && types.symbol_types.at(leaked_symbol).category ==
                                     VerilogTypeCategory::Unknown,
            "module-local type visibility crossed a scope boundary");
}

void test_user_type_elaboration_preserves_parameter_width() {
    const auto parsed = parse_verilog_syntax(
        "package parameter_types;\n"
        "  typedef logic [3:0] nibble_t;\n"
        "endpackage\n"
        "module child #(parameter parameter_types::nibble_t P = 8'hff); endmodule\n"
        "module top;\n"
        "  child #(.P(8'hab)) first();\n"
        "endmodule\n");
    require(!parsed.has_errors(), "typed parameter fixture must parse");
    VerilogElaborationOptions options;
    options.top_module = "top";
    const auto elaborated = elaborate_verilog_design(parsed.syntax, options);
    require(!elaborated.has_errors() && elaborated.top &&
                elaborated.top->children.size() == 1,
            "typedef-typed parameter should elaborate");
    const auto& child = *elaborated.top->children.front();
    require(child.parameter_types.at("P").bit_width == 4 &&
                child.parameters.at("P").bit_width == 4 &&
                child.parameters.at("P").to_uint64() == 0xb,
            "typedef parameter type was not preserved through instance elaboration");
}

void test_enum_parameter_values_are_instance_specific() {
    const auto parsed = parse_verilog_syntax(
        "package enum_types;\n"
        "  typedef enum logic signed [2:0] { NEG = -1, ZERO = 0, ONE = 1 } state_t;\n"
        "endpackage\n"
        "module child #(parameter enum_types::state_t P = enum_types::ONE); endmodule\n"
        "module top;\n"
        "  child #(.P(enum_types::NEG)) negative();\n"
        "  child #(.P(enum_types::ZERO)) zero();\n"
        "endmodule\n");
    require(!parsed.has_errors(), "enum parameter fixture must parse");
    VerilogElaborationOptions options;
    options.top_module = "top";
    const auto elaborated = elaborate_verilog_design(parsed.syntax, options);
    require(!elaborated.has_errors() && elaborated.top &&
                elaborated.top->children.size() == 2,
            "enum-valued parameter overrides should elaborate");
    const auto& negative = *elaborated.top->children[0];
    const auto& zero = *elaborated.top->children[1];
    require(negative.parameter_types.at("P").category ==
                    VerilogTypeCategory::Enumeration &&
                negative.parameter_types.at("P").bit_width == 3 &&
                negative.parameters.at("P").bit_width == 3 &&
                negative.parameters.at("P").is_signed &&
                negative.parameters.at("P").to_int64() == -1 &&
                zero.parameters.at("P").to_int64() == 0,
            "enum type or instance-specific signed values were not preserved");
}

void test_enum_type_matches_slang() {
    constexpr std::string_view source =
        "package enum_reference;\n"
        "  typedef enum logic signed [2:0] { NEG = -1, ZERO = 0, ONE = 1 } state_t;\n"
        "endpackage\n"
        "module enum_type_reference;\n"
        "  localparam enum_reference::state_t VALUE = enum_reference::ONE;\n"
        "endmodule\n";
    const auto parsed = parse_verilog_syntax(std::string(source));
    require(!parsed.has_errors(), "native enum differential fixture must parse");
    const auto semantic = analyze_verilog_semantics(parsed.syntax);
    const auto native = analyze_verilog_expression_types(parsed.syntax, semantic);
    const auto* package =
        lookup_verilog_symbol(*semantic.model.compilation_unit_scope, "enum_reference", false);
    require(package && package->member_scope,
            "native enum differential package scope was not collected");
    const auto* enum_symbol =
        lookup_verilog_symbol(*package->member_scope, "state_t", false);
    require(enum_symbol && native.symbol_types.contains(enum_symbol),
            "native enum differential type was not resolved");
    const auto& native_type = native.symbol_types.at(enum_symbol);

    slang::SourceManager source_manager;
    auto tree = slang::syntax::SyntaxTree::fromText(
        std::string(source), source_manager, "enum_type_reference.sv");
    slang::ast::Compilation compilation;
    compilation.addSyntaxTree(tree);
    require(compilation.getAllDiagnostics().empty(),
            "Slang rejected the supported enum type differential fixture");
    const auto top = std::find_if(
        compilation.getRoot().topInstances.begin(), compilation.getRoot().topInstances.end(),
        [](const auto* instance) { return instance->name == "enum_type_reference"; });
    require(top != compilation.getRoot().topInstances.end(),
            "Slang did not elaborate enum differential module");
    const slang::ast::ParameterSymbol* parameter = nullptr;
    for (const auto& member : (*top)->body.members())
        if (member.kind == slang::ast::SymbolKind::Parameter && member.name == "VALUE")
            parameter = &member.as<slang::ast::ParameterSymbol>();
    require(parameter, "Slang did not expose enum parameter type");
    const auto& slang_type = parameter->getType();
    require(native_type.bit_width == slang_type.getBitWidth() &&
                native_type.is_signed == slang_type.isSigned() &&
                native_type.four_state == slang_type.isFourState(),
            "native enum width/signedness/state differs from Slang");
}

void test_expression_type_with_type_analysis() {
    const auto parsed = parse_verilog_syntax(
        "module type_with_semantic #(parameter int WIDTH = 8) "
        "(input logic [WIDTH-1:0] data, output logic [3:0] result);\n"
        "  localparam logic [2:0] LOCAL = 3'b001;\n"
        "  assign result = data[3:0] + LOCAL;\n"
        "endmodule\n");
    require(!parsed.has_errors(), "type analysis fixture must parse");
    const auto semantic = analyze_verilog_semantics(parsed.syntax);
    const auto type_analysis =
        analyze_verilog_expression_types(parsed.syntax, semantic);
    require(!type_analysis.has_errors(),
            "type analysis must not produce errors");

    const auto* root = semantic.model.compilation_unit_scope.get();
    const auto& module_scope = child_scope(*root, "type_with_semantic");
    const auto* width_symbol =
        lookup_verilog_symbol(module_scope, "WIDTH", false);
    require(width_symbol && type_analysis.symbol_types.contains(width_symbol),
            "WIDTH parameter type was not recorded in type analysis");
    const auto& width_type = type_analysis.symbol_types.at(width_symbol);
    require(width_type.is_known_integral() && *width_type.bit_width == 32 &&
                width_type.is_signed,
            "WIDTH parameter type did not match expected int");

    const auto* local_symbol =
        lookup_verilog_symbol(module_scope, "LOCAL", false);
    require(local_symbol && type_analysis.symbol_types.contains(local_symbol),
            "LOCAL localparam type was not recorded in type analysis");
    const auto& local_type = type_analysis.symbol_types.at(local_symbol);
    require(local_type.is_known_integral() && *local_type.bit_width == 3,
            "LOCAL localparam type did not match expected 3-bit logic");

    VerilogExpression width_expr;
    width_expr.kind = VerilogExpressionKind::Identifier;
    width_expr.text = "WIDTH";
    width_expr.location = parsed.syntax.modules.front().parameters.front().location;
    const auto inferred =
        infer_verilog_expression_type(width_expr, type_analysis);
    require(inferred.type.is_known_integral() && *inferred.type.bit_width == 32 &&
                inferred.type.is_signed,
            "new expression type overload did not resolve WIDTH from type analysis");
}

void test_expression_types_against_slang() {
    constexpr std::string_view expression = "8'hff < 16'h1";
    const auto& native_expression = parse_expression(std::string(expression));
    const auto native = infer_verilog_expression_type(native_expression);

    slang::SourceManager source_manager;
    auto tree = slang::syntax::SyntaxTree::fromText(
        "module type_reference; localparam R = " + std::string(expression) +
            "; endmodule\n",
        source_manager, "type_reference.sv");
    slang::ast::Compilation compilation;
    compilation.addSyntaxTree(tree);
    require(compilation.getAllDiagnostics().empty(),
            "Slang rejected the supported type differential expression");
    const auto& root = compilation.getRoot();
    const auto top = std::find_if(root.topInstances.begin(), root.topInstances.end(),
                                  [](const auto* instance) {
                                      return instance->name == "type_reference";
                                  });
    require(top != root.topInstances.end(),
            "Slang did not elaborate type reference module");
    const slang::ast::ParameterSymbol* parameter = nullptr;
    for (const auto& member : (*top)->body.members())
        if (member.kind == slang::ast::SymbolKind::Parameter && member.name == "R")
            parameter = &member.as<slang::ast::ParameterSymbol>();
    require(parameter, "Slang did not expose the reference parameter type");
    const auto& slang_type = parameter->getType();
    require(native.type.bit_width == slang_type.getBitWidth() &&
                native.type.is_signed == slang_type.isSigned() &&
                native.type.four_state == slang_type.isFourState(),
            "native expression width/signedness/state differs from Slang");
}

const VerilogSemanticScope& child_scope(const VerilogSemanticScope& root,
                                        std::string_view name) {
    const auto found = std::find_if(root.children.begin(), root.children.end(),
                                    [&](const auto& child) {
                                        return child->name == name;
                                    });
    require(found != root.children.end(), "expected scope was not found");
    return **found;
}

} // namespace

int main() {
    try {
        test_literal_and_operator_types();
        test_declarations_assignments_and_scope_types();
        test_instance_parameter_types_are_independent();
        test_supported_binary_type_against_slang();
        test_unknown_user_types_remain_unknown();
        test_generated_and_procedural_declaration_types();
        test_typedef_enum_aggregate_and_array_types();
        test_type_scope_isolation_and_duplicate_diagnostics();
        test_user_type_elaboration_preserves_parameter_width();
        test_enum_parameter_values_are_instance_specific();
        test_enum_type_matches_slang();
        test_expression_type_with_type_analysis();
        test_expression_types_against_slang();
        std::cout << "Native Verilog type-analysis tests passed\n";
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << "Native Verilog type-analysis test failure: " << error.what() << '\n';
        return 1;
    }
}
