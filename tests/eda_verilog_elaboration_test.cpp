#include "labneura/eda/verilog_elaboration.h"

#include <slang/ast/Compilation.h>
#include <slang/ast/symbols/BlockSymbols.h>
#include <slang/ast/symbols/CompilationUnitSymbols.h>
#include <slang/ast/symbols/InstanceSymbols.h>
#include <slang/syntax/SyntaxTree.h>

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

bool has_code(const std::vector<VerilogElaborationDiagnostic>& diagnostics,
              std::string_view code) {
    return std::any_of(diagnostics.begin(), diagnostics.end(),
                       [&](const auto& diagnostic) { return diagnostic.code == code; });
}

bool has_code(const std::vector<VerilogConstantDiagnostic>& diagnostics,
              std::string_view code) {
    return std::any_of(diagnostics.begin(), diagnostics.end(),
                       [&](const auto& diagnostic) { return diagnostic.code == code; });
}

uint64_t parameter_value(const VerilogElaboratedInstance& instance, std::string_view name) {
    const auto found = instance.parameters.find(std::string(name));
    if (found == instance.parameters.end()) {
        std::string keys;
        for (const auto& [key, value] : instance.parameters) {
            (void)value;
            keys += (keys.empty() ? "" : ",") + key;
        }
        throw std::runtime_error("missing elaborated parameter '" + std::string(name) +
                                 "' in instance " + instance.instance_name + "; found: " + keys);
    }
    const auto value = found->second.to_uint64();
    if (!value)
        throw std::runtime_error("parameter '" + std::string(name) +
                                 "' is not a known uint64 value");
    return *value;
}

const VerilogElaboratedGenerateScope& generate_scope(
    const VerilogElaboratedInstance& instance, std::string_view name) {
    const auto found = std::find_if(
        instance.generate_scopes.begin(), instance.generate_scopes.end(),
        [&](const auto& scope) { return scope->name == name; });
    if (found == instance.generate_scopes.end())
        throw std::runtime_error("generated scope '" + std::string(name) + "' was not elaborated");
    return **found;
}

size_t count_sl_examples(const slang::ast::Scope& scope) {
    size_t count = 0;
    for (const auto& member : scope.members()) {
        if (member.kind == slang::ast::SymbolKind::Instance) {
            ++count;
            count += count_sl_examples(member.as<slang::ast::InstanceSymbol>().body);
        }
        else if (member.kind == slang::ast::SymbolKind::GenerateBlock) {
            count += count_sl_examples(member.as<slang::ast::GenerateBlockSymbol>());
        }
        else if (member.kind == slang::ast::SymbolKind::GenerateBlockArray) {
            for (const auto* entry :
                 member.as<slang::ast::GenerateBlockArraySymbol>().entries)
                count += count_sl_examples(*entry);
        }
    }
    return count;
}

const VerilogExpression& only_parameter_expression(std::string expression) {
    auto parsed = parse_verilog_syntax("module eval; parameter R = " + expression +
                                       "; endmodule");
    if (parsed.has_errors() || parsed.syntax.modules.size() != 1 ||
        parsed.syntax.modules.front().parameters.empty() ||
        !parsed.syntax.modules.front().parameters.front().value)
        throw std::runtime_error("could not parse constant-expression test input");
    static thread_local VerilogSyntaxParseResult storage;
    storage = std::move(parsed);
    return *storage.syntax.modules.front().parameters.front().value;
}

uint64_t eval_uint(std::string expression, size_t expected_width, bool expected_signed = false) {
    const auto& ast = only_parameter_expression(std::move(expression));
    const auto result = evaluate_verilog_constant(ast);
    require(!result.has_errors() && result.value.has_value(),
            "constant expression unexpectedly failed to evaluate");
    if (result.value->bit_width != expected_width ||
        result.value->is_signed != expected_signed) {
        throw std::runtime_error(
            "constant expression width or signedness is incorrect: " +
            std::to_string(result.value->bit_width) + " bits, signed=" +
            (result.value->is_signed ? "true" : "false"));
    }
    const auto value = result.value->to_uint64();
    require(value.has_value(), "expected a known unsigned-sized constant");
    return *value;
}

void test_constant_literal_width_and_signedness() {
    require(eval_uint("8'hff + 8'd1", 8) == 0,
            "sized unsigned arithmetic did not wrap to its expression width");
    require(eval_uint("8'shff + 8'sd1", 8, true) == 0,
            "sized signed arithmetic did not preserve width and signedness");
    require(eval_uint("8'sd1 + 8'sd2", 8, true) == 3,
            "sized signed decimal literals were incorrectly sign-extended");
    require(eval_uint("1 + 2", 32, true) == 3,
            "unsized decimal integer was not evaluated as a 32-bit signed value");
    require(eval_uint("1 + 2 * 3", 32, true) == 7 &&
                eval_uint("20 / 3", 32, true) == 6 &&
                eval_uint("20 % 3", 32, true) == 2 &&
                eval_uint("3 ** 3", 32, true) == 27,
            "arithmetic precedence or division/modulo/power evaluation failed");
    require(eval_uint("16'h12_34", 16) == 0x1234,
            "underscored based literal digits were placed incorrectly");

    const auto& signed_negative = only_parameter_expression("-8'sd2");
    const auto negative = evaluate_verilog_constant(signed_negative);
    require(negative.value && negative.value->to_int64() == -2 &&
                negative.value->bit_width == 8 && negative.value->is_signed,
            "signed unary negation is incorrect");
    require(negative.value->location.byte_offset == signed_negative.location.byte_offset,
            "evaluated value did not retain its source location");

    require(eval_uint("8'shff < 8'sd1", 1) == 1,
            "signed comparison did not compare two's-complement values");
    require(eval_uint("8'hff < 8'd1", 1) == 0,
            "unsigned comparison did not compare unsigned values");
    const auto left_shift = eval_uint("8'b0011 << 2", 8);
    require(left_shift == 12,
            "left shift result width or value is incorrect: " + std::to_string(left_shift));
    require(eval_uint("8'b1000 >> 2", 8) == 2,
            "logical right shift result is incorrect");
    require(eval_uint("4'b1010 & 4'b1100", 4) == 8,
            "bitwise operation is incorrect");
    require(eval_uint("&4'b0x11", 1) == 0 && eval_uint("|4'b1x00", 1) == 1,
            "reduction operators ignored controlling bits alongside X values");
    require(eval_uint("1 ? 8'd5 : 8'd6", 8) == 5 &&
                eval_uint("4'b10x1 ? 2'b10 : 2'b10", 2) == 2 &&
                eval_uint("8'b1010_1100[3:0]", 4) == 12,
            "conditional merge or constant bit/part select evaluation failed");
    require(eval_uint("1'b1 && 1'b0", 1) == 0 &&
                eval_uint("4'b0010 == 4'd2", 1) == 1,
            "logical or equality operator result is incorrect");
    const auto& wide_unsigned = only_parameter_expression("64'hffff_ffff_ffff_ffff");
    const auto wide_result = evaluate_verilog_constant(wide_unsigned);
    require(wide_result.value && !wide_result.value->to_int64(),
            "unsigned value outside int64 range was silently converted");
}

void test_unknown_bits_concat_replication_and_errors() {
    const auto& unknown_expression =
        only_parameter_expression("4'bx001 & 4'b0001");
    const auto unknown = evaluate_verilog_constant(unknown_expression);
    require(unknown.value && unknown.value->to_binary_string() == "0001",
            "four-state bitwise logic did not resolve known controlling bits");

    const auto& replicated = only_parameter_expression("{4{2'b10}}");
    const auto replication = evaluate_verilog_constant(replicated);
    require(replication.value && replication.value->bit_width == 8 &&
                replication.value->to_binary_string() == "10101010",
            "replication expression did not preserve concatenation order");

    const auto& divide_by_zero = only_parameter_expression("8'd4 / 8'd0");
    const auto division = evaluate_verilog_constant(divide_by_zero);
    require(!division.value && has_code(division.diagnostics, "DivisionByZero"),
            "divide-by-zero did not produce a structured evaluation diagnostic");

    const auto& invalid_shift = only_parameter_expression("8'd1 << -1");
    const auto shift = evaluate_verilog_constant(invalid_shift);
    require(!shift.value && has_code(shift.diagnostics, "InvalidShift"),
            "negative shift amount did not produce a structured diagnostic");

    const auto& unresolved = only_parameter_expression("NOT_A_PARAMETER + 1");
    const auto unresolved_result = evaluate_verilog_constant(unresolved);
    require(!unresolved_result.value &&
                has_code(unresolved_result.diagnostics, "UnresolvedParameter"),
            "unresolved constant identifier was not diagnosed");

    const auto& clog2 = only_parameter_expression("$clog2(9)");
    const auto clog2_result = evaluate_verilog_constant(clog2);
    require(clog2_result.value && clog2_result.value->to_uint64() == 4 &&
                clog2_result.value->bit_width == 32,
            "$clog2 constant call returned an incorrect value: " +
                (clog2_result.value
                     ? clog2_result.value->to_binary_string() + " width=" +
                           std::to_string(clog2_result.value->bit_width)
                     : "no value"));
    const auto& negative_clog2 = only_parameter_expression("$clog2(-1)");
    const auto negative_clog2_result = evaluate_verilog_constant(negative_clog2);
    require(!negative_clog2_result.value &&
                has_code(negative_clog2_result.diagnostics,
                         "InvalidConstantFunctionArgument"),
            "$clog2 accepted a negative argument");
}

void test_parameter_dependencies_and_instance_overrides() {
    const auto parsed = parse_verilog_syntax(
        "module child #(parameter int WIDTH = 4, parameter DEPTH = WIDTH + 2,\n"
        "               localparam TOTAL = DEPTH * 2) (input logic [WIDTH-1:0] data);\n"
        "endmodule\n"
        "module top;\n"
        "  child #(.WIDTH(5)) first();\n"
        "  child #(3) second();\n"
        "  parameter BASE = 8;\n"
        "  child #(.WIDTH(BASE + 1)) third();\n"
        "endmodule\n");
    require(!parsed.has_errors(), "parameter elaboration fixture must parse");
    if (parsed.syntax.modules.front().parameters.size() != 3) {
        std::string names;
        for (const auto& parameter : parsed.syntax.modules.front().parameters)
            names += (names.empty() ? "" : ",") + parameter.name;
        throw std::runtime_error("child parameter AST names: " + names);
    }
    if (parsed.syntax.modules.front().parameters.front().name != "WIDTH" ||
        parsed.syntax.modules.front().parameters.front().local_parameter) {
        throw std::runtime_error("first child parameter parsed as '" +
                                 parsed.syntax.modules.front().parameters.front().name + "'");
    }
    VerilogElaborationOptions options;
    options.top_module = "top";
    auto result = elaborate_verilog_design(parsed.syntax, options);
    require(!result.has_errors() && result.top && result.top->children.size() == 3,
            "parameterized hierarchy did not elaborate cleanly");
    const auto& first = *result.top->children[0];
    const auto& second = *result.top->children[1];
    const auto& third = *result.top->children[2];
    require(parameter_value(first, "WIDTH") == 5 &&
                parameter_value(first, "DEPTH") == 7 &&
                parameter_value(first, "TOTAL") == 14,
            "named override or declaration-order-independent parameter dependency failed");
    require(parameter_value(second, "WIDTH") == 3 &&
                parameter_value(second, "DEPTH") == 5 &&
                parameter_value(second, "TOTAL") == 10 &&
                parameter_value(first, "WIDTH") != parameter_value(second, "WIDTH"),
            "positional override or isolated instance parameter environment failed");
    require(parameter_value(third, "WIDTH") == 9,
            "instance override did not resolve in the caller parameter environment");

    const auto reordered = parse_verilog_syntax(
        "module forward;\n"
        "  parameter B = A + 1;\n"
        "  parameter A = 6;\n"
        "  localparam C = B * 2;\n"
        "endmodule\n");
    VerilogElaborationOptions forward_options;
    forward_options.top_module = "forward";
    auto forward = elaborate_verilog_design(reordered.syntax, forward_options);
    require(forward.top && parameter_value(*forward.top, "B") == 7 &&
                parameter_value(*forward.top, "C") == 14,
            "forward parameter dependencies or localparam evaluation failed");

    const auto package_syntax = parse_verilog_syntax(
        "package shared; parameter LIMIT = 9; endpackage\n"
        "module package_top; parameter WIDTH = shared::LIMIT + 1; endmodule\n");
    VerilogElaborationOptions package_options;
    package_options.top_module = "package_top";
    const auto package_result = elaborate_verilog_design(package_syntax.syntax, package_options);
    require(package_result.top &&
                parameter_value(*package_result.top, "WIDTH") == 10,
            "package-scoped parameter reference did not resolve");

    const auto sized_parameters = parse_verilog_syntax(
        "module sized;\n"
        "  parameter [7:0] VALUE = 300;\n"
        "  parameter signed [7:0] NEGATIVE = -1;\n"
        "endmodule\n");
    VerilogElaborationOptions sized_options;
    sized_options.top_module = "sized";
    const auto sized_result = elaborate_verilog_design(sized_parameters.syntax, sized_options);
    require(sized_result.top && parameter_value(*sized_result.top, "VALUE") == 44 &&
                sized_result.top->parameters.at("VALUE").bit_width == 8,
            "packed parameter declaration did not size and truncate its value");
    require(sized_result.top->parameters.at("NEGATIVE").to_int64() == -1 &&
                sized_result.top->parameters.at("NEGATIVE").bit_width == 8,
            "signed packed parameter declaration did not preserve signed value");

    const auto invalid_overrides = parse_verilog_syntax(
        "module target #(parameter A = 1, localparam LOCKED = 2); endmodule\n"
        "module invalid_override_top;\n"
        "  target #(.MISSING(3), .LOCKED(4), .A(7), .A(8), 5, 6) bad();\n"
        "endmodule\n");
    VerilogElaborationOptions invalid_options;
    invalid_options.top_module = "invalid_override_top";
    const auto invalid_result =
        elaborate_verilog_design(invalid_overrides.syntax, invalid_options);
    require(has_code(invalid_result.diagnostics, "InvalidParameterOverride") &&
                has_code(invalid_result.diagnostics, "TooManyParameterOverrides") &&
                has_code(invalid_result.diagnostics, "DuplicateParameterOverride"),
            "invalid, duplicate, or excessive parameter overrides were accepted");
}

void test_parameter_cycles_and_invalid_expressions() {
    const auto parsed = parse_verilog_syntax(
        "module invalid_params;\n"
        "  parameter A = B + 1;\n"
        "  parameter B = A + 1;\n"
        "  parameter BAD = 4 / 0;\n"
        "  logic runtime_value;\n"
        "  parameter NOT_CONSTANT = runtime_value + 1;\n"
        "endmodule\n");
    VerilogElaborationOptions options;
    options.top_module = "invalid_params";
    const auto result = elaborate_verilog_design(parsed.syntax, options);
    require(has_code(result.diagnostics, "CircularParameterDependency") &&
                has_code(result.diagnostics, "DivisionByZero") &&
                has_code(result.diagnostics, "UnresolvedParameter"),
            "circular, invalid, and runtime-dependent parameters were not diagnosed");
    const auto division = std::find_if(
        result.diagnostics.begin(), result.diagnostics.end(),
        [](const auto& diagnostic) { return diagnostic.code == "DivisionByZero"; });
    require(division != result.diagnostics.end() && division->location.location.line == 4,
            "constant-evaluation diagnostic lost its source line");
}

void test_hierarchy_top_selection_missing_and_recursive() {
    const auto nested = parse_verilog_syntax(
        "module leaf; endmodule\n"
        "module middle; leaf u_leaf(); endmodule\n"
        "module top; middle u_middle(); endmodule\n");
    const auto automatic_top = elaborate_verilog_design(nested.syntax);
    require(!automatic_top.has_errors() && automatic_top.top &&
                automatic_top.top->module_definition->name == "top" &&
                automatic_top.top->children.size() == 1 &&
                automatic_top.top->children.front()->children.size() == 1,
            "unique top selection or nested hierarchy construction failed");

    const auto ambiguous_syntax =
        parse_verilog_syntax("module first; endmodule module second; endmodule");
    const auto ambiguous = elaborate_verilog_design(ambiguous_syntax.syntax);
    require(has_code(ambiguous.diagnostics, "AmbiguousTopModule") && !ambiguous.top &&
                ambiguous.top_module_candidates.size() == 2,
            "ambiguous top modules were not reported");

    const auto missing_syntax =
        parse_verilog_syntax("module top; missing u(); endmodule");
    VerilogElaborationOptions missing_options;
    missing_options.top_module = "top";
    const auto missing = elaborate_verilog_design(missing_syntax.syntax, missing_options);
    require(has_code(missing.diagnostics, "MissingModuleDefinition"),
            "missing module definition was not reported during elaboration");

    const auto recursive_syntax =
        parse_verilog_syntax("module recurse; recurse u(); endmodule");
    VerilogElaborationOptions recursive_options;
    recursive_options.top_module = "recurse";
    const auto recursive = elaborate_verilog_design(recursive_syntax.syntax, recursive_options);
    require(has_code(recursive.diagnostics, "RecursiveModuleInstantiation"),
            "recursive module instantiation was not detected");
}

void test_macro_generated_parameter_location() {
    const auto parsed = parse_verilog_syntax(
        "`define SIZE_EXPR 6 + 2\n"
        "module macro_top;\n"
        "  parameter WIDTH = `SIZE_EXPR;\n"
        "endmodule\n");
    const auto& expression = *parsed.syntax.modules.front().parameters.front().value;
    require(expression.location.origin_kind == VerilogPreprocessedOriginKind::MacroExpansion &&
                !expression.location.expansion_stack.empty(),
            "macro-generated parameter AST lost its expansion origin");
    VerilogElaborationOptions options;
    options.top_module = "macro_top";
    const auto result = elaborate_verilog_design(parsed.syntax, options);
    require(result.top && parameter_value(*result.top, "WIDTH") == 8,
            "macro-generated parameter expression did not evaluate");
    require(result.top->parameters.at("WIDTH").location.origin_kind ==
                    VerilogPreprocessedOriginKind::MacroExpansion &&
                !result.top->parameters.at("WIDTH").location.expansion_stack.empty(),
            "evaluated macro parameter lost its source-map origin");
}

void test_if_case_and_for_generate() {
    const auto parsed = parse_verilog_syntax(
        "module leaf #(parameter ID = 0); endmodule\n"
        "module top #(parameter ENABLE = 1, parameter MODE = 2);\n"
        "  generate\n"
        "    if (ENABLE) begin : active\n"
        "      logic flag;\n"
        "      leaf #(.ID(1)) active_leaf();\n"
        "    end else begin : inactive\n"
        "      logic unused;\n"
        "    end\n"
        "    case (MODE)\n"
        "      1: begin : mode_one leaf #(.ID(11)) mode_leaf(); end\n"
        "      2, 3: begin : mode_many leaf #(.ID(22)) mode_leaf(); end\n"
        "      default: begin : mode_default leaf #(.ID(99)) mode_leaf(); end\n"
        "    endcase\n"
        "    for (genvar i = 0; i < 3; i = i + 1) begin : lanes\n"
        "      localparam INDEX = i;\n"
        "      logic generated_flag;\n"
        "      leaf #(.ID(i)) lane();\n"
        "    end\n"
        "  endgenerate\n"
        "  for (genvar j = 0; j < 0; j++) begin : empty_loop\n"
        "    leaf zero_iteration();\n"
        "  end\n"
        "endmodule\n");
    require(!parsed.has_errors(), "valid generate constructs should parse without errors");
    const auto& module = parsed.syntax.modules.back();
    require(module.generate_constructs.size() == 2 &&
                module.generate_constructs[0]->kind == VerilogGenerateKind::Region &&
                module.generate_constructs[0]->region_body->constructs.size() == 3 &&
                module.generate_constructs[1]->kind == VerilogGenerateKind::For,
            "explicit generate region or direct for-generate was not retained in the AST");
    require(module.generate_constructs[0]->region_body->constructs[0]->condition &&
                module.generate_constructs[0]->region_body->constructs[2]->initialization &&
                module.generate_constructs[0]->region_body->constructs[2]->loop_condition &&
                module.generate_constructs[0]->region_body->constructs[2]->step,
            "generate conditions and loop expressions were not preserved");

    VerilogElaborationOptions options;
    options.top_module = "top";
    const auto result = elaborate_verilog_design(parsed.syntax, options);
    require(!result.has_errors() && result.top,
            "valid if/case/for-generate design should elaborate cleanly");
    const auto& active = generate_scope(*result.top, "active");
    require(active.declarations.size() == 1 && active.children.size() == 1 &&
                active.children.front()->hierarchy_path == "top.active.active_leaf",
            "selected if-generate branch declarations or instance path are incorrect");
    (void)generate_scope(*result.top, "mode_many");
    const auto& lane0 = generate_scope(*result.top, "lanes[0]");
    const auto& lane1 = generate_scope(*result.top, "lanes[1]");
    const auto& lane2 = generate_scope(*result.top, "lanes[2]");
    require(lane0.declarations.size() == 1 && lane0.children.size() == 1 &&
                lane0.hierarchy_path == "top.lanes[0]" &&
                parameter_value(*lane2.children.front(), "ID") == 2 &&
                lane2.parameters.at("INDEX").to_uint64() == 2 &&
                lane1.children.front()->hierarchy_path == "top.lanes[1].lane" &&
                lane2.parameters.at("INDEX").bit_width == 32,
            "for-generate iteration scopes, indices, declarations, or instances are incorrect");
    require(std::none_of(result.top->generate_scopes.begin(), result.top->generate_scopes.end(),
                         [](const auto& scope) {
                             return scope->name == "inactive" || scope->name == "mode_one" ||
                                    scope->name == "mode_default" || scope->name == "empty_loop[0]";
                         }),
            "an unselected generate branch or zero-iteration loop was elaborated");

    const std::string slang_source =
        "module leaf; endmodule\n"
        "module top; for (genvar i=0; i<3; i++) begin : lanes leaf u(); end endmodule\n";
    slang::SourceManager source_manager;
    auto tree = slang::syntax::SyntaxTree::fromText(slang_source, source_manager, "generate.sv");
    slang::ast::Compilation compilation;
    compilation.addSyntaxTree(tree);
    const auto& root = compilation.getRoot();
    require(compilation.getAllDiagnostics().empty(),
            "Slang rejected the supported generate-loop differential fixture");
    const auto top_instance = std::find_if(
        root.topInstances.begin(), root.topInstances.end(),
        [](const auto* instance) { return instance->name == "top"; });
    require(top_instance != root.topInstances.end() &&
                count_sl_examples((*top_instance)->body) == 3,
            "Slang did not elaborate three generated module instances");
    const auto native_loop = parse_verilog_syntax(slang_source);
    VerilogElaborationOptions native_options;
    native_options.top_module = "top";
    const auto native = elaborate_verilog_design(native_loop.syntax, native_options);
    require(native.top && native.top->generate_scopes.size() == 3 &&
                std::all_of(native.top->generate_scopes.begin(),
                            native.top->generate_scopes.end(),
                            [](const auto& scope) { return scope->children.size() == 1; }),
            "native generate instance structure differs from Slang's elaborated count");
}

void test_generate_else_if_case_wildcards_and_default() {
    const auto parsed = parse_verilog_syntax(
        "module leaf; endmodule\n"
        "module false_branch #(parameter ENABLE = 0);\n"
        "  if (ENABLE) begin : first leaf first_leaf(); end\n"
        "  else if (1) begin : second leaf second_leaf(); end\n"
        "  else begin : last leaf last_leaf(); end\n"
        "endmodule\n"
        "module default_branch #(parameter MODE = 7);\n"
        "  case (MODE)\n"
        "    1: begin : one leaf one_leaf(); end\n"
        "    default: begin : fallback leaf fallback_leaf(); end\n"
        "  endcase\n"
        "endmodule\n"
        "module wildcard_branch;\n"
        "  casez (4'b10x0)\n"
        "    4'b10z0: begin : z_match leaf wildcard_leaf(); end\n"
        "    default: begin : no_match leaf no_match_leaf(); end\n"
        "  endcase\n"
        "endmodule\n"
        "module x_wildcard_branch;\n"
        "  casex (4'b10x0)\n"
        "    4'b1000: begin : x_match leaf x_leaf(); end\n"
        "    default: begin : x_no_match leaf x_no_match_leaf(); end\n"
        "  endcase\n"
        "endmodule\n");
    if (parsed.has_errors()) {
        std::string messages;
        for (const auto& diagnostic : parsed.diagnostics)
            messages += (messages.empty() ? "" : "; ") + diagnostic.code + " at " +
                        std::to_string(diagnostic.location.location.line) + ":" +
                        diagnostic.message;
        throw std::runtime_error("else-if and wildcard generate syntax should parse: " +
                                 messages);
    }

    VerilogElaborationOptions options;
    options.top_module = "false_branch";
    const auto false_result = elaborate_verilog_design(parsed.syntax, options);
    require(false_result.top && false_result.top->generate_scopes.size() == 1 &&
                false_result.top->generate_scopes.front()->name.starts_with("genblk") &&
                false_result.top->generate_scopes.front()->generate_scopes.size() == 1 &&
                false_result.top->generate_scopes.front()->generate_scopes.front()->name ==
                    "second",
            "else-if generate branch was not selected and elaborated");

    options.top_module = "default_branch";
    const auto default_result = elaborate_verilog_design(parsed.syntax, options);
    require(default_result.top && default_result.top->generate_scopes.size() == 1 &&
                default_result.top->generate_scopes.front()->name == "fallback",
            "case-generate default branch was not selected");

    options.top_module = "wildcard_branch";
    const auto wildcard_result = elaborate_verilog_design(parsed.syntax, options);
    require(wildcard_result.top && wildcard_result.top->generate_scopes.size() == 1 &&
                wildcard_result.top->generate_scopes.front()->name == "z_match",
            "casez-generate wildcard item did not match the constant selector");
    options.top_module = "x_wildcard_branch";
    const auto x_wildcard_result = elaborate_verilog_design(parsed.syntax, options);
    require(x_wildcard_result.top && x_wildcard_result.top->generate_scopes.size() == 1 &&
                x_wildcard_result.top->generate_scopes.front()->name == "x_match",
            "casex-generate wildcard selector did not match the constant item");
}

void test_nested_generate_and_instance_specific_conditions() {
    const auto parsed = parse_verilog_syntax(
        "module leaf; endmodule\n"
        "module child #(parameter ENABLE = 0);\n"
        "  if (ENABLE) begin : enabled\n"
        "    for (genvar i=0; i<2; i++) begin : nested\n"
        "      if (i == 1) begin : last leaf u(); end\n"
        "    end\n"
        "  end\n"
        "endmodule\n"
        "module top;\n"
        "  child #(.ENABLE(1)) yes();\n"
        "  child #(.ENABLE(0)) no();\n"
        "endmodule\n");
    require(!parsed.has_errors(), "nested generate fixture should parse cleanly");
    VerilogElaborationOptions options;
    options.top_module = "top";
    const auto result = elaborate_verilog_design(parsed.syntax, options);
    require(!result.has_errors() && result.top && result.top->children.size() == 2,
            "per-instance generate conditions should elaborate without diagnostics");
    const auto& enabled_instance = *result.top->children[0];
    const auto& disabled_instance = *result.top->children[1];
    const auto& enabled = generate_scope(enabled_instance, "enabled");
    require(enabled.generate_scopes.size() == 2 &&
                enabled.generate_scopes[0]->name == "nested[0]" &&
                enabled.generate_scopes[1]->name == "nested[1]" &&
                enabled.generate_scopes[1]->generate_scopes.size() == 1 &&
                enabled.generate_scopes[1]->generate_scopes.front()->name == "last" &&
                enabled.generate_scopes[1]->generate_scopes.front()->children.size() == 1 &&
                enabled.generate_scopes[1]
                        ->generate_scopes.front()
                        ->children.front()
                        ->hierarchy_path == "top.yes.enabled.nested[1].last.u",
            "nested generate scopes or generated hierarchy path is incorrect");
    require(disabled_instance.generate_scopes.empty(),
            "a disabled parameter-dependent generate branch was shared or elaborated");
}

void test_generate_scope_semantic_visibility() {
    const auto parsed = parse_verilog_syntax(
        "module scoped #(parameter LIMIT = 2);\n"
        "  if (LIMIT) begin : outer\n"
        "    localparam OUTER = LIMIT;\n"
        "    if (OUTER) begin : inner logic [OUTER-1:0] value; end\n"
        "    for (genvar i=0; i<LIMIT; i++) begin : lanes\n"
        "      localparam INDEX = i;\n"
        "    end\n"
        "  end\n"
        "endmodule\n");
    const auto semantics = analyze_verilog_semantics(parsed.syntax);
    require(!semantics.has_errors(),
            "generate scopes could not resolve enclosing parameters and declarations");
    VerilogElaborationOptions options;
    options.top_module = "scoped";
    const auto elaborated = elaborate_verilog_design(parsed.syntax, options);
    require(!elaborated.has_errors() && elaborated.top &&
                generate_scope(*elaborated.top, "outer").generate_scopes.size() == 3,
            "generate conditions, dimensions, or loop bounds could not use enclosing scope values");

    const auto region_parameter = parse_verilog_syntax(
        "module region_parameter;\n"
        "  generate\n"
        "    localparam ENABLED = 1;\n"
        "  endgenerate\n"
        "  if (ENABLED) begin : selected logic value; end\n"
        "endmodule\n");
    const auto region_semantics = analyze_verilog_semantics(region_parameter.syntax);
    require(!region_semantics.has_errors(),
            "explicit generate-region parameter was not visible in module scope");
    options.top_module = "region_parameter";
    const auto region_result = elaborate_verilog_design(region_parameter.syntax, options);
    require(!region_result.has_errors() && region_result.top &&
                parameter_value(*region_result.top, "ENABLED") == 1 &&
                generate_scope(*region_result.top, "selected").declarations.size() == 1,
            "explicit generate-region parameter did not control a later generate construct");
}

void test_generate_diagnostics_and_locations() {
    const auto invalid = parse_verilog_syntax(
        "module top;\n"
        "  if (MISSING) begin : bad_condition logic x; end\n"
        "  for (genvar i=0; i<2; i=i) begin : stuck logic y; end\n"
        "endmodule\n");
    VerilogElaborationOptions invalid_options;
    invalid_options.top_module = "top";
    const auto invalid_result = elaborate_verilog_design(invalid.syntax, invalid_options);
    require(has_code(invalid_result.diagnostics, "UnresolvedGenerateCondition") &&
                has_code(invalid_result.diagnostics, "UnresolvedParameter") &&
                has_code(invalid_result.diagnostics, "InvalidGenerateStep"),
            "nonconstant generate condition or non-progressing step was not diagnosed");

    const auto nonconstant_loop = parse_verilog_syntax(
        "module nonconstant_loop;\n"
        "  for (genvar i=0; i<LIMIT; i++) begin : invalid_loop logic x; end\n"
        "endmodule\n");
    VerilogElaborationOptions nonconstant_options;
    nonconstant_options.top_module = "nonconstant_loop";
    const auto nonconstant_result =
        elaborate_verilog_design(nonconstant_loop.syntax, nonconstant_options);
    require(has_code(nonconstant_result.diagnostics, "UnresolvedParameter") &&
                has_code(nonconstant_result.diagnostics, "UnresolvedGenerateCondition") &&
                nonconstant_result.top &&
                nonconstant_result.top->generate_scopes.empty(),
            "nonconstant for-generate condition was not diagnosed or safely stopped");

    const auto unknown_condition = parse_verilog_syntax(
        "module unknown_condition;\n"
        "  if (1'bx) begin : unknown_branch logic x; end\n"
        "endmodule\n");
    VerilogElaborationOptions unknown_options;
    unknown_options.top_module = "unknown_condition";
    const auto unknown_result =
        elaborate_verilog_design(unknown_condition.syntax, unknown_options);
    require(has_code(unknown_result.diagnostics, "UnresolvedGenerateCondition") &&
                unknown_result.top && unknown_result.top->generate_scopes.empty(),
            "unknown four-state generate condition was selected instead of diagnosed");

    const auto top_inference = parse_verilog_syntax(
        "module leaf; endmodule\n"
        "module inferred_top;\n"
        "  if (1) begin : generated_child leaf u(); end\n"
        "endmodule\n");
    const auto inferred = elaborate_verilog_design(top_inference.syntax);
    require(inferred.top && inferred.top->module_definition->name == "inferred_top" &&
                inferred.top->generate_scopes.front()->children.size() == 1,
            "top-module inference ignored module instances inside generate constructs");

    const auto limited = parse_verilog_syntax(
        "module limited;\n"
        "  for (genvar i=0; i<4; i++) begin : iteration logic x; end\n"
        "endmodule\n");
    VerilogElaborationOptions limited_options;
    limited_options.top_module = "limited";
    limited_options.maximum_generate_iterations = 2;
    const auto limited_result = elaborate_verilog_design(limited.syntax, limited_options);
    require(has_code(limited_result.diagnostics, "GenerateIterationLimitExceeded") &&
                limited_result.top && limited_result.top->generate_scopes.size() == 2,
            "generate iteration cap did not stop expansion at its configured limit");

    const auto duplicates = parse_verilog_syntax(
        "module duplicate;\n"
        "  if (1) begin : same logic first; end\n"
        "  if (1) begin : same logic second; end\n"
        "endmodule\n");
    VerilogElaborationOptions duplicate_options;
    duplicate_options.top_module = "duplicate";
    const auto duplicate_result = elaborate_verilog_design(duplicates.syntax, duplicate_options);
    require(has_code(duplicate_result.diagnostics, "DuplicateGeneratedName"),
            "duplicate selected generate block names were not diagnosed");

    const auto macro = parse_verilog_syntax(
        "`define GENERATED if (1) begin : from_macro logic flag; end\n"
        "module macro_top;\n"
        "  `GENERATED\n"
        "endmodule\n");
    VerilogElaborationOptions macro_options;
    macro_options.top_module = "macro_top";
    const auto macro_result = elaborate_verilog_design(macro.syntax, macro_options);
    require(macro_result.top && macro_result.top->generate_scopes.size() == 1 &&
                macro_result.top->generate_scopes.front()->name == "from_macro" &&
                macro_result.top->generate_scopes.front()->location.location.line == 3 &&
                macro_result.top->generate_scopes.front()->location.origin_kind ==
                    VerilogPreprocessedOriginKind::MacroExpansion &&
                !macro_result.top->generate_scopes.front()->location.expansion_stack.empty(),
            "macro-generated scope did not retain invocation location and expansion origin");
}

} // namespace

int main() {
    try {
        test_constant_literal_width_and_signedness();
        test_unknown_bits_concat_replication_and_errors();
        test_parameter_dependencies_and_instance_overrides();
        test_parameter_cycles_and_invalid_expressions();
        test_hierarchy_top_selection_missing_and_recursive();
        test_macro_generated_parameter_location();
        test_if_case_and_for_generate();
        test_generate_else_if_case_wildcards_and_default();
        test_nested_generate_and_instance_specific_conditions();
        test_generate_scope_semantic_visibility();
        test_generate_diagnostics_and_locations();
    }
    catch (const std::exception& error) {
        std::cerr << "EDA Verilog elaboration test failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
