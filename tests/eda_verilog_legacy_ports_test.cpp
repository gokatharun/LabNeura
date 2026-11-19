#include "labneura/eda/verilog_syntax.h"

#include <slang/syntax/SyntaxTree.h>

#include <algorithm>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

using namespace labneura::eda;

void require(bool condition, std::string_view message) {
    if (!condition)
        throw std::runtime_error(std::string(message));
}

bool has_error_code(const VerilogSyntaxParseResult& result, std::string_view code) {
    return std::any_of(result.diagnostics.begin(), result.diagnostics.end(),
                       [&](const auto& diagnostic) { return diagnostic.code == code; });
}

void test_legacy_module_ports() {
    const std::string source =
        "module legacy (a, b, c);\n"
        "  input wire [7:0] a;\n"
        "  output logic b [3:0];\n"
        "  inout c;\n"
        "endmodule\n";
    const auto parsed = parse_verilog_syntax(source);
    require(!parsed.has_errors() && parsed.syntax.modules.size() == 1,
            "valid legacy module port declarations produced diagnostics");

    const auto& ports = parsed.syntax.modules.front().ports;
    require(ports.size() == 3 && ports[0].name == "a" && ports[1].name == "b" &&
                ports[2].name == "c",
            "body declarations changed the order of legacy header ports");
    require(ports[0].direction == VerilogPortDirection::Input &&
                ports[0].type == "wire" && ports[0].packed_dimensions.size() == 1 &&
                ports[0].unpacked_dimensions.empty(),
            "packed input port declaration was not applied to its header entry");
    require(ports[1].direction == VerilogPortDirection::Output &&
                ports[1].type == "logic" && ports[1].unpacked_dimensions.size() == 1,
            "unpacked output port declaration was not applied to its header entry");
    require(ports[2].direction == VerilogPortDirection::Inout,
            "untyped inout declaration was not applied to its header entry");
    require(ports[0].location.location.line == 1 &&
                ports[0].location.source &&
                ports[0].location.origin_kind == VerilogPreprocessedOriginKind::Source,
            "legacy header port source location was not retained");

    slang::SourceManager source_manager;
    const auto slang_tree =
        slang::syntax::SyntaxTree::fromText(source, source_manager, "legacy_ports.sv");
    require(slang_tree && slang_tree->diagnostics().empty(),
            "Slang rejected valid non-ANSI module port syntax");
}

void test_legacy_port_diagnostics_and_recovery() {
    const auto missing = parse_verilog_syntax(
        "module missing (a, b);\n"
        "  input logic a;\n"
        "endmodule\n"
        "module after_missing; endmodule\n");
    require(missing.has_errors() && missing.syntax.modules.size() == 2 &&
                missing.syntax.modules.front().ports.size() == 2 &&
                missing.syntax.modules.front().ports[1].name == "b" &&
                missing.syntax.modules.front().ports[1].direction ==
                    VerilogPortDirection::Unknown &&
                has_error_code(missing, "UndeclaredHeaderPort") &&
                missing.syntax.modules.back().name == "after_missing",
            "undeclared header port was lost or blocked recovery to the next module");

    const auto duplicate = parse_verilog_syntax(
        "module duplicate (a, a);\n"
        "  input logic a;\n"
        "  output logic a;\n"
        "endmodule\n");
    require(has_error_code(duplicate, "DuplicateHeaderPort") &&
                has_error_code(duplicate, "DuplicatePortDeclaration") &&
                has_error_code(duplicate, "IncompatiblePortDeclaration") &&
                duplicate.syntax.modules.front().ports.size() == 2 &&
                duplicate.syntax.modules.front().ports[0].direction ==
                    VerilogPortDirection::Input,
            "duplicate or incompatible port declarations were not diagnosed and retained");

    const auto malformed = parse_verilog_syntax(
        "module malformed (a, b);\n"
        "  input logic a;\n"
        "  output logic ;\n"
        "endmodule\n");
    require(malformed.has_errors() && malformed.syntax.modules.front().ports.size() == 2 &&
                has_error_code(malformed, "ExpectedIdentifier") &&
                has_error_code(malformed, "UndeclaredHeaderPort"),
            "malformed body port declaration silently dropped a header port");
}

const VerilogExpression& find_call(const VerilogStatement& statement,
                                   std::string_view callee) {
    if (statement.call_expression && statement.call_expression->first &&
        statement.call_expression->first->text == callee)
        return *statement.call_expression;
    if (statement.right && statement.right->kind == VerilogExpressionKind::Call &&
        statement.right->first && statement.right->first->text == callee)
        return *statement.right;
    throw std::runtime_error("expected callable expression was not found");
}

void test_named_actuals_and_return_statements() {
    const std::string source =
        "module callable_syntax;\n"
        "  logic x, y, z;\n"
        "  task automatic update(input logic left, input logic right);\n"
        "    return;\n"
        "  endtask\n"
        "  function automatic logic combine(input logic left, input logic right);\n"
        "    return left & right;\n"
        "  endfunction\n"
        "  initial begin\n"
        "    update(.right(y), .left(x));\n"
        "    z = combine(.left(x), .right(y));\n"
        "    update(x, y);\n"
        "  end\n"
        "endmodule\n";
    const auto parsed = parse_verilog_syntax(source);
    require(!parsed.has_errors() && parsed.syntax.modules.size() == 1,
            "valid named callable actuals or return statements produced diagnostics");

    const auto& module = parsed.syntax.modules.front();
    require(module.callables.size() == 2 &&
                module.callables[0].body->statements.front()->kind ==
                    VerilogStatementKind::Return &&
                !module.callables[0].body->statements.front()->expression &&
                module.callables[1].body->statements.front()->kind ==
                    VerilogStatementKind::Return &&
                module.callables[1].body->statements.front()->expression,
            "return statements did not retain optional expressions in the AST");
    const auto& empty_return = *module.callables[0].body->statements.front();
    const auto& value_return = *module.callables[1].body->statements.front();
    require(empty_return.location.location.line == 4 &&
                value_return.location.location.line == 7 &&
                value_return.expression->location.location.line == 7,
            "return statement or expression source locations were not preserved");

    const auto& initial = *module.procedural_blocks.front().body;
    const auto& task_call = find_call(*initial.statements[0], "update");
    require(task_call.operands.size() == 2 && task_call.operand_names.size() == 2 &&
                task_call.operand_names[0] == std::optional<std::string>("right") &&
                task_call.operand_names[1] == std::optional<std::string>("left") &&
                task_call.operands[0]->text == "y" && task_call.operands[1]->text == "x" &&
                task_call.location.location.line == 10 &&
                task_call.operands[0]->location.location.line == 10,
            "named task actual names, operand order, or source locations are incorrect");
    const auto& function_call = find_call(*initial.statements[1], "combine");
    require(function_call.operands.size() == 2 && function_call.operand_names.size() == 2 &&
                function_call.operand_names[0] == std::optional<std::string>("left") &&
                function_call.operand_names[1] == std::optional<std::string>("right"),
            "named function actuals were not represented in the parallel name vector");
    const auto& positional_call = find_call(*initial.statements[2], "update");
    require(positional_call.operands.size() == 2 &&
                positional_call.operand_names.size() == positional_call.operands.size() &&
                !positional_call.operand_names[0] && !positional_call.operand_names[1],
            "positional call AST behavior changed or lacks parallel empty names");

    slang::SourceManager source_manager;
    const auto slang_tree =
        slang::syntax::SyntaxTree::fromText(source, source_manager, "callables.sv");
    require(slang_tree && slang_tree->diagnostics().empty(),
            "Slang rejected valid named callable actual or return syntax");
}

void test_malformed_named_actuals() {
    const auto mixed = parse_verilog_syntax(
        "module mixed;\n"
        "  task t(input logic a, input logic b); endtask\n"
        "  initial t(a, .b(b));\n"
        "endmodule\n");
    require(has_error_code(mixed, "MixedCallArgumentStyles") &&
                mixed.syntax.modules.front().procedural_blocks.front().body->kind ==
                    VerilogStatementKind::TaskCall &&
                mixed.syntax.modules.front().procedural_blocks.front()
                        .body->call_expression->operand_names.size() == 2,
            "mixed actual argument styles were not diagnosed while retaining the call AST");

    const auto duplicate = parse_verilog_syntax(
        "module duplicate_actual;\n"
        "  task t(input logic a); endtask\n"
        "  initial t(.a(a), .a(a));\n"
        "endmodule\n");
    require(has_error_code(duplicate, "DuplicateNamedActualArgument") &&
                duplicate.syntax.modules.front().procedural_blocks.front()
                        .body->call_expression->operand_names.size() == 2,
            "duplicate named actuals were not diagnosed and retained");

    const auto malformed = parse_verilog_syntax(
        "module malformed_actual;\n"
        "  task t(input logic a); endtask\n"
        "  initial t(.a());\n"
        "endmodule\n"
        "module after_actual_error; endmodule\n");
    require(has_error_code(malformed, "MissingNamedCallActual") &&
                malformed.syntax.modules.size() == 2 &&
                malformed.syntax.modules.back().name == "after_actual_error",
            "malformed named actual did not diagnose and recover at the next module");
}

} // namespace

int main() {
    try {
        test_legacy_module_ports();
        test_legacy_port_diagnostics_and_recovery();
        test_named_actuals_and_return_statements();
        test_malformed_named_actuals();
    }
    catch (const std::exception& error) {
        std::cerr << "EDA Verilog legacy ports/parser contract test failed: " << error.what()
                  << '\n';
        return 1;
    }
}
