#include "labneura/eda/verilog_function_evaluation.h"
#include "labneura/eda/verilog_semantic.h"
#include "labneura/eda/verilog_syntax.h"
#include "labneura/eda/verilog_type.h"

#include <slang/ast/Compilation.h>
#include <slang/ast/symbols/CompilationUnitSymbols.h>
#include <slang/ast/symbols/InstanceSymbols.h>
#include <slang/ast/symbols/ParameterSymbols.h>
#include <slang/ast/symbols/SubroutineSymbols.h>
#include <slang/numeric/ConstantValue.h>
#include <slang/syntax/SyntaxTree.h>

#include <algorithm>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {
using namespace labneura::eda;

void require(bool condition, std::string_view message) {
    if (!condition)
        throw std::runtime_error(std::string(message));
}

const VerilogExpression* find_call(const VerilogExpression* expression,
                                   std::string_view name) {
    if (!expression)
        return nullptr;
    if (expression->kind == VerilogExpressionKind::Call && expression->first &&
        expression->first->text == name)
        return expression;
    if (const auto* found = find_call(expression->first.get(), name))
        return found;
    if (const auto* found = find_call(expression->second.get(), name))
        return found;
    if (const auto* found = find_call(expression->third.get(), name))
        return found;
    for (const auto& operand : expression->operands)
        if (const auto* found = find_call(operand.get(), name))
            return found;
    return nullptr;
}

struct NativeFixture {
    VerilogSyntaxParseResult parsed;
    VerilogSemanticAnalysisResult semantic;
    VerilogTypeAnalysisResult types;

    explicit NativeFixture(const std::string& source)
        : parsed(parse_verilog_syntax(source)),
          semantic(analyze_verilog_semantics(parsed.syntax)),
          types(analyze_verilog_expression_types(parsed.syntax, semantic)) {
        require(!parsed.has_errors(), "native control-flow differential fixture did not parse");
        require(!semantic.has_errors(),
                "native control-flow differential fixture has semantic errors");
    }

    const VerilogExpression& call(std::string_view function_name) const {
        for (const auto& module : parsed.syntax.modules)
            for (const auto& parameter : module.parameters)
                if (const auto* found = find_call(parameter.value.get(), function_name))
                    return *found;
        throw std::runtime_error("native fixture has no localparam call for function");
    }
};

struct SlangFixture {
    slang::SourceManager source_manager;
    std::shared_ptr<slang::syntax::SyntaxTree> tree;
    slang::ast::Compilation compilation;

    explicit SlangFixture(const std::string& source)
        : tree(slang::syntax::SyntaxTree::fromText(source, source_manager,
                                                   "control_flow_differential.sv")) {
        compilation.addSyntaxTree(tree);
        const auto diagnostics = compilation.getAllDiagnostics();
        const auto error = std::find_if(diagnostics.begin(), diagnostics.end(),
                                        [](const auto& diagnostic) {
                                            return diagnostic.isError();
                                        });
        if (error != diagnostics.end())
            throw std::runtime_error("Slang rejected control-flow differential fixture: " +
                                     std::string(slang::toString(error->code)));
    }

    const slang::ast::InstanceSymbol& top() {
        const auto& tops = compilation.getRoot().topInstances;
        require(tops.size() == 1, "Slang did not elaborate one differential top module");
        return *tops.front();
    }

    const slang::ast::ParameterSymbol& parameter(std::string_view name) {
        for (const auto& member : top().body.members())
            if (member.kind == slang::ast::SymbolKind::Parameter && member.name == name)
                return member.as<slang::ast::ParameterSymbol>();
        throw std::runtime_error("Slang parameter was not found");
    }
};

void compare_value(const VerilogConstantValue& native,
                   const slang::ast::ParameterSymbol& reference) {
    const auto& value = reference.getValue();
    require(value.isInteger(), "Slang control-flow result is not integral");
    require(native.bit_width == value.integer().getBitWidth(),
            "native control-flow result width differs from Slang");
    for (size_t bit = 0; bit < native.bit_width; ++bit) {
        const auto expected = value.integer()[static_cast<int32_t>(bit)];
        require(!expected.isUnknown(), "Slang control-flow result unexpectedly contains X/Z");
        require((native.bits[bit] == VerilogLogicBit::One) == (expected.value == 1) &&
                    (native.bits[bit] == VerilogLogicBit::Zero ||
                     native.bits[bit] == VerilogLogicBit::One),
                "native control-flow result bits differ from Slang");
    }
}

void compare_function(const NativeFixture& native, SlangFixture& slang,
                      std::string_view function_name, std::string_view parameter_name,
                      uint64_t expected) {
    const auto result = evaluate_verilog_function_call(
        native.parsed.syntax, native.semantic, native.types,
        native.call(function_name));
    require(result.value && result.diagnostics.empty(),
            "native control-flow function did not complete cleanly");
    require(result.value->to_uint64() == expected,
            "native control-flow function returned an unexpected value");
    compare_value(*result.value, slang.parameter(parameter_name));
}

void test_nested_loops_transfer_and_statement_order_against_slang() {
    const std::string source =
        "module control_flow_diff;\n"
        "  function automatic logic [7:0] nested(input logic [7:0] count);\n"
        "    logic [7:0] total; integer i; integer j;\n"
        "    total = 0;\n"
        "    for (i = 0; i < count; i = i + 1) begin\n"
        "      for (j = 0; j < 4; j = j + 1) begin\n"
        "        if (j == 1) continue;\n"
        "        if (i == 2) break;\n"
        "        total = total + 1;\n"
        "      end\n"
        "    end\n"
        "    return total;\n"
        "  endfunction\n"
        "  function automatic logic [7:0] loop_order();\n"
        "    logic [7:0] value; integer i;\n"
        "    value = 0;\n"
        "    repeat (0) value = value + 8'd10;\n"
        "    do value = value + 1; while (value < 2);\n"
        "    while (value < 4) value = value + 1;\n"
        "    for (i = 0; i < 3; i = i + 1) value = value + 1;\n"
        "    return value;\n"
        "  endfunction\n"
        "  localparam logic [7:0] NESTED_RESULT = nested(8'd4);\n"
        "  localparam logic [7:0] LOOP_ORDER_RESULT = loop_order();\n"
        "endmodule\n";
    NativeFixture native(source);
    SlangFixture slang(source);
    compare_function(native, slang, "nested", "NESTED_RESULT", 9);
    compare_function(native, slang, "loop_order", "LOOP_ORDER_RESULT", 7);
}

void test_early_return_four_state_if_and_case_variants_against_slang() {
    const std::string source =
        "module branch_diff;\n"
        "  function automatic logic [7:0] early(input logic [7:0] value);\n"
        "    begin\n"
        "      if (value == 0) return 8'd2;\n"
        "      begin early = 8'd9; return value + 8'd1; end\n"
        "    end\n"
        "  endfunction\n"
        "  function automatic logic [7:0] if_x(input logic condition);\n"
        "    if (condition) if_x = 8'd1; else if_x = 8'd2;\n"
        "  endfunction\n"
        "  function automatic logic [7:0] exact_case(input logic [3:0] value);\n"
        "    case (value)\n"
        "      4'b10x1, 4'b0000: return 8'd3;\n"
        "      default: return 8'd0;\n"
        "    endcase\n"
        "  endfunction\n"
        "  function automatic logic [7:0] z_case(input logic [3:0] value);\n"
        "    casez (value)\n"
        "      4'b1001: return 8'd4;\n"
        "      default: return 8'd0;\n"
        "    endcase\n"
        "  endfunction\n"
        "  function automatic logic [7:0] x_case(input logic [3:0] value);\n"
        "    casex (value)\n"
        "      4'b1001: return 8'd5;\n"
        "      default: return 8'd0;\n"
        "    endcase\n"
        "  endfunction\n"
        "  localparam logic [7:0] EARLY_RESULT = early(8'd7);\n"
        "  localparam logic [7:0] IF_X_RESULT = if_x(1'bx);\n"
        "  localparam logic [7:0] EXACT_RESULT = exact_case(4'b10x1);\n"
        "  localparam logic [7:0] Z_RESULT = z_case(4'b1z01);\n"
        "  localparam logic [7:0] X_RESULT = x_case(4'b1x01);\n"
        "endmodule\n";
    NativeFixture native(source);
    SlangFixture slang(source);
    compare_function(native, slang, "early", "EARLY_RESULT", 8);
    compare_function(native, slang, "if_x", "IF_X_RESULT", 2);
    compare_function(native, slang, "exact_case", "EXACT_RESULT", 3);
    compare_function(native, slang, "z_case", "Z_RESULT", 4);
    compare_function(native, slang, "x_case", "X_RESULT", 5);
}

void test_execution_budget_terminates_forever_loop() {
    const std::string source =
        "module budget_call;\n"
        "  function automatic logic [7:0] endless(); forever begin ; end endfunction\n"
        "  localparam logic [7:0] RESULT = endless();\n"
        "endmodule\n";
    NativeFixture callable_fixture(source);
    VerilogCallableExecutionOptions options;
    options.maximum_statement_steps = 12;
    const auto result = evaluate_verilog_function_call(
        callable_fixture.parsed.syntax, callable_fixture.semantic,
        callable_fixture.types, callable_fixture.call("endless"), {}, options);
    require(!result.value && std::any_of(
                result.diagnostics.begin(), result.diagnostics.end(),
                [](const auto& diagnostic) {
                    return diagnostic.code == "CallableExecutionStepLimitExceeded";
                }),
            "forever loop did not terminate with an execution-budget diagnostic");
}

} // namespace

int main() {
    try {
        test_nested_loops_transfer_and_statement_order_against_slang();
        test_early_return_four_state_if_and_case_variants_against_slang();
        test_execution_budget_terminates_forever_loop();
    }
    catch (const std::exception& error) {
        std::cerr << "EDA Verilog control-flow differential test failed: " << error.what()
                  << '\n';
        return 1;
    }
}
