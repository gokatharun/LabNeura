#include "labneura/eda/verilog_function_evaluation.h"
#include "labneura/eda/verilog_semantic.h"
#include "labneura/eda/verilog_syntax.h"
#include "labneura/eda/verilog_type.h"

#include <slang/ast/Compilation.h>
#include <slang/ast/symbols/CompilationUnitSymbols.h>
#include <slang/ast/symbols/InstanceSymbols.h>
#include <slang/ast/symbols/ParameterSymbols.h>
#include <slang/ast/symbols/SubroutineSymbols.h>
#include <slang/ast/types/Type.h>
#include <slang/numeric/ConstantValue.h>
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

const VerilogSemanticSymbol* native_callable(
    const VerilogSemanticAnalysisResult& semantic, std::string_view module_name,
    std::string_view function_name) {
    const auto* module = lookup_verilog_symbol(
        *semantic.model.compilation_unit_scope, module_name, false);
    return module && module->member_scope
               ? lookup_verilog_symbol(*module->member_scope, function_name, false)
               : nullptr;
}

struct NativeFixture {
    VerilogSyntaxParseResult parsed;
    VerilogSemanticAnalysisResult semantic;
    VerilogTypeAnalysisResult types;

    explicit NativeFixture(std::string source)
        : parsed(parse_verilog_syntax(source)),
          semantic(analyze_verilog_semantics(parsed.syntax)),
          types(analyze_verilog_expression_types(parsed.syntax, semantic)) {
        require(!parsed.has_errors(), "native callable differential fixture did not parse");
        require(!semantic.has_errors(),
                "native callable differential fixture has semantic errors");
    }

    const VerilogExpression& call(std::string_view name) const {
        for (const auto& module : parsed.syntax.modules)
            for (const auto& parameter : module.parameters)
                if (const auto* found = find_call(parameter.value.get(), name))
                    return *found;
        throw std::runtime_error("native fixture has no matching function call");
    }
};

struct SlangFixture {
    slang::SourceManager source_manager;
    std::shared_ptr<slang::syntax::SyntaxTree> tree;
    slang::ast::Compilation compilation;

    explicit SlangFixture(const std::string& source)
        : tree(slang::syntax::SyntaxTree::fromText(source, source_manager,
                                                   "callable_differential.sv")) {
        compilation.addSyntaxTree(tree);
        const auto diagnostics = compilation.getAllDiagnostics();
        const auto error = std::find_if(diagnostics.begin(), diagnostics.end(),
                                        [](const auto& diagnostic) {
                                            return diagnostic.isError();
                                        });
        if (error != diagnostics.end())
            throw std::runtime_error(
                "Slang rejected a callable differential fixture: " +
                std::string(slang::toString(error->code)) + " at line " +
                std::to_string(source_manager.getLineNumber(
                    error->location)));
    }

    const slang::ast::InstanceSymbol& top(std::string_view name) {
        const auto& tops = compilation.getRoot().topInstances;
        const auto found = std::find_if(tops.begin(), tops.end(), [&](const auto* instance) {
            return instance->name == name;
        });
        require(found != tops.end(), "Slang did not elaborate the differential top module");
        return **found;
    }

    const slang::ast::ParameterSymbol& parameter(
        const slang::ast::InstanceSymbol& instance, std::string_view name) const {
        for (const auto& member : instance.body.members())
            if (member.kind == slang::ast::SymbolKind::Parameter && member.name == name)
                return member.as<slang::ast::ParameterSymbol>();
        throw std::runtime_error("Slang parameter symbol was not found");
    }

    const slang::ast::SubroutineSymbol& function(
        const slang::ast::InstanceSymbol& instance, std::string_view name) const {
        for (const auto& member : instance.body.members())
            if (member.kind == slang::ast::SymbolKind::Subroutine && member.name == name)
                return member.as<slang::ast::SubroutineSymbol>();
        throw std::runtime_error("Slang function symbol was not found");
    }
};

void compare_result(const VerilogConstantValue& native_value,
                    const VerilogIntegralType& native_type,
                    const slang::ast::ParameterSymbol& slang_parameter,
                    const slang::ast::SubroutineSymbol& slang_function) {
    const auto& slang_type = slang_function.getReturnType();
    const auto& parameter_type = slang_parameter.getType();
    const auto& slang_value = slang_parameter.getValue();
    require(slang_value.isInteger(), "Slang function result is not integral");
    require(native_type.is_known_integral() && native_type.bit_width &&
                native_type.is_signed == slang_type.isSigned() &&
                *native_type.bit_width == slang_type.getBitWidth() &&
                native_type.four_state == slang_type.isFourState(),
            "native function return type differs from Slang");
    require(parameter_type.getBitWidth() == slang_value.integer().getBitWidth() &&
                native_value.bit_width == slang_value.integer().getBitWidth() &&
                native_value.is_signed == parameter_type.isSigned(),
            "native function result sizing or signedness differs from Slang");
    for (size_t bit = 0; bit < native_value.bit_width; ++bit) {
        const auto native_bit = native_value.bits[bit];
        const auto slang_bit = slang_value.integer()[static_cast<int32_t>(bit)];
        require(!slang_bit.isUnknown(),
                "supported differential fixture unexpectedly produced X or Z in Slang");
        const bool expected_one = slang_bit.value == 1;
        require((native_bit == VerilogLogicBit::One) == expected_one &&
                    (native_bit == VerilogLogicBit::Zero ||
                     native_bit == VerilogLogicBit::One),
                "native function result bits differ from Slang constant evaluation");
    }
}

void test_named_argument_order_and_nested_constant_function_values() {
    const std::string source =
        "module named_value;\n"
        "  function automatic logic [7:0] add_parts(\n"
        "      input logic [7:0] high, input logic [7:0] low);\n"
        "    add_parts = high + low;\n"
        "  endfunction\n"
        "  function automatic logic [7:0] outer(input logic [7:0] value);\n"
        "    outer = add_parts(.low(8'd3), .high(value));\n"
        "  endfunction\n"
        "  localparam logic [7:0] RESULT = outer(8'd9);\n"
        "endmodule\n";
    NativeFixture native(source);
    SlangFixture slang(source);

    const auto* function = native_callable(native.semantic, "named_value", "outer");
    require(function && function->callable && function->kind == VerilogSemanticSymbolKind::Function,
            "native outer function symbol is missing");
    auto evaluated = evaluate_verilog_function_call(
        native.parsed.syntax, native.semantic, native.types, native.call("outer"));
    require(evaluated.value && evaluated.diagnostics.empty(),
            "native nested named function call did not evaluate cleanly");
    const auto& top = slang.top("named_value");
    compare_result(*evaluated.value, native.types.symbol_types.at(function),
                   slang.parameter(top, "RESULT"), slang.function(top, "outer"));
    require(evaluated.value->to_uint64() == 12,
            "named actuals did not bind by formal name in the native evaluator");
}

void test_signed_return_sizing_and_parameterized_formal_type() {
    const std::string source =
        "module parameterized_value #(parameter int WIDTH = 5);\n"
        "  function automatic logic signed [7:0] sign_extend(\n"
        "      input logic signed [3:0] value);\n"
        "    return value;\n"
        "  endfunction\n"
        "  function automatic logic [WIDTH-1:0] identity(\n"
        "      input logic [WIDTH-1:0] value);\n"
        "    identity = value;\n"
        "  endfunction\n"
        "  localparam logic signed [7:0] SIGNED_RESULT = sign_extend(4'sh8);\n"
        "  localparam logic [WIDTH-1:0] PARAMETER_RESULT = identity(5'b10110);\n"
        "endmodule\n";
    NativeFixture native(source);
    SlangFixture slang(source);
    const auto& top = slang.top("parameterized_value");

    const auto* sign_extend = native_callable(native.semantic, "parameterized_value",
                                              "sign_extend");
    require(sign_extend && native.types.symbol_types.contains(sign_extend),
            "native signed function return type is missing");
    auto signed_value = evaluate_verilog_function_call(
        native.parsed.syntax, native.semantic, native.types,
        native.call("sign_extend"));
    require(signed_value.value && signed_value.diagnostics.empty(),
            "native signed return evaluation failed");
    compare_result(*signed_value.value, native.types.symbol_types.at(sign_extend),
                   slang.parameter(top, "SIGNED_RESULT"),
                   slang.function(top, "sign_extend"));
    require(signed_value.value->to_uint64() == 0xf8,
            "signed 4-to-8-bit return did not sign extend as Slang does");

    const auto* identity = native_callable(native.semantic, "parameterized_value", "identity");
    require(identity && native.types.symbol_types.contains(identity),
            "native parameterized function return type is missing");
    auto parameterized_value = evaluate_verilog_function_call(
        native.parsed.syntax, native.semantic, native.types,
        native.call("identity"));
    require(parameterized_value.value && parameterized_value.diagnostics.empty(),
            "native parameterized function evaluation failed");
    compare_result(*parameterized_value.value, native.types.symbol_types.at(identity),
                   slang.parameter(top, "PARAMETER_RESULT"),
                   slang.function(top, "identity"));
    require(parameterized_value.value->to_uint64() == 0b10110,
            "parameterized formal/result width did not preserve the expected value");
}

void test_each_native_invocation_gets_a_fresh_local_frame() {
    const std::string source =
        "module local_frame;\n"
        "  function automatic logic [7:0] count_local(input logic [7:0] seed);\n"
        "    logic [7:0] local_count = 8'd1;\n"
        "    local_count = local_count + seed;\n"
        "    count_local = local_count;\n"
        "  endfunction\n"
        "  localparam logic [7:0] RESULT = count_local(8'd4);\n"
        "endmodule\n";
    NativeFixture native(source);
    SlangFixture slang(source);
    const auto* function = native_callable(native.semantic, "local_frame", "count_local");
    require(function && native.types.symbol_types.contains(function),
            "native local-frame function return type is missing");
    const auto& call = native.call("count_local");
    const auto first = evaluate_verilog_function_call(
        native.parsed.syntax, native.semantic, native.types, call);
    const auto second = evaluate_verilog_function_call(
        native.parsed.syntax, native.semantic, native.types, call);
    require(first.value && second.value && first.diagnostics.empty() &&
                second.diagnostics.empty(),
            "repeated native function evaluation did not complete successfully");
    compare_result(*first.value, native.types.symbol_types.at(function),
                   slang.parameter(slang.top("local_frame"), "RESULT"),
                   slang.function(slang.top("local_frame"), "count_local"));
    require(first.value->to_binary_string() == second.value->to_binary_string() &&
                first.value->to_uint64() == 5 && second.value->to_uint64() == 5,
            "function local state leaked between native invocations");
}

void test_invalid_arity_is_rejected_by_native_binding_and_slang() {
    const std::string source =
        "module invalid_arity;\n"
        "  function automatic logic [7:0] add(input logic [7:0] left,\n"
        "                                      input logic [7:0] right);\n"
        "    add = left + right;\n"
        "  endfunction\n"
        "  localparam logic [7:0] RESULT = add(8'd1);\n"
        "endmodule\n";
    auto parsed = parse_verilog_syntax(source);
    require(!parsed.has_errors(), "native invalid-arity fixture did not parse");
    const auto semantic = analyze_verilog_semantics(parsed.syntax);
    const auto types = analyze_verilog_expression_types(parsed.syntax, semantic);
    const auto* function = native_callable(semantic, "invalid_arity", "add");
    require(function, "native invalid-arity function symbol is missing");
    const auto* call_pointer = find_call(
        parsed.syntax.modules.front().parameters.front().value.get(), "add");
    require(call_pointer, "native invalid-arity call expression is missing");
    const auto binding = bind_verilog_callable_arguments(*function, *call_pointer, types);
    require(!binding.is_valid &&
                std::any_of(binding.diagnostics.begin(), binding.diagnostics.end(),
                            [](const auto& diagnostic) {
                                return diagnostic.code == "MissingCallableArgument" &&
                                       diagnostic.severity ==
                                           VerilogTypeDiagnosticSeverity::Error &&
                                       diagnostic.location.source;
                            }),
            "native binding did not report a source-located missing actual");

    slang::SourceManager source_manager;
    auto tree = slang::syntax::SyntaxTree::fromText(
        source, source_manager, "invalid_callable_arity.sv");
    slang::ast::Compilation compilation;
    compilation.addSyntaxTree(tree);
    const auto diagnostics = compilation.getAllDiagnostics();
    require(std::any_of(diagnostics.begin(), diagnostics.end(),
                        [](const auto& diagnostic) { return diagnostic.isError(); }),
            "Slang unexpectedly accepted a function call with a missing required actual");
}

// Slang's public AST/constant evaluator can evaluate constant function calls,
// but LabNeura task execution is a separate bounded procedural interpreter.
// There is no Slang simulator in this test, so task side effects and output
// writeback are explicitly excluded from value differential claims. Timing,
// event controls, nonblocking assignments, and other unsupported LabNeura
// behavior are also excluded; syntax acceptance alone is not semantic parity.

} // namespace

int main() {
    try {
        test_named_argument_order_and_nested_constant_function_values();
        test_signed_return_sizing_and_parameterized_formal_type();
        test_each_native_invocation_gets_a_fresh_local_frame();
        test_invalid_arity_is_rejected_by_native_binding_and_slang();
        std::cout << "Callable value/type differential tests passed\n";
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << "EDA Verilog callable differential test failed: " << error.what()
                  << '\n';
        return 1;
    }
}
