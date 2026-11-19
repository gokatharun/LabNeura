#include "labneura/eda/verilog_function_evaluation.h"
#include "labneura/eda/verilog_semantic.h"
#include "labneura/eda/verilog_type.h"

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
                                   std::string_view callee) {
    if (!expression)
        return nullptr;
    if (expression->kind == VerilogExpressionKind::Call && expression->first &&
        expression->first->text == callee)
        return expression;
    if (const auto* found = find_call(expression->first.get(), callee))
        return found;
    if (const auto* found = find_call(expression->second.get(), callee))
        return found;
    if (const auto* found = find_call(expression->third.get(), callee))
        return found;
    for (const auto& operand : expression->operands)
        if (const auto* found = find_call(operand.get(), callee))
            return found;
    return nullptr;
}

struct Fixture {
    VerilogSyntaxParseResult parsed;
    VerilogSemanticAnalysisResult semantic;
    VerilogTypeAnalysisResult types;

    explicit Fixture(std::string source) : parsed(parse_verilog_syntax(std::move(source))) {
        require(!parsed.has_errors(), "function evaluation fixture should parse cleanly");
        semantic = analyze_verilog_semantics(parsed.syntax);
        require(!semantic.has_errors(),
                "function evaluation fixture should resolve without semantic errors");
        types = analyze_verilog_expression_types(parsed.syntax, semantic);
    }

    const VerilogExpression& call(std::string_view name) const {
        for (const auto& module : parsed.syntax.modules) {
            for (const auto& assignment : module.assignments) {
                if (const auto* found = find_call(assignment.right.get(), name))
                    return *found;
            }
        }
        throw std::runtime_error("test fixture does not contain the requested function call");
    }
};

VerilogFunctionEvaluationResult evaluate(
    const Fixture& fixture, std::string_view name,
    const VerilogConstantEnvironment& environment = {},
    VerilogCallableExecutionOptions options = {}) {
    return evaluate_verilog_function_call(fixture.parsed.syntax, fixture.semantic,
                                          fixture.types, fixture.call(name), environment,
                                          std::move(options));
}

VerilogConstantValue known_value(size_t width, uint64_t value) {
    VerilogConstantValue result;
    result.bit_width = width;
    result.bits.resize(width, VerilogLogicBit::Zero);
    for (size_t bit = 0; bit < width && bit < 64; ++bit)
        if ((value >> bit) & 1U)
            result.bits[bit] = VerilogLogicBit::One;
    return result;
}

bool has_diagnostic(const VerilogFunctionEvaluationResult& result,
                    std::string_view code) {
    for (const auto& diagnostic : result.diagnostics)
        if (diagnostic.code == code)
            return true;
    return false;
}

void test_named_input_binding_fresh_frame_and_local_initialization() {
    Fixture fixture(
        "module eval_named(output logic [7:0] y);\n"
        "  function automatic logic [7:0] add_local(\n"
        "      input logic [3:0] left, input logic [3:0] right);\n"
        "    logic [7:0] seeded = 8'd5;\n"
        "    logic [3:0] unknown_local;\n"
        "    begin\n"
        "      seeded = seeded + left + right;\n"
        "      add_local = seeded;\n"
        "    end\n"
        "  endfunction\n"
        "  assign y = add_local(.right(4'd3), .left(4'd2));\n"
        "endmodule\n");

    auto caller = VerilogConstantEnvironment{
        {"seeded", known_value(8, 41)}, {"left", known_value(4, 12)}};
    const auto result = evaluate(fixture, "add_local", caller);
    require(result.value && result.value->to_uint64() == 10,
            "named input binding or initialized local assignment returned the wrong value");
    require(caller.at("seeded").to_uint64() == 41 &&
                caller.at("left").to_uint64() == 12,
            "function execution mutated its caller environment");
    require(result.executed_steps >= 3 && result.diagnostics.empty(),
            "successful function execution did not report its executed statements cleanly");
}

void test_uninitialized_four_state_local_and_explicit_return() {
    Fixture fixture(
        "module eval_returns(output logic [7:0] y, output logic [3:0] z,\n"
        "                     output bit [3:0] zeroes);\n"
        "  function automatic logic [3:0] unknown_value(input logic bit_value);\n"
        "    logic [3:0] uninitialized;\n"
        "    unknown_value = uninitialized;\n"
        "  endfunction\n"
        "  function automatic bit [3:0] zero_value(input bit bit_value);\n"
        "    bit [3:0] uninitialized;\n"
        "    zero_value = uninitialized;\n"
        "  endfunction\n"
        "  function automatic logic [7:0] explicit_value(input logic [7:0] a);\n"
        "    begin\n"
        "      explicit_value = 8'd0;\n"
        "      return a + 8'd2;\n"
        "    end\n"
        "  endfunction\n"
        "  assign z = unknown_value(1'b0);\n"
        "  assign zeroes = zero_value(1'b0);\n"
        "  assign y = explicit_value(8'd5);\n"
        "endmodule\n");

    const auto unknown = evaluate(fixture, "unknown_value");
    require(unknown.value && unknown.value->bit_width == 4 &&
                unknown.value->bits.size() == 4 &&
                unknown.value->bits[0] == VerilogLogicBit::X &&
                unknown.value->bits[3] == VerilogLogicBit::X,
            "uninitialized four-state local did not retain its X value");

    const auto zeroes = evaluate(fixture, "zero_value");
    require(zeroes.value && zeroes.value->bit_width == 4 &&
                zeroes.value->to_uint64() == 0,
            "uninitialized two-state local did not start at zero");

    const auto explicit_result = evaluate(fixture, "explicit_value");
    require(explicit_result.value && explicit_result.value->to_uint64() == 7,
            "explicit return expression did not override the function-name variable");
}

void test_bounded_for_loop_and_fully_known_case() {
    Fixture fixture(
        "module eval_loop_case(output logic [7:0] y);\n"
        "  function automatic logic [7:0] count(input logic [7:0] limit);\n"
        "    logic [7:0] total;\n"
        "    begin\n"
        "      total = 0;\n"
        "      for (int i = 0; i < limit; i = i + 1) begin\n"
        "        total = total + 1;\n"
        "      end\n"
        "      case (total)\n"
        "        8'd3: count = total;\n"
        "        default: count = 8'd0;\n"
        "      endcase\n"
        "    end\n"
        "  endfunction\n"
        "  assign y = count(8'd3);\n"
        "endmodule\n");
    const auto result = evaluate(fixture, "count");
    require(result.value && result.value->to_uint64() == 3 &&
                result.diagnostics.empty(),
            "bounded for-loop or known-value case execution returned the wrong value");
}

void test_argument_conversion_and_nested_function_calls() {
    Fixture fixture(
        "module eval_nested(output logic [7:0] y);\n"
        "  function automatic logic [3:0] identity(input logic [3:0] value);\n"
        "    identity = value;\n"
        "  endfunction\n"
        "  function automatic logic [7:0] increment(input logic [7:0] value);\n"
        "    increment = value + 8'd1;\n"
        "  endfunction\n"
        "  function automatic logic [7:0] add_two(input logic [7:0] value);\n"
        "    add_two = value + 8'd2;\n"
        "  endfunction\n"
        "  function automatic logic [7:0] nested_value(input logic [7:0] value);\n"
        "    nested_value = increment(add_two(value));\n"
        "  endfunction\n"
        "  assign y = identity(8'hff) + nested_value(8'd4);\n"
        "endmodule\n");

    const auto narrowed = evaluate(fixture, "identity");
    require(narrowed.value && narrowed.value->bit_width == 4 &&
                narrowed.value->to_uint64() == 15,
            "input formal width conversion did not truncate while preserving value bits");

    const auto nested = evaluate(fixture, "nested_value");
    require(nested.value && nested.value->to_uint64() == 7 &&
                nested.diagnostics.empty(),
            "nested supported function calls did not share the execution state correctly");
}

void test_recursion_and_shared_limits_are_diagnosed_at_call_sites() {
    Fixture recursive(
        "module eval_recursion(output logic [7:0] y);\n"
        "  function automatic logic [7:0] recur(input logic [7:0] value);\n"
        "    recur = recur(value);\n"
        "  endfunction\n"
        "  assign y = recur(8'd1);\n"
        "endmodule\n");
    const auto recursion = evaluate(recursive, "recur");
    require(!recursion.value && has_diagnostic(recursion, "RecursiveCallableExecution") &&
                recursion.diagnostics.front().location.source &&
                recursion.diagnostics.front().related_location.has_value(),
            "recursive function call was not diagnosed with call/declaration locations");

    Fixture limited(
        "module eval_limits(output logic [7:0] y);\n"
        "  function automatic logic [7:0] inner(input logic [7:0] value);\n"
        "    inner = value + 8'd1;\n"
        "  endfunction\n"
        "  function automatic logic [7:0] outer(input logic [7:0] value);\n"
        "    outer = inner(value);\n"
        "  endfunction\n"
        "  assign y = outer(8'd1);\n"
        "endmodule\n");
    VerilogCallableExecutionOptions depth_options;
    depth_options.maximum_call_depth = 1;
    const auto depth = evaluate(limited, "outer", {}, depth_options);
    require(!depth.value && has_diagnostic(depth, "CallableExecutionDepthExceeded") &&
                depth.diagnostics.front().location.source &&
                depth.diagnostics.front().related_location,
            "call-depth exhaustion was not reported at the nested call site");

    VerilogCallableExecutionOptions nested_step_options;
    nested_step_options.maximum_statement_steps = 1;
    const auto nested_steps =
        evaluate(limited, "outer", {}, nested_step_options);
    require(!nested_steps.value &&
                has_diagnostic(nested_steps, "CallableExecutionStepLimitExceeded") &&
                nested_steps.executed_steps == 1,
            "nested function calls did not share the global statement-step limit");

    Fixture steps_fixture(
        "module eval_steps(output logic [7:0] y);\n"
        "  function automatic logic [7:0] multiple_steps(input logic [7:0] value);\n"
        "    begin\n"
        "      multiple_steps = value;\n"
        "      multiple_steps = multiple_steps + 8'd1;\n"
        "    end\n"
        "  endfunction\n"
        "  assign y = multiple_steps(8'd1);\n"
        "endmodule\n");
    VerilogCallableExecutionOptions step_options;
    step_options.maximum_statement_steps = 1;
    const auto steps = evaluate(steps_fixture, "multiple_steps", {}, step_options);
    require(!steps.value &&
                has_diagnostic(steps, "CallableExecutionStepLimitExceeded") &&
                steps.executed_steps == 1,
            "statement-step limit was not enforced or reported accurately");
}

void test_unsupported_statement_fails_without_substituting_a_value() {
    Fixture fixture(
        "module eval_unsupported(output logic [7:0] y);\n"
        "  function automatic logic [7:0] unsupported(input logic [7:0] value);\n"
        "    begin\n"
        "      #1 unsupported = value;\n"
        "    end\n"
        "  endfunction\n"
        "  assign y = unsupported(8'd1);\n"
        "endmodule\n");
    const auto result = evaluate(fixture, "unsupported");
    require(!result.value && has_diagnostic(result, "UnsupportedCallableStatement"),
            "unsupported timing syntax silently produced a function result");
    require(static_cast<bool>(result.diagnostics.front().location.source),
            "unsupported-statement diagnostic is missing its source location");
}

void test_local_state_does_not_leak_between_invocations() {
    Fixture fixture(
        "module eval_fresh_frame(output logic [7:0] y);\n"
        "  function automatic logic [7:0] next_value(input logic [7:0] seed);\n"
        "    logic [7:0] local_count = 8'd0;\n"
        "    local_count = local_count + 8'd1;\n"
        "    next_value = local_count + seed;\n"
        "  endfunction\n"
        "  assign y = next_value(8'd4);\n"
        "endmodule\n");

    const auto first = evaluate(fixture, "next_value");
    const auto second = evaluate(fixture, "next_value");
    require(first.value && second.value && first.value->to_uint64() == 5 &&
                second.value->to_uint64() == 5 && first.diagnostics.empty() &&
                second.diagnostics.empty(),
            "function local initialization leaked state across invocations");
}

void test_return_width_and_signedness_conversions() {
    Fixture fixture(
        "module eval_return_conversion(output logic signed [7:0] y, output logic [3:0] z);\n"
        "  function automatic logic signed [7:0] sign_extend();\n"
        "    return 4'sb1110;\n"
        "  endfunction\n"
        "  function automatic logic [3:0] truncate_value();\n"
        "    return 8'haf;\n"
        "  endfunction\n"
        "  assign y = sign_extend();\n"
        "  assign z = truncate_value();\n"
        "endmodule\n");

    const auto extended = evaluate(fixture, "sign_extend");
    require(extended.value && extended.value->bit_width == 8 &&
                extended.value->is_signed && extended.value->to_uint64() == 0xfe,
            "signed return conversion did not sign-extend and preserve the declared type");

    const auto truncated = evaluate(fixture, "truncate_value");
    require(truncated.value && truncated.value->bit_width == 4 &&
                !truncated.value->is_signed && truncated.value->to_uint64() == 0xf,
            "return conversion did not truncate to the declared function width");
}

void test_parameterized_callable_types_and_two_state_input_conversion() {
    Fixture fixture(
        "module eval_parameterized #(parameter WIDTH = 5) (output bit [WIDTH-1:0] y, output bit [3:0] z);\n"
        "  function automatic bit [WIDTH-1:0] normalize(input bit [WIDTH-1:0] value);\n"
        "    normalize = value;\n"
        "  endfunction\n"
        "  function automatic bit [3:0] remove_unknowns(input logic [3:0] value);\n"
        "    remove_unknowns = value;\n"
        "  endfunction\n"
        "  assign y = normalize(5'd3);\n"
        "  assign z = remove_unknowns(4'b10x1);\n"
        "endmodule\n");

    const auto parameterized = evaluate(fixture, "normalize");
    require(parameterized.value && parameterized.value->bit_width == 5 &&
                parameterized.value->to_uint64() == 3,
            "parameterized formal or return width was not resolved during execution");

    const auto two_state = evaluate(fixture, "remove_unknowns");
    require(two_state.value && two_state.value->bit_width == 4 &&
                two_state.value->is_fully_known() && two_state.value->to_uint64() == 9,
            "four-state input was not converted to a two-state formal value");
}

void test_unsupported_expression_fails_with_location() {
    Fixture fixture(
        "module eval_unsupported_expression(output logic [7:0] y);\n"
        "  function automatic logic [7:0] unsupported(input logic [7:0] value);\n"
        "    unsupported = $display(value);\n"
        "  endfunction\n"
        "  assign y = unsupported(8'd1);\n"
        "endmodule\n");
    const auto result = evaluate(fixture, "unsupported");
    require(!result.value && has_diagnostic(result, "UnsupportedConstantExpression") &&
                !result.diagnostics.empty() && result.diagnostics.front().location.source,
            "unsupported expression did not fail with a source-located diagnostic");
}

void test_selected_and_concatenated_function_assignments() {
    Fixture fixture(
        "module eval_selected_assignments(output logic [7:0] y);\n"
        "  function automatic logic [7:0] compose(input logic [7:0] source);\n"
        "    logic [7:0] local_value;\n"
        "    begin\n"
        "      local_value = 0;\n"
        "      {local_value[7:4], local_value[3:0]} = source;\n"
        "      local_value[7:4] = local_value[3:0];\n"
        "      compose[3:0] = local_value[7:4];\n"
        "      compose[7:4] = local_value[3:0];\n"
        "    end\n"
        "  endfunction\n"
        "  assign y = compose(8'hA5);\n"
        "endmodule\n");
    const auto result = evaluate(fixture, "compose");
    require(result.value && result.value->to_uint64() == 0x55 &&
                result.diagnostics.empty(),
            "function select/concatenation assignment mapping was incorrect");
}

} // namespace

int main() {
    try {
        test_named_input_binding_fresh_frame_and_local_initialization();
        test_uninitialized_four_state_local_and_explicit_return();
        test_bounded_for_loop_and_fully_known_case();
        test_argument_conversion_and_nested_function_calls();
        test_recursion_and_shared_limits_are_diagnosed_at_call_sites();
        test_unsupported_statement_fails_without_substituting_a_value();
        test_local_state_does_not_leak_between_invocations();
        test_return_width_and_signedness_conversions();
        test_parameterized_callable_types_and_two_state_input_conversion();
        test_unsupported_expression_fails_with_location();
        test_selected_and_concatenated_function_assignments();
    }
    catch (const std::exception& error) {
        std::cerr << "EDA Verilog function evaluation test failed: " << error.what()
                  << '\n';
        return 1;
    }
    return 0;
}
