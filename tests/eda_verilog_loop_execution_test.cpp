#include "labneura/eda/verilog_callable_execution.h"
#include "labneura/eda/verilog_syntax.h"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace labneura::eda;

void require(bool condition, std::string_view message) {
    if (!condition)
        throw std::runtime_error(std::string(message));
}

VerilogConstantValue bits(std::string_view msb_first, bool is_signed = false) {
    VerilogConstantValue value;
    value.bit_width = msb_first.size();
    value.is_signed = is_signed;
    for (auto it = msb_first.rbegin(); it != msb_first.rend(); ++it) {
        switch (*it) {
            case '0': value.bits.push_back(VerilogLogicBit::Zero); break;
            case '1': value.bits.push_back(VerilogLogicBit::One); break;
            case 'x': value.bits.push_back(VerilogLogicBit::X); break;
            case 'z': value.bits.push_back(VerilogLogicBit::Z); break;
            default: throw std::runtime_error("invalid logic bit");
        }
    }
    return value;
}

VerilogStatement node(VerilogStatementKind kind, std::string text = {}) {
    VerilogStatement statement;
    statement.kind = kind;
    statement.text = std::move(text);
    return statement;
}

VerilogExpression expression(std::string text) {
    VerilogExpression result;
    result.kind = VerilogExpressionKind::Identifier;
    result.text = std::move(text);
    return result;
}

struct Harness {
    size_t steps = 0;
    size_t step_limit = 1000;
    size_t counter = 0;
    std::vector<std::string> trace;
    std::vector<std::string> diagnostics;

    VerilogProceduralExecutionCallbacks callbacks() {
        VerilogProceduralExecutionCallbacks result;
        result.evaluate_expression = [this](const VerilogExpression& expr)
            -> std::optional<VerilogConstantValue> {
            if (expr.text == "count")
                return bits("0011");
            if (expr.text == "negative")
                return bits("1111", true);
            if (expr.text == "unknown")
                return bits("0x");
            if (expr.text == "condition")
                return bits(counter < 3 ? "1" : "0");
            if (expr.text == "x_condition")
                return bits("x");
            return std::nullopt;
        };
        result.execute_statement = [this](const VerilogStatement& statement) {
            trace.push_back(statement.text);
            if (statement.text == "increment") {
                ++counter;
                return VerilogExecutionOutcome{};
            }
            if (statement.text == "break")
                return VerilogExecutionOutcome{VerilogExecutionOutcomeKind::Break,
                                               std::nullopt};
            if (statement.text == "continue")
                return VerilogExecutionOutcome{VerilogExecutionOutcomeKind::Continue,
                                               std::nullopt};
            if (statement.text == "return")
                return VerilogExecutionOutcome{VerilogExecutionOutcomeKind::Return,
                                               std::nullopt};
            return VerilogExecutionOutcome{};
        };
        result.consume_statement_step = [this](const VerilogSyntaxLocation&) {
            if (steps == step_limit)
                return false;
            ++steps;
            return true;
        };
        result.diagnose = [this](const std::string& code, const std::string&,
                                 const VerilogSyntaxLocation&) {
            diagnostics.push_back(code);
        };
        return result;
    }
};

void test_while_condition_and_false_x() {
    Harness harness;
    VerilogStatement loop = node(VerilogStatementKind::Loop);
    loop.loop_kind = VerilogLoopKind::While;
    loop.expression = std::make_unique<VerilogExpression>(expression("condition"));
    loop.first_statement = std::make_unique<VerilogStatement>(
        node(VerilogStatementKind::Empty, "increment"));
    auto callbacks = harness.callbacks();
    auto result = execute_verilog_procedural_loop(loop, callbacks);
    require(result.kind == VerilogExecutionOutcomeKind::NormalCompletion,
            "while loop did not complete normally");
    require(harness.counter == 3, "while loop condition/body ordering is incorrect");

    Harness x_harness;
    VerilogStatement x_loop = node(VerilogStatementKind::Loop);
    x_loop.loop_kind = VerilogLoopKind::While;
    x_loop.expression = std::make_unique<VerilogExpression>(expression("x_condition"));
    x_loop.first_statement = std::make_unique<VerilogStatement>(
        node(VerilogStatementKind::Empty, "must-not-run"));
    callbacks = x_harness.callbacks();
    result = execute_verilog_procedural_loop(x_loop, callbacks);
    require(result.kind == VerilogExecutionOutcomeKind::NormalCompletion &&
                x_harness.trace.empty(),
            "unknown while condition should behave as false");
}

void test_do_while_body_precedes_condition() {
    Harness harness;
    VerilogStatement loop = node(VerilogStatementKind::Loop);
    loop.loop_kind = VerilogLoopKind::DoWhile;
    loop.expression = std::make_unique<VerilogExpression>(expression("condition"));
    loop.first_statement = std::make_unique<VerilogStatement>(
        node(VerilogStatementKind::Empty, "increment"));
    auto callbacks = harness.callbacks();
    const auto result = execute_verilog_procedural_loop(loop, callbacks);
    require(result.kind == VerilogExecutionOutcomeKind::NormalCompletion &&
                harness.counter == 3,
            "do-while did not execute its body before testing the condition");
}

void test_repeat_zero_negative_and_count() {
    Harness harness;
    VerilogStatement loop = node(VerilogStatementKind::Loop);
    loop.loop_kind = VerilogLoopKind::Repeat;
    loop.expression = std::make_unique<VerilogExpression>(expression("count"));
    loop.first_statement = std::make_unique<VerilogStatement>(
        node(VerilogStatementKind::Empty, "body"));
    auto callbacks = harness.callbacks();
    auto result = execute_verilog_procedural_loop(loop, callbacks);
    require(result.kind == VerilogExecutionOutcomeKind::NormalCompletion &&
                harness.trace.size() == 3,
            "repeat count did not execute the body the requested number of times");

    Harness zero;
    loop.expression = std::make_unique<VerilogExpression>(expression("zero"));
    callbacks = zero.callbacks();
    callbacks.evaluate_expression = [](const VerilogExpression&) {
        return std::optional(bits("0000"));
    };
    result = execute_verilog_procedural_loop(loop, callbacks);
    require(result.kind == VerilogExecutionOutcomeKind::NormalCompletion &&
                zero.trace.empty() && zero.steps == 0,
            "zero repeat count should execute no body iterations");

    Harness negative;
    loop.expression = std::make_unique<VerilogExpression>(expression("negative"));
    callbacks = negative.callbacks();
    result = execute_verilog_procedural_loop(loop, callbacks);
    require(result.kind == VerilogExecutionOutcomeKind::DiagnosticFailure &&
                !negative.diagnostics.empty() &&
                negative.diagnostics.back() == "NegativeCallableRepeatCount",
            "negative repeat count was not rejected explicitly");

    Harness unknown;
    loop.expression = std::make_unique<VerilogExpression>(expression("unknown"));
    callbacks = unknown.callbacks();
    result = execute_verilog_procedural_loop(loop, callbacks);
    require(result.kind == VerilogExecutionOutcomeKind::DiagnosticFailure &&
                unknown.diagnostics.back() == "UnknownCallableRepeatCount",
            "unknown repeat count was not rejected explicitly");
}

void test_for_ordering_continue_and_break() {
    Harness harness;
    VerilogStatement loop = node(VerilogStatementKind::Loop);
    loop.loop_kind = VerilogLoopKind::For;
    loop.loop_initialization = std::make_unique<VerilogStatement>(
        node(VerilogStatementKind::Empty, "init"));
    loop.expression = std::make_unique<VerilogExpression>(expression("condition"));
    loop.first_statement = std::make_unique<VerilogStatement>(
        node(VerilogStatementKind::Empty, "continue"));
    loop.loop_step = std::make_unique<VerilogStatement>(
        node(VerilogStatementKind::Empty, "increment"));
    auto callbacks = harness.callbacks();
    auto result = execute_verilog_procedural_loop(loop, callbacks);
    require(result.kind == VerilogExecutionOutcomeKind::NormalCompletion,
            "for loop with continue did not complete");
    require(harness.counter == 3, "continue failed to execute for-loop update");
    require(harness.trace.size() == 7 && harness.trace.front() == "init" &&
                harness.trace[1] == "continue" && harness.trace[2] == "increment",
            "for-loop initialization/body/update ordering is incorrect");

    Harness breaking;
    loop.first_statement = std::make_unique<VerilogStatement>(
        node(VerilogStatementKind::Empty, "break"));
    callbacks = breaking.callbacks();
    result = execute_verilog_procedural_loop(loop, callbacks);
    require(result.kind == VerilogExecutionOutcomeKind::NormalCompletion &&
                breaking.counter == 0,
            "break did not terminate the loop before its update");
}

void test_return_and_forever_budget() {
    Harness returning;
    VerilogStatement loop = node(VerilogStatementKind::Loop);
    loop.loop_kind = VerilogLoopKind::Repeat;
    loop.expression = std::make_unique<VerilogExpression>(expression("count"));
    loop.first_statement = std::make_unique<VerilogStatement>(
        node(VerilogStatementKind::Empty, "return"));
    auto callbacks = returning.callbacks();
    auto result = execute_verilog_procedural_loop(loop, callbacks);
    require(result.kind == VerilogExecutionOutcomeKind::Return,
            "return did not propagate through the loop");

    Harness bounded;
    bounded.step_limit = 4;
    loop = node(VerilogStatementKind::Loop);
    loop.loop_kind = VerilogLoopKind::Forever;
    callbacks = bounded.callbacks();
    result = execute_verilog_procedural_loop(loop, callbacks);
    require(result.kind == VerilogExecutionOutcomeKind::ExecutionBudgetExhausted &&
                bounded.steps == 4,
            "empty forever loop did not terminate at the execution budget");
}

} // namespace

int main() {
    try {
        test_while_condition_and_false_x();
        test_do_while_body_precedes_condition();
        test_repeat_zero_negative_and_count();
        test_for_ordering_continue_and_break();
        test_return_and_forever_budget();
        std::cout << "All procedural loop execution tests passed\n";
    }
    catch (const std::exception& exception) {
        std::cerr << exception.what() << '\n';
        return 1;
    }
    return 0;
}
