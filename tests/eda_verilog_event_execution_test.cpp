#include "labneura/eda/verilog_task_execution.h"

#include "labneura/eda/verilog_syntax.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace {

using namespace labneura::eda;

void require(bool condition, std::string_view message) {
    if (!condition)
        throw std::runtime_error(std::string(message));
}

VerilogConstantValue scalar(VerilogLogicBit bit) {
    VerilogConstantValue result;
    result.bit_width = 1;
    result.bits = {bit};
    return result;
}

VerilogConstantValue value(uint64_t number, size_t width = 8) {
    VerilogConstantValue result;
    result.bit_width = width;
    result.bits.resize(width, VerilogLogicBit::Zero);
    for (size_t bit = 0; bit < width && bit < 64; ++bit)
        if ((number >> bit) & 1U)
            result.bits[bit] = VerilogLogicBit::One;
    return result;
}

uint64_t known_value(const VerilogConstantEnvironment& environment,
                     std::string_view name) {
    const auto found = environment.find(std::string(name));
    require(found != environment.end(), "expected caller value is missing");
    const auto converted = found->second.to_uint64();
    require(converted.has_value(), "expected caller value is unknown");
    return *converted;
}

const VerilogExpression* find_task_call(const VerilogStatement* statement,
                                        std::string_view task_name) {
    if (!statement)
        return nullptr;
    if (statement->kind == VerilogStatementKind::TaskCall &&
        statement->call_expression && statement->call_expression->first &&
        statement->call_expression->first->text == task_name)
        return statement->call_expression.get();
    if (const auto* found = find_task_call(statement->first_statement.get(), task_name))
        return found;
    if (const auto* found = find_task_call(statement->second_statement.get(), task_name))
        return found;
    for (const auto& nested : statement->statements)
        if (const auto* found = find_task_call(nested.get(), task_name))
            return found;
    for (const auto& item : statement->case_items)
        if (const auto* found = find_task_call(item.statement.get(), task_name))
            return found;
    return nullptr;
}

struct Fixture {
    VerilogSyntaxParseResult parsed;
    VerilogSemanticAnalysisResult semantic;
    VerilogTypeAnalysisResult types;

    explicit Fixture(std::string source)
        : parsed(parse_verilog_syntax(std::move(source))),
          semantic(analyze_verilog_semantics(parsed.syntax)),
          types(analyze_verilog_expression_types(parsed.syntax, semantic)) {
        require(!parsed.has_errors(), "event execution fixture failed to parse");
        if (semantic.has_errors())
            throw std::runtime_error("event execution fixture failed semantic analysis: " +
                                     semantic.diagnostics.front().code + ": " +
                                     semantic.diagnostics.front().message);
    }

    const VerilogExpression& call(std::string_view name) const {
        for (const auto& module : parsed.syntax.modules)
            for (const auto& block : module.procedural_blocks)
                if (const auto* found = find_task_call(block.body.get(), name))
                    return *found;
        throw std::runtime_error("task call not found in event fixture");
    }
};

VerilogTaskExecutionResult run(
    const Fixture& fixture, std::string_view task,
    VerilogConstantEnvironment& environment,
    std::vector<VerilogCallableSignalTransition> transitions) {
    VerilogCallableExecutionOptions options;
    options.signal_transitions = std::move(transitions);
    return execute_verilog_task_call(fixture.parsed.syntax, fixture.semantic,
                                     fixture.types, fixture.call(task), environment,
                                     std::move(options));
}

bool has_code(const VerilogTaskExecutionResult& result, std::string_view code) {
    return std::any_of(result.diagnostics.begin(), result.diagnostics.end(),
                       [&](const auto& diagnostic) {
                           return diagnostic.code == code;
                       });
}

VerilogCallableSignalTransition transition(uint64_t tick, std::string signal,
                                           VerilogLogicBit next) {
    return {tick, std::move(signal), scalar(next)};
}

void test_level_event_waits_for_an_actual_value_change() {
    Fixture fixture(
        "module level_event;\n"
        "  logic signal;\n"
        "  logic [7:0] result;\n"
        "  task automatic wait_level(output logic [7:0] out_value);\n"
        "    @signal out_value = 8'd5;\n"
        "  endtask\n"
        "  initial wait_level(result);\n"
        "endmodule\n");
    VerilogConstantEnvironment environment{{"signal", scalar(VerilogLogicBit::Zero)},
                                           {"result", value(19)}};
    const auto result = run(
        fixture, "wait_level", environment,
        {transition(2, "signal", VerilogLogicBit::Zero),
         transition(6, "signal", VerilogLogicBit::One)});
    require(result.completed &&
                result.status == VerilogTaskExecutionStatus::Completed &&
                result.diagnostics.empty(),
            "level event did not complete on a real value transition");
    require(result.elapsed_delay_ticks == 6 &&
                known_value(environment, "result") == 5,
            "level event consumed the wrong transition or virtual time");
}

void test_positive_and_negative_edges_use_transition_direction() {
    Fixture fixture(
        "module edge_events;\n"
        "  logic clock;\n"
        "  logic [7:0] positive_result, negative_result;\n"
        "  task automatic wait_positive(output logic [7:0] out_value);\n"
        "    @(posedge clock) out_value = 8'd1;\n"
        "  endtask\n"
        "  task automatic wait_negative(output logic [7:0] out_value);\n"
        "    @(negedge clock) out_value = 8'd2;\n"
        "  endtask\n"
        "  initial begin\n"
        "    wait_positive(positive_result);\n"
        "    wait_negative(negative_result);\n"
        "  end\n"
        "endmodule\n");

    VerilogConstantEnvironment positive_environment{
        {"clock", scalar(VerilogLogicBit::Zero)}, {"positive_result", value(90)}};
    const auto positive = run(
        fixture, "wait_positive", positive_environment,
        {transition(1, "clock", VerilogLogicBit::Zero),
         transition(4, "clock", VerilogLogicBit::One)});
    require(positive.completed && positive.diagnostics.empty() &&
                known_value(positive_environment, "positive_result") == 1 &&
                positive.elapsed_delay_ticks == 4,
            "posedge triggered on a level or missed the 0-to-1 transition");

    VerilogConstantEnvironment negative_environment{
        {"clock", scalar(VerilogLogicBit::One)}, {"negative_result", value(90)}};
    const auto negative = run(
        fixture, "wait_negative", negative_environment,
        {transition(3, "clock", VerilogLogicBit::One),
         transition(8, "clock", VerilogLogicBit::Zero)});
    require(negative.completed && negative.diagnostics.empty() &&
                known_value(negative_environment, "negative_result") == 2 &&
                negative.elapsed_delay_ticks == 8,
            "negedge triggered on a level or missed the 1-to-0 transition");
}

void test_four_state_edges_follow_actual_transitions() {
    Fixture fixture(
        "module four_state_edges;\n"
        "  logic clock;\n"
        "  logic [7:0] positive_result, negative_result;\n"
        "  task automatic wait_positive(output logic [7:0] out_value);\n"
        "    @(posedge clock) out_value = 8'd3;\n"
        "  endtask\n"
        "  task automatic wait_negative(output logic [7:0] out_value);\n"
        "    @(negedge clock) out_value = 8'd4;\n"
        "  endtask\n"
        "  initial begin\n"
        "    wait_positive(positive_result);\n"
        "    wait_negative(negative_result);\n"
        "  end\n"
        "endmodule\n");

    VerilogConstantEnvironment positive_environment{
        {"clock", scalar(VerilogLogicBit::X)}, {"positive_result", value(0)}};
    const auto positive = run(
        fixture, "wait_positive", positive_environment,
        {transition(5, "clock", VerilogLogicBit::One)});
    require(positive.completed &&
                known_value(positive_environment, "positive_result") == 3,
            "posedge did not recognize an X-to-1 transition");

    VerilogConstantEnvironment negative_environment{
        {"clock", scalar(VerilogLogicBit::One)}, {"negative_result", value(0)}};
    const auto negative = run(
        fixture, "wait_negative", negative_environment,
        {transition(7, "clock", VerilogLogicBit::X)});
    require(negative.completed &&
                known_value(negative_environment, "negative_result") == 4,
            "negedge did not recognize a 1-to-X transition");
}

void test_event_lists_are_or_combined() {
    Fixture fixture(
        "module combined_events;\n"
        "  logic clock, reset_n;\n"
        "  logic [7:0] result;\n"
        "  task automatic wait_any(output logic [7:0] out_value);\n"
        "    @(posedge clock or negedge reset_n) out_value = 8'd6;\n"
        "  endtask\n"
        "  initial wait_any(result);\n"
        "endmodule\n");
    VerilogConstantEnvironment environment{
        {"clock", scalar(VerilogLogicBit::Zero)},
        {"reset_n", scalar(VerilogLogicBit::One)}, {"result", value(0)}};
    const auto result = run(
        fixture, "wait_any", environment,
        {transition(2, "clock", VerilogLogicBit::Zero),
         transition(5, "reset_n", VerilogLogicBit::Zero)});
    require(result.completed && result.diagnostics.empty() &&
                known_value(environment, "result") == 6 &&
                result.elapsed_delay_ticks == 5,
            "event-list terms were not OR-combined across transitions");
}

void test_repeated_event_controls_consume_later_matching_transitions() {
    Fixture fixture(
        "module repeated_events;\n"
        "  logic clock;\n"
        "  logic [7:0] result;\n"
        "  task automatic count_two_edges(output logic [7:0] out_value);\n"
        "    begin\n"
        "      out_value = 8'd0;\n"
        "      @(posedge clock) out_value = out_value + 8'd1;\n"
        "      @(posedge clock) out_value = out_value + 8'd1;\n"
        "    end\n"
        "  endtask\n"
        "  initial count_two_edges(result);\n"
        "endmodule\n");
    VerilogConstantEnvironment environment{{"clock", scalar(VerilogLogicBit::Zero)},
                                           {"result", value(99)}};
    const auto result = run(
        fixture, "count_two_edges", environment,
        {transition(1, "clock", VerilogLogicBit::One),
         transition(2, "clock", VerilogLogicBit::Zero),
         transition(5, "clock", VerilogLogicBit::One)});
    require(result.completed && result.diagnostics.empty() &&
                known_value(environment, "result") == 2 &&
                result.elapsed_delay_ticks == 5,
            "repeated event controls did not consume distinct transitions in order");
}

void test_exhausted_trace_reports_no_matching_event_and_rolls_back() {
    Fixture fixture(
        "module no_event;\n"
        "  logic clock;\n"
        "  logic [7:0] result;\n"
        "  task automatic wait_positive(output logic [7:0] out_value);\n"
        "    begin\n"
        "      out_value = 8'd1;\n"
        "      @(posedge clock) out_value = 8'd2;\n"
        "    end\n"
        "  endtask\n"
        "  initial wait_positive(result);\n"
        "endmodule\n");
    VerilogConstantEnvironment environment{{"clock", scalar(VerilogLogicBit::Zero)},
                                           {"result", value(77)}};
    const auto result = run(
        fixture, "wait_positive", environment,
        {transition(3, "clock", VerilogLogicBit::Zero)});
    require(!result.completed &&
                result.status == VerilogTaskExecutionStatus::NoMatchingEvent &&
                has_code(result, "NoMatchingEvent") &&
                !result.diagnostics.empty() &&
                result.diagnostics.front().location.source &&
                known_value(environment, "result") == 77,
            "event trace exhaustion must diagnose and leave caller outputs unchanged");
}

void test_wildcard_event_control_is_explicitly_unsupported() {
    Fixture fixture(
        "module wildcard_event;\n"
        "  logic signal;\n"
        "  logic [7:0] result;\n"
        "  task automatic wait_any_change(input logic signal,\n"
        "      output logic [7:0] out_value);\n"
        "    @(*) out_value = 8'd1;\n"
        "  endtask\n"
        "  initial wait_any_change(signal, result);\n"
        "endmodule\n");
    VerilogConstantEnvironment environment{{"signal", scalar(VerilogLogicBit::Zero)},
                                           {"result", value(55)}};
    const auto result = run(
        fixture, "wait_any_change", environment,
        {transition(1, "signal", VerilogLogicBit::One)});
    require(!result.completed && result.status == VerilogTaskExecutionStatus::Failed &&
                !result.diagnostics.empty() &&
                result.diagnostics.front().location.source &&
                known_value(environment, "result") == 55,
            "unsupported wildcard event control must fail explicitly and transactionally");
}

} // namespace

int main() {
    try {
        test_level_event_waits_for_an_actual_value_change();
        test_positive_and_negative_edges_use_transition_direction();
        test_four_state_edges_follow_actual_transitions();
        test_event_lists_are_or_combined();
        test_repeated_event_controls_consume_later_matching_transitions();
        test_exhausted_trace_reports_no_matching_event_and_rolls_back();
        test_wildcard_event_control_is_explicitly_unsupported();
    }
    catch (const std::exception& error) {
        std::cerr << "EDA Verilog event execution test failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
