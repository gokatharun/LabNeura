#include "labneura/eda/verilog_native_lowering.h"
#include "support/eda_sequential_evaluator.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
using namespace labneura::eda;
using namespace labneura::eda::test;

void require(bool condition, std::string_view message) {
    if (!condition)
        throw std::runtime_error(std::string(message));
}

struct NativeFixture {
    VerilogSyntaxParseResult parsed;
    VerilogElaborationResult elaboration;
    VerilogTypeAnalysisResult types;
    VerilogSynthesisEligibilityResult eligibility;
    VerilogNativeLoweringResult lowered;

    explicit NativeFixture(std::string source)
        : parsed(parse_verilog_syntax(std::move(source))),
          elaboration(elaborate_verilog_design(
              parsed.syntax, VerilogElaborationOptions{.top_module = "m"})),
          types(analyze_verilog_expression_types(parsed.syntax,
                                                 elaboration.semantic)),
          eligibility(analyze_verilog_synthesis_eligibility(
              parsed.syntax, elaboration.semantic, types)),
          lowered(lower_verilog_native(parsed.syntax, elaboration, types,
                                       eligibility, "m")) {
        require(!parsed.has_errors() && !elaboration.has_errors(),
                "sequential equivalence source failed to parse/elaborate");
        require(lowered.succeeded(), "sequential equivalence source did not lower");
    }
};

std::string register_name(const Module& module, std::string_view type,
                          std::size_t ordinal = 0) {
    for (const auto& instance : module.get_instances()) {
        if (instance.get_cell_type() != type)
            continue;
        if (ordinal == 0)
            return instance.get_name();
        --ordinal;
    }
    throw std::runtime_error("could not find register instance of type " +
                             std::string(type));
}

SequentialStimulusFrame frame(std::initializer_list<
                              std::pair<const std::string, VerificationBits>> values) {
    return {VerificationPortValues(values)};
}

std::vector<SequentialStimulusFrame> frames(
    std::initializer_list<SequentialStimulusFrame> values) {
    return std::vector<SequentialStimulusFrame>(values);
}

std::uint8_t output_bit(const SequentialEvaluationResult& result,
                        std::size_t frame_index, std::string_view port) {
    const auto& outputs = result.trace.at(frame_index).outputs;
    const auto found = outputs.find(std::string(port));
    if (found == outputs.end() || found->second.size() != 1)
        throw std::runtime_error("sequential trace omitted scalar output " +
                                 std::string(port));
    return found->second.front();
}

SequentialEvaluationResult simulate(
    const Module& module, std::vector<SequentialStimulusFrame> stimulus,
    std::map<std::string, std::uint8_t> initial_state) {
    SequentialEvaluationOptions options;
    options.initial_state = std::move(initial_state);
    return evaluate_sequential_netlist(module, stimulus, options);
}

void test_positive_edge_hold_and_multicycle_behavior() {
    NativeFixture fixture(
        "module m(input logic clk, d, output logic q);\n"
        "  always_ff @(posedge clk) q <= d;\n"
        "endmodule\n");
    const auto state = register_name(*fixture.lowered.module, "DFF");
    const auto result = simulate(
        *fixture.lowered.module,
        frames({frame({{"clk", {0}}, {"d", {0}}}),
                frame({{"clk", {1}}, {"d", {1}}}),
                frame({{"clk", {1}}, {"d", {0}}}),
                frame({{"clk", {0}}, {"d", {0}}}),
                frame({{"clk", {1}}, {"d", {0}}})}),
        {{state, 0}});
    require(result.succeeded(), "positive-edge multi-cycle trace failed");
    require(output_bit(result, 0, "q") == 0 &&
                output_bit(result, 1, "q") == 1 &&
                output_bit(result, 2, "q") == 1 &&
                output_bit(result, 3, "q") == 1 &&
                output_bit(result, 4, "q") == 0,
            "positive-edge sampling or between-edge hold behavior is incorrect");
}

void test_negative_edge_clock() {
    NativeFixture fixture(
        "module m(input logic clk, d, output logic q);\n"
        "  always @(negedge clk) q <= d;\n"
        "endmodule\n");
    const auto state = register_name(*fixture.lowered.module, "DFF");
    const auto result = simulate(
        *fixture.lowered.module,
        frames({frame({{"clk", {1}}, {"d", {0}}}),
                frame({{"clk", {0}}, {"d", {1}}}),
                frame({{"clk", {1}}, {"d", {0}}}),
                frame({{"clk", {0}}, {"d", {0}}})}),
        {{state, 0}});
    require(result.succeeded(), "negative-edge trace failed");
    require(output_bit(result, 1, "q") == 1 &&
                output_bit(result, 2, "q") == 1 &&
                output_bit(result, 3, "q") == 0,
            "inverted-clock edge polarity or hold behavior is incorrect");
}

void test_enable_and_synchronous_reset() {
    NativeFixture enable(
        "module m(input logic clk, en, d, output logic q);\n"
        "  always_ff @(posedge clk) if (en) q <= d;\n"
        "endmodule\n");
    const auto enable_state = register_name(*enable.lowered.module, "DFF");
    const auto enabled = simulate(
        *enable.lowered.module,
        frames({frame({{"clk", {0}}, {"en", {0}}, {"d", {1}}}),
                frame({{"clk", {1}}, {"en", {0}}, {"d", {1}}}),
                frame({{"clk", {0}}, {"en", {1}}, {"d", {1}}}),
                frame({{"clk", {1}}, {"en", {1}}, {"d", {1}}}),
                frame({{"clk", {0}}, {"en", {0}}, {"d", {0}}}),
                frame({{"clk", {1}}, {"en", {0}}, {"d", {0}}})}),
        {{enable_state, 0}});
    require(enabled.succeeded() && output_bit(enabled, 1, "q") == 0 &&
                output_bit(enabled, 3, "q") == 1 &&
                output_bit(enabled, 5, "q") == 1,
            "register enable or implicit hold behavior is incorrect");

    NativeFixture sync(
        "module m(input logic clk, rst, d, output logic q);\n"
        "  always_ff @(posedge clk) if (rst) q <= 1'b0; else q <= d;\n"
        "endmodule\n");
    const auto sync_state = register_name(*sync.lowered.module, "DFF");
    const auto reset = simulate(
        *sync.lowered.module,
        frames({frame({{"clk", {0}}, {"rst", {0}}, {"d", {1}}}),
                frame({{"clk", {1}}, {"rst", {0}}, {"d", {1}}}),
                frame({{"clk", {0}}, {"rst", {1}}, {"d", {1}}}),
                frame({{"clk", {1}}, {"rst", {1}}, {"d", {1}}}),
                frame({{"clk", {0}}, {"rst", {0}}, {"d", {1}}}),
                frame({{"clk", {1}}, {"rst", {0}}, {"d", {1}}})}),
        {{sync_state, 0}});
    require(reset.succeeded() && output_bit(reset, 1, "q") == 1 &&
                output_bit(reset, 2, "q") == 1 &&
                output_bit(reset, 3, "q") == 0 &&
                output_bit(reset, 5, "q") == 1,
            "synchronous reset changed state without/away from a clock edge");
}

void test_async_reset_assertion_without_clock_and_deassertion() {
    NativeFixture reset_fixture(
        "module m(input logic clk, rst_n, d, output logic q);\n"
        "  always_ff @(posedge clk or negedge rst_n)\n"
        "    if (!rst_n) q <= 1'b0; else q <= d;\n"
        "endmodule\n");
    const auto state = register_name(*reset_fixture.lowered.module, "DFFR");
    const auto result = simulate(
        *reset_fixture.lowered.module,
        frames({frame({{"clk", {0}}, {"rst_n", {1}}, {"d", {0}}}),
                frame({{"clk", {0}}, {"rst_n", {0}}, {"d", {1}}}),
                frame({{"clk", {0}}, {"rst_n", {1}}, {"d", {1}}}),
                frame({{"clk", {1}}, {"rst_n", {1}}, {"d", {1}}})}),
        {{state, 1}});
    require(result.succeeded(), "asynchronous reset trace failed");
    require(output_bit(result, 0, "q") == 1 &&
                output_bit(result, 1, "q") == 0 &&
                output_bit(result, 2, "q") == 0 &&
                output_bit(result, 3, "q") == 1,
            "asynchronous reset assertion/deassertion behavior is incorrect");

    const auto initially_asserted = simulate(
        *reset_fixture.lowered.module,
        frames({frame({{"clk", {0}}, {"rst_n", {0}}, {"d", {1}}})}),
        {{state, 1}});
    require(initially_asserted.succeeded() &&
                output_bit(initially_asserted, 0, "q") == 0,
            "an initially asserted active-low reset did not force state before observation");

    NativeFixture set_fixture(
        "module m(input logic clk, rst, d, output logic q);\n"
        "  always_ff @(posedge clk or posedge rst)\n"
        "    if (rst) q <= 1'b1; else q <= d;\n"
        "endmodule\n");
    const auto set_state = register_name(*set_fixture.lowered.module, "DFFS");
    const auto set = simulate(
        *set_fixture.lowered.module,
        frames({frame({{"clk", {0}}, {"rst", {0}}, {"d", {0}}}),
                frame({{"clk", {0}}, {"rst", {1}}, {"d", {0}}}),
                frame({{"clk", {0}}, {"rst", {0}}, {"d", {0}}})}),
        {{set_state, 0}});
    require(set.succeeded() && output_bit(set, 1, "q") == 1 &&
                output_bit(set, 2, "q") == 1,
            "active-high asynchronous set or deassertion behavior is incorrect");

    const auto initially_asserted_set = simulate(
        *set_fixture.lowered.module,
        frames({frame({{"clk", {0}}, {"rst", {1}}, {"d", {0}}})}),
        {{set_state, 0}});
    require(initially_asserted_set.succeeded() &&
                output_bit(initially_asserted_set, 0, "q") == 1,
            "an initially asserted active-high set did not force state before observation");
}

void test_simultaneous_preupdate_state_and_explicit_initialization() {
    NativeFixture fixture(
        "module m(input logic clk, output logic a, b);\n"
        "  always_ff @(posedge clk) begin a <= b; b <= a; end\n"
        "endmodule\n");
    const auto first = register_name(*fixture.lowered.module, "DFF", 0);
    const auto second = register_name(*fixture.lowered.module, "DFF", 1);
    const auto result = simulate(
        *fixture.lowered.module,
        frames({frame({{"clk", {0}}}), frame({{"clk", {1}}})}),
        {{first, 0}, {second, 1}});
    require(result.succeeded() && output_bit(result, 1, "a") == 1 &&
                output_bit(result, 1, "b") == 0,
            "register updates did not use one common pre-edge state");

    SequentialEvaluationOptions missing_initial;
    const auto rejected = evaluate_sequential_netlist(
        *fixture.lowered.module,
        frames({frame({{"clk", {0}}})}), missing_initial);
    require(!rejected.succeeded() &&
                rejected.status == NetlistVerificationStatus::InvalidStimulus,
            "evaluator guessed an initial state instead of requiring one");
}

void test_limits_and_unsupported_sequential_cells() {
    Module unsupported("m", {}, {},
                      {Instance("u0", "DFFSR", {{"D", "d"},
                                                   {"CLK", "clk"},
                                                   {"RESET_B", "rst"},
                                                   {"SET_B", "set"},
                                                   {"Y", "q"}})});
    const auto unsupported_result = evaluate_sequential_netlist(
        unsupported, frames({frame({})}),
        SequentialEvaluationOptions{});
    require(!unsupported_result.succeeded() &&
                unsupported_result.status == NetlistVerificationStatus::UnsupportedCell,
            "unsupported sequential cell was silently accepted");

    NativeFixture fixture(
        "module m(input logic clk, d, output logic q);\n"
        "  always_ff @(posedge clk) q <= d;\n"
        "endmodule\n");
    SequentialEvaluationOptions limited;
    limited.initial_state.emplace(register_name(*fixture.lowered.module, "DFF"), 0);
    limited.max_frames = 1;
    const auto too_many = evaluate_sequential_netlist(
        *fixture.lowered.module,
        frames({frame({{"clk", {0}}, {"d", {0}}}),
                frame({{"clk", {1}}, {"d", {1}}})}),
        limited);
    require(!too_many.succeeded() &&
                too_many.status == NetlistVerificationStatus::ResourceLimit,
            "configured sequential frame bound was not enforced");
}

} // namespace

int main() {
    try {
        test_positive_edge_hold_and_multicycle_behavior();
        test_negative_edge_clock();
        test_enable_and_synchronous_reset();
        test_async_reset_assertion_without_clock_and_deassertion();
        test_simultaneous_preupdate_state_and_explicit_initialization();
        test_limits_and_unsupported_sequential_cells();
    } catch (const std::exception& error) {
        std::cerr << "EDA sequential equivalence test failed: " << error.what()
                  << '\n';
        return 1;
    }
    return 0;
}
