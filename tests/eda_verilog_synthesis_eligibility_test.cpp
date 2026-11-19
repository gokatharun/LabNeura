#include "labneura/eda/verilog_process_analysis.h"

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

struct Fixture {
    VerilogSyntaxParseResult parsed;
    VerilogSemanticAnalysisResult semantic;
    VerilogTypeAnalysisResult types;

    explicit Fixture(std::string source)
        : parsed(parse_verilog_syntax(std::move(source))),
          semantic(analyze_verilog_semantics(parsed.syntax)),
          types(analyze_verilog_expression_types(parsed.syntax, semantic)) {
        require(!parsed.has_errors(), "eligibility fixture did not parse");
        require(!semantic.has_errors(), "eligibility fixture has semantic errors");
    }

    VerilogModuleProcessAnalysis analyze() const {
        const auto result = analyze_verilog_synthesis_eligibility(
            parsed.syntax, semantic, types);
        require(result.modules.size() == 1, "expected one analyzed module");
        return result.modules.front();
    }
};

void test_complete_and_incomplete_combinational_processes() {
    Fixture complete("module m(input logic a, output logic q);\n"
                     "always_comb q = a;\nendmodule\n");
    const auto complete_result = complete.analyze();
    require(complete_result.processes.size() == 1 &&
                complete_result.processes[0].synthesis_eligibility ==
                    VerilogSynthesisEligibility::Eligible,
            "complete always_comb process was not eligible");

    Fixture incomplete("module m(input logic a, output logic q);\n"
                       "always_comb if (a) q = 1'b1;\nendmodule\n");
    const auto incomplete_result = incomplete.analyze();
    require(incomplete_result.processes[0].synthesis_eligibility ==
                VerilogSynthesisEligibility::Unsupported,
            "incomplete always_comb process was not rejected");
}

void test_latch_candidate_and_incomplete_case() {
    Fixture latch("module m(input logic en, d, output logic q);\n"
                  "always_latch if (en) q = d;\nendmodule\n");
    const auto latch_result = latch.analyze();
    require(latch_result.processes[0].assignment_coverage.may_require_storage &&
                latch_result.processes[0].synthesis_eligibility ==
                    VerilogSynthesisEligibility::Eligible,
            "always_latch with a hold path was not recognized");

    Fixture incomplete_case("module m(input logic [1:0] sel, a, output logic q);\n"
                            "always_comb case (sel)\n"
                            "  2'b00: q = a;\n"
                            "  default: ;\n"
                            "endcase\nendmodule\n");
    const auto case_result = incomplete_case.analyze();
    require(case_result.processes[0].assignment_coverage.may_require_storage &&
                case_result.processes[0].synthesis_eligibility ==
                    VerilogSynthesisEligibility::Unsupported,
            "incomplete case assignment was not treated as storage-producing");
}

void test_dynamic_selection_remains_indeterminate() {
    Fixture fixture("module m(input logic [1:0] idx, a, output logic [3:0] q);\n"
                    "always_comb q[idx] = a;\nendmodule\n");
    const auto result = fixture.analyze();
    require(result.processes[0].synthesis_eligibility ==
                VerilogSynthesisEligibility::Indeterminate,
            "dynamic lvalue selection was not kept indeterminate");
}

void test_clocked_process_and_initial_process() {
    Fixture fixture("module m(input logic clk, input logic d, output logic q);\n"
                    "always_ff @(posedge clk) q <= d;\n"
                    "initial q = 1'b0;\nendmodule\n");
    const auto result = fixture.analyze();
    require(result.processes.size() == 2 &&
                result.processes[0].synthesis_eligibility ==
                    VerilogSynthesisEligibility::Eligible &&
                result.processes[1].synthesis_eligibility ==
                    VerilogSynthesisEligibility::Indeterminate,
            "clocked and initialization processes were incorrectly classified");
}

void test_multiple_procedural_writers() {
    Fixture fixture("module m(input logic a, b, output logic q);\n"
                    "always_comb q = a;\n"
                    "always_comb q = b;\nendmodule\n");
    const auto result = fixture.analyze();
    require(result.processes.size() == 2,
            "writer analysis lost one process");
    for (const auto& process : result.processes) {
        require(process.synthesis_eligibility ==
                    VerilogSynthesisEligibility::Unsupported,
                "conflicting procedural writer was not rejected");
        bool found = false;
        for (const auto& diagnostic : process.diagnostics)
            found |= diagnostic.code == "MultipleProcessWriters";
        require(found, "conflicting writer diagnostic was not attached");
    }
}

void test_timing_is_explicitly_unsupported() {
    Fixture fixture("module m(input logic a, output logic q);\n"
                    "initial begin #1 q = a; end\nendmodule\n");
    const auto result = fixture.analyze();
    require(result.processes[0].synthesis_eligibility ==
                VerilogSynthesisEligibility::Unsupported,
            "timed initialization was not explicitly unsupported");
}

void test_resolved_continuous_net_drivers_are_not_procedural_conflicts() {
    Fixture fixture("module m(input logic a, b, output wire y);\n"
                    "assign y = a;\n"
                    "assign y = b;\nendmodule\n");
    const auto result = fixture.analyze();
    require(result.processes.size() == 2,
            "continuous net driver discovery lost an assignment");
    for (const auto& process : result.processes)
        require(process.synthesis_eligibility == VerilogSynthesisEligibility::Eligible,
                "multiple continuous drivers on an explicit net port were blanket-rejected");
}

} // namespace

int main() {
    try {
        test_complete_and_incomplete_combinational_processes();
        test_latch_candidate_and_incomplete_case();
        test_dynamic_selection_remains_indeterminate();
        test_clocked_process_and_initial_process();
        test_multiple_procedural_writers();
        test_timing_is_explicitly_unsupported();
        test_resolved_continuous_net_drivers_are_not_procedural_conflicts();
    } catch (const std::exception& exception) {
        std::cerr << "EDA Verilog synthesis eligibility test failed: "
                  << exception.what() << '\n';
        return 1;
    }
    return 0;
}
