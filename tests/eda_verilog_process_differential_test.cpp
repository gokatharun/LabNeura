#include "labneura/eda/verilog_process_analysis.h"

#include <slang/ast/Compilation.h>
#include <slang/ast/symbols/BlockSymbols.h>
#include <slang/ast/symbols/CompilationUnitSymbols.h>
#include <slang/ast/symbols/InstanceSymbols.h>
#include <slang/syntax/SyntaxTree.h>

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

struct NativeFixture {
    VerilogSyntaxParseResult parsed;
    VerilogSemanticAnalysisResult semantic;
    VerilogTypeAnalysisResult types;
    VerilogSynthesisEligibilityResult eligibility;

    explicit NativeFixture(const std::string& source)
        : parsed(parse_verilog_syntax(source)),
          semantic(analyze_verilog_semantics(parsed.syntax)),
          types(analyze_verilog_expression_types(parsed.syntax, semantic)),
          eligibility(analyze_verilog_synthesis_eligibility(parsed.syntax, semantic,
                                                            types)) {
        require(!parsed.has_errors(), "native process fixture failed to parse");
        require(!semantic.has_errors(), "native process fixture has semantic errors");
        require(eligibility.modules.size() == parsed.syntax.modules.size(),
                "native eligibility did not return one result per module");
    }

    const VerilogModuleProcessAnalysis& module() const {
        require(!eligibility.modules.empty(), "native result has no module analysis");
        return eligibility.modules.front();
    }
};

struct SlangFixture {
    slang::SourceManager source_manager;
    std::shared_ptr<slang::syntax::SyntaxTree> tree;
    slang::ast::Compilation compilation;

    explicit SlangFixture(const std::string& source)
        : tree(slang::syntax::SyntaxTree::fromText(source, source_manager,
                                                   "process_analysis_diff.sv")) {
        compilation.addSyntaxTree(tree);
        for (const auto& diagnostic : compilation.getAllDiagnostics())
            require(!diagnostic.isError(),
                    "Slang rejected syntax or semantic process reference fixture");
    }

    const slang::ast::InstanceSymbol& top() {
        const auto& tops = compilation.getRoot().topInstances;
        require(tops.size() == 1, "Slang did not elaborate one process reference top");
        return *tops.front();
    }
};

size_t slang_process_count(const slang::ast::InstanceSymbol& top) {
    size_t count = 0;
    for (const auto& member : top.body.members())
        if (member.kind == slang::ast::SymbolKind::ProceduralBlock)
            ++count;
    return count;
}

void test_process_discovery_count_against_slang() {
    const std::string source =
        "module process_count(input logic a, clk, output logic y, q, r, i);\n"
        "  assign y = a;\n"
        "  always_comb q = a;\n"
        "  always_ff @(posedge clk) r <= a;\n"
        "  initial i = 1'b0;\n"
        "endmodule\n";
    NativeFixture native(source);
    SlangFixture slang(source);
    require(native.module().processes.size() == 4,
            "native process discovery did not retain continuous and procedural records");
    require(slang_process_count(slang.top()) == 3,
            "Slang reference did not elaborate the three procedural blocks");
    // The comparison is deliberately limited to process inventory: Slang's
    // syntax/AST acceptance is not a reference for native synthesis eligibility.
    require(native.module().processes.size() == slang_process_count(slang.top()) + 1,
            "native process inventory differs from Slang by the continuous driver");
}

void test_complete_and_incomplete_combinational_eligibility() {
    NativeFixture fixture(
        "module comb(input logic a, b, sel, output logic y, z);\n"
        "  always_comb y = a & b;\n"
        "  always_comb if (sel) z = a;\n"
        "endmodule\n");
    require(fixture.module().processes.size() == 2,
            "combinational fixture process count is incorrect");
    const auto& complete = fixture.module().processes[0];
    const auto& incomplete = fixture.module().processes[1];
    require(complete.assignment_coverage.knowledge == VerilogAnalysisKnowledge::Known &&
                !complete.assignment_coverage.may_require_storage &&
                complete.synthesis_eligibility == VerilogSynthesisEligibility::Eligible,
            "complete always_comb assignment was not eligible");
    require(incomplete.assignment_coverage.may_require_storage &&
                incomplete.synthesis_eligibility == VerilogSynthesisEligibility::Unsupported,
            "incomplete always_comb assignment was not rejected as storage inference");
}

void test_partial_vector_and_complete_case_coverage() {
    NativeFixture fixture(
        "module coverage(input logic a, sel, input logic [1:0] d,\n"
        "                output logic [3:0] partial, output logic [1:0] full);\n"
        "  always_comb partial[0] = a;\n"
        "  always_comb case (sel)\n"
        "    1'b0: full = d;\n"
        "    1'b1: full = 2'b00;\n"
        "    default: full = 2'b11;\n"
        "  endcase\n"
        "endmodule\n");
    require(fixture.module().processes.size() == 2,
            "coverage fixture process count is incorrect");
    const auto& partial = fixture.module().processes[0];
    const auto& complete_case = fixture.module().processes[1];
    require(partial.assignment_coverage.may_require_storage &&
                partial.synthesis_eligibility == VerilogSynthesisEligibility::Unsupported,
            "partial packed-vector assignment was incorrectly considered complete");
    require(complete_case.assignment_coverage.knowledge == VerilogAnalysisKnowledge::Known &&
                !complete_case.assignment_coverage.may_require_storage &&
                complete_case.synthesis_eligibility == VerilogSynthesisEligibility::Eligible,
            "case with a default branch was not considered completely assigned");
}

void test_clock_reset_and_nonblocking_classification() {
    NativeFixture fixture(
        "module sequential(input logic clk, rst_n, en, d, output logic q);\n"
        "  always_ff @(posedge clk or negedge rst_n)\n"
        "    if (!rst_n) q <= 1'b0;\n"
        "    else if (en) q <= d;\n"
        "endmodule\n");
    require(fixture.module().processes.size() == 1,
            "sequential fixture process count is incorrect");
    const auto& process = fixture.module().processes.front();
    require(process.sequential.has_nonblocking_assignments &&
                process.sequential.clocks.size() == 1 &&
                process.sequential.clocks.front().symbol &&
                process.sequential.clocks.front().symbol->name == "clk" &&
                !process.sequential.resets.empty() &&
                process.sequential.resets.front().symbol &&
                process.sequential.resets.front().symbol->name == "rst_n" &&
                !process.sequential.resets.front().active_high &&
                process.sequential.resets.front().asynchronous,
            "clock, active-low asynchronous reset, or NBA classification is incorrect");
    require(process.synthesis_eligibility == VerilogSynthesisEligibility::Eligible,
            "supported single-clock reset process was not eligible");
}

void test_unsupported_timing_unknown_events_and_multiple_writers() {
    NativeFixture delayed(
        "module delayed(input logic a, output logic y);\n"
        "  always_comb begin #1 y = a; end\n"
        "endmodule\n");
    require(delayed.module().processes.front().synthesis_eligibility ==
                VerilogSynthesisEligibility::Unsupported,
            "timing-controlled process was not explicitly unsupported");

    NativeFixture ambiguous(
        "module ambiguous(input logic a, b, output logic y);\n"
        "  always @(a) y = a & b;\n"
        "endmodule\n");
    require(ambiguous.module().processes.front().synthesis_eligibility !=
                VerilogSynthesisEligibility::Eligible,
            "unqualified partial sensitivity was silently declared eligible");

    NativeFixture mixed_events(
        "module mixed_events(input logic clk, a, output logic y);\n"
        "  always @(posedge clk or a) y = a;\n"
        "endmodule\n");
    require(mixed_events.module().processes.front().synthesis_eligibility ==
                VerilogSynthesisEligibility::Unsupported,
            "mixed edge/level events were silently treated as combinational");

    NativeFixture writers(
        "module writers(input logic a, b, output logic y);\n"
        "  always_comb y = a;\n"
        "  always_comb y = b;\n"
        "endmodule\n");
    require(writers.module().processes.size() == 2,
            "multiple-writer fixture process count is incorrect");
    for (const auto& process : writers.module().processes)
        require(process.synthesis_eligibility == VerilogSynthesisEligibility::Unsupported,
                "cross-process procedural writer conflict was not rejected");
}

} // namespace

int main() {
    try {
        test_process_discovery_count_against_slang();
        test_complete_and_incomplete_combinational_eligibility();
        test_partial_vector_and_complete_case_coverage();
        test_clock_reset_and_nonblocking_classification();
        test_unsupported_timing_unknown_events_and_multiple_writers();
    } catch (const std::exception& error) {
        std::cerr << "EDA Verilog process differential test failed: " << error.what()
                  << '\n';
        return 1;
    }
    return 0;
}
