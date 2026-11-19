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
    VerilogProcess process;

    explicit Fixture(std::string source)
        : parsed(parse_verilog_syntax(source)),
          semantic(analyze_verilog_semantics(parsed.syntax)),
          types(analyze_verilog_expression_types(parsed.syntax, semantic)) {
        require(!parsed.has_errors(), "coverage fixture failed to parse");
        require(!semantic.has_errors(), "coverage fixture has semantic errors");
        const auto& module = parsed.syntax.modules.front();
        const auto discovery = discover_verilog_processes(module, semantic);
        require(discovery.processes.size() == 1,
                "coverage fixture did not produce exactly one process");
        process = discovery.processes.front();
    }

    VerilogProcessAssignmentCoverage analyze() const {
        return analyze_verilog_assignment_coverage(process, semantic, types);
    }

    const VerilogSymbolAssignmentCoverage& signal(
        const VerilogProcessAssignmentCoverage& coverage,
        std::string_view name) const {
        for (const auto& item : coverage.symbols)
            if (item.symbol && item.symbol->name == name)
                return item;
        throw std::runtime_error("signal missing from assignment coverage");
    }
};

void test_branch_and_bit_coverage() {
    Fixture fixture(
        "module m(input logic sel, output logic [3:0] q);\n"
        "always_comb if (sel) q[0] = 1'b1; else q[0] = 1'b0;\n"
        "endmodule\n");
    const auto coverage = fixture.analyze();
    const auto& q = fixture.signal(coverage, "q");
    require(q.bits.size() == 4 &&
                q.bits[0] == VerilogAssignmentCoverageState::DefinitelyAssigned &&
                q.bits[1] == VerilogAssignmentCoverageState::NotAssigned &&
                q.whole_signal == VerilogAssignmentCoverageState::PossiblyAssigned,
            "branch coverage did not preserve per-bit definite assignment");
    require(coverage.may_require_storage,
            "known incomplete always_comb assignment did not flag storage");
}

void test_case_default_and_concatenation() {
    Fixture fixture(
        "module m(input logic sel, output logic [3:0] q);\n"
        "always_comb case (sel)\n"
        "  1'b0: {q[3:2], q[1:0]} = 4'b0;\n"
        "  default: {q[3:2], q[1:0]} = 4'b1;\n"
        "endcase\nendmodule\n");
    const auto coverage = fixture.analyze();
    const auto& q = fixture.signal(coverage, "q");
    require(q.whole_signal == VerilogAssignmentCoverageState::DefinitelyAssigned,
            "case branches with a default did not cover all concatenated bits");
    require(!coverage.may_require_storage,
            "complete always_comb assignment was marked as storage requiring");
}

void test_zero_iteration_and_guaranteed_do_body() {
    Fixture zero("module m(output logic [1:0] q);\n"
                 "always_comb repeat (0) q = 2'b0;\nendmodule\n");
    const auto zero_coverage = zero.analyze();
    require(zero_coverage.symbols.empty(),
            "zero repeat incorrectly credited a body assignment");

    Fixture once("module m(output logic [1:0] q);\n"
                 "always_comb do q = 2'b0; while (1'b0);\nendmodule\n");
    const auto once_coverage = once.analyze();
    require(once.signal(once_coverage, "q").whole_signal ==
                VerilogAssignmentCoverageState::DefinitelyAssigned,
            "do-while failed to count its guaranteed first execution");
}

void test_dynamic_select_stays_unknown() {
    Fixture fixture("module m(input logic [1:0] index, output logic [3:0] q);\n"
                    "always_comb q[index] = 1'b1;\nendmodule\n");
    const auto coverage = fixture.analyze();
    const auto& q = fixture.signal(coverage, "q");
    require(q.whole_signal == VerilogAssignmentCoverageState::Unknown &&
                !coverage.diagnostics.empty(),
            "dynamic select was treated as known bit coverage");
}

} // namespace

int main() {
    try {
        test_branch_and_bit_coverage();
        test_case_default_and_concatenation();
        test_zero_iteration_and_guaranteed_do_body();
        test_dynamic_select_stays_unknown();
    } catch (const std::exception& exception) {
        std::cerr << "EDA Verilog assignment coverage test failed: "
                  << exception.what() << '\n';
        return 1;
    }
    return 0;
}
