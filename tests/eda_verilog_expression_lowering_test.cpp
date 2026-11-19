#include "labneura/eda/verilog_native_lowering.h"

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

struct Fixture {
    VerilogSyntaxParseResult parsed;
    VerilogElaborationResult elaboration;
    VerilogTypeAnalysisResult types;
    VerilogSynthesisEligibilityResult eligibility;
    VerilogNativeLoweringResult lowered;

    explicit Fixture(std::string source)
        : parsed(parse_verilog_syntax(std::move(source))),
          elaboration(elaborate_verilog_design(
              parsed.syntax, VerilogElaborationOptions{.top_module = "m"})),
          types(analyze_verilog_expression_types(parsed.syntax,
                                                 elaboration.semantic)),
          eligibility(analyze_verilog_synthesis_eligibility(
              parsed.syntax, elaboration.semantic, types)),
          lowered(lower_verilog_native(parsed.syntax, elaboration, types,
                                       eligibility, "m")) {
        if (parsed.has_errors()) {
            std::string message = "expression fixture did not parse";
            for (const auto& diagnostic : parsed.diagnostics)
                message += "\n" + diagnostic.code + ": " + diagnostic.message;
            throw std::runtime_error(message);
        }
        require(!elaboration.has_errors(), "expression fixture did not elaborate");
    }
};

bool has_cell(const Module& module, std::string_view type) {
    return std::any_of(module.get_instances().begin(), module.get_instances().end(),
                       [&](const auto& instance) {
                           return instance.get_cell_type() == type;
                       });
}

void test_bitwise_logic_arithmetic_and_conditional_cells() {
    Fixture fixture(
        "module m(input logic [3:0] a, b, input logic sel, output logic [3:0] q,\n"
        "          output logic truth, equal);\n"
        "  always_comb begin\n"
        "    q = sel ? (a + b) : (a ^ b);\n"
        "    truth = (a && b);\n"
        "    equal = (a == b);\n"
        "  end\n"
        "endmodule\n");
    require(fixture.lowered.succeeded(),
            "supported operators did not lower through the public API");
    require(has_cell(*fixture.lowered.module, "AND2") &&
                has_cell(*fixture.lowered.module, "XOR2") &&
                has_cell(*fixture.lowered.module, "MUX2"),
            "arithmetic, bitwise/logical, comparison, or conditional gates are missing");
}

void test_static_select_concatenation_and_replication() {
    Fixture fixture(
        "module m(input logic [7:0] a, output logic [3:0] q);\n"
        "  always_comb q = {a[7:6], {2{a[1:0]}}};\n"
        "endmodule\n");
    require(fixture.lowered.succeeded(),
            "static packed selects, concatenation, and replication did not lower");
}

void test_unsupported_dynamic_select_and_four_state_literal_diagnose() {
    Fixture dynamic(
        "module m(input logic [3:0] a, input logic [1:0] index, output logic q);\n"
        "  always_comb q = a[index];\n"
        "endmodule\n");
    require(!dynamic.lowered.succeeded() && !dynamic.lowered.diagnostics.empty(),
            "dynamic select was accepted or omitted its diagnostic");

    Fixture four_state(
        "module m(input logic a, output logic q);\n"
        "  always_comb q = a & 1'bx;\n"
        "endmodule\n");
    require(!four_state.lowered.succeeded() &&
                !four_state.lowered.diagnostics.empty(),
            "X-valued expression was accepted or omitted its diagnostic");
}

void test_unsupported_arithmetic_operator_diagnoses() {
    Fixture fixture(
        "module m(input logic [3:0] a, b, output logic [3:0] q);\n"
        "  always_comb q = a * b;\n"
        "endmodule\n");
    require(!fixture.lowered.succeeded() && !fixture.lowered.diagnostics.empty(),
            "unsupported nonconstant multiplication was accepted silently");
}

} // namespace

int main() {
    try {
        test_bitwise_logic_arithmetic_and_conditional_cells();
        test_static_select_concatenation_and_replication();
        test_unsupported_dynamic_select_and_four_state_literal_diagnose();
        test_unsupported_arithmetic_operator_diagnoses();
    } catch (const std::exception& exception) {
        std::cerr << "EDA Verilog expression lowering test failed: "
                  << exception.what() << '\n';
        return 1;
    }
    return 0;
}
