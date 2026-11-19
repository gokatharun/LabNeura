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
          elaboration(elaborate_verilog_design(parsed.syntax,
                                               VerilogElaborationOptions{
                                                   .top_module = "m"})),
          types(analyze_verilog_expression_types(parsed.syntax,
                                                 elaboration.semantic)),
          eligibility(analyze_verilog_synthesis_eligibility(
              parsed.syntax, elaboration.semantic, types)),
          lowered(lower_verilog_native(parsed.syntax, elaboration, types,
                                       eligibility, "m")) {
        require(!parsed.has_errors(), "combinational fixture did not parse");
        require(!elaboration.has_errors(), "combinational fixture did not elaborate");
    }
};

bool has_cell(const Module& module, std::string_view type) {
    return std::any_of(module.get_instances().begin(), module.get_instances().end(),
                       [&](const auto& instance) {
                           return instance.get_cell_type() == type;
                       });
}

void test_ordered_blocking_assignments_and_if_merge() {
    Fixture fixture(
        "module m(input logic a, b, sel, output logic q);\n"
        "  logic tmp;\n"
        "  always_comb begin\n"
        "    tmp = a;\n"
        "    tmp = tmp & b;\n"
        "    if (sel) q = tmp; else q = a;\n"
        "  end\n"
        "endmodule\n");
    require(fixture.lowered.succeeded(),
            "ordered blocking assignment and if process did not lower");
    require(has_cell(*fixture.lowered.module, "AND2") &&
                has_cell(*fixture.lowered.module, "MUX2") &&
                has_cell(*fixture.lowered.module, "BUF"),
            "lowering omitted expected combinational operator, mux, or output buffer");
}

void test_case_default_and_static_concatenated_destination() {
    Fixture fixture(
        "module m(input logic [1:0] sel, input logic [3:0] a, b,\n"
        "          output logic [3:0] q);\n"
        "  always_comb case (sel)\n"
        "    2'b00: {q[3:2], q[1:0]} = a;\n"
        "    2'b01, 2'b10: {q[3:2], q[1:0]} = b;\n"
        "    default: {q[3:2], q[1:0]} = 4'b0;\n"
        "  endcase\n"
        "endmodule\n");
    require(fixture.lowered.succeeded(),
            "complete case with static concatenated lvalue did not lower");
    require(has_cell(*fixture.lowered.module, "MUX2") &&
                has_cell(*fixture.lowered.module, "BUF"),
            "case lowering omitted decision muxes or destination buffers");
}

void test_incomplete_process_is_not_given_an_implicit_zero() {
    Fixture fixture("module m(input logic en, d, output logic q);\n"
                    "  always_comb if (en) q = d;\n"
                    "endmodule\n");
    require(!fixture.lowered.succeeded() &&
                fixture.lowered.status == VerilogNativeLoweringStatus::Unsupported,
            "incomplete combinational assignment was silently lowered");
    require(!fixture.lowered.diagnostics.empty(),
            "incomplete combinational process produced no diagnostic");
}

void test_nonblocking_combinational_process_is_rejected() {
    Fixture nonblocking("module m(input logic a, output logic q);\n"
                        "  always_comb q <= a;\n"
                        "endmodule\n");
    require(!nonblocking.lowered.succeeded() &&
                !nonblocking.lowered.diagnostics.empty(),
            "nonblocking assignment was accepted by combinational lowering");

}

} // namespace

int main() {
    try {
        test_ordered_blocking_assignments_and_if_merge();
        test_case_default_and_static_concatenated_destination();
        test_incomplete_process_is_not_given_an_implicit_zero();
        test_nonblocking_combinational_process_is_rejected();
    } catch (const std::exception& exception) {
        std::cerr << "EDA Verilog combinational lowering test failed: "
                  << exception.what() << '\n';
        return 1;
    }
    return 0;
}
