#include "labneura/eda/verilog_native_lowering.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {
using namespace labneura::eda;

void require(bool condition, std::string_view message) {
    if (!condition) throw std::runtime_error(std::string(message));
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
        require(!parsed.has_errors(), "native fixture did not parse");
        require(!elaboration.has_errors(), "native fixture did not elaborate");
    }
};

bool has_cell(const Module& module, std::string_view type) {
    return std::any_of(module.get_instances().begin(), module.get_instances().end(),
                       [&](const auto& cell) {
                           return cell.get_cell_type() == type;
                       });
}

void flat_ports_and_continuous_gate_lower_to_existing_netlist() {
    Fixture fixture("module m(input logic a, b, output logic y);\n"
                    "  logic n;\n"
                    "  assign n = a & b;\n"
                    "  assign y = n;\n"
                    "endmodule\n");
    require(fixture.lowered.succeeded(), "flat module failed to lower");
    const auto& module = *fixture.lowered.module;
    require(module.get_ports().size() == 3, "port declarations were not preserved");
    require(has_cell(module, "AND2") && has_cell(module, "BUF"),
            "continuous expression or BUF connection was not lowered");
    require(to_verilog(module) == to_verilog(*fixture.lowered.module),
            "native netlist rendering was not deterministic");
}

void packed_ranges_preserve_declared_bit_names() {
    Fixture fixture("module m(input logic [0:3] a, output logic [0:3] y);\n"
                    "  assign y = a;\n"
                    "endmodule\n");
    require(fixture.lowered.succeeded(), "packed vector assignment failed");
    const auto& module = *fixture.lowered.module;
    require(module.get_ports().front().get_left_index() == 0 &&
                module.get_ports().front().get_right_index() == 3,
            "ascending port range was not preserved");
    const auto contains = [&](std::string_view name) {
        return std::any_of(module.get_nets().begin(), module.get_nets().end(),
                           [&](const auto& net) { return net.get_name() == name; });
    };
    require(contains("a[3]") && contains("a[0]") &&
                contains("y[3]") && contains("y[0]"),
            "packed nets did not use declared indices");
}

void unsupported_types_and_four_state_literals_fail_explicitly() {
    Fixture unpacked("module m(input logic [3:0] a, output logic [3:0] y);\n"
                     "  logic q [1:0];\n"
                     "  assign y = a;\n"
                     "endmodule\n");
    require(!unpacked.lowered.succeeded() && !unpacked.lowered.diagnostics.empty(),
            "unpacked module declaration was silently accepted");

    Fixture unknown("module m(input logic a, output logic y);\n"
                    "  assign y = 1'bx;\n"
                    "endmodule\n");
    require(!unknown.lowered.succeeded() && !unknown.lowered.diagnostics.empty(),
            "four-state constant was silently lowered");
}

} // namespace

int main() {
    try {
        flat_ports_and_continuous_gate_lower_to_existing_netlist();
        packed_ranges_preserve_declared_bit_names();
        unsupported_types_and_four_state_literals_fail_explicitly();
    } catch (const std::exception& exception) {
        std::cerr << "EDA Verilog module lowering test failed: "
                  << exception.what() << '\n';
        return 1;
    }
    return 0;
}
