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
        require(!parsed.has_errors(), "native lowering source did not parse");
        require(!elaboration.has_errors(), "native lowering source did not elaborate");
    }
};

bool has_cell(const Module& module, std::string_view type) {
    return std::any_of(module.get_instances().begin(), module.get_instances().end(),
                       [&](const auto& instance) {
                           return instance.get_cell_type() == type;
                       });
}

bool has_diagnostic(const VerilogNativeLoweringResult& result,
                    std::string_view code) {
    return std::any_of(result.diagnostics.begin(), result.diagnostics.end(),
                       [&](const auto& diagnostic) {
                           return diagnostic.code == code;
                       });
}

void test_wire_logic_and_constant_output_structure() {
    Fixture fixture(
        "module m(input logic a, b, output logic y, z);\n"
        "  assign y = a & b;\n"
        "  assign z = 1'b1;\n"
        "endmodule\n");
    require(fixture.lowered.succeeded(),
            "supported flat continuous assignments did not lower");
    const auto& module = *fixture.lowered.module;
    require(module.get_name() == "m" && module.get_ports().size() == 4,
            "native lowering lost the top module or ports");
    require(has_cell(module, "AND2") && has_cell(module, "TIEHI") &&
                has_cell(module, "BUF"),
            "generated native structure lacks AND, constant, or output connectivity");
    const auto emitted = to_verilog(module);
    // The structural Verilog writer emits generic cells as blackbox modules
    // whose names are a deterministic hex encoding of the original cell type.
    require(emitted.find("module m") != std::string::npos &&
                emitted.find("__labneura_cell__414e4432") != std::string::npos &&
                emitted.find("__labneura_cell__5449454849") != std::string::npos,
            "actual structural Verilog omitted the encoded AND2/TIEHI cells");
}

void test_deterministic_lowering_and_failed_result_atomicity() {
    const std::string source =
        "module m(input logic [1:0] a, output logic [1:0] y);\n"
        "  assign y = {a[0], a[1]};\n"
        "endmodule\n";
    Fixture first(source);
    Fixture second(source);
    require(first.lowered.succeeded() && second.lowered.succeeded(),
            "supported concatenation did not lower");
    require(to_verilog(*first.lowered.module) == to_verilog(*second.lowered.module),
            "native netlist naming or ordering was not deterministic");

    Fixture unsupported(
        "module m(input logic [3:0] a, b, output logic [3:0] y);\n"
        "  always_comb y = a * b;\n"
        "endmodule\n");
    require(!unsupported.lowered.succeeded() && !unsupported.lowered.module &&
                !unsupported.lowered.diagnostics.empty(),
            "unsupported hardware operation was not failure-atomic with a diagnostic");
}

void test_unresolved_driver_and_hierarchy_are_not_silently_lowered() {
    Fixture undriven("module m(input logic a, output logic y); endmodule\n");
    require(!undriven.lowered.succeeded() && !undriven.lowered.module &&
                has_diagnostic(undriven.lowered, "UndrivenNativeOutput"),
            "undriven output was not diagnosed");

    Fixture hierarchy(
        "module child(input logic a, output logic y); assign y = a; endmodule\n"
        "module m(input logic a, output logic y); child u(.a(a), .y(y)); endmodule\n");
    require(!hierarchy.lowered.succeeded() && !hierarchy.lowered.module &&
                has_diagnostic(hierarchy.lowered, "UnsupportedNativeHierarchy"),
            "unsupported hierarchy was not explicitly rejected");
}
} // namespace

int main() {
    try {
        test_wire_logic_and_constant_output_structure();
        test_deterministic_lowering_and_failed_result_atomicity();
        test_unresolved_driver_and_hierarchy_are_not_silently_lowered();
    } catch (const std::exception& error) {
        std::cerr << "EDA Verilog native lowering test failed: " << error.what()
                  << '\n';
        return 1;
    }
    return 0;
}
