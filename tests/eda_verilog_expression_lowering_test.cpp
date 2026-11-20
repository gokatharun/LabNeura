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

bool has_buffer(const Module& module, std::string_view source,
                std::string_view destination) {
    return std::any_of(module.get_instances().begin(), module.get_instances().end(),
                       [&](const auto& instance) {
        if (instance.get_cell_type() != "BUF")
            return false;
        const auto& connections = instance.get_connections();
        const auto input = std::find_if(connections.begin(), connections.end(),
                                        [](const auto& connection) {
                                            return connection.get_pin() == "A";
                                        });
        const auto output = std::find_if(connections.begin(), connections.end(),
                                         [](const auto& connection) {
                                             return connection.get_pin() == "Y";
                                         });
        return input != connections.end() && output != connections.end() &&
               input->get_net() == source && output->get_net() == destination;
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

void test_identifier_select_range_direction_and_part_mapping() {
    Fixture fixture(
        "module m(input logic [7:4] descending, input logic [0:3] ascending,\n"
        "         output logic high, low, output logic [3:0] part, part_up);\n"
        "  assign high = descending[7];\n"
        "  assign low = ascending[0];\n"
        "  assign part = descending[7:4];\n"
        "  assign part_up = ascending[3 -: 4];\n"
        "endmodule\n");
    require(fixture.lowered.succeeded(),
            "identifier selects on ascending or descending ranges did not lower");
    const auto& module = *fixture.lowered.module;
    require(has_buffer(module, "descending[7]", "high") &&
                has_buffer(module, "ascending[0]", "low"),
            "bit-select range direction changed the selected source net");
    require(has_buffer(module, "descending[4]", "part[0]") &&
                has_buffer(module, "descending[7]", "part[3]"),
            "static part-select bit ordering changed during lowering");
    require(has_buffer(module, "ascending[0]", "part_up[0]") &&
                has_buffer(module, "ascending[3]", "part_up[3]"),
            "indexed part-select mapping changed on an ascending declaration");
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

void test_unsupported_division_diagnoses() {
    Fixture fixture(
        "module m(input logic [3:0] a, b, output logic [3:0] q);\n"
        "  always_comb q = a / b;\n"
        "endmodule\n");
    require(!fixture.lowered.succeeded() && !fixture.lowered.diagnostics.empty(),
            "unsupported nonconstant division was accepted silently");
}

} // namespace

int main() {
    try {
        test_bitwise_logic_arithmetic_and_conditional_cells();
        test_static_select_concatenation_and_replication();
        test_identifier_select_range_direction_and_part_mapping();
        test_unsupported_dynamic_select_and_four_state_literal_diagnose();
        test_unsupported_division_diagnoses();
    } catch (const std::exception& exception) {
        std::cerr << "EDA Verilog expression lowering test failed: "
                  << exception.what() << '\n';
        return 1;
    }
    return 0;
}
