#include "labneura/eda/verilog_native_lowering.h"
#include "support/eda_netlist_evaluator.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {
using namespace labneura::eda;
using namespace labneura::eda::test;

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

std::string diagnostic_text(const VerilogNativeLoweringResult& result) {
    std::string text;
    for (const auto& diagnostic : result.diagnostics)
        text += " [" + diagnostic.code + "] " + diagnostic.message;
    return text;
}

VerificationBits bits(std::uint32_t value, size_t width) {
    VerificationBits result(width);
    for (size_t bit = 0; bit < width; ++bit)
        result[bit] = static_cast<std::uint8_t>((value >> bit) & 1U);
    return result;
}

bool evaluate_select(const Module& module, std::uint32_t data,
                     size_t data_width, std::uint32_t index,
                     size_t index_width) {
    const auto result = evaluate_combinational_netlist(
        module, {{"data", bits(data, data_width)},
                 {"index", bits(index, index_width)}});
    require(result.succeeded(), "mapped dynamic-select netlist did not evaluate");
    const auto output = result.outputs.find("y");
    require(output != result.outputs.end() && output->second.size() == 1,
            "mapped dynamic-select output is missing or not scalar");
    return output->second.front() != 0;
}

void exhaustively_check_dynamic_select(const std::string& source,
                                       size_t data_width, size_t index_width,
                                       bool ascending) {
    Fixture fixture(source);
    require(fixture.lowered.succeeded(), "bounded dynamic bit-select did not lower");
    const auto& module = *fixture.lowered.module;
    const std::uint32_t data_patterns = std::uint32_t{1} << data_width;
    const std::uint32_t indices = std::uint32_t{1} << index_width;
    for (std::uint32_t data = 0; data < data_patterns; ++data) {
        for (std::uint32_t index = 0; index < indices; ++index) {
            const size_t position = ascending ? data_width - 1 - index : index;
            const bool expected = ((data >> position) & 1U) != 0;
            require(evaluate_select(module, data, data_width, index, index_width) ==
                        expected,
                    "mapped dynamic bit-select differs from independent truth table");
        }
    }
}

void exhaustively_check_nonzero_dynamic_select(const std::string& source,
                                               bool ascending) {
    Fixture fixture(source);
    require(fixture.lowered.succeeded(),
            "range-proven nonzero dynamic bit-select did not lower:" +
                diagnostic_text(fixture.lowered));
    const auto& module = *fixture.lowered.module;
    for (std::uint32_t data = 0; data < 16; ++data) {
        for (std::uint32_t index = 0; index < 4; ++index) {
            // The selector values are 4..7.  The input test bits are stored
            // LSB first, so descending [7:4] maps 4 to bit 0 while ascending
            // [4:7] maps 4 to bit 3.
            const size_t position = ascending ? 3 - index : index;
            require(evaluate_select(module, data, 4, index, 2) ==
                        (((data >> position) & 1U) != 0),
                    "nonzero-range dynamic select differs from independent truth table");
        }
    }
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

void test_bounded_dynamic_select_and_range_rejection() {
    exhaustively_check_dynamic_select(
        "module m(input logic [1:0] data, input logic index, output logic y);\n"
        " assign y = data[index];\nendmodule\n", 2, 1, false);
    exhaustively_check_dynamic_select(
        "module m(input logic [3:0] data, input logic [1:0] index, output logic y);\n"
        " assign y = data[index];\nendmodule\n", 4, 2, false);
    exhaustively_check_dynamic_select(
        "module m(input logic [7:0] data, input logic [2:0] index, output logic y);\n"
        " assign y = data[index];\nendmodule\n", 8, 3, false);
    exhaustively_check_dynamic_select(
        "module m(input logic [0:3] data, input logic [1:0] index, output logic y);\n"
        " assign y = data[index];\nendmodule\n", 4, 2, true);

    Fixture sixteen(
        "module m(input logic [15:0] data, input logic [3:0] index, output logic y);\n"
        " assign y = data[index];\nendmodule\n");
    require(sixteen.lowered.succeeded(), "16-way dynamic select did not lower");
    for (std::uint32_t hot_bit = 0; hot_bit < 16; ++hot_bit) {
        const auto data = std::uint32_t{1} << hot_bit;
        for (std::uint32_t index = 0; index < 16; ++index)
            require(evaluate_select(*sixteen.lowered.module, data, 16, index, 4) ==
                        (index == hot_bit),
                    "16-way one-hot dynamic select differs from independent expectation");
    }
    for (const auto data : {std::uint32_t{0x0000}, std::uint32_t{0xffff},
                            std::uint32_t{0xaaaa}, std::uint32_t{0x5555},
                            std::uint32_t{0x8001}, std::uint32_t{0x1248},
                            std::uint32_t{0x9e37}}) {
        for (std::uint32_t index = 0; index < 16; ++index)
            require(evaluate_select(*sixteen.lowered.module, data, 16, index, 4) ==
                        (((data >> index) & 1U) != 0),
                    "16-way dynamic select differs from independent expectation");
    }

    Fixture computed(
        "module m(input logic [3:0] data, input logic [1:0] index, output logic y);\n"
        " assign y = data[index ^ 2'b01];\nendmodule\n");
    require(computed.lowered.succeeded(), "computed bounded index did not lower");
    for (std::uint32_t data = 0; data < 16; ++data)
        for (std::uint32_t index = 0; index < 4; ++index)
            require(evaluate_select(*computed.lowered.module, data, 4, index, 2) ==
                        (((data >> (index ^ 1U)) & 1U) != 0),
                    "computed dynamic index differs from independent expectation");

    exhaustively_check_nonzero_dynamic_select(
        "module m(input logic [7:4] data, input logic [1:0] index, output logic y);\n"
        " logic [2:0] selected_index;\n"
        " assign selected_index = {1'b1, index};\n"
        " assign y = data[selected_index];\n"
        "endmodule\n", false);
    exhaustively_check_nonzero_dynamic_select(
        "module m(input logic [4:7] data, input logic [1:0] index, output logic y);\n"
        " assign y = data[{1'b1, index}];\n"
        "endmodule\n", true);

    Fixture zero_based_offset(
        "module m(input logic [3:0] data, input logic [1:0] index, output logic y);\n"
        " assign y = data[index + 2'd0];\n"
        "endmodule\n");
    require(zero_based_offset.lowered.succeeded(),
            "overflow-safe zero-based constant offset did not lower");
    for (std::uint32_t data = 0; data < 16; ++data)
        for (std::uint32_t index = 0; index < 4; ++index)
            require(evaluate_select(*zero_based_offset.lowered.module, data, 4, index, 2) ==
                        (((data >> index) & 1U) != 0),
                    "zero-based constant-offset select differs from independent truth table");

    exhaustively_check_nonzero_dynamic_select(
        "module m(input logic [7:4] data, input logic [1:0] index, output logic y);\n"
        " assign y = data[index + 3'd4];\n"
        "endmodule\n", false);
    exhaustively_check_nonzero_dynamic_select(
        "module m(input logic [7:4] data, input logic [1:0] index, output logic y);\n"
        " assign y = data[{1'b1, index}];\n"
        "endmodule\n", false);
    exhaustively_check_dynamic_select(
        "module m(input logic [3:0] data, input logic [1:0] index, output logic y);\n"
        " assign y = data[{1'b1, index} - 3'd4];\n"
        "endmodule\n", 4, 2, false);

    Fixture arithmetic_overflow(
        "module m(input logic [3:0] data, input logic [1:0] index, output logic y);\n"
        " assign y = data[index + 2'd1];\n"
        "endmodule\n");
    require(!arithmetic_overflow.lowered.succeeded() &&
                has_diagnostic(arithmetic_overflow.lowered,
                               "DynamicNativeSelectRangeUnproven"),
            "wrapping dynamic index arithmetic was accepted");

    Fixture arithmetic_underflow(
        "module m(input logic [3:0] data, input logic [1:0] index, output logic y);\n"
        " assign y = data[index - 2'd1];\n"
        "endmodule\n");
    require(!arithmetic_underflow.lowered.succeeded() &&
                has_diagnostic(arithmetic_underflow.lowered,
                               "DynamicNativeSelectRangeUnproven"),
            "underflowing dynamic index arithmetic was accepted");

    Fixture unsafe_intersection(
        "module m(input logic [7:4] data, input logic [1:0] index, output logic y);\n"
        " assign y = data[index + 3'd5];\n"
        "endmodule\n");
    require(!unsafe_intersection.lowered.succeeded() &&
                has_diagnostic(unsafe_intersection.lowered,
                               "DynamicNativeSelectRangeUnproven"),
            "partially out-of-range dynamic index was accepted");

    Fixture sparse_concatenation(
        "module m(input logic [3:0] data, input logic index, output logic y);\n"
        " assign y = data[{index, index}];\n"
        "endmodule\n");
    require(!sparse_concatenation.lowered.succeeded() &&
                has_diagnostic(sparse_concatenation.lowered,
                               "DynamicNativeSelectRangeUnproven"),
            "sparse repeated-index concatenation was accepted");

    Fixture too_wide(
        "module m(input logic [3:0] data, input logic [2:0] index, output logic y);\n"
        " assign y = data[index];\n"
        "endmodule\n");
    require(!too_wide.lowered.succeeded() &&
                has_diagnostic(too_wide.lowered, "DynamicNativeSelectRangeUnproven"),
            "potentially out-of-range index was accepted");

    Fixture signed_index(
        "module m(input logic [3:0] data, input logic signed [1:0] index, output logic y);\n"
        " assign y = data[index];\nendmodule\n");
    require(!signed_index.lowered.succeeded() &&
                has_diagnostic(signed_index.lowered, "DynamicNativeSelectUnsupported"),
            "signed dynamic index was accepted");

    Fixture x_index(
        "module m(input logic [3:0] data, input logic [1:0] index, output logic y);\n"
        " assign y = data[index ^ 2'bx0];\nendmodule\n");
    require(!x_index.lowered.succeeded() &&
                has_diagnostic(x_index.lowered, "FourStateConstantUnsupported"),
            "explicit X index expression was accepted");

    Fixture z_index(
        "module m(input logic [3:0] data, input logic [1:0] index, output logic y);\n"
        " assign y = data[index ^ 2'bz0];\nendmodule\n");
    require(!z_index.lowered.succeeded() &&
                has_diagnostic(z_index.lowered, "FourStateConstantUnsupported"),
            "explicit Z index expression was accepted");

    Fixture dynamic_part(
        "module m(input logic [7:0] data, input logic [2:0] index, output logic [1:0] y);\n"
        " assign y = data[index +: 2];\nendmodule\n");
    require(!dynamic_part.lowered.succeeded() &&
                has_diagnostic(dynamic_part.lowered, "DynamicNativeSelectUnsupported"),
            "dynamic part select was accepted");

    Fixture dynamic_write(
        "module m(input logic a, input logic index, output logic [1:0] y);\n"
        " always_comb y[index] = a;\nendmodule\n");
    require(!dynamic_write.lowered.succeeded() && !dynamic_write.lowered.module,
            "dynamic procedural write was accepted");

    Fixture static_selects(
        "module m(input logic [3:0] data, output logic [1:0] y);\n"
        " assign y = data[3:2];\nendmodule\n");
    require(static_selects.lowered.succeeded(),
            "existing static part-select lowering regressed");
}
} // namespace

int main() {
    try {
        test_wire_logic_and_constant_output_structure();
        test_deterministic_lowering_and_failed_result_atomicity();
        test_unresolved_driver_and_hierarchy_are_not_silently_lowered();
        test_bounded_dynamic_select_and_range_rejection();
    } catch (const std::exception& error) {
        std::cerr << "EDA Verilog native lowering test failed: " << error.what()
                  << '\n';
        return 1;
    }
    return 0;
}
