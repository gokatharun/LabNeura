#include "labneura/eda/verilog_native_lowering.h"
#include "support/eda_netlist_evaluator.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <optional>
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

std::uint32_t evaluate_output(const Module& module,
                              const VerificationPortValues& inputs,
                              size_t width) {
    const auto result = evaluate_combinational_netlist(module, inputs);
    require(result.succeeded(), "mapped arithmetic netlist did not evaluate");
    const auto output = result.outputs.find("y");
    require(output != result.outputs.end() && output->second.size() == width,
            "mapped arithmetic output is missing or has the wrong width");
    std::uint32_t value = 0;
    for (size_t bit = 0; bit < width; ++bit)
        value |= static_cast<std::uint32_t>(output->second[bit]) << bit;
    return value;
}

int signed_value(std::uint32_t bits_value, size_t width) {
    const auto sign = std::uint32_t{1} << (width - 1);
    return (bits_value & sign) == 0
        ? static_cast<int>(bits_value)
        : static_cast<int>(bits_value) - static_cast<int>(std::uint32_t{1} << width);
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
        "  always_comb y = a / b;\n"
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

void test_arithmetic_sizing_and_signedness() {
    Fixture mixed_add(
        "module m(input logic signed [1:0] a, input logic [3:0] b, "
        "output logic [3:0] y);\n"
        " assign y = a + b;\nendmodule\n");
    require(mixed_add.lowered.succeeded(),
            "mixed signed/unsigned addition did not lower");
    for (std::uint32_t a = 0; a < 4; ++a)
        for (std::uint32_t b = 0; b < 16; ++b)
            require(evaluate_output(*mixed_add.lowered.module,
                                    {{"a", bits(a, 2)}, {"b", bits(b, 4)}}, 4) ==
                        ((a + b) & 0xfU),
                    "mixed signed/unsigned addition did not zero-extend its signed operand");

    Fixture signed_add_subtract(
        "module m(input logic signed [1:0] a, input logic signed [3:0] b, "
        "output logic signed [3:0] y);\n"
        " assign y = a + b;\nendmodule\n");
    require(signed_add_subtract.lowered.succeeded(),
            "signed widening addition did not lower");
    for (std::uint32_t a = 0; a < 4; ++a)
        for (std::uint32_t b = 0; b < 16; ++b) {
            const auto expected = static_cast<std::uint32_t>(
                signed_value(a, 2) + signed_value(b, 4)) & 0xfU;
            require(evaluate_output(*signed_add_subtract.lowered.module,
                                    {{"a", bits(a, 2)}, {"b", bits(b, 4)}}, 4) ==
                        expected,
                    "signed widening addition did not sign-extend its operand");
        }

    Fixture unsigned_subtract(
        "module m(input logic [3:0] a, b, output logic [3:0] y);\n"
        " assign y = a - b;\nendmodule\n");
    require(unsigned_subtract.lowered.succeeded(), "unsigned subtraction did not lower");
    for (std::uint32_t a = 0; a < 16; ++a)
        for (std::uint32_t b = 0; b < 16; ++b)
            require(evaluate_output(*unsigned_subtract.lowered.module,
                                    {{"a", bits(a, 4)}, {"b", bits(b, 4)}}, 4) ==
                        ((a - b) & 0xfU),
                    "unsigned subtraction borrow behavior differs from modulo arithmetic");

    Fixture carried_add(
        "module m(input logic [3:0] a, b, output logic [4:0] y);\n"
        " assign y = a + b;\nendmodule\n");
    require(carried_add.lowered.succeeded(), "context-widened addition did not lower");
    for (std::uint32_t a = 0; a < 16; ++a)
        for (std::uint32_t b = 0; b < 16; ++b)
            require(evaluate_output(*carried_add.lowered.module,
                                    {{"a", bits(a, 4)}, {"b", bits(b, 4)}}, 5) == a + b,
                    "context-widened addition lost carry-out");

    Fixture parameterized_add(
        "module m #(parameter W = 3) (input logic [W-1:0] a, b, "
        "output logic [W:0] y);\n"
        " assign y = a + b;\nendmodule\n");
    require(parameterized_add.lowered.succeeded(),
            "parameterized-width addition did not lower");
    for (std::uint32_t a = 0; a < 8; ++a)
        for (std::uint32_t b = 0; b < 8; ++b)
            require(evaluate_output(*parameterized_add.lowered.module,
                                    {{"a", bits(a, 3)}, {"b", bits(b, 3)}}, 4) == a + b,
                    "parameterized-width addition differs from independent truth table");

    Fixture signed_less(
        "module m(input logic signed [2:0] a, b, output logic y);\n"
        " assign y = a < b;\nendmodule\n");
    Fixture unsigned_less(
        "module m(input logic [2:0] a, b, output logic y);\n"
        " assign y = a < b;\nendmodule\n");
    require(signed_less.lowered.succeeded() && unsigned_less.lowered.succeeded(),
            "signed or unsigned comparison did not lower");
    for (std::uint32_t a = 0; a < 8; ++a)
        for (std::uint32_t b = 0; b < 8; ++b) {
            require(evaluate_output(*signed_less.lowered.module,
                                    {{"a", bits(a, 3)}, {"b", bits(b, 3)}}, 1) ==
                        static_cast<std::uint32_t>(signed_value(a, 3) < signed_value(b, 3)),
                    "signed comparison differs from independent reference");
            require(evaluate_output(*unsigned_less.lowered.module,
                                    {{"a", bits(a, 3)}, {"b", bits(b, 3)}}, 1) ==
                        static_cast<std::uint32_t>(a < b),
                    "unsigned comparison differs from independent reference");
        }

    Fixture wide_boundary_add(
        "module m(input logic [15:0] a, b, output logic [16:0] y);\n"
        " assign y = a + b;\nendmodule\n");
    require(wide_boundary_add.lowered.succeeded(),
            "16-bit context-widened addition did not lower");
    for (const auto a : {std::uint32_t{0}, std::uint32_t{1}, std::uint32_t{0x7fff},
                         std::uint32_t{0x8000}, std::uint32_t{0xfffe},
                         std::uint32_t{0xffff}})
        for (const auto b : {std::uint32_t{0}, std::uint32_t{1},
                             std::uint32_t{0x7fff}, std::uint32_t{0x8000},
                             std::uint32_t{0xfffe}, std::uint32_t{0xffff}})
            require(evaluate_output(*wide_boundary_add.lowered.module,
                                    {{"a", bits(a, 16)}, {"b", bits(b, 16)}}, 17) == a + b,
                    "16-bit context-widened addition lost a boundary carry");

    Fixture narrow_multiply(
        "module m(input logic [1:0] a, b, output logic [1:0] y);\n"
        " assign y = a * b;\nendmodule\n");
    Fixture wide_multiply(
        "module m(input logic [3:0] a, b, output logic [7:0] y);\n"
        " assign y = a * b;\nendmodule\n");
    Fixture signed_multiply(
        "module m(input logic signed [2:0] a, b, output logic signed [5:0] y);\n"
        " assign y = a * b;\nendmodule\n");
    Fixture parameterized_multiply(
        "module m #(parameter W = 3) (input logic [W-1:0] a, b, "
        "output logic [2*W-1:0] y);\n"
        " assign y = a * b;\nendmodule\n");
    require(narrow_multiply.lowered.succeeded() && wide_multiply.lowered.succeeded() &&
                signed_multiply.lowered.succeeded() &&
                parameterized_multiply.lowered.succeeded(),
            "supported multiplication did not lower");
    for (std::uint32_t a = 0; a < 4; ++a)
        for (std::uint32_t b = 0; b < 4; ++b)
            require(evaluate_output(*narrow_multiply.lowered.module,
                                    {{"a", bits(a, 2)}, {"b", bits(b, 2)}}, 2) ==
                        ((a * b) & 0x3U),
                    "narrow multiplication did not truncate at its expression width");
    for (std::uint32_t a = 0; a < 16; ++a)
        for (std::uint32_t b = 0; b < 16; ++b)
            require(evaluate_output(*wide_multiply.lowered.module,
                                    {{"a", bits(a, 4)}, {"b", bits(b, 4)}}, 8) == a * b,
                    "context-widened multiplication differs from independent truth table");
    for (std::uint32_t a = 0; a < 8; ++a)
        for (std::uint32_t b = 0; b < 8; ++b) {
            const auto expected = static_cast<std::uint32_t>(
                signed_value(a, 3) * signed_value(b, 3)) & 0x3fU;
            require(evaluate_output(*signed_multiply.lowered.module,
                                    {{"a", bits(a, 3)}, {"b", bits(b, 3)}}, 6) == expected,
                    "signed multiplication differs from independent truth table");
            require(evaluate_output(*parameterized_multiply.lowered.module,
                                    {{"a", bits(a, 3)}, {"b", bits(b, 3)}}, 6) == a * b,
                    "parameterized multiplication differs from independent truth table");
        }

    Fixture oversized_multiply(
        "module m(input logic [64:0] a, b, output logic [64:0] y);\n"
        " assign y = a * b;\nendmodule\n");
    require(!oversized_multiply.lowered.succeeded() &&
                has_diagnostic(oversized_multiply.lowered,
                               "UnsupportedNativeMultiplyWidth"),
            "oversized multiplication was not explicitly rejected");

    Fixture constant_divide(
        "module m(input logic [3:0] a, output logic [3:0] y);\n"
        " assign y = a / 4'd3;\nendmodule\n");
    Fixture constant_remainder(
        "module m(input logic [3:0] a, output logic [3:0] y);\n"
        " assign y = a % 4'd3;\nendmodule\n");
    Fixture dynamic_left_shift(
        "module m(input logic [3:0] a, input logic [2:0] amount, output logic [3:0] y);\n"
        " assign y = a << amount;\nendmodule\n");
    Fixture dynamic_arithmetic_right_shift(
        "module m(input logic signed [3:0] a, input logic [2:0] amount, "
        "output logic signed [3:0] y);\n"
        " assign y = a >>> amount;\nendmodule\n");
    require(constant_divide.lowered.succeeded() && constant_remainder.lowered.succeeded() &&
                dynamic_left_shift.lowered.succeeded() &&
                dynamic_arithmetic_right_shift.lowered.succeeded(),
            "supported division, remainder, or dynamic shift did not lower");
    for (std::uint32_t a = 0; a < 16; ++a) {
        require(evaluate_output(*constant_divide.lowered.module, {{"a", bits(a, 4)}}, 4) == a / 3,
                "constant unsigned division differs from independent truth table");
        require(evaluate_output(*constant_remainder.lowered.module, {{"a", bits(a, 4)}}, 4) == a % 3,
                "constant unsigned remainder differs from independent truth table");
        for (std::uint32_t amount = 0; amount < 8; ++amount) {
            require(evaluate_output(*dynamic_left_shift.lowered.module,
                                    {{"a", bits(a, 4)}, {"amount", bits(amount, 3)}}, 4) ==
                        (amount >= 4 ? 0U : ((a << amount) & 0xfU)),
                    "dynamic logical left shift differs from independent truth table");
            const auto signed_a = signed_value(a, 4);
            const auto expected = amount >= 4
                ? (signed_a < 0 ? 0xfU : 0U)
                : (static_cast<std::uint32_t>(signed_a >> amount) & 0xfU);
            require(evaluate_output(*dynamic_arithmetic_right_shift.lowered.module,
                                    {{"a", bits(a, 4)}, {"amount", bits(amount, 3)}}, 4) == expected,
                    "dynamic arithmetic right shift differs from independent truth table");
        }
    }

    Fixture variable_divide(
        "module m(input logic [3:0] a, b, output logic [3:0] y);\n"
        " assign y = a / b;\nendmodule\n");
    require(!variable_divide.lowered.succeeded() &&
                has_diagnostic(variable_divide.lowered, "UnsupportedNativeDivision"),
            "variable division was not explicitly rejected");
}

void test_native_memory_representation_contract() {
    Memory memory("mem", 8, 16, 4, "clk", "we",
                  {"addr[0]", "addr[1]", "addr[2]", "addr[3]"},
                  {"din[0]", "din[1]", "din[2]", "din[3]", "din[4]", "din[5]", "din[6]", "din[7]"},
                  {"addr[0]", "addr[1]", "addr[2]", "addr[3]"},
                  {"dout[0]", "dout[1]", "dout[2]", "dout[3]", "dout[4]", "dout[5]", "dout[6]", "dout[7]"});
    Module module("m", {}, {}, {}, {}, false, {}, {}, {memory});
    require(module.get_memories().size() == 1, "native memory was not retained by Module");
    const auto& retained = module.get_memories().front();
    require(retained.get_name() == "mem" && retained.get_word_width() == 8 &&
                retained.get_depth() == 16 && retained.get_address_width() == 4 &&
                retained.get_clock() == "clk" && retained.get_write_enable() == "we" &&
                retained.get_write_address().size() == 4 &&
                retained.get_write_data().size() == 8 && retained.get_read_data().size() == 8 &&
                retained.get_read_during_write() == MemoryReadDuringWrite::ReadBeforeWrite,
            "native memory metadata or read-before-write contract changed");
}

void test_rtl_memory_inference_and_cycle_contract() {
    Fixture fixture(
        "module m #(parameter WIDTH = 4, parameter DEPTH = 4) (\n"
        " input logic clk, we, input logic [1:0] addr,\n"
        " input logic [WIDTH-1:0] din, output logic [WIDTH-1:0] dout);\n"
        " logic [WIDTH-1:0] mem [0:DEPTH-1];\n"
        " always @(posedge clk) begin\n"
        "   if (we) mem[addr] <= din;\n"
        "   dout <= mem[addr];\n"
        " end\nendmodule\n");
    require(fixture.lowered.succeeded(),
            "canonical synchronous RTL memory did not infer:" +
                diagnostic_text(fixture.lowered));
    require(fixture.lowered.module->get_memories().size() == 1,
            "RTL memory inference did not retain exactly one Memory node");
    const auto& memory = fixture.lowered.module->get_memories().front();
    require(memory.get_name() == "mem" && memory.get_word_width() == 4 &&
                memory.get_depth() == 4 && memory.get_address_width() == 2 &&
                memory.get_clock() == "clk" && memory.get_write_enable() == "we" &&
                memory.get_write_address().size() == 2 &&
                memory.get_read_address().size() == 2 &&
                memory.get_write_data().size() == 4 &&
                memory.get_read_data().size() == 4 &&
                memory.get_read_during_write() == MemoryReadDuringWrite::ReadBeforeWrite,
            "inferred Memory node metadata is incorrect");
    require(!has_cell(*fixture.lowered.module, "DFF"),
            "inferred memory was incorrectly expanded into flip-flops");
    const auto remapped = generic_map(*fixture.lowered.module);
    require(remapped.get_memories().size() == 1,
            "generic mapping silently dropped a retained Memory node");
    bool exporter_rejected = false;
    try {
        (void)to_verilog(remapped);
    } catch (const std::runtime_error&) {
        exporter_rejected = true;
    }
    require(exporter_rejected,
            "Verilog exporter silently dropped a retained Memory node");

    // Independent cycle reference. Uninitialized words remain unknown. The
    // registered read observes pre-edge storage and the write commits after it.
    std::vector<std::optional<std::uint32_t>> storage(memory.get_depth());
    auto cycle = [&](bool write_enable, std::uint32_t address,
                     std::uint32_t data) -> std::optional<std::uint32_t> {
        require(address < storage.size(), "memory test address exceeds inferred depth");
        const auto observed = storage[address];
        if (write_enable)
            storage[address] = data & 0xfU;
        return observed;
    };
    require(!cycle(false, 0, 0), "uninitialized memory was assumed to be zero");
    require(!cycle(true, 0, 0xa), "first same-address write did not read old unknown data");
    require(cycle(false, 0, 0).value_or(0) == 0xa,
            "memory data did not persist after write");
    require(cycle(true, 0, 0x5).value_or(0) == 0xa,
            "same-address collision was not read-before-write");
    require(cycle(false, 0, 0).value_or(0) == 0x5,
            "replacement memory data was not visible on the following cycle");
    require(!cycle(true, 3, 0xf), "upper-bound first write lost unknown state");
    require(cycle(false, 3, 0).value_or(0) == 0xf,
            "upper-bound memory address did not retain data");

    Fixture partial_write(
        "module m(input logic clk, we, input logic addr, din, output logic dout);\n"
        " logic mem [0:1];\n"
        " always @(posedge clk) begin\n"
        "  if (we) mem[addr][0] <= din;\n"
        "  dout <= mem[addr];\n"
        " end\nendmodule\n");
    require(!partial_write.lowered.succeeded() &&
                has_diagnostic(partial_write.lowered, "UnsupportedNativeMemoryPorts"),
            "partial memory write was not explicitly rejected");

    Fixture eight_by_eight(
        "module m #(parameter WIDTH=8, parameter DEPTH=8) "
        "(input logic clk,we,input logic [2:0] addr,input logic [WIDTH-1:0] din,"
        "output logic [WIDTH-1:0] dout); logic [WIDTH-1:0] mem [0:DEPTH-1];"
        "always @(posedge clk) begin if(we) mem[addr] <= din; dout <= mem[addr]; end "
        "endmodule\n");
    Fixture sixteen_by_eight(
        "module m #(parameter WIDTH=8, parameter DEPTH=16) "
        "(input logic clk,we,input logic [3:0] addr,input logic [WIDTH-1:0] din,"
        "output logic [WIDTH-1:0] dout); logic [WIDTH-1:0] mem [0:DEPTH-1];"
        "always @(posedge clk) begin if(we) mem[addr] <= din; dout <= mem[addr]; end "
        "endmodule\n");
    require(eight_by_eight.lowered.succeeded() && sixteen_by_eight.lowered.succeeded() &&
                eight_by_eight.lowered.module->get_memories().front().get_depth() == 8 &&
                sixteen_by_eight.lowered.module->get_memories().front().get_depth() == 16,
            "parameterized 8x8 or 16x8 memory did not infer");

    Fixture wide_address(
        "module m(input logic clk,we,input logic [2:0] addr,input logic [3:0] din,"
        "output logic [3:0] dout); logic [3:0] mem [0:3];"
        "always @(posedge clk) begin if(we) mem[addr] <= din; dout <= mem[addr]; end "
        "endmodule\n");
    require(!wide_address.lowered.succeeded() &&
                has_diagnostic(wide_address.lowered,
                               "UnsupportedNativeMemoryPortWidth"),
            "potentially out-of-range memory address was accepted");

    Fixture multiple_clocks(
        "module m(input logic clk0,clk1,we,input logic addr,din,output logic dout);"
        "logic mem [0:1]; always @(posedge clk0 or posedge clk1) begin "
        "if(we) mem[addr] <= din; dout <= mem[addr]; end endmodule\n");
    require(!multiple_clocks.lowered.succeeded() &&
                has_diagnostic(multiple_clocks.lowered,
                               "UnsupportedNativeMemoryClock"),
            "multiple-clock memory was accepted");
}
} // namespace

int main() {
    try {
        test_wire_logic_and_constant_output_structure();
        test_deterministic_lowering_and_failed_result_atomicity();
        test_unresolved_driver_and_hierarchy_are_not_silently_lowered();
        test_bounded_dynamic_select_and_range_rejection();
        test_arithmetic_sizing_and_signedness();
        test_native_memory_representation_contract();
        test_rtl_memory_inference_and_cycle_contract();
    } catch (const std::exception& error) {
        std::cerr << "EDA Verilog native lowering test failed: " << error.what()
                  << '\n';
        return 1;
    }
    return 0;
}
