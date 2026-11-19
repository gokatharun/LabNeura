#include "labneura/eda/verilog_assignment.h"

#include "labneura/eda/verilog_syntax.h"

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

const VerilogStatement* first_assignment(const VerilogStatement* statement) {
    if (!statement)
        return nullptr;
    if (statement->kind == VerilogStatementKind::ProceduralAssignment)
        return statement;
    if (const auto* found = first_assignment(statement->first_statement.get()))
        return found;
    if (const auto* found = first_assignment(statement->second_statement.get()))
        return found;
    for (const auto& child : statement->statements)
        if (const auto* found = first_assignment(child.get()))
            return found;
    return nullptr;
}

struct Fixture {
    VerilogSyntaxParseResult parsed;
    VerilogSemanticAnalysisResult semantic;
    VerilogTypeAnalysisResult types;
    const VerilogExpression* lvalue = nullptr;

    explicit Fixture(std::string lvalue_text, std::string declarations =
                                                   "logic [7:0] bus;")
        : parsed(parse_verilog_syntax("module m; " + declarations +
                                      " initial begin " + lvalue_text +
                                      " = 0; end endmodule")),
          semantic(analyze_verilog_semantics(parsed.syntax)),
          types(analyze_verilog_expression_types(parsed.syntax, semantic)) {
        require(!parsed.has_errors(), "lvalue fixture failed to parse");
        require(!semantic.has_errors(), "lvalue fixture failed semantic analysis");
        const auto* assignment = first_assignment(
            parsed.syntax.modules.front().procedural_blocks.front().body.get());
        require(assignment && assignment->left, "lvalue assignment was not parsed");
        lvalue = assignment->left.get();
    }

    VerilogLvalueResolutionResult resolve(
        const VerilogConstantEnvironment& environment = {}) const {
        return resolve_verilog_lvalue(
            *lvalue, semantic, types,
            [&](const VerilogExpression& expression)
                -> std::optional<VerilogConstantValue> {
                return evaluate_verilog_constant(expression, environment).value;
            });
    }
};

void test_identifier_and_bit_select_mapping() {
    Fixture full("bus");
    auto full_result = full.resolve();
    require(full_result.target.has_value(), "identifier lvalue did not resolve");
    require(full_result.target->bit_width == 8 && full_result.target->is_integral &&
                !full_result.target->is_signed && full_result.target->four_state,
            "identifier type metadata was not preserved");
    const auto& full_positions = full_result.target->destinations.front().bit_positions;
    require(full_positions.size() == 8 && full_positions[0] == 0 &&
                full_positions[7] == 7,
            "full identifier bit positions are incorrect");

    Fixture bit("bus[3]");
    auto bit_result = bit.resolve();
    require(bit_result.target && bit_result.target->bit_width == 1 &&
                bit_result.target->is_integral && !bit_result.target->is_signed,
            "bit-select metadata is incorrect");
    require(bit_result.target->destinations.front().bit_positions[0] == 3,
            "descending bit-select mapped to the wrong physical bit");
}

void test_part_selects_and_ascending_ranges() {
    Fixture descending("bus[6:3]");
    auto desc = descending.resolve();
    require(desc.target && desc.target->bit_width == 4,
            "descending part-select did not resolve");
    const auto& desc_bits = desc.target->destinations.front().bit_positions;
    require(desc_bits[0] == 3 && desc_bits[1] == 4 && desc_bits[3] == 6,
            "descending part-select bit order is incorrect");

    Fixture ascending("up[2:5]", "logic [0:7] up;");
    auto asc = ascending.resolve();
    require(asc.target && asc.target->bit_width == 4,
            "ascending part-select did not resolve");
    const auto& asc_bits = asc.target->destinations.front().bit_positions;
    require(asc_bits[0] == 2 && asc_bits[1] == 3 && asc_bits[3] == 5,
            "ascending part-select bit order is incorrect");
}

void test_indexed_part_selects_and_runtime_index() {
    Fixture up("bus[2 +: 3]");
    const auto up_result = up.resolve();
    require(up_result.target && up_result.target->bit_width == 3,
            "indexed upward select did not resolve");
    const auto& up_bits = up_result.target->destinations.front().bit_positions;
    require(up_bits[0] == 2 && up_bits[1] == 3 && up_bits[2] == 4,
            "indexed upward select mapped the wrong bits");

    Fixture down("bus[6 -: 3]");
    const auto down_result = down.resolve();
    require(down_result.target && down_result.target->bit_width == 3,
            "indexed downward select did not resolve");
    const auto& down_bits = down_result.target->destinations.front().bit_positions;
    require(down_bits[0] == 4 && down_bits[1] == 5 && down_bits[2] == 6,
            "indexed downward select mapped the wrong bits");

    Fixture runtime("bus[index]", "logic [7:0] bus; integer index;");
    VerilogConstantEnvironment values;
    VerilogConstantValue index;
    index.bit_width = 32;
    index.is_signed = true;
    index.bits.assign(32, VerilogLogicBit::Zero);
    index.bits[0] = VerilogLogicBit::One;
    index.bits[2] = VerilogLogicBit::One;
    values["index"] = index;
    const auto runtime_result = runtime.resolve(values);
    require(runtime_result.target &&
                runtime_result.target->destinations.front().bit_positions[0] == 5,
            "runtime index expression did not use the supplied frame evaluator");
}

void test_concatenation_order_and_out_of_range() {
    Fixture concat("{bus[1:0], other[3:2]}",
                   "logic [7:0] bus; logic [7:0] other;");
    const auto result = concat.resolve();
    require(result.target && result.target->bit_width == 4 &&
                result.target->destinations.size() == 2,
            "concatenated lvalue did not resolve");
    require(result.target->destinations[0].storage_name == "other" &&
                result.target->destinations[0].target_lsb_offset == 0 &&
                result.target->destinations[0].bit_positions[0] == 2 &&
                result.target->destinations[1].storage_name == "bus" &&
                result.target->destinations[1].target_lsb_offset == 2 &&
                result.target->destinations[1].bit_positions[0] == 0,
            "concatenated lvalue ordering does not match Verilog bit ordering");

    Fixture out_of_range("bus[10]");
    const auto oor = out_of_range.resolve();
    require(oor.target && oor.diagnostics.empty() &&
                !oor.target->destinations.front().bit_positions.front(),
            "legal out-of-range bit select was not represented as ignored storage");
}

void test_unknown_index_and_invalid_target_diagnostics() {
    Fixture unknown("bus[index]", "logic [7:0] bus; logic [1:0] index;");
    VerilogConstantEnvironment values;
    VerilogConstantValue index;
    index.bit_width = 2;
    index.bits = {VerilogLogicBit::X, VerilogLogicBit::Zero};
    values["index"] = index;
    const auto unknown_result = unknown.resolve(values);
    require(!unknown_result.target && !unknown_result.diagnostics.empty() &&
                unknown_result.diagnostics.front().code == "UnknownLvalueIndex",
            "unknown lvalue index did not produce an explicit diagnostic");

    Fixture nonconstant_part("bus[upper:lower]",
                             "logic [7:0] bus; integer upper; integer lower;");
    const auto nonconstant_part_result = nonconstant_part.resolve();
    require(!nonconstant_part_result.target &&
                !nonconstant_part_result.diagnostics.empty() &&
                nonconstant_part_result.diagnostics.front().code ==
                    "UnsupportedLvalueSelectBounds",
            "runtime variables were accepted as constant part-select bounds");

    Fixture invalid("bus");
    VerilogExpression non_lvalue;
    non_lvalue.kind = VerilogExpressionKind::Literal;
    non_lvalue.location = invalid.lvalue->location;
    const auto invalid_result = resolve_verilog_lvalue(
        non_lvalue, invalid.semantic, invalid.types,
        [](const VerilogExpression&) -> std::optional<VerilogConstantValue> {
            return std::nullopt;
        });
    require(!invalid_result.target && !invalid_result.diagnostics.empty() &&
                invalid_result.diagnostics.front().code == "UnsupportedLvalueTarget",
            "non-assignable literal target did not produce an explicit diagnostic");

    Fixture invalid_width("bus[2 +: 0]");
    const auto width_result = invalid_width.resolve();
    require(!width_result.target && !width_result.diagnostics.empty() &&
                width_result.diagnostics.front().code == "InvalidLvalueSelectWidth",
            "zero-width indexed part-select did not produce an explicit diagnostic");

    Fixture overflow("bus[index]", "logic [7:0] bus; logic [63:0] index;");
    VerilogConstantValue too_wide;
    too_wide.bit_width = 64;
    too_wide.bits.assign(64, VerilogLogicBit::One);
    VerilogConstantEnvironment overflow_values{{"index", too_wide}};
    const auto overflow_result = overflow.resolve(overflow_values);
    require(overflow_result.target && overflow_result.diagnostics.empty() &&
                !overflow_result.target->destinations.front().bit_positions.front(),
            "large known unsigned index should be a legal out-of-range selection");

    Fixture large_part(
        "bus[64'hffff_ffff_ffff_ffff:64'hffff_ffff_ffff_ffff]");
    const auto large_part_result = large_part.resolve();
    require(large_part_result.target && large_part_result.target->bit_width == 1 &&
                !large_part_result.target->destinations.front().bit_positions.front(),
            "large unsigned constant part-select should preserve legal out-of-range bits");
}

} // namespace

int main() {
    try {
        test_identifier_and_bit_select_mapping();
        test_part_selects_and_ascending_ranges();
        test_indexed_part_selects_and_runtime_index();
        test_concatenation_order_and_out_of_range();
        test_unknown_index_and_invalid_target_diagnostics();
        std::cout << "native Verilog lvalue resolution tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "native Verilog lvalue resolution test failure: " << error.what()
                  << '\n';
        return 1;
    }
}
