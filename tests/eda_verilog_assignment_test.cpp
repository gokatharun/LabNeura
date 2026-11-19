#include "labneura/eda/verilog_assignment.h"

#include "labneura/eda/verilog_syntax.h"

#include <algorithm>
#include <cstdint>
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

VerilogConstantValue bits(std::string_view msb_first, bool is_signed = false) {
    VerilogConstantValue result;
    result.bit_width = msb_first.size();
    result.is_signed = is_signed;
    for (auto it = msb_first.rbegin(); it != msb_first.rend(); ++it) {
        switch (*it) {
            case '0': result.bits.push_back(VerilogLogicBit::Zero); break;
            case '1': result.bits.push_back(VerilogLogicBit::One); break;
            case 'x': case 'X': result.bits.push_back(VerilogLogicBit::X); break;
            case 'z': case 'Z': result.bits.push_back(VerilogLogicBit::Z); break;
            default: throw std::runtime_error("invalid test bit spelling");
        }
    }
    return result;
}

VerilogIntegralType integral_type(size_t width, bool is_signed = false,
                                  bool four_state = true) {
    VerilogIntegralType result;
    result.category = VerilogTypeCategory::Integral;
    result.bit_width = width;
    result.is_signed = is_signed;
    result.four_state = four_state;
    return result;
}

VerilogResolvedAssignmentTarget single_target(std::string name, size_t width,
                                               bool is_signed = false,
                                               bool four_state = true) {
    VerilogResolvedAssignmentTarget target;
    target.bit_width = width;
    target.is_integral = true;
    target.is_signed = is_signed;
    target.four_state = four_state;
    VerilogResolvedAssignmentDestination destination;
    destination.storage_name = std::move(name);
    destination.declared_type = integral_type(width, is_signed, four_state);
    for (size_t bit = 0; bit < width; ++bit)
        destination.bit_positions.push_back(bit);
    target.destinations.push_back(std::move(destination));
    return target;
}

uint64_t known(const VerilogConstantValue& value) {
    auto result = value.to_uint64();
    require(result.has_value(), "expected known test value");
    return *result;
}

const VerilogStatement* find_assignment(const VerilogStatement* statement,
                                        std::string_view root_name) {
    if (!statement)
        return nullptr;
    if (statement->kind == VerilogStatementKind::ProceduralAssignment &&
        statement->left) {
        const VerilogExpression* root = statement->left.get();
        while ((root->kind == VerilogExpressionKind::Select ||
                root->kind == VerilogExpressionKind::MemberAccess) && root->first)
            root = root->first.get();
        if ((root->kind == VerilogExpressionKind::Identifier &&
             root->text == root_name) ||
            root->kind == VerilogExpressionKind::Concatenation)
            return statement;
    }
    if (const auto* found = find_assignment(statement->first_statement.get(), root_name))
        return found;
    if (const auto* found = find_assignment(statement->second_statement.get(), root_name))
        return found;
    for (const auto& nested : statement->statements)
        if (const auto* found = find_assignment(nested.get(), root_name))
            return found;
    for (const auto& item : statement->case_items)
        if (const auto* found = find_assignment(item.statement.get(), root_name))
            return found;
    return nullptr;
}

struct Fixture {
    VerilogSyntaxParseResult parsed;
    VerilogSemanticAnalysisResult semantic;
    VerilogTypeAnalysisResult types;

    explicit Fixture(std::string source)
        : parsed(parse_verilog_syntax(std::move(source))),
          semantic(analyze_verilog_semantics(parsed.syntax)),
          types(analyze_verilog_expression_types(parsed.syntax, semantic)) {
        require(!parsed.has_errors(), "assignment fixture did not parse");
        require(!semantic.has_errors(), "assignment fixture has semantic errors");
    }

    const VerilogExpression& target(std::string_view root) const {
        const auto& module = parsed.syntax.modules.front();
        for (const auto& block : module.procedural_blocks)
            if (const auto* assignment = find_assignment(block.body.get(), root))
                return *assignment->left;
        for (const auto& callable : module.callables)
            if (const auto* assignment = find_assignment(callable.body.get(), root))
                return *assignment->left;
        throw std::runtime_error("assignment target '" + std::string(root) +
                                 "' not found in fixture");
    }
};

VerilogLvalueResolutionResult resolve(const Fixture& fixture,
                                       const VerilogExpression& expression,
                                       const VerilogConstantEnvironment& environment = {}) {
    return resolve_verilog_lvalue(
        expression, fixture.semantic, fixture.types,
        [&](const VerilogExpression& index)
            -> std::optional<VerilogConstantValue> {
            auto result = evaluate_verilog_constant(index, environment);
            return result.value;
        });
}

void test_identifier_resolution_and_width_boundary_conversion() {
    Fixture fixture(
        "module assignment_widths;\n"
        "  logic signed [7:0] signed_target;\n"
        "  logic [7:0] unsigned_target;\n"
        "  initial begin\n"
        "    signed_target = 0;\n"
        "    unsigned_target = 0;\n"
        "  end\n"
        "endmodule\n");
    auto signed_target = resolve(fixture, fixture.target("signed_target"));
    require(signed_target.target && signed_target.target->bit_width == 8 &&
                signed_target.target->is_signed &&
                signed_target.target->destinations.front().bit_positions.size() == 8,
            "full identifier target lost declared width or signedness");

    // Exhaustively exercise every source/target width pair across byte and
    // machine-word boundaries. Assignment extension follows RHS signedness.
    std::vector<size_t> widths;
    for (size_t width = 1; width <= 65; ++width)
        widths.push_back(width);
    widths.push_back(128);
    for (const size_t target_width : widths) {
        for (const size_t source_width : widths) {
            for (const bool source_signed : {false, true}) {
                VerilogResolvedAssignmentTarget target =
                    single_target("storage", target_width, true, true);
                VerilogConstantEnvironment environment{
                    {"storage", bits(std::string(target_width, '0'))}};
                std::string pattern(source_width, '0');
                pattern.front() = '1'; // Keep the sign bit observable.
                const auto source = bits(pattern, source_signed);
                const auto result = assign_verilog_lvalue(target, source, environment);
                require(result.assigned && result.diagnostics.empty(),
                        "width-boundary assignment conversion failed");
                const auto& actual = environment.at("storage");
                require(actual.bit_width == target_width && actual.is_signed,
                        "assignment did not retain destination width and signedness");
                for (size_t bit = 0; bit < target_width; ++bit) {
                    const auto expected = bit < source_width
                                              ? source.bits[bit]
                                              : (source_signed ? VerilogLogicBit::One
                                                               : VerilogLogicBit::Zero);
                    require(actual.bits[bit] == expected,
                            "assignment source extension/truncation produced wrong bit");
                }
            }
        }
    }
}

void test_packed_bit_part_and_indexed_select_mapping() {
    Fixture fixture(
        "module select_targets;\n"
        "  logic [7:0] bus;\n"
        "  initial begin\n"
        "    bus[6] = 1'b0;\n"
        "    bus[7:4] = 4'b0000;\n"
        "    bus[2 +: 3] = 3'b000;\n"
        "    bus[6 -: 3] = 3'b000;\n"
        "  end\n"
        "endmodule\n");
    const auto check_positions = [&](std::string_view root, size_t expected_width,
                                     const std::vector<size_t>& expected) {
        const auto resolution = resolve(fixture, fixture.target(root));
        require(resolution.target && resolution.target->bit_width == expected_width &&
                    !resolution.target->is_signed,
                "packed select target has incorrect width or signedness");
        const auto& actual = resolution.target->destinations.front().bit_positions;
        require(actual.size() == expected.size(), "packed select mapping size differs");
        for (size_t i = 0; i < expected.size(); ++i)
            require(actual[i] && *actual[i] == expected[i],
                    "packed select mapped the wrong storage bit");
    };
    check_positions("bus", 1, {6});
    // The helper finds the first matching base name, so use dedicated parsed
    // sources below for each distinct select form.
    Fixture part_fixture(
        "module part_select; logic [7:0] bus; initial bus[7:4] = 4'b0; endmodule");
    auto part = resolve(part_fixture, part_fixture.target("bus"));
    require(part.target && part.target->bit_width == 4 &&
                part.target->destinations.front().bit_positions[0] == 4 &&
                part.target->destinations.front().bit_positions[3] == 7,
            "descending part-select LSB order is incorrect");
    Fixture up_fixture(
        "module up_select; logic [7:0] bus; initial bus[2 +: 3] = 3'b0; endmodule");
    auto up = resolve(up_fixture, up_fixture.target("bus"));
    require(up.target && up.target->bit_width == 3 &&
                up.target->destinations.front().bit_positions[0] == 2 &&
                up.target->destinations.front().bit_positions[2] == 4,
            "indexed-up select bit order is incorrect");
    Fixture down_fixture(
        "module down_select; logic [7:0] bus; initial bus[6 -: 3] = 3'b0; endmodule");
    auto down = resolve(down_fixture, down_fixture.target("bus"));
    require(down.target && down.target->bit_width == 3 &&
                down.target->destinations.front().bit_positions[0] == 4 &&
                down.target->destinations.front().bit_positions[2] == 6,
            "indexed-down select bit order is incorrect");
}

void test_concat_mapping_assignment_and_readback() {
    Fixture fixture(
        "module concat_targets;\n"
        "  logic [3:0] high, low;\n"
        "  initial {high, low} = 8'h00;\n"
        "endmodule\n");
    const auto resolution = resolve(fixture, fixture.target("high"));
    require(resolution.target && resolution.target->bit_width == 8 &&
                resolution.target->destinations.size() == 2 &&
                resolution.target->destinations[0].storage_name == "low" &&
                resolution.target->destinations[0].target_lsb_offset == 0 &&
                resolution.target->destinations[1].storage_name == "high" &&
                resolution.target->destinations[1].target_lsb_offset == 4,
            "concatenation components were not ordered LSB-first");
    VerilogConstantEnvironment environment{{"high", bits("0000")},
                                           {"low", bits("0000")}};
    const auto assignment = assign_verilog_lvalue(*resolution.target,
                                                   bits("10100101"), environment);
    require(assignment.assigned && known(environment.at("high")) == 0xa &&
                known(environment.at("low")) == 0x5,
            "concatenation assignment scattered the wrong source bits");
    const auto readback = read_verilog_lvalue(*resolution.target, environment);
    require(readback.value && readback.value->to_binary_string() == "10100101",
            "concatenation lvalue read did not gather mapped bits in lvalue order");
}

void test_out_of_range_bits_read_x_and_ignore_writes() {
    VerilogResolvedAssignmentTarget target;
    target.bit_width = 2;
    target.is_integral = true;
    target.four_state = true;
    VerilogResolvedAssignmentDestination destination;
    destination.storage_name = "small";
    destination.declared_type = integral_type(2);
    destination.bit_positions = {0, std::nullopt};
    target.destinations.push_back(destination);
    VerilogConstantEnvironment environment{{"small", bits("10")}};
    auto readback = read_verilog_lvalue(target, environment);
    require(readback.value && readback.value->bits[0] == VerilogLogicBit::Zero &&
                readback.value->bits[1] == VerilogLogicBit::X,
            "legal out-of-range lvalue read bit should be X");
    const auto assigned = assign_verilog_lvalue(target, bits("11"), environment);
    require(assigned.assigned && environment.at("small").bits[0] == VerilogLogicBit::One &&
                environment.at("small").bits[1] == VerilogLogicBit::One,
            "null out-of-range destination bit should be ignored on assignment");
}

void test_two_state_conversion_and_transactional_duplicate_rejection() {
    auto target = single_target("two_state", 4, false, false);
    VerilogConstantEnvironment environment{{"two_state", bits("1010")}};
    const auto converted = assign_verilog_lvalue(target, bits("xz10"), environment);
    require(converted.assigned && environment.at("two_state").to_binary_string() == "0010",
            "two-state destination did not map X/Z assignment bits to zero");

    VerilogResolvedAssignmentTarget overlap;
    overlap.bit_width = 2;
    overlap.is_integral = true;
    overlap.four_state = true;
    auto duplicate = single_target("same", 1).destinations.front();
    duplicate.target_lsb_offset = 0;
    auto duplicate_high = duplicate;
    duplicate_high.target_lsb_offset = 1;
    overlap.destinations = {duplicate, duplicate_high};
    VerilogConstantEnvironment transactional{{"same", bits("0")}};
    const auto rejected = assign_verilog_lvalue(overlap, bits("11"), transactional);
    require(!rejected.assigned &&
                std::any_of(rejected.diagnostics.begin(), rejected.diagnostics.end(),
                            [](const auto& diagnostic) {
                                return diagnostic.code ==
                                       "OverlappingAssignmentDestinations";
                            }) &&
                transactional.at("same").to_binary_string() == "0",
            "overlapping destinations must fail without publishing staged writes");
}

void test_invalid_storage_or_width_failures_are_transactional() {
    VerilogConstantEnvironment environment{{"left", bits("0101")}};
    VerilogResolvedAssignmentTarget target;
    target.bit_width = 4;
    target.is_integral = true;
    auto first = single_target("left", 4).destinations.front();
    auto missing = single_target("missing", 4).destinations.front();
    missing.target_lsb_offset = 0;
    first.target_lsb_offset = 0;
    target.destinations = {first, missing};
    auto result = assign_verilog_lvalue(target, bits("1010"), environment);
    require(!result.assigned && environment.at("left").to_binary_string() == "0101",
            "missing later storage should not partially publish earlier writes");

    auto small = single_target("left", 4);
    result = assign_verilog_lvalue(small, bits("1010"), environment,
                                   VerilogAssignmentOptions{3});
    require(!result.assigned && environment.at("left").to_binary_string() == "0101",
            "configured width limit failure must preserve storage");
}

} // namespace

int main() {
    try {
        test_identifier_resolution_and_width_boundary_conversion();
        test_packed_bit_part_and_indexed_select_mapping();
        test_concat_mapping_assignment_and_readback();
        test_out_of_range_bits_read_x_and_ignore_writes();
        test_two_state_conversion_and_transactional_duplicate_rejection();
        test_invalid_storage_or_width_failures_are_transactional();
    }
    catch (const std::exception& error) {
        std::cerr << "EDA Verilog assignment test failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
