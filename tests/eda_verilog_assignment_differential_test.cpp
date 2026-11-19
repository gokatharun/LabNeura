#include "labneura/eda/verilog_assignment.h"
#include "labneura/eda/verilog_function_evaluation.h"
#include "labneura/eda/verilog_syntax.h"
#include "labneura/eda/verilog_task_execution.h"

#include <slang/ast/Compilation.h>
#include <slang/ast/symbols/CompilationUnitSymbols.h>
#include <slang/ast/symbols/InstanceSymbols.h>
#include <slang/ast/symbols/ParameterSymbols.h>
#include <slang/numeric/ConstantValue.h>
#include <slang/syntax/SyntaxTree.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using namespace labneura::eda;

void require(bool condition, std::string_view message) {
    if (!condition)
        throw std::runtime_error(std::string(message));
}

VerilogConstantValue value(uint64_t number, size_t width, bool is_signed = false) {
    VerilogConstantValue result;
    result.bit_width = width;
    result.is_signed = is_signed;
    result.bits.resize(width, VerilogLogicBit::Zero);
    for (size_t bit = 0; bit < width && bit < 64; ++bit)
        if ((number >> bit) & 1U)
            result.bits[bit] = VerilogLogicBit::One;
    return result;
}

uint64_t known(const VerilogConstantEnvironment& environment, std::string_view name) {
    const auto found = environment.find(std::string(name));
    require(found != environment.end(), "assignment storage value is missing");
    const auto converted = found->second.to_uint64();
    require(converted.has_value(), "assignment storage value is unknown");
    return *converted;
}

struct Fixture {
    VerilogSyntaxParseResult parsed;
    VerilogSemanticAnalysisResult semantic;
    VerilogTypeAnalysisResult types;

    explicit Fixture(std::string source)
        : parsed(parse_verilog_syntax(std::move(source))),
          semantic(analyze_verilog_semantics(parsed.syntax)),
          types(analyze_verilog_expression_types(parsed.syntax, semantic)) {
        require(!parsed.has_errors(), "assignment fixture failed to parse");
        require(!semantic.has_errors(), "assignment fixture failed semantic analysis");
    }
};

const VerilogStatement& find_assignment(const VerilogCallableDeclaration& callable,
                                       VerilogExpressionKind lhs_kind,
                                       std::string_view base_name,
                                       std::optional<VerilogSelectKind> select_kind = {}) {
    const VerilogStatement* found = nullptr;
    const auto visit = [&](const auto& self, const VerilogStatement* statement) -> void {
        if (!statement)
            return;
        if (statement->kind == VerilogStatementKind::ProceduralAssignment &&
            statement->left && statement->left->kind == lhs_kind) {
            const VerilogExpression* base = statement->left.get();
            while (base->kind == VerilogExpressionKind::Select && base->first)
                base = base->first.get();
            if ((base->kind == VerilogExpressionKind::Identifier &&
                 base->text == base_name &&
                 (!select_kind || statement->left->select_kind == *select_kind)) ||
                lhs_kind == VerilogExpressionKind::Concatenation)
                found = statement;
        }
        self(self, statement->first_statement.get());
        self(self, statement->second_statement.get());
        for (const auto& child : statement->statements)
            self(self, child.get());
        for (const auto& item : statement->case_items)
            self(self, item.statement.get());
    };
    visit(visit, callable.body.get());
    if (!found)
        throw std::runtime_error("assignment fixture statement was not found");
    return *found;
}

const VerilogCallableDeclaration& callable(const Fixture& fixture,
                                           std::string_view name) {
    for (const auto& module : fixture.parsed.syntax.modules)
        for (const auto& item : module.callables)
            if (item.name == name)
                return item;
    throw std::runtime_error("assignment fixture callable was not found");
}

VerilogAssignmentExpressionEvaluator evaluator(
    const VerilogConstantEnvironment& environment) {
    return [&](const VerilogExpression& expression)
        -> std::optional<VerilogConstantValue> {
        auto result = evaluate_verilog_constant(expression, environment);
        return result.value;
    };
}

VerilogResolvedAssignmentTarget resolve(const Fixture& fixture,
                                       const VerilogExpression& expression,
                                       const VerilogConstantEnvironment& environment) {
    const auto result = resolve_verilog_lvalue(expression, fixture.semantic, fixture.types,
                                               evaluator(environment));
    require(result.target.has_value() && result.diagnostics.empty(),
            "supported lvalue failed to resolve");
    return *result.target;
}

void test_concatenation_and_select_assignment_expected_values() {
    Fixture fixture(
        "module assignments;\n"
        "  logic [3:0] upper, lower;\n"
        "  logic [7:0] bus;\n"
        "  task automatic apply(input logic [7:0] source, input int base);\n"
        "    {upper, lower} = source;\n"
        "    bus[7:4] = upper;\n"
        "    bus[base +: 3] = source[2:0];\n"
        "  endtask\n"
        "endmodule\n");
    VerilogConstantEnvironment environment{{"upper", value(0, 4)},
                                           {"lower", value(0, 4)},
                                           {"bus", value(0, 8)},
                                           {"source", value(0xa5, 8)},
                                           {"base", value(1, 32, true)}};

    const auto& concat = find_assignment(callable(fixture, "apply"),
                                         VerilogExpressionKind::Concatenation, "upper");
    auto concat_target = resolve(fixture, *concat.left, environment);
    const auto concat_value = evaluate_verilog_constant(*concat.right, environment);
    require(concat_value.value.has_value(), "concatenation RHS failed to evaluate");
    const auto concat_result = assign_verilog_lvalue(concat_target, *concat_value.value,
                                                     environment);
    require(concat_result.assigned && concat_result.diagnostics.empty() &&
                known(environment, "upper") == 0xa && known(environment, "lower") == 5,
            "concatenation assignment did not distribute source bits MSB-first");
    const auto packed_read = read_verilog_lvalue(concat_target, environment);
    require(packed_read.value && packed_read.diagnostics.empty() &&
                packed_read.value->to_uint64() == 0xa5 &&
                packed_read.value->bit_width == 8,
            "concatenation lvalue read order did not reconstruct the packed value");

    const auto& range = find_assignment(callable(fixture, "apply"),
                                        VerilogExpressionKind::Select, "bus",
                                        VerilogSelectKind::Part);
    auto range_target = resolve(fixture, *range.left, environment);
    VerilogConstantEnvironment range_values{{"bus", value(0, 8)}};
    const auto range_rhs = evaluate_verilog_constant(*range.right, environment);
    require(range_rhs.value.has_value(), "range assignment RHS failed to evaluate");
    const auto range_result = assign_verilog_lvalue(range_target, *range_rhs.value,
                                                    range_values);
    require(range_result.assigned && known(range_values, "bus") == 0xa0,
            "descending declared range [7:4] mapped to the wrong storage bits");

    const VerilogStatement* indexed_assignment = nullptr;
    const auto visit = [&](const auto& self, const VerilogStatement* statement) -> void {
        if (!statement)
            return;
        if (statement->kind == VerilogStatementKind::ProceduralAssignment &&
            statement->left && statement->left->kind == VerilogExpressionKind::Select &&
            statement->left->select_kind == VerilogSelectKind::IndexedUp)
            indexed_assignment = statement;
        self(self, statement->first_statement.get());
        for (const auto& child : statement->statements)
            self(self, child.get());
    };
    visit(visit, callable(fixture, "apply").body.get());
    require(indexed_assignment, "indexed part-select assignment was not found");
    auto indexed_target = resolve(fixture, *indexed_assignment->left, environment);
    VerilogConstantEnvironment indexed_values{{"bus", value(0, 8)}};
    const auto indexed_rhs = evaluate_verilog_constant(*indexed_assignment->right,
                                                       environment);
    require(indexed_rhs.value.has_value(), "indexed assignment RHS failed to evaluate");
    const auto indexed_result = assign_verilog_lvalue(indexed_target,
                                                      *indexed_rhs.value,
                                                      indexed_values);
    require(indexed_result.assigned && known(indexed_values, "bus") == 0x0a,
            "runtime indexed part-select did not honor its base and width");
}

void test_conversion_four_state_and_two_state_expected_values() {
    VerilogIntegralType four_state = infer_verilog_declared_type("logic", {});
    VerilogIntegralType two_state = infer_verilog_declared_type("bit", {});
    four_state.bit_width = 4;
    two_state.bit_width = 4;
    VerilogResolvedAssignmentTarget four_target;
    four_target.bit_width = 4;
    four_target.is_integral = true;
    four_target.four_state = true;
    four_target.destinations.push_back(
        {nullptr, "four", four_state, 0, {0, 1, 2, 3}});
    VerilogResolvedAssignmentTarget two_target;
    two_target.bit_width = 4;
    two_target.is_integral = true;
    two_target.four_state = false;
    two_target.destinations.push_back(
        {nullptr, "two", two_state, 0, {0, 1, 2, 3}});
    VerilogConstantEnvironment environment{{"four", value(0, 4)},
                                           {"two", value(0, 4)}};
    VerilogConstantValue source;
    source.bit_width = 8;
    source.bits = {VerilogLogicBit::One, VerilogLogicBit::Z, VerilogLogicBit::X,
                   VerilogLogicBit::One, VerilogLogicBit::Zero, VerilogLogicBit::One,
                   VerilogLogicBit::One, VerilogLogicBit::One};

    require(assign_verilog_lvalue(four_target, source, environment).assigned &&
                environment.at("four").bit_width == 4 &&
                environment.at("four").bits[0] == VerilogLogicBit::One &&
                environment.at("four").bits[1] == VerilogLogicBit::Z &&
                environment.at("four").bits[2] == VerilogLogicBit::X &&
                environment.at("four").bits[3] == VerilogLogicBit::One,
            "assignment conversion did not truncate while preserving four-state bits");
    require(assign_verilog_lvalue(two_target, source, environment).assigned &&
                environment.at("two").bit_width == 4 &&
                environment.at("two").bits[1] == VerilogLogicBit::Zero &&
                environment.at("two").bits[2] == VerilogLogicBit::Zero,
            "two-state assignment did not normalize X/Z to zero");

    VerilogIntegralType signed_type = infer_verilog_declared_type("logic", {});
    signed_type.bit_width = 8;
    signed_type.is_signed = true;
    VerilogResolvedAssignmentTarget signed_target;
    signed_target.bit_width = 8;
    signed_target.is_integral = true;
    signed_target.is_signed = true;
    signed_target.four_state = true;
    std::vector<std::optional<size_t>> signed_positions;
    for (size_t bit = 0; bit < 8; ++bit)
        signed_positions.push_back(bit);
    signed_target.destinations.push_back(
        {nullptr, "signed", signed_type, 0, std::move(signed_positions)});
    VerilogConstantEnvironment signed_environment{{"signed", value(0, 8, true)}};
    VerilogConstantValue signed_source;
    signed_source.bit_width = 4;
    signed_source.is_signed = true;
    signed_source.bits = {VerilogLogicBit::Zero, VerilogLogicBit::Zero,
                          VerilogLogicBit::Zero, VerilogLogicBit::One};
    const auto signed_assignment = assign_verilog_lvalue(
        signed_target, signed_source, signed_environment);
    require(signed_assignment.assigned &&
                signed_environment.at("signed").bits ==
                    std::vector<VerilogLogicBit>{VerilogLogicBit::Zero, VerilogLogicBit::Zero,
                                                VerilogLogicBit::Zero, VerilogLogicBit::One,
                                                VerilogLogicBit::One, VerilogLogicBit::One,
                                                VerilogLogicBit::One, VerilogLogicBit::One},
            "signed RHS assignment conversion did not sign-extend its most-significant bit");
}

void test_out_of_range_and_overlap_are_transactional() {
    Fixture fixture(
        "module assignment_bounds;\n"
        "  logic [3:0] bus;\n"
        "  task automatic update();\n"
        "    bus[6:3] = 4'b1011;\n"
        "    {bus[3:0], bus[2:1]} = 6'b111000;\n"
        "  endtask\n"
        "endmodule\n");
    const auto& assignment = callable(fixture, "update");
    VerilogConstantEnvironment environment{{"bus", value(0, 4)}};
    const auto& out_of_range = find_assignment(assignment, VerilogExpressionKind::Select,
                                               "bus");
    auto target = resolve(fixture, *out_of_range.left, environment);
    size_t ignored = 0;
    for (const auto& destination : target.destinations)
        ignored += static_cast<size_t>(std::count(destination.bit_positions.begin(),
                                                 destination.bit_positions.end(),
                                                 std::nullopt));
    require(ignored == 3, "legal out-of-range select bits were not marked as ignored");
    const auto read = read_verilog_lvalue(target, environment);
    require(read.value && read.diagnostics.empty() && read.value->bit_width == 4 &&
                read.value->bits[0] == VerilogLogicBit::Zero &&
                read.value->bits[1] == VerilogLogicBit::X &&
                read.value->bits[2] == VerilogLogicBit::X &&
                read.value->bits[3] == VerilogLogicBit::X,
            "out-of-range selected bits should read as X in lvalue order");
    const auto before = environment.at("bus").to_binary_string();
    const auto rhs = evaluate_verilog_constant(*out_of_range.right, environment);
    require(rhs.value.has_value(), "out-of-range RHS failed to evaluate");
    const auto ignored_result = assign_verilog_lvalue(target, *rhs.value, environment);
    require(ignored_result.assigned && ignored_result.diagnostics.empty() &&
                environment.at("bus").bits[0] == VerilogLogicBit::Zero &&
                environment.at("bus").bits[1] == VerilogLogicBit::Zero &&
                environment.at("bus").bits[2] == VerilogLogicBit::Zero &&
                environment.at("bus").bits[3] == VerilogLogicBit::One &&
                before != environment.at("bus").to_binary_string(),
            "out-of-range selected bits were not ignored while valid bits assigned");

    const auto& overlap = find_assignment(assignment,
                                         VerilogExpressionKind::Concatenation, "bus");
    auto overlap_target = resolve(fixture, *overlap.left, environment);
    const auto overlap_rhs = evaluate_verilog_constant(*overlap.right, environment);
    require(overlap_rhs.value.has_value(), "overlap RHS failed to evaluate");
    const auto rollback_value = environment.at("bus").to_binary_string();
    const auto overlap_result = assign_verilog_lvalue(overlap_target,
                                                      *overlap_rhs.value,
                                                      environment);
    require(!overlap_result.assigned && !overlap_result.diagnostics.empty() &&
                overlap_result.diagnostics.front().code ==
                    "OverlappingAssignmentDestinations" &&
                environment.at("bus").to_binary_string() == rollback_value,
            "overlapping destination bits must diagnose and roll back atomically");
}

void test_function_assignment_value_against_slang() {
    const std::string source =
        "module assignment_reference;\n"
        "  function automatic logic [7:0] pack_parts(input logic [3:0] high,\n"
        "                                               input logic [3:0] low);\n"
        "    logic [7:0] result;\n"
        "    result = 8'h00;\n"
        "    result[7:4] = high;\n"
        "    result[3:0] = low;\n"
        "    pack_parts = result;\n"
        "  endfunction\n"
        "  localparam logic [7:0] RESULT = pack_parts(4'ha, 4'h5);\n"
        "endmodule\n";
    Fixture native(source);
    const VerilogExpression* call = nullptr;
    for (const auto& parameter : native.parsed.syntax.modules.front().parameters)
        if (parameter.value && parameter.value->kind == VerilogExpressionKind::Call)
            call = parameter.value.get();
    require(call, "native function call for assignment differential was not found");
    const auto* function_symbol = lookup_verilog_symbol(
        *native.semantic.model.compilation_unit_scope, "assignment_reference", false);
    require(function_symbol && function_symbol->member_scope,
            "native differential module scope is missing");
    const auto* pack = lookup_verilog_symbol(*function_symbol->member_scope, "pack_parts", false);
    require(pack, "native differential function symbol is missing");
    const auto native_result = evaluate_verilog_function_call(
        native.parsed.syntax, native.semantic, native.types, *call);
    require(native_result.value && native_result.value->to_uint64() == 0xa5 &&
                native_result.value->bit_width == 8,
            "native part-select assignment returned a wrong independently expected value");

    slang::SourceManager source_manager;
    auto tree = slang::syntax::SyntaxTree::fromText(source, source_manager,
                                                    "assignment_reference.sv");
    slang::ast::Compilation compilation;
    compilation.addSyntaxTree(tree);
    const auto diagnostics = compilation.getAllDiagnostics();
    require(std::none_of(diagnostics.begin(), diagnostics.end(), [](const auto& item) {
                return item.isError();
            }),
            "Slang rejected supported constant-function assignment fixture");
    const auto& tops = compilation.getRoot().topInstances;
    require(!tops.empty(), "Slang did not elaborate assignment differential module");
    const slang::ast::ParameterSymbol* parameter = nullptr;
    for (const auto& member : tops.front()->body.members())
        if (member.kind == slang::ast::SymbolKind::Parameter && member.name == "RESULT")
            parameter = &member.as<slang::ast::ParameterSymbol>();
    require(parameter && parameter->getValue().isInteger() &&
                parameter->getValue().integer().getBitWidth() == 8 &&
                parameter->getValue().integer().as<uint64_t>() ==
                    native_result.value->to_uint64(),
            "native function assignment result differs from Slang constant evaluation");
}

void test_task_output_writeback_to_selected_actual() {
    Fixture fixture(
        "module selected_output;\n"
        "  logic [7:0] bus;\n"
        "  task automatic produce(output logic [3:0] value);\n"
        "    value = 4'ha;\n"
        "  endtask\n"
        "  initial produce(bus[7:4]);\n"
        "endmodule\n");
    const auto* task_symbol = lookup_verilog_symbol(
        *fixture.semantic.model.compilation_unit_scope, "selected_output", false);
    require(task_symbol && task_symbol->member_scope,
            "selected output module scope is missing");
    const auto* producer = lookup_verilog_symbol(*task_symbol->member_scope, "produce", false);
    require(producer, "selected output task symbol is missing");
    const auto& initial = fixture.parsed.syntax.modules.front().procedural_blocks.front();
    require(initial.body && initial.body->kind == VerilogStatementKind::TaskCall &&
                initial.body->call_expression,
            "selected output task call was not parsed");
    VerilogConstantEnvironment environment{{"bus", value(0x05, 8)}};
    const auto result = execute_verilog_task_call(
        fixture.parsed.syntax, fixture.semantic, fixture.types,
        *initial.body->call_expression, environment);
    require(result.completed && result.diagnostics.empty() &&
                known(environment, "bus") == 0xa5,
            "task output writeback to a selected actual did not preserve other bits");
}

} // namespace

int main() {
    try {
        test_concatenation_and_select_assignment_expected_values();
        test_conversion_four_state_and_two_state_expected_values();
        test_out_of_range_and_overlap_are_transactional();
        test_function_assignment_value_against_slang();
        test_task_output_writeback_to_selected_actual();
        std::cout << "EDA Verilog assignment differential tests passed\n";
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << "EDA Verilog assignment differential test failed: " << error.what()
                  << '\n';
        return 1;
    }
}
