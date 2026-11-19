#include "labneura/eda/verilog_callable_binding.h"

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

const VerilogExpression* find_call(const VerilogExpression* expression,
                                   std::string_view callee) {
    if (!expression)
        return nullptr;
    if (expression->kind == VerilogExpressionKind::Call && expression->first &&
        expression->first->text == callee)
        return expression;
    if (const auto* found = find_call(expression->first.get(), callee))
        return found;
    if (const auto* found = find_call(expression->second.get(), callee))
        return found;
    if (const auto* found = find_call(expression->third.get(), callee))
        return found;
    for (const auto& operand : expression->operands)
        if (const auto* found = find_call(operand.get(), callee))
            return found;
    return nullptr;
}

const VerilogStatement* find_task_call(const VerilogStatement* statement,
                                       std::string_view callee) {
    if (!statement)
        return nullptr;
    if (statement->kind == VerilogStatementKind::TaskCall &&
        statement->call_expression && statement->call_expression->first &&
        statement->call_expression->first->text == callee)
        return statement;
    if (const auto* found = find_task_call(statement->first_statement.get(), callee))
        return found;
    if (const auto* found = find_task_call(statement->second_statement.get(), callee))
        return found;
    for (const auto& child : statement->statements)
        if (const auto* found = find_task_call(child.get(), callee))
            return found;
    for (const auto& item : statement->case_items)
        if (const auto* found = find_task_call(item.statement.get(), callee))
            return found;
    return nullptr;
}

const VerilogSemanticSymbol* find_symbol(const VerilogSemanticAnalysisResult& semantic,
                                         std::string_view unit,
                                         std::string_view callable_name) {
    const auto* unit_symbol =
        lookup_verilog_symbol(*semantic.model.compilation_unit_scope, unit, false);
    return unit_symbol && unit_symbol->member_scope
               ? lookup_verilog_symbol(*unit_symbol->member_scope, callable_name, false)
               : nullptr;
}

void test_formal_order_types_and_declaration_lookup() {
    const auto parsed = parse_verilog_syntax(
        "module binding(input logic [7:0] src, output logic [7:0] wide,\n"
        "               output logic signed [3:0] signed_value);\n"
        "  logic [3:0] narrow;\n"
        "  logic [3:0] unsigned_value;\n"
        "  function logic [7:0] identity(input logic [7:0] value);\n"
        "    identity = value;\n"
        "  endfunction\n"
        "  task automatic transfer(input logic [7:0] in_value,\n"
        "      output logic [3:0] out_value,\n"
        "      inout logic signed [3:0] io_value,\n"
        "      ref logic [3:0] ref_value);\n"
        "    begin end\n"
        "  endtask\n"
        "  assign wide = identity(src);\n"
        "  always_comb transfer(src, wide, unsigned_value, narrow);\n"
        "endmodule\n");
    require(!parsed.has_errors(), "binding fixture should parse cleanly");

    const auto semantic = analyze_verilog_semantics(parsed.syntax);
    const auto types = analyze_verilog_expression_types(parsed.syntax, semantic);
    const auto* function_symbol = find_symbol(semantic, "binding", "identity");
    const auto* task_symbol = find_symbol(semantic, "binding", "transfer");
    require(function_symbol && task_symbol && function_symbol->callable &&
                task_symbol->callable,
            "semantic callable symbols are missing");
    require(find_verilog_callable_declaration(parsed.syntax, *function_symbol) ==
                &parsed.syntax.modules.front().callables.front(),
            "function syntax declaration lookup did not use its source location");
    require(find_verilog_callable_declaration(parsed.syntax, *task_symbol) ==
                &parsed.syntax.modules.front().callables.back(),
            "task syntax declaration lookup did not use its source location");

    const auto* function_call =
        find_call(parsed.syntax.modules.front().assignments.front().right.get(), "identity");
    require(function_call, "function call expression was not found");
    const auto function_binding =
        bind_verilog_callable_arguments(*function_symbol, *function_call, types);
    require(function_binding.is_valid && function_binding.arguments.size() == 1 &&
                function_binding.arguments[0].formal_index == 0 &&
                function_binding.arguments[0].formal &&
                function_binding.arguments[0].formal->name == "value" &&
                function_binding.arguments[0].actual == function_call->operands[0].get(),
            "function positional actual did not bind in formal order");

    const auto* task_statement =
        find_task_call(parsed.syntax.modules.front().procedural_blocks.front().body.get(),
                       "transfer");
    require(task_statement, "task call statement was not found");
    const auto& task_call = *task_statement->call_expression;
    const auto task_binding =
        bind_verilog_callable_arguments(*task_symbol, task_call, types);
    require(task_binding.arguments.size() == 4 &&
                task_binding.arguments[0].formal->name == "in_value" &&
                task_binding.arguments[0].actual == task_call.operands[0].get() &&
                task_binding.arguments[1].formal->name == "out_value" &&
                task_binding.arguments[1].actual == task_call.operands[1].get() &&
                task_binding.arguments[2].formal->name == "io_value" &&
                task_binding.arguments[2].actual == task_call.operands[2].get() &&
                task_binding.arguments[3].formal->name == "ref_value" &&
                task_binding.arguments[3].actual == task_call.operands[3].get(),
            "task input/output/inout/ref bindings did not follow declaration order");

    const auto width_warning = std::find_if(
        task_binding.diagnostics.begin(), task_binding.diagnostics.end(),
        [](const auto& diagnostic) {
            return diagnostic.code == "CallableArgumentWidthConversion";
        });
    const auto signedness_warning = std::find_if(
        task_binding.diagnostics.begin(), task_binding.diagnostics.end(),
        [](const auto& diagnostic) {
            return diagnostic.code == "CallableArgumentSignednessConversion";
        });
    require(width_warning != task_binding.diagnostics.end() &&
                width_warning->severity == VerilogTypeDiagnosticSeverity::Warning &&
                width_warning->location.source &&
                width_warning->related_location &&
                width_warning->related_location->location.line == 9,
            "known output width conversion did not retain call and formal locations");
    require(signedness_warning != task_binding.diagnostics.end() &&
                signedness_warning->severity == VerilogTypeDiagnosticSeverity::Warning &&
                signedness_warning->related_location &&
                signedness_warning->related_location->location.line == 10,
            "known inout signedness conversion did not retain formal location");
    require(std::none_of(task_binding.diagnostics.begin(), task_binding.diagnostics.end(),
                         [](const auto& diagnostic) {
                             return diagnostic.code == "CallableRefTypeMismatch";
                         }),
            "matching ref category/width/sign/state should be accepted");
}

void test_required_actual_and_lvalue_diagnostic_locations() {
    const auto parsed = parse_verilog_syntax(
        "module invalid_binding(output logic out_value);\n"
        "  task automatic write_out(output logic result);\n"
        "    begin end\n"
        "  endtask\n"
        "  always_comb write_out(1'b1);\n"
        "endmodule\n");
    require(!parsed.has_errors(), "invalid actual fixture should parse");
    const auto semantic = analyze_verilog_semantics(parsed.syntax);
    const auto types = analyze_verilog_expression_types(parsed.syntax, semantic);
    const auto* task_symbol = find_symbol(semantic, "invalid_binding", "write_out");
    const auto* statement =
        find_task_call(parsed.syntax.modules.front().procedural_blocks.front().body.get(),
                       "write_out");
    require(task_symbol && statement, "output callable or its invocation is missing");
    const auto result =
        bind_verilog_callable_arguments(*task_symbol, *statement->call_expression, types);
    const auto invalid = std::find_if(
        result.diagnostics.begin(), result.diagnostics.end(), [](const auto& diagnostic) {
            return diagnostic.code == "InvalidCallableArgumentLValue";
        });
    require(!result.is_valid && invalid != result.diagnostics.end() &&
                invalid->severity == VerilogTypeDiagnosticSeverity::Error &&
                invalid->location.source &&
                invalid->location.location.line == 5 &&
                invalid->related_location &&
                invalid->related_location->location.line == 2,
            "non-lvalue output actual diagnostic lost source or related formal location");

    VerilogExpression missing_call;
    missing_call.kind = VerilogExpressionKind::Call;
    missing_call.location = statement->call_expression->location;
    const auto missing =
        bind_verilog_callable_arguments(*task_symbol, missing_call, types);
    require(!missing.is_valid && missing.arguments.size() == 1 &&
                missing.arguments[0].actual == nullptr &&
                std::any_of(missing.diagnostics.begin(), missing.diagnostics.end(),
                            [](const auto& diagnostic) {
                                return diagnostic.code == "MissingCallableArgument" &&
                                       diagnostic.related_location.has_value();
                            }),
            "missing actual should preserve a nullable binding and formal location");
}

void test_assignable_select_member_and_concatenation_actuals() {
    const auto parsed = parse_verilog_syntax(
        "module complex_lvalues(input logic a, input logic b);\n"
        "  typedef struct packed { logic bit_field; } packed_t;\n"
        "  packed_t record;\n"
        "  logic [1:0] bus;\n"
        "  task automatic capture(output logic selected,\n"
        "      output logic member_value, output logic [1:0] pair_value);\n"
        "    begin end\n"
        "  endtask\n"
        "  always_comb capture(bus[0], record.bit_field, {a, b});\n"
        "endmodule\n");
    require(!parsed.has_errors(), "complex lvalue fixture should parse cleanly");
    const auto semantic = analyze_verilog_semantics(parsed.syntax);
    const auto types = analyze_verilog_expression_types(parsed.syntax, semantic);
    const auto* task_symbol = find_symbol(semantic, "complex_lvalues", "capture");
    const auto* statement =
        find_task_call(parsed.syntax.modules.front().procedural_blocks.front().body.get(),
                       "capture");
    require(task_symbol && statement, "lvalue task symbol or invocation is missing");
    const auto result =
        bind_verilog_callable_arguments(*task_symbol, *statement->call_expression, types);
    require(result.is_valid && result.arguments.size() == 3 &&
                result.arguments[0].actual->kind == VerilogExpressionKind::Select &&
                result.arguments[1].actual->kind == VerilogExpressionKind::MemberAccess &&
                result.arguments[2].actual->kind == VerilogExpressionKind::Concatenation &&
                std::none_of(result.diagnostics.begin(), result.diagnostics.end(),
                             [](const auto& diagnostic) {
                                 return diagnostic.severity ==
                                        VerilogTypeDiagnosticSeverity::Error;
                             }),
            "assignable select/member/concatenation actuals should pass binding");
}

void test_ref_strict_type_mismatch_and_function_direction() {
    const auto parsed = parse_verilog_syntax(
        "module ref_binding;\n"
        "  logic [3:0] value;\n"
        "  task automatic refer(ref logic signed [3:0] item);\n"
        "    begin end\n"
        "  endtask\n"
        "  always_comb refer(value);\n"
        "endmodule\n");
    require(!parsed.has_errors(), "ref mismatch fixture should parse");
    const auto semantic = analyze_verilog_semantics(parsed.syntax);
    const auto types = analyze_verilog_expression_types(parsed.syntax, semantic);
    const auto* task_symbol = find_symbol(semantic, "ref_binding", "refer");
    const auto* statement =
        find_task_call(parsed.syntax.modules.front().procedural_blocks.front().body.get(),
                       "refer");
    require(task_symbol && statement, "ref callable or invocation is missing");
    const auto binding =
        bind_verilog_callable_arguments(*task_symbol, *statement->call_expression, types);
    const auto ref_mismatch = std::find_if(
        binding.diagnostics.begin(), binding.diagnostics.end(), [](const auto& diagnostic) {
            return diagnostic.code == "CallableRefTypeMismatch";
        });
    require(!binding.is_valid && ref_mismatch != binding.diagnostics.end() &&
                ref_mismatch->related_location &&
                ref_mismatch->related_location->location.line == 3,
            "ref signedness mismatch should be an error with the formal location");

    const auto* formal = task_symbol->callable->formal_arguments.front();
    const auto* actual = statement->call_expression->operands.front().get();
    auto identity_types = types;
    auto& formal_identity = identity_types.symbol_types[formal];
    auto& actual_identity = identity_types.expression_types[actual];
    formal_identity.category = VerilogTypeCategory::Enumeration;
    actual_identity.category = VerilogTypeCategory::Enumeration;
    formal_identity.bit_width = 4;
    actual_identity.bit_width = 4;
    formal_identity.is_signed = actual_identity.is_signed;
    formal_identity.identity_name = "left_enum_t";
    actual_identity.identity_name = "right_enum_t";
    const auto identity_binding = bind_verilog_callable_arguments(
        *task_symbol, *statement->call_expression, identity_types);
    require(std::any_of(identity_binding.diagnostics.begin(),
                        identity_binding.diagnostics.end(), [](const auto& diagnostic) {
                            return diagnostic.code == "CallableRefTypeMismatch";
                        }),
            "same-width named enum types with different identities must not bind by ref");

    auto packed_dimension_types = types;
    auto& formal_packed = packed_dimension_types.symbol_types[formal];
    auto& actual_packed = packed_dimension_types.expression_types[actual];
    formal_packed.is_signed = actual_packed.is_signed;
    formal_packed.packed_dimensions = {{3, 0, 4}};
    actual_packed.packed_dimensions = {{0, 3, 4}};
    const auto packed_dimension_binding = bind_verilog_callable_arguments(
        *task_symbol, *statement->call_expression, packed_dimension_types);
    require(std::any_of(packed_dimension_binding.diagnostics.begin(),
                        packed_dimension_binding.diagnostics.end(),
                        [](const auto& diagnostic) {
                            return diagnostic.code == "CallableRefTypeMismatch";
                        }),
            "same-width ref packed dimensions with different bounds must not match");

    auto unpacked_dimension_types = types;
    auto& formal_unpacked = unpacked_dimension_types.symbol_types[formal];
    auto& actual_unpacked = unpacked_dimension_types.expression_types[actual];
    formal_unpacked.is_signed = actual_unpacked.is_signed;
    formal_unpacked.unpacked_dimensions = {{1, 0, 2}};
    actual_unpacked.unpacked_dimensions = {{2, 1, 2}};
    const auto unpacked_dimension_binding = bind_verilog_callable_arguments(
        *task_symbol, *statement->call_expression, unpacked_dimension_types);
    require(std::any_of(unpacked_dimension_binding.diagnostics.begin(),
                        unpacked_dimension_binding.diagnostics.end(),
                        [](const auto& diagnostic) {
                            return diagnostic.code == "CallableRefTypeMismatch";
                        }),
            "ref unpacked dimensions with different bounds must not match");

    VerilogSemanticSymbol invalid_function;
    invalid_function.name = task_symbol->name;
    invalid_function.kind = VerilogSemanticSymbolKind::Function;
    invalid_function.callable.emplace();
    invalid_function.callable->formal_arguments =
        task_symbol->callable->formal_arguments;
    const auto as_function =
        bind_verilog_callable_arguments(invalid_function, *statement->call_expression, types);
    require(std::any_of(as_function.diagnostics.begin(), as_function.diagnostics.end(),
                        [](const auto& diagnostic) {
                            return diagnostic.code == "InvalidFunctionFormalDirection";
                        }),
            "non-input function formal direction should be rejected");
}

void test_parameterized_width_and_input_conversions() {
    const auto parsed = parse_verilog_syntax(
        "module parameterized_binding #(parameter N = 8) (input logic [3:0] narrow);\n"
        "  task automatic accept(input logic [N-1:0] wide,\n"
        "                       input logic signed [3:0] signed_value);\n"
        "    begin end\n"
        "  endtask\n"
        "  always_comb accept(narrow, narrow);\n"
        "endmodule\n");
    require(!parsed.has_errors(), "parameterized binding fixture should parse cleanly");
    const auto semantic = analyze_verilog_semantics(parsed.syntax);
    const auto types = analyze_verilog_expression_types(parsed.syntax, semantic);
    const auto* task_symbol = find_symbol(semantic, "parameterized_binding", "accept");
    const auto* statement = find_task_call(
        parsed.syntax.modules.front().procedural_blocks.front().body.get(), "accept");
    require(task_symbol && statement, "parameterized callable or call is missing");
    const auto binding =
        bind_verilog_callable_arguments(*task_symbol, *statement->call_expression, types);
    require(binding.is_valid, "input conversions should remain warnings");

    const auto width_warning = std::find_if(
        binding.diagnostics.begin(), binding.diagnostics.end(), [](const auto& diagnostic) {
            return diagnostic.code == "CallableArgumentWidthConversion";
        });
    const auto sign_warning = std::find_if(
        binding.diagnostics.begin(), binding.diagnostics.end(), [](const auto& diagnostic) {
            return diagnostic.code == "CallableArgumentSignednessConversion";
        });
    require(width_warning != binding.diagnostics.end() &&
                width_warning->severity == VerilogTypeDiagnosticSeverity::Warning &&
                width_warning->message.find("4-bit actual to 8-bit formal") !=
                    std::string::npos &&
                width_warning->related_location &&
                width_warning->related_location->location.line == 2,
            "parameterized formal width was not resolved for input compatibility");
    require(sign_warning != binding.diagnostics.end() &&
                sign_warning->severity == VerilogTypeDiagnosticSeverity::Warning &&
                sign_warning->related_location &&
                sign_warning->related_location->location.line == 3,
            "input signedness conversion did not produce a formal-located warning");
}

void test_ref_four_state_compatibility_and_positional_underarity() {
    const auto parsed = parse_verilog_syntax(
        "module state_binding;\n"
        "  logic [3:0] four_state_value;\n"
        "  bit [3:0] two_state_value;\n"
        "  task automatic refer(ref logic [3:0] item); begin end endtask\n"
        "  always_comb refer(two_state_value);\n"
        "endmodule\n");
    require(!parsed.has_errors(), "four-state ref fixture should parse cleanly");
    const auto semantic = analyze_verilog_semantics(parsed.syntax);
    const auto types = analyze_verilog_expression_types(parsed.syntax, semantic);
    const auto* task_symbol = find_symbol(semantic, "state_binding", "refer");
    const auto* statement = find_task_call(
        parsed.syntax.modules.front().procedural_blocks.front().body.get(), "refer");
    require(task_symbol && statement, "ref callable or call is missing");
    const auto& actual = *statement->call_expression->operands.front();
    auto incompatible_types = types;
    require(incompatible_types.symbol_types.contains(task_symbol->callable->formal_arguments[0]) &&
                incompatible_types.expression_types.contains(&actual),
            "ref formal or actual type was not resolved");
    auto& formal_type = incompatible_types.symbol_types.at(
        task_symbol->callable->formal_arguments[0]);
    auto& actual_type = incompatible_types.expression_types.at(&actual);
    require(formal_type.four_state && !actual_type.four_state,
            "fixture should represent four-state formal and two-state actual types");
    const auto mismatch = bind_verilog_callable_arguments(
        *task_symbol, *statement->call_expression, incompatible_types);
    require(!mismatch.is_valid &&
                std::any_of(mismatch.diagnostics.begin(), mismatch.diagnostics.end(),
                            [](const auto& diagnostic) {
                                return diagnostic.code == "CallableRefTypeMismatch" &&
                                       diagnostic.related_location.has_value();
                            }),
            "ref four-state/two-state mismatch should be an error");

    formal_type.four_state = false;
    const auto matching = bind_verilog_callable_arguments(
        *task_symbol, *statement->call_expression, incompatible_types);
    require(matching.is_valid && matching.diagnostics.empty(),
            "matching two-state ref types should bind without diagnostics");

    const auto consume_source = parse_verilog_syntax(
        "module positional_arity;\n"
        "  task automatic consume(input logic first, input logic second); begin end endtask\n"
        "  always_comb consume(1'b0);\n"
        "endmodule\n");
    require(!consume_source.has_errors(), "positional under-arity fixture should parse cleanly");
    const auto consume_semantic = analyze_verilog_semantics(consume_source.syntax);
    const auto consume_types =
        analyze_verilog_expression_types(consume_source.syntax, consume_semantic);
    const auto* consume_symbol =
        find_symbol(consume_semantic, "positional_arity", "consume");
    const auto* consume_statement = find_task_call(
        consume_source.syntax.modules.front().procedural_blocks.front().body.get(), "consume");
    require(consume_symbol && consume_statement, "positional callable or call is missing");
    const auto underarity = bind_verilog_callable_arguments(
        *consume_symbol, *consume_statement->call_expression, consume_types);
    require(!underarity.is_valid &&
                std::any_of(underarity.diagnostics.begin(), underarity.diagnostics.end(),
                            [](const auto& diagnostic) {
                                return diagnostic.code == "MissingCallableArgument" &&
                                       diagnostic.related_location.has_value();
                            }),
            "missing positional actual should identify its required formal");
}

std::unique_ptr<VerilogExpression> identifier_expression(std::string name) {
    auto expression = std::make_unique<VerilogExpression>();
    expression->kind = VerilogExpressionKind::Identifier;
    expression->text = std::move(name);
    return expression;
}

VerilogExpression make_named_call(
    const std::vector<std::optional<std::string>>& operand_names) {
    VerilogExpression call;
    call.kind = VerilogExpressionKind::Call;
    for (size_t i = 0; i < operand_names.size(); ++i) {
        call.operands.push_back(identifier_expression("actual" + std::to_string(i)));
        call.operand_names.push_back(operand_names[i]);
    }
    return call;
}

void test_named_binding_and_argument_style_diagnostics() {
    const auto parsed = parse_verilog_syntax(
        "module named_binding(input logic a, input logic b, output logic y);\n"
        "  task automatic consume(input logic first, input logic second);\n"
        "    begin end\n"
        "  endtask\n"
        "  always_comb consume(.second(b), .first(a));\n"
        "endmodule\n");
    require(!parsed.has_errors(), "named call fixture should parse cleanly");
    const auto semantic = analyze_verilog_semantics(parsed.syntax);
    const auto types = analyze_verilog_expression_types(parsed.syntax, semantic);
    const auto* task_symbol = find_symbol(semantic, "named_binding", "consume");
    const auto* statement =
        find_task_call(parsed.syntax.modules.front().procedural_blocks.front().body.get(),
                       "consume");
    require(task_symbol && statement, "named task symbol or call was not found");
    const auto named = bind_verilog_callable_arguments(
        *task_symbol, *statement->call_expression, types);
    require(named.is_valid && named.arguments.size() == 2 &&
                named.arguments[0].formal->name == "first" &&
                named.arguments[0].actual->text == "a" &&
                named.arguments[1].formal->name == "second" &&
                named.arguments[1].actual->text == "b",
            "named actuals did not bind by formal name into declaration order");

    const auto mixed_call = make_named_call({std::string("first"), std::nullopt});
    const auto mixed = bind_verilog_callable_arguments(*task_symbol, mixed_call, types);
    require(std::any_of(mixed.diagnostics.begin(), mixed.diagnostics.end(),
                        [](const auto& diagnostic) {
                            return diagnostic.code == "MixedCallableArgumentStyles";
                        }),
            "mixed named/positional arguments were not diagnosed");

    const auto duplicate_call =
        make_named_call({std::string("first"), std::string("first")});
    const auto duplicate =
        bind_verilog_callable_arguments(*task_symbol, duplicate_call, types);
    require(std::any_of(duplicate.diagnostics.begin(), duplicate.diagnostics.end(),
                        [](const auto& diagnostic) {
                            return diagnostic.code == "DuplicateNamedCallableArgument";
                        }),
            "duplicate named formal was not diagnosed");

    const auto unexpected_call = make_named_call({std::string("absent")});
    const auto unexpected =
        bind_verilog_callable_arguments(*task_symbol, unexpected_call, types);
    require(std::any_of(unexpected.diagnostics.begin(), unexpected.diagnostics.end(),
                        [](const auto& diagnostic) {
                            return diagnostic.code == "UnexpectedNamedCallableArgument";
                        }),
            "unknown named formal was not diagnosed");

    const auto missing_named_call = make_named_call({std::string("first")});
    const auto missing_named =
        bind_verilog_callable_arguments(*task_symbol, missing_named_call, types);
    require(std::any_of(missing_named.diagnostics.begin(), missing_named.diagnostics.end(),
                        [](const auto& diagnostic) {
                            return diagnostic.code == "MissingCallableArgument" &&
                                   diagnostic.related_location.has_value();
                        }),
            "omitted named formal was not diagnosed at its formal location");

    const auto excess_positional_call = make_named_call({std::nullopt, std::nullopt,
                                                         std::nullopt});
    const auto excess =
        bind_verilog_callable_arguments(*task_symbol, excess_positional_call, types);
    require(std::any_of(excess.diagnostics.begin(), excess.diagnostics.end(),
                        [](const auto& diagnostic) {
                            return diagnostic.code == "ExcessPositionalCallableArguments";
                        }),
            "excess positional actual was not diagnosed");
}

} // namespace

int main() {
    try {
        test_formal_order_types_and_declaration_lookup();
        test_required_actual_and_lvalue_diagnostic_locations();
        test_assignable_select_member_and_concatenation_actuals();
        test_ref_strict_type_mismatch_and_function_direction();
        test_named_binding_and_argument_style_diagnostics();
        test_parameterized_width_and_input_conversions();
        test_ref_four_state_compatibility_and_positional_underarity();
    } catch (const std::exception& error) {
        std::cerr << "EDA Verilog callable binding test failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
