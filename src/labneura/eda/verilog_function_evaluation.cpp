#include "labneura/eda/verilog_function_evaluation.h"

#include "labneura/eda/verilog_assignment.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>

namespace labneura::eda {
namespace {

using Symbol = VerilogSemanticSymbol;
using Type = VerilogIntegralType;

using StatementResult = VerilogExecutionOutcome;

struct EvaluationContext {
    const VerilogCompilationUnit& syntax;
    const VerilogSemanticAnalysisResult& semantic;
    const VerilogTypeAnalysisResult& type_analysis;
    VerilogCallableExecutionState& state;
    size_t next_placeholder = 0;
};

const Symbol* find_callee_symbol(const VerilogSemanticAnalysisResult& semantic,
                                 const VerilogExpression& call) {
    if (!call.first)
        return nullptr;
    const auto range = semantic.references_at(call.first->location);
    for (auto entry = range.end(); entry != range.begin();) {
        --entry;
        const auto& reference = semantic.references[entry->reference_ordinal];
        if (reference.declaration &&
            (reference.declaration->kind == VerilogSemanticSymbolKind::Function ||
             reference.declaration->kind == VerilogSemanticSymbolKind::Task))
            return reference.declaration;
    }
    return nullptr;
}

const Type* symbol_type(const EvaluationContext& context, const Symbol* symbol) {
    if (!symbol)
        return nullptr;
    const auto found = context.type_analysis.symbol_types.find(symbol);
    return found == context.type_analysis.symbol_types.end() ? nullptr : &found->second;
}

bool is_integral_value_type(const Type* type) {
    return type && type->is_known_integral();
}

std::optional<VerilogConstantValue> convert_value(
    const VerilogConstantValue& value, const Type* destination,
    const VerilogCallableExecutionOptions& options,
    VerilogCallableExecutionState& state, const VerilogSyntaxLocation& location) {
    if (!destination || destination->category == VerilogTypeCategory::Unknown)
        return value;
    if (!destination->is_known_integral()) {
        state.diagnose("UnsupportedCallableValueType",
                       "callable execution currently supports only known integral values",
                       location, destination->declaration_location);
        return std::nullopt;
    }
    const size_t width = *destination->bit_width;
    if (width > options.constant_options.maximum_width) {
        state.diagnose("CallableValueWidthExceeded",
                       "callable value width exceeds the configured constant-evaluation limit",
                       location, destination->declaration_location);
        return std::nullopt;
    }

    VerilogConstantValue converted;
    converted.bit_width = width;
    converted.is_signed = destination->is_signed;
    converted.location = location;
    converted.bits.resize(width, VerilogLogicBit::Zero);
    if (value.unbased_unsized && !value.bits.empty()) {
        std::fill(converted.bits.begin(), converted.bits.end(), value.bits.front());
        converted.unbased_unsized = true;
    }
    else {
        const size_t copied = std::min(width, value.bits.size());
        std::copy_n(value.bits.begin(), copied, converted.bits.begin());
        if (copied < width && value.is_signed && !value.bits.empty())
            std::fill(converted.bits.begin() + static_cast<std::ptrdiff_t>(copied),
                      converted.bits.end(), value.bits.back());
    }
    if (!destination->four_state) {
        for (auto& bit : converted.bits)
            if (bit == VerilogLogicBit::X || bit == VerilogLogicBit::Z)
                bit = VerilogLogicBit::Zero;
    }
    return converted;
}

std::optional<VerilogConstantValue> default_value(
    const Type* type, bool diagnose_unknown, const VerilogCallableExecutionOptions& options,
    VerilogCallableExecutionState& state, const VerilogSyntaxLocation& location,
    std::string_view subject) {
    if (!is_integral_value_type(type)) {
        if (diagnose_unknown) {
            state.diagnose("UnsupportedCallableLocalType",
                           "cannot initialize " + std::string(subject) +
                               " without a known integral type",
                           location, type ? std::optional(type->declaration_location)
                                          : std::nullopt);
        }
        return std::nullopt;
    }
    if (*type->bit_width > options.constant_options.maximum_width) {
        state.diagnose("CallableValueWidthExceeded",
                       "callable value width exceeds the configured constant-evaluation limit",
                       location, type->declaration_location);
        return std::nullopt;
    }
    VerilogConstantValue value;
    value.bit_width = *type->bit_width;
    value.is_signed = type->is_signed;
    value.location = location;
    value.bits.assign(value.bit_width,
                      type->four_state ? VerilogLogicBit::X : VerilogLogicBit::Zero);
    return value;
}

std::unique_ptr<VerilogExpression> clone_expression(const VerilogExpression& source) {
    auto result = std::make_unique<VerilogExpression>();
    result->kind = source.kind;
    result->select_kind = source.select_kind;
    result->location = source.location;
    result->text = source.text;
    result->member_separator = source.member_separator;
    if (source.first)
        result->first = clone_expression(*source.first);
    if (source.second)
        result->second = clone_expression(*source.second);
    if (source.third)
        result->third = clone_expression(*source.third);
    result->operands.reserve(source.operands.size());
    for (const auto& operand : source.operands)
        result->operands.push_back(operand ? clone_expression(*operand) : nullptr);
    result->operand_names = source.operand_names;
    return result;
}

std::optional<VerilogConstantValue> execute_function_call(
    EvaluationContext& context, const VerilogExpression& call,
    const VerilogConstantEnvironment& caller_environment);

bool replace_nested_function_calls(EvaluationContext& context,
                                  VerilogExpression& expression,
                                  const VerilogExpression& original,
                                  VerilogConstantEnvironment& environment) {
    if (expression.kind == VerilogExpressionKind::Call) {
        const auto* callable = find_callee_symbol(context.semantic, original);
        if (callable && callable->kind == VerilogSemanticSymbolKind::Function) {
            auto value = execute_function_call(context, original, environment);
            if (!value)
                return false;
            std::string placeholder("\x1f" "labneura.function.");
            placeholder += std::to_string(context.next_placeholder++);
            environment[placeholder] = std::move(*value);
            expression.kind = VerilogExpressionKind::Identifier;
            expression.text = std::move(placeholder);
            expression.first.reset();
            expression.second.reset();
            expression.third.reset();
            expression.operands.clear();
            expression.operand_names.clear();
            return true;
        }
        if (callable && callable->kind == VerilogSemanticSymbolKind::Task) {
            context.state.diagnose("TaskUsedAsFunction",
                                   "a task call cannot be evaluated in a function expression",
                                   expression.location,
                                   callable->declaration_location);
            return false;
        }
    }

    if (expression.first &&
        original.first &&
        !replace_nested_function_calls(context, *expression.first, *original.first,
                                       environment))
        return false;
    if (expression.second &&
        original.second &&
        !replace_nested_function_calls(context, *expression.second, *original.second,
                                       environment))
        return false;
    if (expression.third &&
        original.third &&
        !replace_nested_function_calls(context, *expression.third, *original.third,
                                       environment))
        return false;
    for (size_t index = 0; index < expression.operands.size(); ++index) {
        if (expression.operands[index] && index < original.operands.size() &&
            original.operands[index] &&
            !replace_nested_function_calls(context, *expression.operands[index],
                                           *original.operands[index], environment))
            return false;
    }
    return true;
}

std::optional<VerilogConstantValue> evaluate_expression(
    EvaluationContext& context, const VerilogExpression& expression,
    const VerilogConstantEnvironment& environment) {
    auto prepared = clone_expression(expression);
    VerilogConstantEnvironment values = environment;
    if (!replace_nested_function_calls(context, *prepared, expression, values))
        return std::nullopt;
    auto evaluated = evaluate_verilog_constant(
        *prepared, values, {}, context.state.options.constant_options);
    for (const auto& diagnostic : evaluated.diagnostics)
        context.state.diagnose(diagnostic.code, diagnostic.message, diagnostic.location);
    if (!evaluated.value && evaluated.diagnostics.empty())
        context.state.diagnose("CallableExpressionEvaluationFailed",
                               "expression could not be evaluated",
                               expression.location);
    return evaluated.value;
}

const Symbol* local_symbol(const Symbol& callable, std::string_view name) {
    if (!callable.member_scope)
        return nullptr;
    const auto found = callable.member_scope->symbols_by_name.find(std::string(name));
    return found == callable.member_scope->symbols_by_name.end() ? nullptr
                                                                 : found->second;
}

bool is_callable_local(const Symbol* symbol, const Symbol& callable) {
    if (!symbol)
        return false;
    for (const auto* scope = symbol->scope; scope; scope = scope->parent)
        if (scope == callable.member_scope)
            return true;
    return symbol == &callable;
}

StatementResult execute_statement(
    EvaluationContext& context, const Symbol& callable,
    const VerilogStatement* statement, VerilogConstantEnvironment& environment,
    std::unordered_map<std::string, Type>& loop_types,
    bool& function_result_assigned);

StatementResult execute_loop(
    EvaluationContext& context, const Symbol& callable,
    const VerilogStatement& statement, VerilogConstantEnvironment& environment,
    std::unordered_map<std::string, Type>& loop_types,
    bool& function_result_assigned) {
    const std::string loop_name = statement.loop_variable_name;
    const bool scoped_variable = !loop_name.empty() &&
                                 !statement.loop_variable_type.empty();
    std::optional<VerilogConstantValue> old_value;
    std::optional<Type> old_type;
    if (scoped_variable) {
        if (const auto found = environment.find(loop_name); found != environment.end())
            old_value = found->second;
        if (const auto found = loop_types.find(loop_name); found != loop_types.end())
            old_type = found->second;
        auto type = infer_verilog_declared_type(statement.loop_variable_type);
        if (!type.is_known_integral()) {
            context.state.diagnose("UnsupportedForLoopVariableType",
                                   "for-loop variables require a known integral type",
                                   statement.loop_variable_location);
            return {VerilogExecutionOutcomeKind::DiagnosticFailure, std::nullopt};
        }
        loop_types[loop_name] = type;
        auto initial = default_value(&type, true, context.state.options,
                                     context.state, statement.loop_variable_location,
                                     "for-loop variable");
        if (!initial)
            return {VerilogExecutionOutcomeKind::DiagnosticFailure, std::nullopt};
        environment[loop_name] = std::move(*initial);
    }
    struct LoopVariableRestore {
        VerilogConstantEnvironment& environment;
        std::unordered_map<std::string, Type>& types;
        std::string name;
        bool active;
        std::optional<VerilogConstantValue> value;
        std::optional<Type> type;
        ~LoopVariableRestore() {
            if (!active)
                return;
            if (value)
                environment[name] = std::move(*value);
            else
                environment.erase(name);
            if (type)
                types[name] = std::move(*type);
            else
                types.erase(name);
        }
    } restore{environment, loop_types, loop_name, scoped_variable,
              std::move(old_value), std::move(old_type)};

    VerilogProceduralExecutionCallbacks callbacks;
    callbacks.evaluate_expression = [&](const VerilogExpression& expression) {
        return evaluate_expression(context, expression, environment);
    };
    callbacks.execute_statement = [&](const VerilogStatement& child) {
        return execute_statement(context, callable, &child, environment, loop_types,
                                 function_result_assigned);
    };
    callbacks.consume_statement_step = [&](const VerilogSyntaxLocation& location) {
        return context.state.consume_statement_step(location);
    };
    callbacks.diagnose = [&](const std::string& code, const std::string& message,
                             const VerilogSyntaxLocation& location) {
        context.state.diagnose(code, message, location);
    };
    return execute_verilog_procedural_loop(statement, callbacks);
}

StatementResult execute_statement(
    EvaluationContext& context, const Symbol& callable,
    const VerilogStatement* statement, VerilogConstantEnvironment& environment,
    std::unordered_map<std::string, Type>& loop_types,
    bool& function_result_assigned) {
    if (!statement)
        return {};
    if (!context.state.consume_statement_step(statement->location))
        return {VerilogExecutionOutcomeKind::ExecutionBudgetExhausted, std::nullopt};

    switch (statement->kind) {
        case VerilogStatementKind::Block:
            for (const auto& child : statement->statements) {
                auto result = execute_statement(context, callable, child.get(), environment,
                                                loop_types, function_result_assigned);
                if (!result.succeeded() || result.returned())
                    return result;
            }
            return {};
        case VerilogStatementKind::Empty:
            return {};
        case VerilogStatementKind::ProceduralAssignment: {
            if (statement->assignment_kind != VerilogAssignmentKind::Blocking) {
                context.state.diagnose("UnsupportedNonblockingAssignment",
                                       "nonblocking assignments are not supported in function execution",
                                       statement->location);
                return {VerilogExecutionOutcomeKind::UnsupportedExecution,
                        std::nullopt};
            }
            if (!statement->left) {
                context.state.diagnose("UnsupportedCallableLvalue",
                                       "function blocking assignment has no lvalue",
                                       statement->location);
                return {VerilogExecutionOutcomeKind::DiagnosticFailure, std::nullopt};
            }
            if (!statement->right) {
                context.state.diagnose("MissingCallableAssignmentValue",
                                       "blocking assignment has no right-hand expression",
                                       statement->location);
                return {VerilogExecutionOutcomeKind::DiagnosticFailure, std::nullopt};
            }
            auto value = evaluate_expression(context, *statement->right, environment);
            if (!value)
                return {VerilogExecutionOutcomeKind::DiagnosticFailure, std::nullopt};

            const auto resolved = resolve_verilog_lvalue(
                *statement->left, context.semantic, context.type_analysis,
                [&](const VerilogExpression& expression) {
                    return evaluate_expression(context, expression, environment);
                });
            if (!resolved.target) {
                // The compact loop frame supports declarations synthesized by
                // for-loop execution that have no semantic symbol.
                if (statement->left->kind == VerilogExpressionKind::Identifier) {
                    const auto loop = loop_types.find(statement->left->text);
                    if (loop != loop_types.end()) {
                        auto converted = convert_value(
                            *value, &loop->second, context.state.options,
                            context.state, statement->right->location);
                        if (!converted)
                            return {VerilogExecutionOutcomeKind::DiagnosticFailure, std::nullopt};
                        environment[statement->left->text] = std::move(*converted);
                        return {};
                    }
                }
                for (const auto& diagnostic : resolved.diagnostics)
                    context.state.diagnose(diagnostic.code, diagnostic.message,
                                           diagnostic.location,
                                           diagnostic.related_location);
                return {VerilogExecutionOutcomeKind::DiagnosticFailure, std::nullopt};
            }

            auto assignment_target = *resolved.target;
            const bool function_name_target =
                statement->left->kind == VerilogExpressionKind::Identifier &&
                statement->left->text == callable.name;
            if (function_name_target) {
                if (const auto* return_type = symbol_type(context, &callable)) {
                    assignment_target.is_integral = return_type->is_known_integral();
                    assignment_target.is_signed = return_type->is_signed;
                    assignment_target.four_state = return_type->four_state;
                    for (auto& destination : assignment_target.destinations)
                        destination.declared_type = *return_type;
                }
            }
            for (const auto& destination : assignment_target.destinations) {
                if (!is_callable_local(destination.symbol, callable)) {
                    context.state.diagnose(
                        "UnsupportedCallableLvalue",
                        "function execution cannot assign outside its local frame",
                        statement->left->location,
                        destination.symbol
                            ? std::optional(destination.symbol->declaration_location)
                            : std::nullopt);
                    return {VerilogExecutionOutcomeKind::DiagnosticFailure, std::nullopt};
                }
            }
            const auto assigned = assign_verilog_lvalue(
                assignment_target, *value, environment,
                {context.state.options.constant_options.maximum_width});
            if (!assigned.assigned) {
                for (const auto& diagnostic : assigned.diagnostics)
                    context.state.diagnose(diagnostic.code, diagnostic.message,
                                           diagnostic.location,
                                           diagnostic.related_location);
                return {VerilogExecutionOutcomeKind::DiagnosticFailure, std::nullopt};
            }
            if (std::any_of(assignment_target.destinations.begin(),
                            assignment_target.destinations.end(),
                            [&](const auto& destination) {
                                return destination.symbol == &callable ||
                                       (function_name_target &&
                                        destination.storage_name == callable.name);
                            }))
                function_result_assigned = true;
            return {};
        }
        case VerilogStatementKind::If:
        case VerilogStatementKind::Case: {
            VerilogProceduralExecutionCallbacks callbacks;
            callbacks.evaluate_expression = [&](const VerilogExpression& expression) {
                return evaluate_expression(context, expression, environment);
            };
            callbacks.execute_statement = [&](const VerilogStatement& child) {
                return execute_statement(context, callable, &child, environment,
                                         loop_types, function_result_assigned);
            };
            callbacks.diagnose = [&](const std::string& code,
                                     const std::string& message,
                                     const VerilogSyntaxLocation& location) {
                context.state.diagnose(code, message, location);
            };
            return execute_verilog_procedural_branch(*statement, callbacks);
        }
        case VerilogStatementKind::Loop:
            return execute_loop(context, callable, *statement, environment, loop_types,
                                function_result_assigned);
        case VerilogStatementKind::Break:
            return {VerilogExecutionOutcomeKind::Break, std::nullopt};
        case VerilogStatementKind::Continue:
            return {VerilogExecutionOutcomeKind::Continue, std::nullopt};
        case VerilogStatementKind::Return: {
            if (!statement->expression)
                return {VerilogExecutionOutcomeKind::Return, std::nullopt};
            auto value = evaluate_expression(context, *statement->expression, environment);
            if (!value)
                return {VerilogExecutionOutcomeKind::DiagnosticFailure, std::nullopt};
            value = convert_value(*value, symbol_type(context, &callable),
                                  context.state.options, context.state,
                                  statement->expression->location);
            if (!value)
                return {VerilogExecutionOutcomeKind::DiagnosticFailure, std::nullopt};
            return {VerilogExecutionOutcomeKind::Return, std::move(value)};
        }
        case VerilogStatementKind::TaskCall:
            context.state.diagnose("UnsupportedNestedTaskCall",
                                   "nested task calls are not supported by function evaluation",
                                   statement->call_expression
                                       ? statement->call_expression->location
                                       : statement->location);
            return {VerilogExecutionOutcomeKind::UnsupportedExecution,
                    std::nullopt};
        case VerilogStatementKind::EventControl:
        case VerilogStatementKind::DelayControl:
        case VerilogStatementKind::Wait:
        case VerilogStatementKind::Unsupported:
            context.state.diagnose("UnsupportedCallableStatement",
                                   "this procedural statement is not supported in function execution",
                                   statement->location);
            return {VerilogExecutionOutcomeKind::UnsupportedExecution,
                    std::nullopt};
    }
    context.state.diagnose("UnsupportedCallableStatement",
                           "unknown procedural statement kind in function execution",
                           statement->location);
    return {VerilogExecutionOutcomeKind::DiagnosticFailure, std::nullopt};
}

std::optional<VerilogConstantValue> execute_function_call(
    EvaluationContext& context, const VerilogExpression& call,
    const VerilogConstantEnvironment& caller_environment) {
    if (call.kind != VerilogExpressionKind::Call) {
        context.state.diagnose("ExpectedFunctionCall",
                               "function evaluation requires a call expression",
                               call.location);
        return std::nullopt;
    }
    const auto* callable_symbol = find_callee_symbol(context.semantic, call);
    if (!callable_symbol ||
        callable_symbol->kind != VerilogSemanticSymbolKind::Function ||
        !callable_symbol->callable) {
        context.state.diagnose("UnresolvedFunctionCall",
                               "call expression does not resolve to a declared function",
                               call.location,
                               callable_symbol
                                   ? std::optional(callable_symbol->declaration_location)
                                   : std::nullopt);
        return std::nullopt;
    }

    const size_t diagnostic_start = context.state.diagnostics.size();
    if (!context.state.enter_callable(*callable_symbol, call.location))
        return std::nullopt;
    struct ActiveCallGuard {
        VerilogCallableExecutionState& state;
        ~ActiveCallGuard() { state.leave_callable(); }
    } guard{context.state};

    const auto* declaration =
        find_verilog_callable_declaration(context.syntax, *callable_symbol);
    if (!declaration || declaration->kind != VerilogCallableKind::Function) {
        context.state.diagnose("MissingFunctionDeclaration",
                               "resolved function has no matching syntax declaration",
                               call.location,
                               callable_symbol->declaration_location);
        return std::nullopt;
    }
    if (declaration->body) {
        const auto control_flow = analyze_verilog_control_flow(*declaration->body);
        if (!control_flow.valid) {
            for (const auto& diagnostic : control_flow.diagnostics)
                context.state.diagnose(diagnostic.code, diagnostic.message,
                                       diagnostic.location);
            return std::nullopt;
        }
    }
    auto binding =
        bind_verilog_callable_arguments(*callable_symbol, call, context.type_analysis);
    for (const auto& diagnostic : binding.diagnostics) {
        if (diagnostic.severity == VerilogTypeDiagnosticSeverity::Error)
            context.state.diagnose(diagnostic.code, diagnostic.message,
                                   diagnostic.location,
                                   diagnostic.related_location);
    }
    if (!binding.is_valid ||
        context.state.diagnostics.size() != diagnostic_start) {
        if (context.state.diagnostics.size() == diagnostic_start)
            context.state.diagnose("InvalidFunctionArgumentBinding",
                                   "function call arguments could not be bound",
                                   call.location,
                                   callable_symbol->declaration_location);
        return std::nullopt;
    }

    VerilogConstantEnvironment frame = caller_environment;
    const Type* return_type = symbol_type(context, callable_symbol);
    auto initial_result = default_value(
        return_type, true, context.state.options, context.state,
        callable_symbol->declaration_location, "function result");
    if (!initial_result)
        return std::nullopt;
    frame[callable_symbol->name] = std::move(*initial_result);

    for (const auto& argument : binding.arguments) {
        if (!argument.formal || !argument.actual) {
            context.state.diagnose("MissingFunctionArgument",
                                   "every function formal requires a bound input actual",
                                   call.location,
                                   argument.formal
                                       ? std::optional(argument.formal->declaration_location)
                                       : std::nullopt);
            return std::nullopt;
        }
        auto value = evaluate_expression(context, *argument.actual, caller_environment);
        if (!value)
            return std::nullopt;
        value = convert_value(*value, symbol_type(context, argument.formal),
                              context.state.options, context.state,
                              argument.actual->location);
        if (!value)
            return std::nullopt;
        frame[argument.formal->name] = std::move(*value);
    }

    for (const auto& local : declaration->local_declarations) {
        const auto* local_declaration = local_symbol(*callable_symbol, local.name);
        const Type* local_type = symbol_type(context, local_declaration);
        auto value = default_value(local_type, true, context.state.options,
                                   context.state, local.location,
                                   "local variable '" + local.name + "'");
        if (!value)
            return std::nullopt;
        frame[local.name] = std::move(*value);
    }
    for (const auto& local : declaration->local_declarations) {
        if (!local.initializer)
            continue;
        auto value = evaluate_expression(context, *local.initializer, frame);
        if (!value)
            return std::nullopt;
        const Type* local_type =
            symbol_type(context, local_symbol(*callable_symbol, local.name));
        value = convert_value(*value, local_type, context.state.options,
                              context.state, local.initializer->location);
        if (!value)
            return std::nullopt;
        frame[local.name] = std::move(*value);
    }

    bool function_result_assigned = false;
    std::unordered_map<std::string, Type> loop_types;
    auto execution = execute_statement(context, *callable_symbol, declaration->body.get(),
                                       frame, loop_types, function_result_assigned);
    if (!execution.succeeded())
        return std::nullopt;

    std::optional<VerilogConstantValue> result;
    if (execution.returned() && execution.return_value)
        result = std::move(execution.return_value);
    else if (function_result_assigned)
        result = frame[callable_symbol->name];
    if (!result) {
        context.state.diagnose("FunctionReturnValueMissing",
                               "function completed without an explicit return value or function-name assignment",
                               declaration->location,
                               callable_symbol->declaration_location);
        return std::nullopt;
    }
    if (context.state.diagnostics.size() != diagnostic_start)
        return std::nullopt;
    return result;
}

} // namespace

VerilogFunctionEvaluationResult evaluate_verilog_function_call(
    const VerilogCompilationUnit& syntax,
    const VerilogSemanticAnalysisResult& semantic,
    const VerilogTypeAnalysisResult& type_analysis,
    const VerilogExpression& call,
    const VerilogConstantEnvironment& caller_environment,
    VerilogCallableExecutionState& state) {
    const size_t diagnostics_before = state.diagnostics.size();
    const size_t steps_before = state.statement_steps;
    EvaluationContext context{syntax, semantic, type_analysis, state};
    auto value = execute_function_call(context, call, caller_environment);
    std::vector<VerilogCallableExecutionDiagnostic> diagnostics(
        state.diagnostics.begin() + static_cast<std::ptrdiff_t>(diagnostics_before),
        state.diagnostics.end());
    return {std::move(value), std::move(diagnostics),
            state.statement_steps - steps_before};
}

VerilogFunctionEvaluationResult evaluate_verilog_function_call(
    const VerilogCompilationUnit& syntax,
    const VerilogSemanticAnalysisResult& semantic,
    const VerilogTypeAnalysisResult& type_analysis,
    const VerilogExpression& call,
    const VerilogConstantEnvironment& caller_environment,
    VerilogCallableExecutionOptions options) {
    VerilogCallableExecutionState state(std::move(options));
    return evaluate_verilog_function_call(syntax, semantic, type_analysis, call,
                                          caller_environment, state);
}

} // namespace labneura::eda
