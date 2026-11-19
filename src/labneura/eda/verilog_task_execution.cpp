#include "labneura/eda/verilog_task_execution.h"

#include "labneura/eda/verilog_callable_binding.h"
#include "labneura/eda/verilog_constant.h"
#include "labneura/eda/verilog_assignment.h"
#include "labneura/eda/verilog_function_evaluation.h"
#include "labneura/eda/verilog_syntax.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace labneura::eda {
namespace {

using StatementFlow = VerilogExecutionOutcomeKind;

struct CallableGuard {
    VerilogCallableExecutionState& state;
    bool entered = false;

    CallableGuard(VerilogCallableExecutionState& execution_state,
                  const VerilogSemanticSymbol& callable,
                  const VerilogSyntaxLocation& call_location)
        : state(execution_state),
          entered(state.enter_callable(callable, call_location)) {}

    ~CallableGuard() {
        if (entered)
            state.leave_callable();
    }
};

struct LoopVariableGuard {
    VerilogConstantEnvironment& environment;
    std::string name;
    bool active = false;
    bool existed = false;
    std::optional<VerilogConstantValue> previous_value;

    LoopVariableGuard(VerilogConstantEnvironment& values,
                      const VerilogStatement& statement)
        : environment(values), name(statement.loop_variable_name),
          active(statement.loop_kind == VerilogLoopKind::For &&
                 !statement.loop_variable_type.empty() && !name.empty()) {
        if (active) {
            const auto found = environment.find(name);
            if (found != environment.end()) {
                existed = true;
                previous_value = found->second;
            }
        }
    }

    ~LoopVariableGuard() {
        if (!active)
            return;
        if (existed)
            environment[name] = std::move(*previous_value);
        else
            environment.erase(name);
    }
};

const VerilogSemanticSymbol* resolve_callable(
    const VerilogSemanticAnalysisResult& semantic,
    const VerilogExpression& call) {
    if (!call.first)
        return nullptr;

    const auto same_location = [&](const VerilogSemanticReference& reference) {
        return reference.location.source == call.first->location.source &&
               reference.location.byte_offset == call.first->location.byte_offset;
    };
    auto found = std::find_if(semantic.references.begin(), semantic.references.end(),
                              [&](const auto& reference) {
                                  return same_location(reference) &&
                                         reference.declaration &&
                                         (reference.declaration->kind ==
                                              VerilogSemanticSymbolKind::Task ||
                                          reference.declaration->kind ==
                                              VerilogSemanticSymbolKind::Function);
                              });
    if (found != semantic.references.end())
        return found->declaration;

    found = std::find_if(semantic.references.begin(), semantic.references.end(),
                         [&](const auto& reference) {
                             return reference.declaration &&
                                    reference.name == call.first->text &&
                                    (reference.declaration->kind ==
                                         VerilogSemanticSymbolKind::Task ||
                                     reference.declaration->kind ==
                                         VerilogSemanticSymbolKind::Function);
                         });
    return found == semantic.references.end() ? nullptr : found->declaration;
}

bool is_callable_local_symbol(const VerilogSemanticSymbol* symbol,
                              const VerilogSemanticSymbol& callable) {
    if (!symbol)
        return false;
    if (symbol == &callable)
        return true;
    for (const auto* scope = symbol->scope; scope; scope = scope->parent)
        if (scope == callable.member_scope)
            return true;
    return false;
}

std::vector<VerilogCallableExecutionDiagnostic> diagnostics_since(
    const VerilogCallableExecutionState& state, size_t start) {
    if (start >= state.diagnostics.size())
        return {};
    return {state.diagnostics.begin() + static_cast<std::ptrdiff_t>(start),
            state.diagnostics.end()};
}

VerilogTaskExecutionResult make_result(
    const VerilogCallableExecutionState& state, size_t diagnostic_start,
    size_t step_start, uint64_t time_start, bool completed) {
    VerilogTaskExecutionResult result;
    result.completed = completed;
    result.elapsed_delay_ticks = state.elapsed_delay_ticks - time_start;
    result.executed_steps = state.statement_steps - step_start;
    result.diagnostics = diagnostics_since(state, diagnostic_start);
    result.status = completed ? VerilogTaskExecutionStatus::Completed
                              : VerilogTaskExecutionStatus::Failed;
    if (!completed && std::any_of(
            result.diagnostics.begin(), result.diagnostics.end(),
            [](const auto& diagnostic) {
                return diagnostic.code == "NoMatchingEvent";
            }))
        result.status = VerilogTaskExecutionStatus::NoMatchingEvent;
    return result;
}

const VerilogIntegralType* expression_type(
    const VerilogExpression& expression,
    const VerilogTypeAnalysisResult& type_analysis) {
    const auto found = type_analysis.expression_types.find(&expression);
    return found == type_analysis.expression_types.end() ? nullptr : &found->second;
}

const VerilogIntegralType* symbol_type(
    const VerilogSemanticSymbol* symbol,
    const VerilogTypeAnalysisResult& type_analysis) {
    if (!symbol)
        return nullptr;
    const auto found = type_analysis.symbol_types.find(symbol);
    return found == type_analysis.symbol_types.end() ? nullptr : &found->second;
}

class TaskExecutor {
public:
    TaskExecutor(const VerilogCompilationUnit& syntax,
                 const VerilogSemanticAnalysisResult& semantic,
                 const VerilogTypeAnalysisResult& type_analysis,
                 VerilogCallableExecutionState& state)
        : syntax_(syntax), semantic_(semantic), type_analysis_(type_analysis),
          state_(state) {}

    bool execute(const VerilogExpression& call,
                 VerilogConstantEnvironment& caller_environment) {
        const auto* callable = resolve_callable(semantic_, call);
        if (!callable || callable->kind != VerilogSemanticSymbolKind::Task) {
            state_.diagnose(
                callable ? "CallableKindMismatch" : "UnresolvedTaskCall",
                callable ? "a function cannot be invoked as a task"
                         : "task call does not resolve to a task declaration",
                call.location,
                callable ? std::optional<VerilogSyntaxLocation>(
                               callable->declaration_location)
                         : std::nullopt);
            return false;
        }
        return execute_task(*callable, call, caller_environment);
    }

private:
    const VerilogCompilationUnit& syntax_;
    const VerilogSemanticAnalysisResult& semantic_;
    const VerilogTypeAnalysisResult& type_analysis_;
    VerilogCallableExecutionState& state_;
    bool failed_ = false;

    static bool values_equal(const VerilogConstantValue& left,
                             const VerilogConstantValue& right) {
        return left.bit_width == right.bit_width && left.bits == right.bits;
    }

    static bool is_posedge(VerilogLogicBit previous, VerilogLogicBit current) {
        if (previous == VerilogLogicBit::Zero)
            return current == VerilogLogicBit::One ||
                   current == VerilogLogicBit::X || current == VerilogLogicBit::Z;
        return (previous == VerilogLogicBit::X || previous == VerilogLogicBit::Z) &&
               current == VerilogLogicBit::One;
    }

    static bool is_negedge(VerilogLogicBit previous, VerilogLogicBit current) {
        if (previous == VerilogLogicBit::One)
            return current == VerilogLogicBit::Zero ||
                   current == VerilogLogicBit::X || current == VerilogLogicBit::Z;
        return (previous == VerilogLogicBit::X || previous == VerilogLogicBit::Z) &&
               current == VerilogLogicBit::Zero;
    }

    void fail(std::string code, std::string message,
              const VerilogSyntaxLocation& location,
              std::optional<VerilogSyntaxLocation> related = {}) {
        state_.diagnose(std::move(code), std::move(message), location,
                        std::move(related));
        failed_ = true;
    }

    bool apply_signal_transition(
        const VerilogCallableSignalTransition& transition,
        VerilogConstantEnvironment& environment,
        const VerilogSyntaxLocation& control_location) {
        if (transition.time_ticks < state_.elapsed_delay_ticks) {
            fail("InvalidCallableSignalTransitionOrder",
                 "scheduled signal transitions must be ordered by nondecreasing logical time",
                 control_location);
            return false;
        }
        const auto current = environment.find(transition.signal_name);
        if (current == environment.end()) {
            fail("UnknownCallableSignalTransition",
                 "scheduled transition names signal '" + transition.signal_name +
                     "' which is not visible in the task frame",
                 control_location);
            return false;
        }
        if (transition.value.bit_width == 0 ||
            transition.value.bits.size() != transition.value.bit_width ||
            transition.value.bit_width != current->second.bit_width) {
            fail("InvalidCallableSignalTransitionValue",
                 "scheduled signal transition must preserve the visible signal width",
                 control_location);
            return false;
        }
        current->second = transition.value;
        state_.elapsed_delay_ticks = transition.time_ticks;
        ++state_.next_signal_transition;
        return true;
    }

    bool advance_delay(uint64_t ticks,
                       VerilogConstantEnvironment& environment,
                       const VerilogSyntaxLocation& location) {
        if (ticks > std::numeric_limits<uint64_t>::max() -
                        state_.elapsed_delay_ticks) {
            fail("CallableDelayOverflow",
                 "cumulative procedural delay exceeds the logical tick range",
                 location);
            return false;
        }
        const uint64_t target_time = state_.elapsed_delay_ticks + ticks;
        // A zero delay is an immediate ordered continuation. This interpreter
        // has no delta-cycle region, so it does not drain same-time updates.
        if (ticks == 0)
            return true;
        while (state_.next_signal_transition <
                   state_.options.signal_transitions.size() &&
               state_.options.signal_transitions[state_.next_signal_transition]
                       .time_ticks <= target_time) {
            if (!apply_signal_transition(
                    state_.options.signal_transitions[state_.next_signal_transition],
                    environment, location))
                return false;
        }
        state_.elapsed_delay_ticks = target_time;
        return true;
    }

    bool execute_event_control(const VerilogEventControl& control,
                               VerilogConstantEnvironment& environment) {
        if (control.wildcard) {
            fail("UnsupportedCallableWildcardEventControl",
                 "wildcard event controls are not supported during callable execution",
                 control.location);
            return false;
        }
        if (control.terms.empty()) {
            fail("UnsupportedCallableEventControl",
                 "event control has no supported event terms", control.location);
            return false;
        }
        for (const auto& term : control.terms) {
            if (!term.expression) {
                fail("UnsupportedCallableEventExpression",
                     "event term is missing its expression", term.location);
                return false;
            }
            if (term.edge != VerilogEventEdge::Any) {
                const auto* type = expression_type(*term.expression, type_analysis_);
                if (type && type->bit_width && *type->bit_width != 1) {
                    fail("UnsupportedCallableEventExpression",
                         "posedge and negedge currently require a one-bit expression",
                         term.expression->location);
                    return false;
                }
            }
        }

        while (state_.next_signal_transition <
               state_.options.signal_transitions.size()) {
            const auto& transition = state_.options.signal_transitions[
                state_.next_signal_transition];
            std::vector<std::optional<VerilogConstantValue>> previous_values;
            previous_values.reserve(control.terms.size());
            for (const auto& term : control.terms) {
                auto previous = evaluate_expression(term.expression.get(), environment);
                if (!previous)
                    return false;
                previous_values.push_back(std::move(previous));
            }
            if (!apply_signal_transition(transition, environment, control.location))
                return false;
            for (size_t index = 0; index < control.terms.size(); ++index) {
                const auto& term = control.terms[index];
                auto current = evaluate_expression(term.expression.get(), environment);
                if (!current)
                    return false;
                const auto& previous = *previous_values[index];
                bool matched = false;
                if (term.edge == VerilogEventEdge::Any) {
                    matched = !values_equal(previous, *current);
                } else if (previous.bit_width == 1 && current->bit_width == 1 &&
                           !previous.bits.empty() && !current->bits.empty()) {
                    matched = term.edge == VerilogEventEdge::Posedge
                                  ? is_posedge(previous.bits.front(), current->bits.front())
                                  : is_negedge(previous.bits.front(), current->bits.front());
                } else {
                    fail("UnsupportedCallableEventExpression",
                         "edge-sensitive event expression must evaluate to one bit",
                         term.expression->location);
                    return false;
                }
                if (matched)
                    return true;
            }
        }
        fail("NoMatchingEvent",
             "event trace ended before the task's event control was triggered",
             control.location);
        return false;
    }

    bool report_constant_diagnostics(
        const std::vector<VerilogConstantDiagnostic>& diagnostics) {
        for (const auto& diagnostic : diagnostics)
            fail(diagnostic.code, diagnostic.message, diagnostic.location);
        return diagnostics.empty();
    }

    std::optional<VerilogConstantValue> convert_for_type(
        VerilogConstantValue value, const VerilogIntegralType* type,
        const VerilogSyntaxLocation& location) {
        if (!type || type->category == VerilogTypeCategory::Unknown)
            return value;
        if (!type->is_known_integral()) {
            fail("UnsupportedCallableValueType",
                 "task execution currently supports only known integral values",
                 location, type->declaration_location);
            return std::nullopt;
        }
        if (*type->bit_width > state_.options.constant_options.maximum_width) {
            fail("CallableValueWidthExceeded",
                 "task value width exceeds the configured constant-evaluation limit",
                 location, type->declaration_location);
            return std::nullopt;
        }
        VerilogConstantValue converted;
        converted.bit_width = *type->bit_width;
        converted.is_signed = type->is_signed;
        converted.location = location;
        converted.bits.resize(converted.bit_width, VerilogLogicBit::Zero);
        if (value.unbased_unsized && !value.bits.empty()) {
            std::fill(converted.bits.begin(), converted.bits.end(),
                      value.bits.front());
            converted.unbased_unsized = true;
        } else {
            const size_t copied = std::min(converted.bit_width, value.bits.size());
            std::copy_n(value.bits.begin(), copied, converted.bits.begin());
            if (copied < converted.bit_width && value.is_signed &&
                !value.bits.empty()) {
                std::fill(converted.bits.begin() + static_cast<std::ptrdiff_t>(copied),
                          converted.bits.end(), value.bits.back());
            }
        }
        if (!type->four_state) {
            for (auto& bit : converted.bits)
                if (bit == VerilogLogicBit::X || bit == VerilogLogicBit::Z)
                    bit = VerilogLogicBit::Zero;
        }
        return converted;
    }

    std::optional<VerilogConstantValue> default_for_type(
        const VerilogIntegralType& type,
        const VerilogSyntaxLocation& location) {
        if (!type.is_known_integral())
            return std::nullopt;
        if (*type.bit_width > state_.options.constant_options.maximum_width) {
            fail("CallableValueWidthExceeded",
                 "task value width exceeds the configured constant-evaluation limit",
                 location, type.declaration_location);
            return std::nullopt;
        }
        VerilogConstantValue value;
        value.bit_width = *type.bit_width;
        value.is_signed = type.is_signed;
        value.location = location;
        value.bits.assign(value.bit_width,
                          type.four_state ? VerilogLogicBit::X
                                          : VerilogLogicBit::Zero);
        return value;
    }

    std::unique_ptr<VerilogExpression> constant_expression(
        const VerilogConstantValue& value,
        const VerilogSyntaxLocation& location) {
        if (value.bit_width == 0 || value.bits.size() != value.bit_width) {
            fail("UnsupportedCallableExpressionValue",
                 "callable expression produced a value without an integral width",
                 location);
            return {};
        }
        auto result = std::make_unique<VerilogExpression>();
        result->kind = VerilogExpressionKind::Literal;
        result->location = location;
        result->text = std::to_string(value.bit_width) + "'b";
        for (auto bit = value.bits.rbegin(); bit != value.bits.rend(); ++bit) {
            switch (*bit) {
                case VerilogLogicBit::Zero:
                    result->text.push_back('0');
                    break;
                case VerilogLogicBit::One:
                    result->text.push_back('1');
                    break;
                case VerilogLogicBit::X:
                    result->text.push_back('x');
                    break;
                case VerilogLogicBit::Z:
                    result->text.push_back('z');
                    break;
            }
        }
        return result;
    }

    std::unique_ptr<VerilogExpression> prepare_expression(
        const VerilogExpression& expression,
        const VerilogConstantEnvironment& environment) {
        if (expression.kind == VerilogExpressionKind::Call) {
            const auto* callable = resolve_callable(semantic_, expression);
            if (callable &&
                callable->kind == VerilogSemanticSymbolKind::Function) {
                const size_t diagnostic_start = state_.diagnostics.size();
                auto result = evaluate_verilog_function_call(
                    syntax_, semantic_, type_analysis_, expression, environment,
                    state_);
                if (!result.value) {
                    if (state_.diagnostics.size() == diagnostic_start)
                        fail("FunctionCallEvaluationFailed",
                             "user function call in task expression failed to evaluate",
                             expression.location,
                             callable->declaration_location);
                    failed_ = true;
                    return {};
                }
                return constant_expression(*result.value, expression.location);
            }
        }

        auto result = std::make_unique<VerilogExpression>();
        result->kind = expression.kind;
        result->select_kind = expression.select_kind;
        result->location = expression.location;
        result->text = expression.text;
        result->member_separator = expression.member_separator;
        const auto copy_child = [&](const std::unique_ptr<VerilogExpression>& child,
                                    std::unique_ptr<VerilogExpression>& target) {
            if (!child)
                return true;
            target = prepare_expression(*child, environment);
            return target != nullptr;
        };
        if (!copy_child(expression.first, result->first) ||
            !copy_child(expression.second, result->second) ||
            !copy_child(expression.third, result->third))
            return {};
        result->operands.reserve(expression.operands.size());
        for (const auto& operand : expression.operands) {
            if (!operand) {
                fail("MalformedCallableExpression",
                     "callable expression contains an empty operand",
                     expression.location);
                return {};
            }
            auto prepared = prepare_expression(*operand, environment);
            if (!prepared)
                return {};
            result->operands.push_back(std::move(prepared));
        }
        result->operand_names = expression.operand_names;
        return result;
    }

    std::optional<VerilogConstantValue> evaluate_expression(
        const VerilogExpression* expression,
        const VerilogConstantEnvironment& environment) {
        if (!expression) {
            fail("MissingCallableExpression",
                 "callable statement requires an expression",
                 VerilogSyntaxLocation{});
            return std::nullopt;
        }
        auto prepared = prepare_expression(*expression, environment);
        if (!prepared)
            return std::nullopt;
        auto result = evaluate_verilog_constant(
            *prepared, environment, {}, state_.options.constant_options);
        if (!report_constant_diagnostics(result.diagnostics))
            return std::nullopt;
        if (!result.value) {
            fail("UnsupportedCallableExpression",
                 "callable expression could not be evaluated",
                 expression->location);
            return std::nullopt;
        }
        return result.value;
    }

    bool assign(const VerilogExpression* left, const VerilogExpression* right,
                VerilogConstantEnvironment& environment,
                const VerilogSyntaxLocation& location) {
        if (!left) {
            fail("UnsupportedCallableLValue",
                 "task blocking assignment has no lvalue", location);
            return false;
        }
        const auto resolved = resolve_verilog_lvalue(
            *left, semantic_, type_analysis_,
            [&](const VerilogExpression& expression) {
                return evaluate_expression(&expression, environment);
            });
        if (!resolved.target) {
            report_assignment_diagnostics(resolved.diagnostics);
            return false;
        }
        const auto* active_callable = state_.active_callables.empty()
                                         ? nullptr
                                         : state_.active_callables.back();
        if (!active_callable || !active_callable->member_scope) {
            fail("UnsupportedCallableLValue",
                 "task assignment has no active callable-local scope",
                 left->location);
            return false;
        }
        for (const auto& destination : resolved.target->destinations) {
            if (!is_callable_local_symbol(destination.symbol, *active_callable)) {
                fail("UnsupportedCallableLValue",
                     "task execution cannot assign outside its local frame",
                     left->location,
                     destination.symbol
                         ? std::optional(destination.symbol->declaration_location)
                         : std::nullopt);
                return false;
            }
        }
        auto value = evaluate_expression(right, environment);
        if (!value)
            return false;
        const auto assigned = assign_verilog_lvalue(
            *resolved.target, *value, environment,
            {state_.options.constant_options.maximum_width});
        if (!assigned.assigned)
            report_assignment_diagnostics(assigned.diagnostics);
        return assigned.assigned;
    }

    void report_assignment_diagnostics(
        const std::vector<VerilogAssignmentDiagnostic>& diagnostics) {
        for (const auto& diagnostic : diagnostics)
            fail(diagnostic.code, diagnostic.message, diagnostic.location,
                 diagnostic.related_location);
    }

    VerilogProceduralExecutionCallbacks execution_callbacks(
        VerilogConstantEnvironment& environment) {
        VerilogProceduralExecutionCallbacks callbacks;
        callbacks.evaluate_expression = [this, &environment](
                                           const VerilogExpression& expression) {
            return evaluate_expression(&expression, environment);
        };
        callbacks.execute_statement = [this, &environment](
                                          const VerilogStatement& statement) {
            return VerilogExecutionOutcome{execute_statement(statement, environment),
                                           std::nullopt};
        };
        callbacks.consume_statement_step = [this](
                                               const VerilogSyntaxLocation& location) {
            const bool consumed = state_.consume_statement_step(location);
            if (!consumed)
                failed_ = true;
            return consumed;
        };
        callbacks.diagnose = [this](const std::string& code,
                                    const std::string& message,
                                    const VerilogSyntaxLocation& location) {
            fail(code, message, location);
        };
        return callbacks;
    }

    StatementFlow execute_loop(const VerilogStatement& statement,
                               VerilogConstantEnvironment& environment) {
        LoopVariableGuard loop_variable_guard(environment, statement);
        if (loop_variable_guard.active) {
            const auto type = infer_verilog_declared_type(
                statement.loop_variable_type, {}, environment);
            const auto initial =
                default_for_type(type, statement.loop_variable_location);
            if (!initial) {
                if (!failed_)
                    fail("UnsupportedCallableLoopVariableType",
                         "for-loop variables require a known integral type",
                         statement.loop_variable_location);
                return StatementFlow::Failure;
            }
            environment[statement.loop_variable_name] = *initial;
        }
        auto callbacks = execution_callbacks(environment);
        return execute_verilog_procedural_loop(statement, callbacks).kind;
    }

    StatementFlow execute_statement(const VerilogStatement& statement,
                                    VerilogConstantEnvironment& environment) {
        if (!state_.consume_statement_step(statement.location))
            return StatementFlow::ExecutionBudgetExhausted;
        switch (statement.kind) {
            case VerilogStatementKind::Block:
                for (const auto& nested : statement.statements) {
                    if (!nested)
                        continue;
                    const auto flow = execute_statement(*nested, environment);
                    if (flow != StatementFlow::NormalCompletion)
                        return flow;
                }
                return StatementFlow::NormalCompletion;
            case VerilogStatementKind::If:
            case VerilogStatementKind::Case:
                return execute_verilog_procedural_branch(
                           statement, execution_callbacks(environment))
                    .kind;
            case VerilogStatementKind::Loop:
                return execute_loop(statement, environment);
            case VerilogStatementKind::ProceduralAssignment:
                if (statement.assignment_kind != VerilogAssignmentKind::Blocking) {
                    fail("UnsupportedNonblockingCallableAssignment",
                         "nonblocking assignments are not supported during callable execution",
                         statement.location);
                    return StatementFlow::UnsupportedExecution;
                }
                return assign(statement.left.get(), statement.right.get(), environment,
                              statement.location)
                           ? StatementFlow::NormalCompletion
                           : StatementFlow::Failure;
            case VerilogStatementKind::TaskCall: {
                if (!statement.call_expression) {
                    fail("MalformedCallableTaskCall",
                         "task-call statement has no call expression",
                         statement.location);
                    return StatementFlow::Failure;
                }
                if (!execute(*statement.call_expression, environment))
                    return StatementFlow::Failure;
                return StatementFlow::NormalCompletion;
            }
            case VerilogStatementKind::Return:
                if (statement.expression) {
                    fail("TaskReturnValueUnsupported",
                         "tasks may return without a value only",
                         statement.expression->location);
                    return StatementFlow::Failure;
                }
                return StatementFlow::Return;
            case VerilogStatementKind::Break:
                return StatementFlow::Break;
            case VerilogStatementKind::Continue:
                return StatementFlow::Continue;
            case VerilogStatementKind::Empty:
                return StatementFlow::NormalCompletion;
            case VerilogStatementKind::EventControl:
                if (!statement.event_control ||
                    !execute_event_control(*statement.event_control, environment)) {
                    if (!state_.diagnostics.empty() &&
                        state_.diagnostics.back().code == "NoMatchingEvent")
                        return StatementFlow::NoMatchingEvent;
                    return StatementFlow::DiagnosticFailure;
                }
                return statement.first_statement
                           ? execute_statement(*statement.first_statement, environment)
                           : StatementFlow::NormalCompletion;
            case VerilogStatementKind::DelayControl: {
                auto delay = evaluate_expression(statement.expression.get(), environment);
                if (!delay) {
                    fail("UnsupportedCallableDelayExpression",
                         "procedural delay expression could not be evaluated as an integral constant",
                         statement.expression ? statement.expression->location
                                              : statement.location);
                    return StatementFlow::Failure;
                }
                if (delay->bit_width == 0 || delay->bit_width > 64 ||
                    delay->bits.size() != delay->bit_width ||
                    !delay->is_fully_known() ||
                    (delay->is_signed && delay->bits.back() == VerilogLogicBit::One)) {
                    fail("InvalidCallableDelayValue",
                         "procedural delay must be a known, nonnegative integral value no wider than 64 bits",
                         statement.expression ? statement.expression->location
                                              : statement.location);
                    return StatementFlow::Failure;
                }
                const auto ticks = delay->to_uint64();
                if (!ticks) {
                    fail("InvalidCallableDelayValue",
                         "procedural delay value is outside the supported logical tick range",
                         statement.expression ? statement.expression->location
                                              : statement.location);
                    return StatementFlow::Failure;
                }
                if (!advance_delay(*ticks, environment, statement.location))
                    return StatementFlow::Failure;
                return statement.first_statement
                           ? execute_statement(*statement.first_statement, environment)
                           : StatementFlow::NormalCompletion;
            }
            case VerilogStatementKind::Wait:
                fail("UnsupportedCallableWait",
                     "wait statements are not supported during callable execution",
                     statement.location);
                return StatementFlow::UnsupportedExecution;
            case VerilogStatementKind::Unsupported:
                fail("UnsupportedCallableStatement",
                     statement.text.empty()
                         ? "statement is not supported during callable execution"
                         : "unsupported callable statement: " + statement.text,
                     statement.location);
                return StatementFlow::UnsupportedExecution;
        }
        fail("UnsupportedCallableStatement",
             "statement kind is not supported during callable execution",
             statement.location);
        return StatementFlow::UnsupportedExecution;
    }

    std::optional<VerilogIntegralType> type_for_symbol(
        const VerilogSemanticSymbol* symbol,
        const VerilogConstantEnvironment& environment) const {
        if (const auto* found = symbol_type(symbol, type_analysis_))
            return *found;
        if (!symbol)
            return std::nullopt;
        auto inferred = infer_verilog_declared_type(
            symbol->declared_type, symbol->packed_dimensions, environment);
        if (inferred.category == VerilogTypeCategory::Unknown ||
            !inferred.bit_width)
            return std::nullopt;
        return inferred;
    }

    std::optional<VerilogConstantValue> default_for_symbol(
        const VerilogSemanticSymbol* symbol,
        const VerilogConstantEnvironment& environment) {
        const auto type = type_for_symbol(symbol, environment);
        if (!type)
            return std::nullopt;
        return default_for_type(*type, symbol->declaration_location);
    }

    bool copy_binding_value(const VerilogCallableArgumentBinding& binding,
                            const VerilogConstantEnvironment& source,
                            VerilogConstantEnvironment& frame,
                            const VerilogResolvedAssignmentTarget* inout_target) {
        if (!binding.formal || !binding.actual)
            return false;
        const auto direction = binding.formal->argument_direction.value_or(
            VerilogPortDirection::Unknown);

        if (direction == VerilogPortDirection::Ref) {
            fail("UnsupportedCallableRefArgument",
                 "ref argument aliasing is not supported during task execution",
                 binding.actual->location,
                 binding.formal->declaration_location);
            return false;
        }
        std::optional<VerilogConstantValue> value;
        if (direction == VerilogPortDirection::Inout) {
            if (!inout_target) {
                fail("MissingCallableInoutBinding",
                     "inout actual lvalue was not resolved at task-call binding",
                     binding.actual->location);
                return false;
            }
            auto read = read_verilog_lvalue(
                *inout_target, source,
                state_.options.constant_options.maximum_width);
            if (!read.value) {
                report_assignment_diagnostics(read.diagnostics);
                return false;
            }
            value = std::move(read.value);
        } else {
            value = evaluate_expression(binding.actual, source);
        }
        if (!value)
            return false;
        const auto formal_type = type_for_symbol(binding.formal, source);
        if (formal_type) {
            value = convert_for_type(std::move(*value), &*formal_type,
                                     binding.actual->location);
            if (!value)
                return false;
        }
        frame[binding.formal->name] = std::move(*value);
        return true;
    }

    bool copy_out_bindings(
        const std::vector<VerilogCallableArgumentBinding>& bindings,
        const std::vector<std::optional<VerilogResolvedAssignmentTarget>>& targets,
        const VerilogConstantEnvironment& frame,
        VerilogConstantEnvironment& caller_environment) {
        auto candidate = caller_environment;
        if (targets.size() != bindings.size()) {
            fail("InvalidCallableOutputBindings",
                 "task output target cache does not match its argument bindings",
                 {});
            return false;
        }
        for (size_t index = 0; index < bindings.size(); ++index) {
            const auto& binding = bindings[index];
            if (!binding.formal || !binding.actual)
                continue;
            const auto direction = binding.formal->argument_direction.value_or(
                VerilogPortDirection::Unknown);
            if (direction == VerilogPortDirection::Ref) {
                fail("UnsupportedCallableRefArgument",
                     "ref argument aliasing is not supported during task execution",
                     binding.actual->location,
                     binding.formal->declaration_location);
                return false;
            }
            if (direction != VerilogPortDirection::Output &&
                direction != VerilogPortDirection::Inout)
                continue;
            const auto found = frame.find(binding.formal->name);
            if (found == frame.end()) {
                fail("UninitializedCallableOutput",
                     "task output formal '" + binding.formal->name +
                         "' has no execution value",
                     binding.formal->declaration_location);
                return false;
            }
            if (!targets[index]) {
                fail("MissingCallableOutputBinding",
                     "task output actual lvalue was not resolved at task-call binding",
                     binding.actual->location);
                return false;
            }
            const auto assigned = assign_verilog_lvalue(
                *targets[index], found->second, candidate,
                {state_.options.constant_options.maximum_width});
            if (!assigned.assigned) {
                report_assignment_diagnostics(assigned.diagnostics);
                return false;
            }
        }
        caller_environment = std::move(candidate);
        return true;
    }

    bool execute_task(const VerilogSemanticSymbol& callable,
                      const VerilogExpression& call,
                      VerilogConstantEnvironment& caller_environment) {
        CallableGuard guard(state_, callable, call.location);
        if (!guard.entered)
            return false;
        const auto* declaration =
            find_verilog_callable_declaration(syntax_, callable);
        if (!declaration || !declaration->body) {
            fail("MissingTaskDeclaration",
                 "resolved task declaration or body could not be found",
                 call.location, callable.declaration_location);
            return false;
        }
        const auto control_flow =
            analyze_verilog_control_flow(*declaration->body);
        if (!control_flow.valid) {
            for (const auto& diagnostic : control_flow.diagnostics) {
                state_.diagnose(diagnostic.code, diagnostic.message,
                                diagnostic.location);
                failed_ = true;
            }
            return false;
        }
        const auto binding =
            bind_verilog_callable_arguments(callable, call, type_analysis_);
        for (const auto& diagnostic : binding.diagnostics) {
            if (diagnostic.severity == VerilogTypeDiagnosticSeverity::Error) {
                state_.diagnose(diagnostic.code, diagnostic.message,
                                diagnostic.location, diagnostic.related_location);
                failed_ = true;
            }
        }
        if (!binding.is_valid)
            return false;

        VerilogConstantEnvironment frame = caller_environment;
        const auto& formals = binding.arguments;
        std::vector<std::optional<VerilogResolvedAssignmentTarget>> output_targets(
            formals.size());
        for (size_t index = 0; index < formals.size(); ++index) {
            const auto& argument = formals[index];
            if (!argument.formal)
                continue;
            const auto direction = argument.formal->argument_direction.value_or(
                VerilogPortDirection::Unknown);
            if (direction == VerilogPortDirection::Output ||
                direction == VerilogPortDirection::Inout) {
                if (!argument.actual) {
                    fail("MissingCallableOutputActual",
                         "task output or inout argument has no actual expression",
                         argument.formal->declaration_location);
                    return false;
                }
                const auto resolved = resolve_verilog_lvalue(
                    *argument.actual, semantic_, type_analysis_,
                    [&](const VerilogExpression& expression) {
                        return evaluate_expression(&expression, caller_environment);
                    });
                if (!resolved.target) {
                    report_assignment_diagnostics(resolved.diagnostics);
                    return false;
                }
                output_targets[index] = *resolved.target;
            }
            if (direction == VerilogPortDirection::Output) {
                const auto value = default_for_symbol(argument.formal, frame);
                if (!value) {
                    if (!failed_)
                        fail("UnsupportedCallableFormalType",
                             "output formal '" + argument.formal->name +
                                 "' does not have a supported known integral type",
                             argument.formal->declaration_location);
                    return false;
                }
                frame[argument.formal->name] = *value;
            } else if (!copy_binding_value(
                           argument, caller_environment, frame,
                           direction == VerilogPortDirection::Inout
                               ? &*output_targets[index]
                               : nullptr)) {
                if (!failed_)
                    fail("UnsupportedCallableArgument",
                         "task argument could not be initialized",
                         argument.actual ? argument.actual->location : call.location,
                         argument.formal->declaration_location);
                return false;
            }
        }

        for (const auto& local : declaration->local_declarations) {
            const VerilogSemanticSymbol* local_symbol =
                callable.member_scope
                    ? lookup_verilog_symbol(*callable.member_scope, local.name, false)
                    : nullptr;
            if (local.initializer) {
                auto value = evaluate_expression(local.initializer.get(), frame);
                if (!value)
                    return false;
                if (const auto type = type_for_symbol(local_symbol, frame)) {
                    value = convert_for_type(std::move(*value), &*type,
                                             local.initializer->location);
                    if (!value)
                        return false;
                }
                frame[local.name] = std::move(*value);
            } else if (const auto value = default_for_symbol(local_symbol, frame)) {
                frame[local.name] = *value;
            } else {
                if (!failed_)
                    fail("UnsupportedCallableLocalType",
                         "uninitialized local '" + local.name +
                             "' does not have a supported known integral type",
                         local.location);
                return false;
            }
        }

        const bool previous_failure = failed_;
        failed_ = false;
        StatementFlow flow = execute_statement(*declaration->body, frame);
        const bool execution_failed =
            failed_ || (flow != StatementFlow::NormalCompletion &&
                        flow != StatementFlow::Return);
        failed_ = previous_failure || execution_failed;
        if (execution_failed)
            return false;
        return copy_out_bindings(binding.arguments, output_targets, frame,
                                 caller_environment);
    }
};

} // namespace

VerilogTaskExecutionResult execute_verilog_task_call(
    const VerilogCompilationUnit& syntax,
    const VerilogSemanticAnalysisResult& semantic,
    const VerilogTypeAnalysisResult& type_analysis,
    const VerilogExpression& call,
    VerilogConstantEnvironment& caller_environment,
    VerilogCallableExecutionState& state) {
    const size_t diagnostic_start = state.diagnostics.size();
    const size_t step_start = state.statement_steps;
    const uint64_t time_start = state.elapsed_delay_ticks;
    TaskExecutor executor(syntax, semantic, type_analysis, state);
    const bool completed = executor.execute(call, caller_environment);
    return make_result(state, diagnostic_start, step_start, time_start, completed);
}

VerilogTaskExecutionResult execute_verilog_task_call(
    const VerilogCompilationUnit& syntax,
    const VerilogSemanticAnalysisResult& semantic,
    const VerilogTypeAnalysisResult& type_analysis,
    const VerilogExpression& call,
    VerilogConstantEnvironment& caller_environment,
    VerilogCallableExecutionOptions options) {
    VerilogCallableExecutionState state(std::move(options));
    return execute_verilog_task_call(syntax, semantic, type_analysis, call,
                                     caller_environment, state);
}

} // namespace labneura::eda
