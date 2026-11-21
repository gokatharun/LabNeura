#include "labneura/eda/verilog_callable_execution.h"

#include "labneura/eda/verilog_syntax.h"

#include <cstdint>
#include <limits>

namespace labneura::eda {
namespace {

using OutcomeKind = VerilogExecutionOutcomeKind;

VerilogExecutionOutcome outcome(OutcomeKind kind) {
    return {kind, std::nullopt};
}

void diagnose(const VerilogProceduralExecutionCallbacks& callbacks,
              const char* code, const char* message,
              const VerilogSyntaxLocation& location) {
    if (callbacks.diagnose)
        callbacks.diagnose(code, message, location);
}

bool condition_value(const VerilogConstantValue& value) {
    // Procedural if/loop conditions take the true branch only when the
    // four-state value is definitely nonzero. X/Z therefore act as false.
    for (const auto bit : value.bits)
        if (bit == VerilogLogicBit::One)
            return true;
    return false;
}

bool evaluate_condition(const VerilogStatement& statement,
                        const VerilogProceduralExecutionCallbacks& callbacks,
                        bool& condition) {
    if (!statement.expression) {
        if (statement.loop_kind == VerilogLoopKind::For) {
            condition = true;
            return true;
        }
        diagnose(callbacks, "MissingCallableLoopCondition",
                 "while loop requires a condition expression", statement.location);
        return false;
    }
    if (!callbacks.evaluate_expression) {
        diagnose(callbacks, "MissingProceduralExpressionEvaluator",
                 "loop execution requires an expression evaluator",
                 statement.expression->location);
        return false;
    }
    auto value = callbacks.evaluate_expression(*statement.expression);
    if (!value) {
        diagnose(callbacks, "UnsupportedCallableLoopCondition",
                 "loop condition could not be evaluated", statement.expression->location);
        return false;
    }
    condition = condition_value(*value);
    return true;
}

bool evaluate_repeat_count(const VerilogStatement& statement,
                           const VerilogProceduralExecutionCallbacks& callbacks,
                           uint64_t& count) {
    if (!statement.expression) {
        diagnose(callbacks, "MissingRepeatCount",
                 "repeat loop has no count expression", statement.location);
        return false;
    }
    if (!callbacks.evaluate_expression) {
        diagnose(callbacks, "MissingProceduralExpressionEvaluator",
                 "loop execution requires an expression evaluator",
                 statement.expression->location);
        return false;
    }
    const auto value = callbacks.evaluate_expression(*statement.expression);
    if (!value) {
        diagnose(callbacks, "UnsupportedCallableRepeatCount",
                 "repeat count could not be evaluated", statement.expression->location);
        return false;
    }
    if (value->bits.empty() || !value->is_fully_known()) {
        diagnose(callbacks, "UnknownCallableRepeatCount",
                 "repeat loop count must be a fully known integral value",
                 statement.expression->location);
        return false;
    }
    if (value->is_signed && value->bits.back() == VerilogLogicBit::One) {
        diagnose(callbacks, "NegativeCallableRepeatCount",
                 "repeat loop count may not be negative",
                 statement.expression->location);
        return false;
    }
    if (value->bits.size() > std::numeric_limits<uint64_t>::digits) {
        for (size_t bit = std::numeric_limits<uint64_t>::digits;
             bit < value->bits.size(); ++bit) {
            if (value->bits[bit] == VerilogLogicBit::One) {
                diagnose(callbacks, "CallableRepeatCountOutOfRange",
                         "repeat loop count exceeds the supported execution range",
                         statement.expression->location);
                return false;
            }
        }
    }
    count = 0;
    for (size_t bit = 0;
         bit < value->bits.size() && bit < std::numeric_limits<uint64_t>::digits;
         ++bit) {
        if (value->bits[bit] == VerilogLogicBit::One)
            count |= uint64_t{1} << bit;
    }
    return true;
}

VerilogExecutionOutcome execute_child(
    const VerilogStatement* statement,
    const VerilogProceduralExecutionCallbacks& callbacks) {
    if (!statement)
        return outcome(OutcomeKind::NormalCompletion);
    if (!callbacks.execute_statement) {
        diagnose(callbacks, "MissingProceduralStatementExecutor",
                 "loop execution requires a statement executor", statement->location);
        return outcome(OutcomeKind::DiagnosticFailure);
    }
    return callbacks.execute_statement(*statement);
}

VerilogExecutionOutcome budget_step(
    const VerilogStatement& statement,
    const VerilogProceduralExecutionCallbacks& callbacks) {
    if (!callbacks.consume_statement_step) {
        diagnose(callbacks, "MissingProceduralStepBudget",
                 "loop execution requires a shared statement-step budget",
                 statement.location);
        return outcome(OutcomeKind::DiagnosticFailure);
    }
    if (callbacks.consume_statement_step(statement.location))
        return outcome(OutcomeKind::NormalCompletion);
    return outcome(OutcomeKind::ExecutionBudgetExhausted);
}

} // namespace

VerilogExecutionOutcome execute_verilog_procedural_loop(
    const VerilogStatement& statement,
    const VerilogProceduralExecutionCallbacks& callbacks) {
    if (statement.kind != VerilogStatementKind::Loop) {
        diagnose(callbacks, "ExpectedProceduralLoop",
                 "loop executor requires a loop statement", statement.location);
        return outcome(OutcomeKind::UnsupportedExecution);
    }

    auto initialization = execute_child(statement.loop_initialization.get(), callbacks);
    if (initialization.kind != OutcomeKind::NormalCompletion)
        return initialization;

    if (statement.loop_kind == VerilogLoopKind::Repeat) {
        uint64_t count = 0;
        if (!evaluate_repeat_count(statement, callbacks, count))
            return outcome(OutcomeKind::DiagnosticFailure);
        for (uint64_t iteration = 0; iteration < count; ++iteration) {
            auto budget = budget_step(statement, callbacks);
            if (budget.kind != OutcomeKind::NormalCompletion)
                return budget;
            auto body = execute_child(statement.first_statement.get(), callbacks);
            if (body.kind == OutcomeKind::Break)
                return outcome(OutcomeKind::NormalCompletion);
            if (body.kind == OutcomeKind::Continue)
                continue;
            if (body.kind != OutcomeKind::NormalCompletion)
                return body;
        }
        return outcome(OutcomeKind::NormalCompletion);
    }

    if (statement.loop_kind != VerilogLoopKind::For &&
        statement.loop_kind != VerilogLoopKind::While &&
        statement.loop_kind != VerilogLoopKind::DoWhile &&
        statement.loop_kind != VerilogLoopKind::Forever) {
        diagnose(callbacks, "UnsupportedCallableLoopKind",
                 "loop kind is not supported by procedural execution",
                 statement.location);
        return outcome(OutcomeKind::UnsupportedExecution);
    }

    if (statement.loop_kind == VerilogLoopKind::DoWhile) {
        for (;;) {
            auto budget = budget_step(statement, callbacks);
            if (budget.kind != OutcomeKind::NormalCompletion)
                return budget;
            auto body = execute_child(statement.first_statement.get(), callbacks);
            if (body.kind == OutcomeKind::Break)
                return outcome(OutcomeKind::NormalCompletion);
            if (body.kind != OutcomeKind::NormalCompletion &&
                body.kind != OutcomeKind::Continue)
                return body;
            bool condition = false;
            if (!evaluate_condition(statement, callbacks, condition))
                return outcome(OutcomeKind::DiagnosticFailure);
            if (!condition)
                return outcome(OutcomeKind::NormalCompletion);
        }
    }

    for (;;) {
        // Charge each test/iteration, including forever loops with empty bodies.
        auto budget = budget_step(statement, callbacks);
        if (budget.kind != OutcomeKind::NormalCompletion)
            return budget;

        if (statement.loop_kind != VerilogLoopKind::Forever) {
            bool condition = false;
            if (!evaluate_condition(statement, callbacks, condition))
                return outcome(OutcomeKind::DiagnosticFailure);
            if (!condition)
                return outcome(OutcomeKind::NormalCompletion);
        }

        auto body = execute_child(statement.first_statement.get(), callbacks);
        if (body.kind == OutcomeKind::Break)
            return outcome(OutcomeKind::NormalCompletion);
        if (body.kind != OutcomeKind::NormalCompletion &&
            body.kind != OutcomeKind::Continue)
            return body;

        // A for-loop continue still executes the update expression.
        if (statement.loop_kind == VerilogLoopKind::For && statement.loop_step) {
            auto step = execute_child(statement.loop_step.get(), callbacks);
            if (step.kind == OutcomeKind::Break)
                return outcome(OutcomeKind::NormalCompletion);
            if (step.kind != OutcomeKind::NormalCompletion)
                return step;
        }
    }
}

} // namespace labneura::eda
