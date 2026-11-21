#include "labneura/eda/verilog_callable_execution.h"

#include <algorithm>
#include <string>

namespace labneura::eda {
namespace {

VerilogExecutionOutcome diagnostic_failure(
    const VerilogProceduralExecutionCallbacks& callbacks,
    std::string code, std::string message,
    const VerilogSyntaxLocation& location) {
    if (callbacks.diagnose)
        callbacks.diagnose(code, message, location);
    return {VerilogExecutionOutcomeKind::DiagnosticFailure, std::nullopt};
}

VerilogLogicBit extended_bit(const VerilogConstantValue& value, size_t index,
                             bool sign_extend) {
    if (index < value.bits.size())
        return value.bits[index];
    return sign_extend && !value.bits.empty() ? value.bits.back()
                                              : VerilogLogicBit::Zero;
}

bool case_values_match(const VerilogConstantValue& selector,
                       const VerilogConstantValue& item,
                       VerilogCaseKind case_kind) {
    const size_t width = std::max(selector.bits.size(), item.bits.size());
    const bool sign_extend = selector.is_signed && item.is_signed;
    for (size_t index = 0; index < width; ++index) {
        const auto lhs = extended_bit(selector, index, sign_extend);
        const auto rhs = extended_bit(item, index, sign_extend);
        const bool selector_wildcard =
            (case_kind == VerilogCaseKind::CaseZ && lhs == VerilogLogicBit::Z) ||
            (case_kind == VerilogCaseKind::CaseX &&
             (lhs == VerilogLogicBit::X || lhs == VerilogLogicBit::Z));
        const bool item_wildcard =
            (case_kind == VerilogCaseKind::CaseZ && rhs == VerilogLogicBit::Z) ||
            (case_kind == VerilogCaseKind::CaseX &&
             (rhs == VerilogLogicBit::X || rhs == VerilogLogicBit::Z));
        if (!selector_wildcard && !item_wildcard && lhs != rhs)
            return false;
    }
    return true;
}

VerilogExecutionOutcome execute_selected(
    const VerilogStatement* selected,
    const VerilogProceduralExecutionCallbacks& callbacks) {
    if (!selected)
        return {};
    if (!callbacks.execute_statement) {
        if (callbacks.diagnose)
            callbacks.diagnose(
                "MissingCallableStatementExecutor",
                "selected procedural branch cannot execute in this context",
                selected->location);
        return {VerilogExecutionOutcomeKind::UnsupportedExecution, std::nullopt};
    }
    return callbacks.execute_statement(*selected);
}

} // namespace

VerilogExecutionOutcome execute_verilog_procedural_branch(
    const VerilogStatement& statement,
    const VerilogProceduralExecutionCallbacks& callbacks) {
    if (statement.kind == VerilogStatementKind::If) {
        if (!statement.expression) {
            return diagnostic_failure(
                callbacks, "MissingCallableCondition",
                "procedural conditional has no condition expression",
                statement.location);
        }
        if (!callbacks.evaluate_expression) {
            return diagnostic_failure(
                callbacks, "MissingCallableExpressionEvaluator",
                "procedural conditional cannot be evaluated in this execution context",
                statement.expression->location);
        }
        auto condition = callbacks.evaluate_expression(*statement.expression);
        if (!condition) {
            return {VerilogExecutionOutcomeKind::DiagnosticFailure, std::nullopt};
        }
        if (condition->bits.empty()) {
            return diagnostic_failure(
                callbacks, "InvalidCallableCondition",
                "procedural condition must produce a nonempty integral value",
                statement.expression->location);
        }

        // A procedural if takes its true branch only when the expression's
        // logical truth value is 1. X/Z-only conditions therefore take else.
        const bool truth = std::any_of(
            condition->bits.begin(), condition->bits.end(),
            [](VerilogLogicBit bit) { return bit == VerilogLogicBit::One; });
        return execute_selected(
            truth ? statement.first_statement.get()
                  : statement.second_statement.get(),
            callbacks);
    }

    if (statement.kind != VerilogStatementKind::Case) {
        return diagnostic_failure(
            callbacks, "UnsupportedProceduralBranch",
            "branch executor received a statement other than if or case",
            statement.location);
    }
    if (!statement.expression) {
        return diagnostic_failure(
            callbacks, "MissingCallableCaseSelector",
            "case statement has no selector expression", statement.location);
    }
    if (!callbacks.evaluate_expression) {
        return diagnostic_failure(
            callbacks, "MissingCallableExpressionEvaluator",
            "case statement cannot be evaluated in this execution context",
            statement.expression->location);
    }
    auto selector = callbacks.evaluate_expression(*statement.expression);
    if (!selector)
        return {VerilogExecutionOutcomeKind::DiagnosticFailure, std::nullopt};
    if (selector->bits.empty()) {
        return diagnostic_failure(
            callbacks, "InvalidCallableCaseSelector",
            "case selector must produce a nonempty integral value",
            statement.expression->location);
    }

    switch (statement.case_kind) {
        case VerilogCaseKind::Case:
        case VerilogCaseKind::CaseZ:
        case VerilogCaseKind::CaseX:
            break;
        default:
            return diagnostic_failure(
                callbacks, "UnsupportedCallableCaseKind",
                "case variant is not supported by procedural execution",
                statement.location);
    }

    const VerilogCaseItem* default_item = nullptr;
    for (const auto& item : statement.case_items) {
        if (item.is_default) {
            if (!default_item)
                default_item = &item;
            continue;
        }
        for (const auto& expression : item.expressions) {
            if (!expression) {
                return diagnostic_failure(
                    callbacks, "MissingCallableCaseItemExpression",
                    "case item is missing an expression", item.location);
            }
            auto candidate = callbacks.evaluate_expression(*expression);
            if (!candidate)
                return {VerilogExecutionOutcomeKind::DiagnosticFailure,
                        std::nullopt};
            if (candidate->bits.empty()) {
                return diagnostic_failure(
                    callbacks, "InvalidCallableCaseItem",
                    "case item must produce a nonempty integral value",
                    expression->location);
            }
            if (case_values_match(*selector, *candidate, statement.case_kind))
                return execute_selected(item.statement.get(), callbacks);
        }
    }
    return execute_selected(default_item ? default_item->statement.get() : nullptr,
                            callbacks);
}

} // namespace labneura::eda
