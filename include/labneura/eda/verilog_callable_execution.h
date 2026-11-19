#pragma once

#include "labneura/eda/verilog_constant.h"
#include "labneura/eda/verilog_semantic.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace labneura::eda {

// Shared outcome for structured procedural execution. Timing waits retain a
// distinct terminal outcome so callers can preserve the existing task status.
enum class VerilogExecutionOutcomeKind : uint8_t {
    NormalCompletion,
    Break,
    Continue,
    Return,
    UnsupportedExecution,
    ExecutionBudgetExhausted,
    DiagnosticFailure,
    NoMatchingEvent,
    // Compatibility name for failure paths until all callable paths classify
    // diagnostics and unsupported outcomes separately.
    Failure = DiagnosticFailure
};

struct VerilogExecutionOutcome {
    VerilogExecutionOutcomeKind kind =
        VerilogExecutionOutcomeKind::NormalCompletion;
    std::optional<VerilogConstantValue> return_value;

    VerilogExecutionOutcome() = default;
    VerilogExecutionOutcome(VerilogExecutionOutcomeKind outcome_kind,
                            std::optional<VerilogConstantValue> value = {})
        : kind(outcome_kind), return_value(std::move(value)) {}
    bool succeeded() const {
        return kind == VerilogExecutionOutcomeKind::NormalCompletion ||
               kind == VerilogExecutionOutcomeKind::Return;
    }
    bool returned() const {
        return kind == VerilogExecutionOutcomeKind::Return;
    }
};

struct VerilogProceduralExecutionCallbacks {
    std::function<std::optional<VerilogConstantValue>(const VerilogExpression&)>
        evaluate_expression;
    std::function<VerilogExecutionOutcome(const VerilogStatement&)>
        execute_statement;
    std::function<bool(const VerilogSyntaxLocation&)> consume_statement_step;
    std::function<void(const std::string&, const std::string&,
                       const VerilogSyntaxLocation&)>
        diagnose;
};

struct VerilogControlFlowAnalysisResult {
    struct Diagnostic {
        std::string code;
        std::string message;
        VerilogSyntaxLocation location;
    };
    bool valid = true;
    size_t maximum_loop_nesting = 0;
    bool may_return = false;
    bool contains_unsupported = false;
    std::vector<Diagnostic> diagnostics;
};

VerilogExecutionOutcome execute_verilog_procedural_branch(
    const VerilogStatement& statement,
    const VerilogProceduralExecutionCallbacks& callbacks);
VerilogExecutionOutcome execute_verilog_procedural_loop(
    const VerilogStatement& statement,
    const VerilogProceduralExecutionCallbacks& callbacks);
VerilogControlFlowAnalysisResult analyze_verilog_control_flow(
    const VerilogStatement& statement);

// A deterministic external signal update used by timing-aware task execution.
// time_ticks is an absolute logical tick; no wall-clock delay is performed.
struct VerilogCallableSignalTransition {
    uint64_t time_ticks = 0;
    // Resolves in the current task frame (including a visible formal name).
    std::string signal_name;
    // The new value; the previous value comes from the task frame itself.
    VerilogConstantValue value;
};

struct VerilogCallableExecutionOptions {
    size_t maximum_statement_steps = 100000;
    size_t maximum_call_depth = 64;
    VerilogConstantEvaluationOptions constant_options;
    // Must be ordered by nondecreasing absolute logical time. This is finite
    // deterministic stimulus, not an event scheduler or a wall-clock source.
    std::vector<VerilogCallableSignalTransition> signal_transitions;
};

struct VerilogCallableExecutionDiagnostic {
    std::string code;
    std::string message;
    VerilogSyntaxLocation location;
    std::optional<VerilogSyntaxLocation> related_location;
};

struct VerilogCallableExecutionState {
    explicit VerilogCallableExecutionState(
        VerilogCallableExecutionOptions execution_options = {})
        : options(std::move(execution_options)) {}

    VerilogCallableExecutionOptions options;
    size_t statement_steps = 0;
    uint64_t elapsed_delay_ticks = 0;
    size_t next_signal_transition = 0;
    std::vector<const VerilogSemanticSymbol*> active_callables;
    std::vector<VerilogCallableExecutionDiagnostic> diagnostics;

    void diagnose(std::string code, std::string message,
                  const VerilogSyntaxLocation& location,
                  std::optional<VerilogSyntaxLocation> related_location = {}) {
        diagnostics.push_back({std::move(code), std::move(message), location,
                               std::move(related_location)});
    }

    bool consume_statement_step(const VerilogSyntaxLocation& location) {
        if (statement_steps >= options.maximum_statement_steps) {
            diagnose("CallableExecutionStepLimitExceeded",
                     "callable execution exceeded the configured statement-step limit",
                     location);
            return false;
        }
        ++statement_steps;
        return true;
    }

    bool enter_callable(const VerilogSemanticSymbol& callable,
                        const VerilogSyntaxLocation& call_location) {
        if (std::find(active_callables.begin(), active_callables.end(), &callable) !=
            active_callables.end()) {
            diagnose("RecursiveCallableExecution",
                     "recursive function/task execution is not supported",
                     call_location, callable.declaration_location);
            return false;
        }
        if (active_callables.size() >= options.maximum_call_depth) {
            diagnose("CallableExecutionDepthExceeded",
                     "callable execution exceeded the configured call-depth limit",
                     call_location, callable.declaration_location);
            return false;
        }
        active_callables.push_back(&callable);
        return true;
    }

    void leave_callable() {
        if (!active_callables.empty())
            active_callables.pop_back();
    }
};

} // namespace labneura::eda
