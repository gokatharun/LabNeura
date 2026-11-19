#include "labneura/eda/verilog_callable_execution.h"
#include "labneura/eda/verilog_syntax.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>

namespace {

using namespace labneura::eda;

void require(bool condition, std::string_view message) {
    if (!condition)
        throw std::runtime_error(std::string(message));
}

VerilogConstantValue value(std::initializer_list<VerilogLogicBit> bits,
                           bool is_signed = false) {
    VerilogConstantValue result;
    result.bit_width = bits.size();
    result.bits.assign(bits.begin(), bits.end());
    result.is_signed = is_signed;
    return result;
}

std::unique_ptr<VerilogExpression> expr(std::string name) {
    auto result = std::make_unique<VerilogExpression>();
    result->kind = VerilogExpressionKind::Identifier;
    result->text = std::move(name);
    return result;
}

VerilogCaseItem case_item(std::initializer_list<std::string> expressions,
                          std::string body, bool is_default = false) {
    VerilogCaseItem item;
    item.is_default = is_default;
    for (const auto& name : expressions)
        item.expressions.push_back(expr(name));
    item.statement = std::make_unique<VerilogStatement>();
    item.statement->kind = VerilogStatementKind::Empty;
    item.statement->text = std::move(body);
    return item;
}

struct Harness {
    VerilogProceduralExecutionCallbacks callbacks;
    std::unordered_map<std::string, VerilogConstantValue> values;
    std::string executed;
    std::string diagnostic;
    VerilogExecutionOutcomeKind body_outcome =
        VerilogExecutionOutcomeKind::Return;

    Harness() {
        callbacks.evaluate_expression = [this](const VerilogExpression& expression)
            -> std::optional<VerilogConstantValue> {
            const auto found = values.find(expression.text);
            return found == values.end() ? std::nullopt
                                         : std::optional(found->second);
        };
        callbacks.execute_statement = [this](const VerilogStatement& statement) {
            executed = statement.text;
            return VerilogExecutionOutcome{body_outcome, std::nullopt};
        };
        callbacks.diagnose = [this](const std::string& code,
                                    const std::string&,
                                    const VerilogSyntaxLocation&) {
            diagnostic = code;
        };
    }
};

void test_unknown_if_condition_takes_else_and_propagates_outcome() {
    VerilogStatement statement;
    statement.kind = VerilogStatementKind::If;
    statement.expression = expr("condition");
    statement.first_statement = std::make_unique<VerilogStatement>();
    statement.first_statement->text = "then";
    statement.second_statement = std::make_unique<VerilogStatement>();
    statement.second_statement->text = "else";
    Harness harness;
    harness.values.emplace("condition", value({VerilogLogicBit::X}));

    const auto outcome = execute_verilog_procedural_branch(statement,
                                                            harness.callbacks);
    require(harness.executed == "else",
            "unknown procedural if condition did not take the else branch");
    require(outcome.kind == VerilogExecutionOutcomeKind::Return,
            "if branch did not propagate return outcome");
    require(harness.diagnostic.empty(),
            "unknown if condition should not be treated as an execution error");
}

void test_if_truth_uses_any_one_and_false_for_zero_or_unknown_only() {
    for (const auto& [condition, expected] : {
             std::pair{value({VerilogLogicBit::X, VerilogLogicBit::One}), "then"},
             std::pair{value({VerilogLogicBit::X, VerilogLogicBit::Zero}), "else"},
             std::pair{value({VerilogLogicBit::Z}), "else"}}) {
        VerilogStatement statement;
        statement.kind = VerilogStatementKind::If;
        statement.expression = expr("condition");
        statement.first_statement = std::make_unique<VerilogStatement>();
        statement.first_statement->text = "then";
        statement.second_statement = std::make_unique<VerilogStatement>();
        statement.second_statement->text = "else";
        Harness harness;
        harness.values.emplace("condition", condition);
        execute_verilog_procedural_branch(statement, harness.callbacks);
        require(harness.executed == expected,
                "procedural if four-state truth behavior was incorrect");
    }
}

void test_case_exact_match_allows_unknown_selector_and_uses_first_match() {
    VerilogStatement statement;
    statement.kind = VerilogStatementKind::Case;
    statement.expression = expr("selector");
    statement.case_items.push_back(case_item({"not_match"}, "wrong"));
    statement.case_items.push_back(case_item({"first", "second"}, "first-match"));
    statement.case_items.push_back(case_item({}, "default", true));
    Harness harness;
    harness.values.emplace("selector", value({VerilogLogicBit::X, VerilogLogicBit::One}));
    harness.values.emplace("not_match", value({VerilogLogicBit::X, VerilogLogicBit::Zero}));
    harness.values.emplace("first", value({VerilogLogicBit::X, VerilogLogicBit::One}));
    harness.values.emplace("second", value({VerilogLogicBit::X, VerilogLogicBit::One}));

    const auto outcome = execute_verilog_procedural_branch(statement,
                                                            harness.callbacks);
    require(harness.executed == "first-match",
            "case did not exactly match four-state values in source order");
    require(outcome.kind == VerilogExecutionOutcomeKind::Return,
            "case did not propagate selected branch outcome");
}

void test_casez_and_casex_wildcard_both_operands() {
    for (const auto [kind, selector_bit, item_bit] : {
             std::tuple{VerilogCaseKind::CaseZ, VerilogLogicBit::Z,
                        VerilogLogicBit::One},
             std::tuple{VerilogCaseKind::CaseZ, VerilogLogicBit::Zero,
                        VerilogLogicBit::Z},
             std::tuple{VerilogCaseKind::CaseX, VerilogLogicBit::X,
                        VerilogLogicBit::One}}) {
        VerilogStatement statement;
        statement.kind = VerilogStatementKind::Case;
        statement.case_kind = kind;
        statement.expression = expr("selector");
        statement.case_items.push_back(case_item({"item"}, "matched"));
        Harness harness;
        harness.values.emplace("selector", value({selector_bit}));
        harness.values.emplace("item", value({item_bit}));
        execute_verilog_procedural_branch(statement, harness.callbacks);
        require(harness.executed == "matched",
                "casez/casex did not wildcard either operand correctly");
    }
}

void test_default_and_diagnostic_failure() {
    VerilogStatement statement;
    statement.kind = VerilogStatementKind::Case;
    statement.expression = expr("selector");
    statement.case_items.push_back(case_item({"item"}, "wrong"));
    statement.case_items.push_back(case_item({}, "default", true));
    Harness harness;
    harness.values.emplace("selector", value({VerilogLogicBit::Zero}));
    harness.values.emplace("item", value({VerilogLogicBit::One}));
    execute_verilog_procedural_branch(statement, harness.callbacks);
    require(harness.executed == "default", "case default was not selected");

    statement.expression.reset();
    harness.executed.clear();
    const auto outcome = execute_verilog_procedural_branch(statement,
                                                            harness.callbacks);
    require(outcome.kind == VerilogExecutionOutcomeKind::DiagnosticFailure &&
                harness.diagnostic == "MissingCallableCaseSelector",
            "malformed case did not return an explicit diagnostic failure");
}

void test_control_transfer_outcomes_are_preserved() {
    for (const auto expected : {VerilogExecutionOutcomeKind::Break,
                                VerilogExecutionOutcomeKind::Continue}) {
        VerilogStatement statement;
        statement.kind = VerilogStatementKind::If;
        statement.expression = expr("condition");
        statement.first_statement = std::make_unique<VerilogStatement>();
        statement.first_statement->text = "selected";
        Harness harness;
        harness.values.emplace("condition", value({VerilogLogicBit::One}));
        harness.body_outcome = expected;
        const auto outcome = execute_verilog_procedural_branch(
            statement, harness.callbacks);
        require(outcome.kind == expected,
                "if branch swallowed a break or continue outcome");
    }
}

} // namespace

int main() {
    try {
        test_unknown_if_condition_takes_else_and_propagates_outcome();
        test_if_truth_uses_any_one_and_false_for_zero_or_unknown_only();
        test_case_exact_match_allows_unknown_selector_and_uses_first_match();
        test_casez_and_casex_wildcard_both_operands();
        test_default_and_diagnostic_failure();
        test_control_transfer_outcomes_are_preserved();
        std::cout << "All procedural branch execution tests passed\n";
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
