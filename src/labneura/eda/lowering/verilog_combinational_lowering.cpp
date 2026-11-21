#include "verilog_native_lowering_internal.h"

#include <algorithm>

namespace labneura::eda::detail {
namespace {

using Value = VerilogNativeLoweringValue;
using Environment = VerilogNativeValueEnvironment;

bool execute_statement(VerilogNativeLoweringSession& session,
                       const VerilogStatement& statement,
                       const Environment& current, Environment& updated);

Value resize_value(VerilogNativeLoweringSession& session, Value value,
                   size_t width, bool signed_destination,
                   const VerilogSyntaxLocation& location) {
    if (width == 0) {
        session.diagnose("InvalidNativeAssignmentWidth",
                         "assignment destination has zero width", location);
        return {};
    }
    if (value.bits.empty()) {
        session.diagnose("InvalidNativeExpressionWidth",
                         "assignment source has zero width", location);
        return {};
    }
    if (value.bits.size() > width) {
        value.bits.resize(width);
    } else if (value.bits.size() < width) {
        const auto extension = value.is_signed ? value.bits.back()
                                               : session.constant(false);
        value.bits.resize(width, extension);
    }
    value.is_signed = signed_destination;
    return value;
}

std::optional<Value> lower_expression(
    VerilogNativeLoweringSession& session, const VerilogExpression& expression,
    const Environment& environment,
    std::optional<size_t> context_width = std::nullopt) {
    const VerilogNativeSymbolResolver resolve = [&](const auto* symbol) {
        return session.resolve_value(symbol, &environment);
    };
    return lower_verilog_native_expression(session, expression, resolve,
                                           context_width);
}

std::optional<std::string> condition_bit(
    VerilogNativeLoweringSession& session, const Value& condition,
    const VerilogSyntaxLocation& location) {
    if (condition.bits.empty()) {
        session.diagnose("InvalidNativeCondition",
                         "procedural condition has no value bits", location);
        return std::nullopt;
    }
    auto result = condition.bits.front();
    for (size_t bit = 1; bit < condition.bits.size(); ++bit)
        result = session.emit_binary("OR2", result, condition.bits[bit]);
    return result;
}

bool assign_value(VerilogNativeLoweringSession& session,
                  const VerilogStatement& statement,
                  const Environment& current, Environment& updated) {
    if (!statement.left || !statement.right) {
        session.diagnose("MissingNativeAssignmentOperand",
                         "procedural assignment requires both an lvalue and a value",
                         statement.location);
        return false;
    }
    if (statement.assignment_kind != VerilogAssignmentKind::Blocking) {
        session.diagnose("UnsupportedNativeCombinationalAssignment",
                         "combinational native lowering supports blocking assignments only",
                         statement.location);
        return false;
    }

    const auto target = session.resolve_lvalue(*statement.left);
    if (!target.target) {
        if (target.diagnostics.empty())
            session.diagnose("UnresolvedNativeAssignmentTarget",
                             "assignment lvalue could not be resolved",
                             statement.left->location);
        else
            for (const auto& diagnostic : target.diagnostics)
                session.diagnose(diagnostic.code, diagnostic.message,
                                 diagnostic.location,
                                 diagnostic.related_location);
        return false;
    }
    if (!target.target->is_integral || target.target->bit_width == 0) {
        session.diagnose("UnsupportedNativeAssignmentTarget",
                         "combinational lowering requires an integral static lvalue",
                         statement.left->location);
        return false;
    }

    auto source = lower_expression(session, *statement.right, current,
                                   target.target->bit_width);
    if (!source)
        return false;
    auto converted = resize_value(session, std::move(*source),
                                 target.target->bit_width,
                                 target.target->is_signed,
                                 statement.right->location);
    if (converted.bits.size() != target.target->bit_width)
        return false;

    struct BitDestination {
        const VerilogSemanticSymbol* symbol = nullptr;
        size_t physical_bit = 0;
    };
    std::vector<std::optional<BitDestination>> destinations(
        target.target->bit_width);
    for (const auto& destination : target.target->destinations) {
        if (!destination.symbol || !session.signal(destination.symbol)) {
            session.diagnose("UnsupportedNativeAssignmentStorage",
                             "procedural lvalue must resolve to a declared module signal",
                             statement.left->location);
            return false;
        }
        for (size_t bit = 0; bit < destination.bit_positions.size(); ++bit) {
            if (!destination.bit_positions[bit]) {
                session.diagnose("OutOfRangeNativeAssignment",
                                 "out-of-range lvalue bits are not lowered by the native subset",
                                 statement.left->location);
                return false;
            }
            const size_t target_bit = destination.target_lsb_offset + bit;
            if (target_bit >= destinations.size() || destinations[target_bit]) {
                session.diagnose("OverlappingNativeAssignmentDestinations",
                                 "assignment destination bits overlap or exceed the resolved width",
                                 statement.left->location);
                return false;
            }
            destinations[target_bit] = BitDestination{
                destination.symbol, *destination.bit_positions[bit]};
        }
    }
    if (std::any_of(destinations.begin(), destinations.end(),
                    [](const auto& destination) { return !destination; })) {
        session.diagnose("IncompleteNativeAssignmentMapping",
                         "not every lvalue bit maps to declared storage",
                         statement.left->location);
        return false;
    }

    // Resolve the old version once per symbol. A blocking assignment updates
    // its environment immediately, so later statements see this new version.
    for (size_t bit = 0; bit < destinations.size(); ++bit) {
        const auto& destination = *destinations[bit];
        auto found = updated.find(destination.symbol);
        if (found == updated.end()) {
            auto original = session.resolve_value(destination.symbol, &current);
            if (!original) {
                session.diagnose("UnresolvedNativeAssignmentStorage",
                                 "assignment target has no lowered signal value",
                                 statement.left->location);
                return false;
            }
            found = updated.emplace(destination.symbol, std::move(*original)).first;
        }
        if (destination.physical_bit >= found->second.bits.size()) {
            session.diagnose("NativeAssignmentBitOutOfRange",
                             "resolved destination bit exceeds the lowered signal width",
                             statement.left->location);
            return false;
        }
        found->second.bits[destination.physical_bit] = converted.bits[bit];
    }
    return true;
}

std::optional<Value> merge_environments(
    VerilogNativeLoweringSession& session, const Environment& when_true,
    const Environment& when_false, const Environment& base,
    const std::string& select, const VerilogSyntaxLocation& location,
    Environment& output) {
    std::vector<const VerilogSemanticSymbol*> symbols;
    std::unordered_set<const VerilogSemanticSymbol*> seen;
    for (const auto& [symbol, value] : base) {
        (void)value;
        if (seen.insert(symbol).second)
            symbols.push_back(symbol);
    }
    for (const auto& [symbol, value] : when_true) {
        (void)value;
        if (seen.insert(symbol).second)
            symbols.push_back(symbol);
    }
    for (const auto& [symbol, value] : when_false) {
        (void)value;
        if (seen.insert(symbol).second)
            symbols.push_back(symbol);
    }
    std::sort(symbols.begin(), symbols.end(), [&](const auto* lhs, const auto* rhs) {
        const auto* left_signal = session.signal(lhs);
        const auto* right_signal = session.signal(rhs);
        if (left_signal && right_signal && left_signal->name != right_signal->name)
            return left_signal->name < right_signal->name;
        if (left_signal != nullptr || right_signal != nullptr)
            return left_signal != nullptr;
        return lhs->name < rhs->name;
    });

    for (const auto* symbol : symbols) {
        auto true_value = session.resolve_value(symbol, &when_true);
        auto false_value = session.resolve_value(symbol, &when_false);
        if (!true_value || !false_value) {
            session.diagnose("UnresolvedNativeBranchValue",
                             "branch merge could not resolve a signal value",
                             location);
            return std::nullopt;
        }
        if (true_value->bits.size() != false_value->bits.size()) {
            session.diagnose("NativeBranchWidthMismatch",
                             "procedural branches produce different signal widths",
                             location);
            return std::nullopt;
        }
        Value merged = *true_value;
        for (size_t bit = 0; bit < merged.bits.size(); ++bit) {
            if (true_value->bits[bit] != false_value->bits[bit])
                merged.bits[bit] = session.emit_mux(false_value->bits[bit],
                                                    true_value->bits[bit], select);
        }
        const auto base_value = session.resolve_value(symbol, &base);
        const auto existing_true = when_true.find(symbol);
        const auto existing_false = when_false.find(symbol);
        const bool was_modified = existing_true != when_true.end() ||
                                  existing_false != when_false.end();
        if (was_modified || !base_value || merged.bits != base_value->bits)
            output[symbol] = std::move(merged);
    }
    return Value{};
}

std::optional<Value> execute_case(
    VerilogNativeLoweringSession& session, const VerilogStatement& statement,
    const Environment& current, Environment& updated) {
    if (statement.case_kind != VerilogCaseKind::Case &&
        statement.case_kind != VerilogCaseKind::CaseZ &&
        statement.case_kind != VerilogCaseKind::CaseX) {
        session.diagnose("UnsupportedNativeCaseKind",
                         "case statement kind is outside the native subset",
                         statement.location);
        return std::nullopt;
    }
    if (!statement.expression || statement.case_items.empty()) {
        session.diagnose("InvalidNativeCaseStatement",
                         "case statement requires a selector and at least one item",
                         statement.location);
        return std::nullopt;
    }
    auto selector = lower_expression(session, *statement.expression, current);
    if (!selector || selector->bits.empty())
        return std::nullopt;

    const VerilogCaseItem* default_item = nullptr;
    std::vector<const VerilogCaseItem*> matching_items;
    for (const auto& item : statement.case_items) {
        if (item.is_default) {
            if (default_item) {
                session.diagnose("DuplicateNativeCaseDefault",
                                 "case statement contains multiple default items",
                                 item.location);
                return std::nullopt;
            }
            default_item = &item;
            continue;
        }
        if (item.expressions.empty()) {
            session.diagnose("EmptyNativeCaseItem",
                             "case item has no expressions", item.location);
            return std::nullopt;
        }
        matching_items.push_back(&item);
    }
    if (!default_item) {
        session.diagnose("NativeCaseRequiresDefault",
                         "native combinational case lowering requires an explicit default branch",
                         statement.location);
        return std::nullopt;
    }

    Environment default_environment = current;
    if (default_item->statement &&
        !execute_statement(session, *default_item->statement, current,
                           default_environment))
        return std::nullopt;

    // Construct a source-priority mux chain. The default branch is the
    // initial value, then each earlier case item is layered over later items.
    Environment result = std::move(default_environment);
    for (auto item_it = matching_items.rbegin(); item_it != matching_items.rend();
         ++item_it) {
        const auto& item = **item_it;
        Environment item_environment = current;
        if (item.statement &&
            !execute_statement(session, *item.statement, current,
                               item_environment))
            return std::nullopt;

        std::optional<std::string> item_match;
        for (const auto& item_expression : item.expressions) {
            if (!item_expression) {
                session.diagnose("MissingNativeCaseValue",
                                 "case item expression is missing", item.location);
                return std::nullopt;
            }
            auto value = lower_expression(session, *item_expression, current);
            if (!value || value->bits.empty())
                return std::nullopt;
            const size_t compare_width =
                std::max(selector->bits.size(), value->bits.size());
            const bool compare_signed = selector->is_signed && value->is_signed;
            auto lhs = resize_value(session, *selector, compare_width,
                                    compare_signed, statement.expression->location);
            auto rhs = resize_value(session, *value, compare_width,
                                    compare_signed, item_expression->location);
            std::string equal = session.constant(true);
            for (size_t bit = 0; bit < compare_width; ++bit) {
                const auto different = session.emit_binary("XOR2", lhs.bits[bit],
                                                           rhs.bits[bit]);
                const auto same = session.emit_not(different);
                equal = session.emit_binary("AND2", equal, same);
            }
            item_match = item_match
                ? std::optional<std::string>{session.emit_binary(
                      "OR2", *item_match, equal)}
                : std::optional<std::string>{equal};
        }
        if (!item_match)
            return std::nullopt;
        Environment merged;
        if (!merge_environments(session, item_environment, result, current,
                                *item_match, item.location, merged))
            return std::nullopt;
        result = std::move(merged);
    }
    updated = std::move(result);
    return Value{};
}

bool execute_statement(VerilogNativeLoweringSession& session,
                       const VerilogStatement& statement,
                       const Environment& current, Environment& updated) {
    switch (statement.kind) {
        case VerilogStatementKind::Empty:
            updated = current;
            return true;
        case VerilogStatementKind::Block: {
            updated = current;
            for (const auto& child : statement.statements) {
                if (!child)
                    continue;
                Environment next;
                if (!execute_statement(session, *child, updated, next))
                    return false;
                updated = std::move(next);
            }
            return true;
        }
        case VerilogStatementKind::ProceduralAssignment:
            updated = current;
            return assign_value(session, statement, current, updated);
        case VerilogStatementKind::If: {
            if (!statement.expression || !statement.first_statement) {
                session.diagnose("InvalidNativeIfStatement",
                                 "if statement requires a condition and true branch",
                                 statement.location);
                return false;
            }
            auto condition = lower_expression(session, *statement.expression, current);
            if (!condition)
                return false;
            auto select = condition_bit(session, *condition, statement.location);
            if (!select)
                return false;
            Environment true_environment;
            if (!execute_statement(session, *statement.first_statement, current,
                                  true_environment))
                return false;
            Environment false_environment = current;
            if (statement.second_statement &&
                !execute_statement(session, *statement.second_statement, current,
                                   false_environment))
                return false;
            updated.clear();
            return merge_environments(session, true_environment, false_environment,
                                      current, *select, statement.location,
                                      updated).has_value();
        }
        case VerilogStatementKind::Case:
            updated = current;
            return execute_case(session, statement, current, updated).has_value();
        case VerilogStatementKind::Loop:
            session.diagnose("UnsupportedNativeCombinationalLoop",
                             "native combinational lowering accepts no procedural loops yet",
                             statement.location);
            return false;
        case VerilogStatementKind::EventControl:
        case VerilogStatementKind::DelayControl:
        case VerilogStatementKind::Wait:
            session.diagnose("UnsupportedNativeCombinationalTiming",
                             "timing-controlled procedural statements cannot be lowered",
                             statement.location);
            return false;
        case VerilogStatementKind::TaskCall:
        case VerilogStatementKind::Return:
        case VerilogStatementKind::Break:
        case VerilogStatementKind::Continue:
        case VerilogStatementKind::Unsupported:
            session.diagnose("UnsupportedNativeCombinationalStatement",
                             "statement is outside the native combinational lowering subset",
                             statement.location);
            return false;
    }
    session.diagnose("UnknownNativeCombinationalStatement",
                     "unrecognized statement kind in native combinational lowering",
                     statement.location);
    return false;
}

} // namespace

bool lower_verilog_native_combinational_process(
    VerilogNativeLoweringSession& session,
    const VerilogProcessAnalysis& process_analysis) {
    const auto& process = process_analysis.process;
    if (process.kind != VerilogProcessKind::AlwaysComb &&
        process.kind != VerilogProcessKind::Always) {
        session.diagnose("InvalidNativeCombinationalProcessKind",
                         "combinational lowerer received a non-combinational process",
                         process.location, std::nullopt, false);
        return false;
    }
    if (process_analysis.synthesis_eligibility !=
        VerilogSynthesisEligibility::Eligible) {
        session.diagnose("IneligibleNativeCombinationalProcess",
                         "process analysis did not mark this process eligible",
                         process.location, std::nullopt, false);
        return false;
    }
    if (!process.procedural_block || !process.procedural_block->body) {
        session.diagnose("MissingNativeCombinationalBody",
                         "combinational process has no procedural body",
                         process.location);
        return false;
    }
    if (process.kind == VerilogProcessKind::Always &&
        std::any_of(process.events.begin(), process.events.end(),
                    [](const auto& event) {
                        return event.edge != VerilogEventEdge::Any;
                    })) {
        session.diagnose("SequentialProcessSentToCombinationalLowerer",
                         "edge-sensitive process cannot use combinational lowering",
                         process.location, std::nullopt, false);
        return false;
    }
    if (process_analysis.assignment_coverage.knowledge !=
            VerilogAnalysisKnowledge::Known ||
        process_analysis.assignment_coverage.may_require_storage) {
        session.diagnose("IncompleteNativeCombinationalCoverage",
                         "combinational process must have known complete assignment coverage",
                         process.location);
        return false;
    }
    if (process_analysis.dependencies.knowledge !=
            VerilogAnalysisKnowledge::Known ||
        process_analysis.dependencies.unresolved_callable_side_effects) {
        session.diagnose("UnresolvedNativeCombinationalDependencies",
                         "combinational process has unresolved dependencies",
                         process.location);
        return false;
    }

    Environment initial;
    Environment final_environment;
    if (!execute_statement(session, *process.procedural_block->body, initial,
                           final_environment))
        return false;

    std::vector<const VerilogSemanticSymbol*> outputs;
    std::unordered_set<const VerilogSemanticSymbol*> seen_outputs;
    for (const auto* symbol : process_analysis.dependencies.writes)
        if (symbol && session.signal(symbol) && seen_outputs.insert(symbol).second)
            outputs.push_back(symbol);
    std::sort(outputs.begin(), outputs.end(), [&](const auto* lhs, const auto* rhs) {
        return session.signal(lhs)->name < session.signal(rhs)->name;
    });
    if (outputs.empty()) {
        session.diagnose("NativeCombinationalProcessHasNoOutputs",
                         "combinational process has no declared signal destinations",
                         process.location);
        return false;
    }
    for (const auto* symbol : outputs) {
        const auto value = session.resolve_value(symbol, &final_environment);
        const auto* signal = session.signal(symbol);
        if (!value || !signal || value->bits.size() != signal->bits.size()) {
            session.diagnose("UnresolvedNativeCombinationalOutput",
                             "lowered process output does not match its declared signal width",
                             process.location);
            return false;
        }
        for (size_t bit = 0; bit < signal->bits.size(); ++bit)
            session.emit_buffer(value->bits[bit], signal->bits[bit]);
    }
    return true;
}

} // namespace labneura::eda::detail
