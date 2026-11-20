#include "labneura/eda/verilog_process_analysis.h"

#include <algorithm>

namespace labneura::eda {
namespace {

const VerilogSemanticSymbol* ref_at(const VerilogExpression* e,
                                    const VerilogSemanticAnalysisResult& sem) {
    if (!e || e->kind != VerilogExpressionKind::Identifier) return nullptr;
    const auto references = sem.references_at(e->location);
    if (!references.empty())
        return sem.references[references.begin()->reference_ordinal].declaration;
    return nullptr;
}

struct Test { const VerilogSemanticSymbol* signal = nullptr; bool high = true; };
Test test_signal(const VerilogExpression* e, const VerilogSemanticAnalysisResult& sem) {
    if (!e) return {};
    if (e->kind == VerilogExpressionKind::Unary && e->text == "!" && e->first) {
        auto t = test_signal(e->first.get(), sem); if (t.signal) t.high = !t.high; return t;
    }
    if (e->kind == VerilogExpressionKind::Binary && e->first && e->second &&
        (e->text == "==" || e->text == "===" || e->text == "!=" || e->text == "!==")) {
        const auto* s = ref_at(e->first.get(), sem);
        const auto* lit = e->second.get();
        bool swapped = false;
        if (!s) { s = ref_at(e->second.get(), sem); lit = e->first.get(); swapped = true; }
        if (s && lit && lit->kind == VerilogExpressionKind::Literal &&
            (lit->text == "0" || lit->text == "1" || lit->text == "1'b0" || lit->text == "1'b1")) {
            bool value = lit->text.back() == '1';
            bool high = (e->text == "==" || e->text == "===") ? value : !value;
            (void)swapped; return {s, high};
        }
    }
    return {ref_at(e, sem), true};
}

struct AssignmentLiteralState {
    bool found = false;
    bool all_literal = true;
};

AssignmentLiteralState assignment_literal_state(const VerilogStatement* s) {
    if (!s) return {};
    AssignmentLiteralState result;
    if (s->kind == VerilogStatementKind::ProceduralAssignment) {
        result.found = true;
        result.all_literal = s->right &&
                             s->right->kind == VerilogExpressionKind::Literal;
    }
    const auto merge = [&](const AssignmentLiteralState& child) {
        result.found |= child.found;
        if (child.found) result.all_literal &= child.all_literal;
    };
    merge(assignment_literal_state(s->first_statement.get()));
    merge(assignment_literal_state(s->second_statement.get()));
    merge(assignment_literal_state(s->loop_initialization.get()));
    merge(assignment_literal_state(s->loop_step.get()));
    for (const auto& child : s->statements)
        merge(assignment_literal_state(child.get()));
    for (const auto& item : s->case_items)
        merge(assignment_literal_state(item.statement.get()));
    return result;
}

std::optional<bool> reset_polarity_in_if(
    const VerilogStatement* statement, const VerilogSemanticSymbol* signal,
    const VerilogSemanticAnalysisResult& semantic) {
    if (!statement || statement->kind != VerilogStatementKind::If ||
        !statement->expression)
        return {};
    const auto test = test_signal(statement->expression.get(), semantic);
    if (test.signal != signal)
        return {};
    const auto true_branch = assignment_literal_state(statement->first_statement.get());
    const auto false_branch = assignment_literal_state(statement->second_statement.get());
    if (true_branch.found && true_branch.all_literal && false_branch.found &&
        !false_branch.all_literal)
        return test.high;
    if (false_branch.found && false_branch.all_literal && true_branch.found &&
        !true_branch.all_literal)
        return !test.high;
    return {};
}

std::optional<bool> find_reset_polarity(
    const VerilogStatement* root, const VerilogSemanticSymbol* signal,
    const VerilogSemanticAnalysisResult& semantic) {
    if (!root) return {};
    if (const auto polarity = reset_polarity_in_if(root, signal, semantic))
        return polarity;
    if (const auto polarity =
            find_reset_polarity(root->first_statement.get(), signal, semantic))
        return polarity;
    if (const auto polarity =
            find_reset_polarity(root->second_statement.get(), signal, semantic))
        return polarity;
    for (const auto& child : root->statements)
        if (const auto polarity = find_reset_polarity(child.get(), signal, semantic))
            return polarity;
    for (const auto& item : root->case_items)
        if (const auto polarity =
                find_reset_polarity(item.statement.get(), signal, semantic))
            return polarity;
    return {};
}

void collect_assignments(const VerilogStatement* s, VerilogSequentialProcessAnalysis& out,
                         const VerilogSemanticAnalysisResult& sem,
                         std::vector<Test>& conditions) {
    if (!s) return;
    if (s->kind == VerilogStatementKind::ProceduralAssignment) {
        out.has_blocking_assignments |= s->assignment_kind == VerilogAssignmentKind::Blocking;
        out.has_nonblocking_assignments |= s->assignment_kind == VerilogAssignmentKind::Nonblocking;
        if (!conditions.empty()) out.has_enable_or_hold_pattern = true;
        return;
    }
    if (s->kind == VerilogStatementKind::If) {
        auto t = test_signal(s->expression.get(), sem);
        if (t.signal) conditions.push_back(t);
        collect_assignments(s->first_statement.get(), out, sem, conditions);
        if (s->second_statement) {
            if (t.signal) conditions.back().high = !conditions.back().high;
            collect_assignments(s->second_statement.get(), out, sem, conditions);
            if (t.signal) conditions.back().high = !conditions.back().high;
        }
        if (t.signal) conditions.pop_back();
        return;
    }
    if (s->kind == VerilogStatementKind::Block || s->kind == VerilogStatementKind::Loop ||
        s->kind == VerilogStatementKind::EventControl || s->kind == VerilogStatementKind::DelayControl) {
        collect_assignments(s->first_statement.get(), out, sem, conditions);
        for (const auto& child : s->statements) collect_assignments(child.get(), out, sem, conditions);
        return;
    }
    for (const auto& item : s->case_items) collect_assignments(item.statement.get(), out, sem, conditions);
}

} // namespace

VerilogSequentialProcessAnalysis analyze_verilog_sequential_process(
    const VerilogProcess& process, const VerilogSemanticAnalysisResult& semantic,
    const VerilogTypeAnalysisResult&, const VerilogProcessDependencyResult& dependencies) {
    VerilogSequentialProcessAnalysis out;
    out.has_blocking_assignments = std::any_of(process.assignments.begin(), process.assignments.end(),
        [](const auto& a) { return a.kind == VerilogProcessAssignmentKind::Blocking; });
    out.has_nonblocking_assignments = std::any_of(process.assignments.begin(), process.assignments.end(),
        [](const auto& a) { return a.kind == VerilogProcessAssignmentKind::Nonblocking; });

    std::vector<VerilogSequentialClock> candidates;
    std::vector<VerilogSequentialReset> async_resets;
    for (const auto& e : process.events) {
        if (e.edge == VerilogEventEdge::Any) continue;
        const auto* sym = e.symbol ? e.symbol : ref_at(e.expression, semantic);
        if (!sym) continue;
        const auto reset_polarity = process.procedural_block
            ? find_reset_polarity(process.procedural_block->body.get(), sym, semantic)
            : std::optional<bool>{};
        const bool has_other_edge = std::any_of(process.events.begin(), process.events.end(),
            [&](const auto& other) { return other.edge != VerilogEventEdge::Any &&
                (other.symbol ? other.symbol : ref_at(other.expression, semantic)) != sym; });
        const bool edge_matches_polarity = (e.edge == VerilogEventEdge::Posedge) == reset_polarity.value_or(false);
        if (reset_polarity && has_other_edge && edge_matches_polarity) {
            async_resets.push_back({sym, *reset_polarity, true, false,
                e.location, process.procedural_block->body->location, true});
        } else candidates.push_back({sym, e.edge, e.location});
    }
    std::vector<VerilogSequentialClock> unique;
    for (const auto& c : candidates)
        if (std::none_of(unique.begin(), unique.end(), [&](const auto& x) { return x.symbol == c.symbol && x.edge == c.edge; })) unique.push_back(c);
    if (unique.size() == 1) out.clocks = unique;
    else if (unique.size() > 1) {
        out.has_ambiguous_clock_or_reset = true;
        out.knowledge = VerilogAnalysisKnowledge::Unsupported;
        out.diagnostics.push_back({"AmbiguousSequentialClock", "multiple edge-sensitive event signals prevent single-clock classification", process.location, {}});
    }

    std::vector<Test> conditions;
    if (process.procedural_block) collect_assignments(process.procedural_block->body.get(), out, semantic, conditions);

    if (unique.size() == 1)
        out.resets = async_resets;

    // A synchronous reset candidate must gate a constant-valued branch and
    // use a signal distinct from the clock. This avoids calling every enable
    // condition a reset.
    if (unique.size() == 1 && process.procedural_block) {
        std::vector<const VerilogStatement*> stack{process.procedural_block->body.get()};
        while (!stack.empty()) {
            auto* s = stack.back(); stack.pop_back(); if (!s) continue;
            if (s->kind == VerilogStatementKind::If && s->expression) {
                const auto t = test_signal(s->expression.get(), semantic);
                const auto polarity = t.signal == unique[0].symbol
                    ? std::optional<bool>{}
                    : reset_polarity_in_if(s, t.signal, semantic);
                if (t.signal && t.signal != unique[0].symbol && polarity) {
                    const bool exists = std::any_of(out.resets.begin(), out.resets.end(),
                        [&](const auto& r) { return r.symbol == t.signal; });
                    if (!exists) out.resets.push_back({t.signal, *polarity, false, true,
                        s->location, s->expression->location, true});
                }
            }
            stack.push_back(s->first_statement.get()); stack.push_back(s->second_statement.get());
            for (const auto& x : s->statements) stack.push_back(x.get());
        }
    }

    // Conditional non-reset gating commonly implements enable/hold. Keep this
    // conservative and only report it when assignments are nested under ifs.
    out.has_enable_or_hold_pattern = out.has_enable_or_hold_pattern ||
        (!process.assignments.empty() && process.procedural_block &&
         process.procedural_block->body &&
         process.procedural_block->body->kind == VerilogStatementKind::If);
    if (out.knowledge != VerilogAnalysisKnowledge::Unsupported)
        out.knowledge = out.clocks.empty() ? VerilogAnalysisKnowledge::Unknown : VerilogAnalysisKnowledge::Known;
    if (dependencies.knowledge != VerilogAnalysisKnowledge::Known && out.knowledge == VerilogAnalysisKnowledge::Known)
        out.knowledge = dependencies.knowledge;
    return out;
}

} // namespace labneura::eda
