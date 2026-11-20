#include "labneura/eda/verilog_process_analysis.h"

#include "labneura/eda/verilog_constant.h"

#include <algorithm>
#include <set>
#include <tuple>

namespace labneura::eda {
namespace {

const VerilogSemanticSymbol* symbol_for(const VerilogExpression& expression,
                                        const VerilogSemanticAnalysisResult& semantic) {
    const auto references = semantic.references_at(expression.location);
    if (!references.empty())
        return semantic.references[references.begin()->reference_ordinal].declaration;
    return nullptr;
}

bool signal_symbol(const VerilogSemanticSymbol* symbol) {
    if (!symbol) return false;
    return symbol->kind == VerilogSemanticSymbolKind::Port ||
           symbol->kind == VerilogSemanticSymbolKind::Net ||
           symbol->kind == VerilogSemanticSymbolKind::Variable;
}

struct Analyzer {
    const VerilogProcess& process;
    const VerilogCompilationUnit& syntax;
    const VerilogSemanticAnalysisResult& semantic;
    const VerilogTypeAnalysisResult& types;
    std::set<std::string> active_callables;
    VerilogProcessDependencyResult result;
    std::set<const VerilogSemanticSymbol*> reads;
    std::set<const VerilogSemanticSymbol*> writes;

    std::optional<size_t> constant_index(const VerilogExpression* e) const {
        if (!e) return std::nullopt;
        auto value = evaluate_verilog_constant(*e);
        if (!value.value) return std::nullopt;
        auto integer = value.value->to_int64();
        if (!integer || *integer < 0) return std::nullopt;
        return static_cast<size_t>(*integer);
    }

    void add(const VerilogSemanticSymbol* symbol, bool read, bool write,
             VerilogSyntaxLocation loc, std::optional<size_t> lsb = {},
             std::optional<size_t> width = {}, bool dynamic = false,
             VerilogAnalysisKnowledge knowledge = VerilogAnalysisKnowledge::Known) {
        if (!signal_symbol(symbol)) return;
        result.accesses.push_back({symbol, read, write, lsb, width, dynamic, knowledge, loc});
        if (read) reads.insert(symbol);
        if (write) writes.insert(symbol);
        if (knowledge != VerilogAnalysisKnowledge::Known &&
            result.knowledge == VerilogAnalysisKnowledge::Known)
            result.knowledge = knowledge;
    }

    void expression(const VerilogExpression* e) {
        if (!e) return;
        if (e->kind == VerilogExpressionKind::Identifier) {
            const auto* symbol = symbol_for(*e, semantic);
            if (signal_symbol(symbol)) add(symbol, true, false, e->location);
            else if (!symbol) {
                result.knowledge = VerilogAnalysisKnowledge::Unknown;
                result.diagnostics.push_back({"UnresolvedProcessDependency",
                    "identifier reference could not be resolved for process dependencies",
                    e->location, std::nullopt});
            }
            return;
        }
        if (e->kind == VerilogExpressionKind::Call) {
            // Arguments are always dependencies. Callable side effects are
            // deliberately conservative until body summaries are available.
            for (const auto& operand : e->operands) expression(operand.get());
            std::string name = e->text;
            if (name.empty() && e->first) name = e->first->text;
            const bool pure_system = name == "$clog2" || name == "$bits" ||
                name == "$signed" || name == "$unsigned" || name == "$itor" ||
                name == "$rtoi" || name == "$isunknown";
            bool summarized = false;
            if (!pure_system) {
                for (const auto& module : syntax.modules) {
                    if (module.name != process.module_name)
                        continue;
                    for (const auto& callable : module.callables) {
                        if (callable.name != name || callable.kind != VerilogCallableKind::Function)
                            continue;
                        if (active_callables.contains(name)) break;
                        active_callables.insert(name);
                        statement(callable.body.get());
                        active_callables.erase(name);
                        summarized = true;
                    }
                }
            }
            if (!pure_system && !summarized) {
                result.unresolved_callable_side_effects = true;
                result.knowledge = VerilogAnalysisKnowledge::Unknown;
                result.diagnostics.push_back({"UnresolvedCallableSideEffects",
                    "callable side effects are not summarized by native process dependency analysis",
                    e->location, std::nullopt});
            }
            return;
        }
        if (e->kind == VerilogExpressionKind::Select && e->first &&
            e->first->kind == VerilogExpressionKind::Identifier) {
            // Reads of selected values retain a known range when the index is
            // constant. Dynamic indices are themselves reads and widen the
            // selected signal dependency conservatively.
            const auto* base = symbol_for(*e->first, semantic);
            expression(e->second.get());
            expression(e->third.get());
            const auto a = constant_index(e->second.get());
            const auto b = constant_index(e->third.get());
            if (a && (e->select_kind == VerilogSelectKind::Bit || b)) {
                size_t low = *a, width = 1;
                if (b) { low = std::min(*a, *b); width = (*a > *b ? *a - *b : *b - *a) + 1; }
                add(base, true, false, e->location, low, width);
            } else {
                add(base, true, false, e->location, {}, {}, true,
                    VerilogAnalysisKnowledge::Unknown);
            }
            return;
        }
        if (e->kind == VerilogExpressionKind::MemberAccess) {
            expression(e->first.get());
            return;
        }
        expression(e->first.get()); expression(e->second.get()); expression(e->third.get());
        for (const auto& operand : e->operands) expression(operand.get());
    }

    void destination(const VerilogExpression* e) {
        if (!e) return;
        if (e->kind == VerilogExpressionKind::Concatenation) {
            for (const auto& operand : e->operands) destination(operand.get());
            return;
        }
        if (e->kind == VerilogExpressionKind::Select && e->first) {
            expression(e->second.get()); expression(e->third.get());
            const auto* base = e->first->kind == VerilogExpressionKind::Identifier
                ? symbol_for(*e->first, semantic) : nullptr;
            const auto a = constant_index(e->second.get());
            const auto b = constant_index(e->third.get());
            if (a && (e->select_kind == VerilogSelectKind::Bit || b)) {
                size_t low = *a, width = 1;
                if (b) { low = std::min(*a, *b); width = (*a > *b ? *a - *b : *b - *a) + 1; }
                add(base, false, true, e->location, low, width);
            } else {
                add(base, false, true, e->location, {}, {}, true,
                    VerilogAnalysisKnowledge::Unknown);
            }
            if (!base) expression(e->first.get());
            return;
        }
        if (e->kind == VerilogExpressionKind::Identifier) {
            add(symbol_for(*e, semantic), false, true, e->location);
            return;
        }
        // Index expressions and any unresolved destination components may
        // read signals even though the destination itself cannot be mapped.
        expression(e);
        result.knowledge = VerilogAnalysisKnowledge::Unknown;
    }

    void statement(const VerilogStatement* s) {
        if (!s) return;
        switch (s->kind) {
        case VerilogStatementKind::Block:
            for (const auto& child : s->statements) statement(child.get()); break;
        case VerilogStatementKind::If:
            expression(s->expression.get()); statement(s->first_statement.get());
            statement(s->second_statement.get()); break;
        case VerilogStatementKind::Case:
            expression(s->expression.get());
            for (const auto& item : s->case_items) {
                for (const auto& value : item.expressions) expression(value.get());
                statement(item.statement.get());
            } break;
        case VerilogStatementKind::Loop:
            statement(s->loop_initialization.get()); expression(s->expression.get());
            statement(s->first_statement.get()); statement(s->loop_step.get()); break;
        case VerilogStatementKind::ProceduralAssignment:
            destination(s->left.get()); expression(s->right.get()); break;
        case VerilogStatementKind::TaskCall:
            expression(s->call_expression.get()); break;
        case VerilogStatementKind::EventControl:
        case VerilogStatementKind::DelayControl:
            if (s->event_control) for (const auto& term : s->event_control->terms)
                expression(term.expression.get());
            expression(s->expression.get()); statement(s->first_statement.get()); break;
        case VerilogStatementKind::Wait:
        case VerilogStatementKind::Return:
            expression(s->expression.get()); statement(s->first_statement.get()); break;
        default: break;
        }
    }

    VerilogProcessDependencyResult run() {
        if (process.continuous_assignment) {
            destination(process.continuous_assignment->left.get());
            expression(process.continuous_assignment->right.get());
        } else if (process.procedural_block) {
            statement(process.procedural_block->body.get());
        }
        // Shared assignment records cover synthesized/process records without
        // a direct AST owner. Avoid double counting records already walked.
        if (!process.procedural_block && !process.continuous_assignment) {
            for (const auto& assignment : process.assignments) {
                destination(assignment.lvalue);
                expression(assignment.value);
            }
        }
        result.raw_reads.assign(reads.begin(), reads.end());
        result.writes.assign(writes.begin(), writes.end());
        if (process.implicit_sensitivity) result.implicit_sensitivity = result.raw_reads;
        auto symbol_order = [](const auto* a, const auto* b) {
            return std::tie(a->declaration_location.byte_offset, a->name) <
                   std::tie(b->declaration_location.byte_offset, b->name);
        };
        std::sort(result.raw_reads.begin(), result.raw_reads.end(), symbol_order);
        std::sort(result.writes.begin(), result.writes.end(), symbol_order);
        std::sort(result.implicit_sensitivity.begin(), result.implicit_sensitivity.end(), symbol_order);
        std::sort(result.accesses.begin(), result.accesses.end(), [](const auto& a, const auto& b) {
            return std::tie(a.symbol->declaration_location.byte_offset, a.symbol->name,
                            a.lsb, a.width, a.dynamic_select, a.location.byte_offset,
                            a.read, a.write) <
                   std::tie(b.symbol->declaration_location.byte_offset, b.symbol->name,
                            b.lsb, b.width, b.dynamic_select, b.location.byte_offset,
                            b.read, b.write);
        });
        (void)types;
        return std::move(result);
    }
};
} // namespace

VerilogProcessDependencyResult analyze_verilog_process_dependencies(
    const VerilogProcess& process, const VerilogCompilationUnit& syntax,
    const VerilogSemanticAnalysisResult& semantic,
    const VerilogTypeAnalysisResult& type_analysis) {
    return Analyzer{process, syntax, semantic, type_analysis, {}, {}, {}, {}}.run();
}

} // namespace labneura::eda
