#include "labneura/eda/verilog_process_analysis.h"

#include <algorithm>
#include <string>
#include <tuple>

namespace labneura::eda {
namespace {

const VerilogSemanticSymbol* find_module_symbol(
    const VerilogModuleDeclaration& module,
    const VerilogSemanticAnalysisResult& semantic) {
    if (!semantic.model.compilation_unit_scope)
        return nullptr;
    const auto found =
        semantic.model.compilation_unit_scope->symbols_by_name.find(module.name);
    if (found == semantic.model.compilation_unit_scope->symbols_by_name.end() ||
        found->second->kind != VerilogSemanticSymbolKind::Module)
        return nullptr;
    return found->second;
}

const VerilogSemanticSymbol* resolve_event_symbol(
    const VerilogExpression& expression,
    const VerilogSemanticAnalysisResult& semantic) {
    if (expression.kind != VerilogExpressionKind::Identifier)
        return nullptr;
    for (const auto& reference : semantic.references) {
        if (reference.location.source == expression.location.source &&
            reference.location.byte_offset == expression.location.byte_offset &&
            reference.declaration)
            return reference.declaration;
    }
    return nullptr;
}

void record_event_control(
    VerilogProcess& process, const VerilogEventControl* control,
    const VerilogSemanticAnalysisResult& semantic) {
    if (!control)
        return;
    process.has_explicit_event_control = true;
    process.wildcard_sensitivity = control->wildcard;
    if (control->wildcard) {
        process.implicit_sensitivity = true;
        return;
    }

    for (const auto& term : control->terms) {
        VerilogProcessEvent event;
        event.edge = term.edge;
        event.expression = term.expression.get();
        event.location = term.location;
        if (term.expression) {
            event.symbol = resolve_event_symbol(*term.expression, semantic);
            if (event.symbol) {
                const bool is_signal =
                    event.symbol->kind == VerilogSemanticSymbolKind::Port ||
                    event.symbol->kind == VerilogSemanticSymbolKind::Net ||
                    event.symbol->kind == VerilogSemanticSymbolKind::Variable;
                event.resolution = is_signal ? VerilogAnalysisKnowledge::Known
                                             : VerilogAnalysisKnowledge::Unsupported;
                if (!is_signal) {
                    process.diagnostics.push_back(
                        {"UnsupportedProcessEventExpression",
                         "process event expression does not resolve to a signal symbol",
                         term.location, event.symbol->declaration_location});
                }
            } else {
                event.resolution = VerilogAnalysisKnowledge::Unknown;
                process.diagnostics.push_back(
                    {"UnresolvedProcessEventSignal",
                     "process event expression could not be resolved to a semantic signal",
                     term.location, std::nullopt});
            }
        } else {
            event.resolution = VerilogAnalysisKnowledge::Unsupported;
            process.diagnostics.push_back(
                {"MissingProcessEventExpression",
                 "process event term is missing its expression", term.location,
                 std::nullopt});
        }
        process.events.push_back(std::move(event));
    }
}

void inspect_statement(VerilogProcess& process,
                       const VerilogStatement* statement) {
    if (!statement)
        return;
    switch (statement->kind) {
        case VerilogStatementKind::Block:
            for (const auto& child : statement->statements)
                inspect_statement(process, child.get());
            return;
        case VerilogStatementKind::If:
            inspect_statement(process, statement->first_statement.get());
            inspect_statement(process, statement->second_statement.get());
            return;
        case VerilogStatementKind::Case:
            for (const auto& item : statement->case_items)
                inspect_statement(process, item.statement.get());
            return;
        case VerilogStatementKind::Loop:
            inspect_statement(process, statement->loop_initialization.get());
            inspect_statement(process, statement->first_statement.get());
            inspect_statement(process, statement->loop_step.get());
            return;
        case VerilogStatementKind::EventControl:
        case VerilogStatementKind::DelayControl:
            process.contains_timing_control = true;
            inspect_statement(process, statement->first_statement.get());
            return;
        case VerilogStatementKind::ProceduralAssignment: {
            VerilogProcessAssignment assignment;
            assignment.kind =
                statement->assignment_kind == VerilogAssignmentKind::Blocking
                    ? VerilogProcessAssignmentKind::Blocking
                    : VerilogProcessAssignmentKind::Nonblocking;
            assignment.lvalue = statement->left.get();
            assignment.value = statement->right.get();
            assignment.statement = statement;
            assignment.location = statement->location;
            process.assignments.push_back(std::move(assignment));
            return;
        }
        case VerilogStatementKind::Wait:
        case VerilogStatementKind::Unsupported:
            process.contains_unsupported_statement = true;
            if (statement->kind == VerilogStatementKind::Wait)
                process.contains_timing_control = true;
            return;
        case VerilogStatementKind::TaskCall:
        case VerilogStatementKind::Return:
        case VerilogStatementKind::Break:
        case VerilogStatementKind::Continue:
        case VerilogStatementKind::Empty:
            return;
    }
    process.contains_unsupported_statement = true;
}

VerilogProcessKind process_kind(VerilogProceduralBlockKind kind) {
    switch (kind) {
        case VerilogProceduralBlockKind::Always:
            return VerilogProcessKind::Always;
        case VerilogProceduralBlockKind::AlwaysComb:
            return VerilogProcessKind::AlwaysComb;
        case VerilogProceduralBlockKind::AlwaysFF:
            return VerilogProcessKind::AlwaysFF;
        case VerilogProceduralBlockKind::AlwaysLatch:
            return VerilogProcessKind::AlwaysLatch;
        case VerilogProceduralBlockKind::Initial:
            return VerilogProcessKind::Initial;
    }
    return VerilogProcessKind::Always;
}

VerilogProcessClassification classification(VerilogProcessKind kind) {
    switch (kind) {
        case VerilogProcessKind::ContinuousAssignment:
            return VerilogProcessClassification::ContinuousDriver;
        case VerilogProcessKind::Always:
            // An unqualified always is declared generic intent. Its event
            // structure is retained separately for later inference.
            return VerilogProcessClassification::GenericAlways;
        case VerilogProcessKind::AlwaysComb:
            return VerilogProcessClassification::CombinationalCandidate;
        case VerilogProcessKind::AlwaysFF:
            return VerilogProcessClassification::SequentialCandidate;
        case VerilogProcessKind::AlwaysLatch:
            return VerilogProcessClassification::LatchCandidate;
        case VerilogProcessKind::Initial:
            return VerilogProcessClassification::InitializationProcess;
    }
    return VerilogProcessClassification::Unknown;
}

} // namespace

VerilogProcessDiscoveryResult discover_verilog_processes(
    const VerilogModuleDeclaration& module,
    const VerilogSemanticAnalysisResult& semantic) {
    VerilogProcessDiscoveryResult result;
    const auto* module_symbol = find_module_symbol(module, semantic);
    if (!module_symbol) {
        result.diagnostics.push_back(
            {"UnresolvedProcessModule",
             "process discovery could not resolve the module semantic symbol",
             module.location, std::nullopt});
    }

    struct Candidate {
        size_t source_order = 0;
        const VerilogProceduralBlock* procedural = nullptr;
        const VerilogContinuousAssignment* continuous = nullptr;
    };
    std::vector<Candidate> candidates;
    candidates.reserve(module.assignments.size() + module.procedural_blocks.size());
    for (const auto& assignment : module.assignments)
        candidates.push_back({assignment.location.byte_offset, nullptr, &assignment});
    for (const auto& block : module.procedural_blocks)
        candidates.push_back({block.location.byte_offset, &block, nullptr});
    std::stable_sort(candidates.begin(), candidates.end(),
                     [](const Candidate& lhs, const Candidate& rhs) {
                         return lhs.source_order < rhs.source_order;
                     });

    result.processes.reserve(candidates.size());
    for (const auto& candidate : candidates) {
        VerilogProcess process;
        process.index = result.processes.size();
        process.module_name = module.name;
        process.module_symbol = module_symbol;
        if (candidate.continuous) {
            process.kind = VerilogProcessKind::ContinuousAssignment;
            process.classification = VerilogProcessClassification::ContinuousDriver;
            process.location = candidate.continuous->location;
            process.continuous_assignment = candidate.continuous;
            process.assignments.push_back(
                {VerilogProcessAssignmentKind::Continuous,
                 candidate.continuous->left.get(),
                 candidate.continuous->right.get(), nullptr,
                 candidate.continuous->location});
        } else {
            process.kind = process_kind(candidate.procedural->kind);
            process.classification = classification(process.kind);
            process.location = candidate.procedural->location;
            process.procedural_block = candidate.procedural;
            if (process.kind == VerilogProcessKind::AlwaysComb ||
                process.kind == VerilogProcessKind::AlwaysLatch)
                process.implicit_sensitivity = true;
            record_event_control(process, candidate.procedural->event_control.get(),
                                 semantic);
            inspect_statement(process, candidate.procedural->body.get());
        }
        result.diagnostics.insert(result.diagnostics.end(),
                                  process.diagnostics.begin(),
                                  process.diagnostics.end());
        result.processes.push_back(std::move(process));
    }
    return result;
}

} // namespace labneura::eda
