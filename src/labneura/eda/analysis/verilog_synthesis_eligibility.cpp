#include "labneura/eda/verilog_process_analysis.h"

#include <algorithm>
#include <unordered_map>

namespace labneura::eda {
namespace {

void add_diagnostic(std::vector<VerilogProcessDiagnostic>& diagnostics,
                    std::string code, std::string message,
                    const VerilogSyntaxLocation& location,
                    std::optional<VerilogSyntaxLocation> related = std::nullopt) {
    diagnostics.push_back(
        {std::move(code), std::move(message), location, std::move(related)});
}

bool is_signal(const VerilogSemanticSymbol* symbol) {
    return symbol && (symbol->kind == VerilogSemanticSymbolKind::Port ||
                      symbol->kind == VerilogSemanticSymbolKind::Net ||
                      symbol->kind == VerilogSemanticSymbolKind::Variable);
}

bool is_module_signal(const VerilogSemanticSymbol* symbol,
                      const VerilogSemanticSymbol* module) {
    if (!module || !module->member_scope || !is_signal(symbol))
        return false;
    for (auto* scope = symbol->scope; scope; scope = scope->parent)
        if (scope == module->member_scope)
            return true;
    return false;
}

bool is_net_port(const VerilogSemanticSymbol* symbol,
                 const VerilogModuleDeclaration& module) {
    if (!symbol || symbol->kind != VerilogSemanticSymbolKind::Port)
        return false;
    const auto port = std::find_if(module.ports.begin(), module.ports.end(),
                                   [&](const auto& candidate) {
                                       return candidate.name == symbol->name;
                                   });
    if (port == module.ports.end())
        return false;
    const auto& type = port->type;
    return type.find("wire") != std::string::npos ||
           type.find("tri") != std::string::npos ||
           type.find("wand") != std::string::npos ||
           type.find("wor") != std::string::npos;
}

struct Writer {
    size_t process_index;
    VerilogProcessKind kind;
    VerilogSyntaxLocation location;
};

} // namespace

VerilogSynthesisEligibilityResult analyze_verilog_synthesis_eligibility(
    const VerilogCompilationUnit& syntax,
    const VerilogSemanticAnalysisResult& semantic,
    const VerilogTypeAnalysisResult& type_analysis) {
    VerilogSynthesisEligibilityResult result;

    for (const auto& module : syntax.modules) {
        const auto discovery = discover_verilog_processes(module, semantic);
        VerilogModuleProcessAnalysis module_result;
        module_result.module_name = module.name;
        if (!discovery.processes.empty())
            module_result.module_symbol = discovery.processes.front().module_symbol;

        std::unordered_map<const VerilogSemanticSymbol*, std::vector<Writer>> writers;
        module_result.processes.reserve(discovery.processes.size());
        for (const auto& process : discovery.processes) {
            VerilogProcessAnalysis analysis;
            analysis.process = process;
            analysis.dependencies = analyze_verilog_process_dependencies(
                process, syntax, semantic, type_analysis);
            if (process.kind != VerilogProcessKind::ContinuousAssignment &&
                process.kind != VerilogProcessKind::Initial) {
                analysis.assignment_coverage = analyze_verilog_assignment_coverage(
                    process, semantic, type_analysis);
            }
            analysis.sequential = analyze_verilog_sequential_process(
                process, semantic, type_analysis, analysis.dependencies);
            analysis.diagnostics.insert(analysis.diagnostics.end(),
                                        process.diagnostics.begin(),
                                        process.diagnostics.end());
            analysis.diagnostics.insert(analysis.diagnostics.end(),
                                        analysis.dependencies.diagnostics.begin(),
                                        analysis.dependencies.diagnostics.end());
            analysis.diagnostics.insert(analysis.diagnostics.end(),
                                        analysis.assignment_coverage.diagnostics.begin(),
                                        analysis.assignment_coverage.diagnostics.end());
            analysis.diagnostics.insert(analysis.diagnostics.end(),
                                        analysis.sequential.diagnostics.begin(),
                                        analysis.sequential.diagnostics.end());

            for (const auto& access : analysis.dependencies.accesses) {
                if (process.kind != VerilogProcessKind::Initial && access.write &&
                    is_module_signal(access.symbol, module_result.module_symbol)) {
                    auto& signal_writers = writers[access.symbol];
                    if (std::none_of(signal_writers.begin(), signal_writers.end(),
                                     [&](const auto& writer) {
                                         return writer.process_index == process.index;
                                     })) {
                        signal_writers.push_back(
                            {process.index, process.kind, access.location});
                    }
                }
            }

            const bool has_unknown_dependencies =
                analysis.dependencies.knowledge != VerilogAnalysisKnowledge::Known ||
                analysis.dependencies.unresolved_callable_side_effects;
            const bool has_unknown_events = std::any_of(
                process.events.begin(), process.events.end(), [](const auto& event) {
                    return event.resolution == VerilogAnalysisKnowledge::Unknown;
                });
            const bool has_unsupported_events = std::any_of(
                process.events.begin(), process.events.end(), [](const auto& event) {
                    return event.resolution == VerilogAnalysisKnowledge::Unsupported;
                });
            const bool unsupported_ast = process.contains_unsupported_statement ||
                                         process.contains_timing_control;
            auto mark_unsupported = [&](std::string code, std::string message) {
                add_diagnostic(analysis.diagnostics, std::move(code),
                               std::move(message), process.location);
                analysis.synthesis_eligibility = VerilogSynthesisEligibility::Unsupported;
            };
            auto mark_indeterminate = [&](std::string code, std::string message) {
                add_diagnostic(analysis.diagnostics, std::move(code),
                               std::move(message), process.location);
                if (analysis.synthesis_eligibility !=
                    VerilogSynthesisEligibility::Unsupported)
                    analysis.synthesis_eligibility =
                        VerilogSynthesisEligibility::Indeterminate;
            };

            if (unsupported_ast) {
                mark_unsupported("UnsupportedProcessConstruct",
                                 "process contains timing or statement constructs outside the native synthesis subset");
            }
            if (has_unknown_dependencies) {
                mark_indeterminate("UnresolvedProcessDependencies",
                                   "process dependencies or callable side effects could not be resolved conservatively");
            }
            if (has_unknown_events) {
                mark_indeterminate("UnresolvedProcessEvents",
                                   "one or more event expressions could not be resolved to a semantic signal");
            }
            if (has_unsupported_events) {
                mark_unsupported("UnsupportedProcessEventExpression",
                                 "one or more event expressions are outside the supported signal subset");
            }

            if (analysis.synthesis_eligibility !=
                    VerilogSynthesisEligibility::Unsupported &&
                !has_unknown_dependencies && !has_unknown_events) {
                switch (process.kind) {
                    case VerilogProcessKind::ContinuousAssignment:
                        if (analysis.dependencies.knowledge ==
                            VerilogAnalysisKnowledge::Known)
                            analysis.synthesis_eligibility =
                                VerilogSynthesisEligibility::Eligible;
                        break;
                    case VerilogProcessKind::AlwaysComb:
                    case VerilogProcessKind::AlwaysLatch:
                    case VerilogProcessKind::Always: {
                        const bool sequential_intent =
                            process.kind == VerilogProcessKind::Always &&
                            !process.events.empty() &&
                            std::all_of(process.events.begin(), process.events.end(),
                                        [](const auto& event) {
                                            return event.edge != VerilogEventEdge::Any &&
                                                   event.resolution ==
                                                       VerilogAnalysisKnowledge::Known;
                                        });
                        if (sequential_intent) {
                            if (analysis.sequential.knowledge ==
                                VerilogAnalysisKnowledge::Unsupported)
                                mark_unsupported("AmbiguousSequentialStructure",
                                                 "edge-controlled process has ambiguous clock or reset events");
                            else if (analysis.sequential.knowledge ==
                                    VerilogAnalysisKnowledge::Known &&
                                analysis.sequential.clocks.size() == 1)
                                analysis.synthesis_eligibility =
                                    VerilogSynthesisEligibility::Eligible;
                            else
                                mark_indeterminate("UnknownSequentialStructure",
                                                   "edge-controlled process lacks a resolved single-clock/reset structure");
                        } else if (process.kind == VerilogProcessKind::Always &&
                                   std::any_of(process.events.begin(),
                                               process.events.end(),
                                               [](const auto& event) {
                                                   return event.edge !=
                                                          VerilogEventEdge::Any;
                                               })) {
                            mark_unsupported("MixedSequentialEventControl",
                                             "generic always process mixes edge-sensitive and level-sensitive events");
                        } else if (process.kind == VerilogProcessKind::Always &&
                                   process.events.empty() &&
                                   !process.implicit_sensitivity) {
                            mark_unsupported("UnclassifiedAlwaysProcess",
                                             "generic always process has no supported combinational or clocked event structure");
                        } else if (analysis.assignment_coverage.knowledge !=
                                   VerilogAnalysisKnowledge::Known) {
                            mark_indeterminate("UnknownAssignmentCoverage",
                                               "process assignment completeness is not known");
                        } else if (process.kind == VerilogProcessKind::AlwaysLatch) {
                            if (analysis.assignment_coverage.may_require_storage)
                                analysis.synthesis_eligibility =
                                    VerilogSynthesisEligibility::Eligible;
                            else
                                mark_unsupported("AlwaysLatchWithoutLatch",
                                                 "always_latch process does not have an incomplete assignment path");
                        } else if (analysis.assignment_coverage.may_require_storage) {
                            if (process.kind == VerilogProcessKind::AlwaysComb)
                                mark_unsupported("IncompleteAlwaysCombAssignment",
                                                 "always_comb process has a reachable incomplete assignment path");
                            else
                                analysis.synthesis_eligibility =
                                    VerilogSynthesisEligibility::Eligible;
                        } else {
                            bool complete_sensitivity = true;
                            if (process.kind == VerilogProcessKind::Always &&
                                process.has_explicit_event_control &&
                                !process.wildcard_sensitivity) {
                                for (const auto* read : analysis.dependencies.raw_reads) {
                                    const bool listed = std::any_of(
                                        process.events.begin(), process.events.end(),
                                        [&](const auto& event) {
                                            return event.symbol == read;
                                        });
                                    if (!listed) {
                                        complete_sensitivity = false;
                                        add_diagnostic(
                                            analysis.diagnostics,
                                            "IncompleteAlwaysSensitivity",
                                            "generic level-sensitive always event control omits a signal read by the process",
                                            process.location);
                                        break;
                                    }
                                }
                            }
                            analysis.synthesis_eligibility = complete_sensitivity
                                ? VerilogSynthesisEligibility::Eligible
                                : VerilogSynthesisEligibility::Unsupported;
                        }
                        break;
                    }
                    case VerilogProcessKind::AlwaysFF:
                        if (analysis.sequential.knowledge ==
                            VerilogAnalysisKnowledge::Unsupported)
                            mark_unsupported("AmbiguousSequentialStructure",
                                             "always_ff process has ambiguous clock or reset events");
                        else if (analysis.sequential.knowledge ==
                                VerilogAnalysisKnowledge::Known &&
                            analysis.sequential.clocks.size() == 1)
                            analysis.synthesis_eligibility =
                                VerilogSynthesisEligibility::Eligible;
                        else
                            mark_indeterminate("UnknownSequentialStructure",
                                               "always_ff process lacks a resolved single-clock structure");
                        break;
                    case VerilogProcessKind::Initial:
                        mark_indeterminate("InitializationProcessNotLowered",
                                           "initial process is retained for initialization analysis but is not lowered in this milestone");
                        break;
                }
            }
            module_result.processes.push_back(std::move(analysis));
        }

        for (const auto& [symbol, symbol_writers] : writers) {
            if (symbol_writers.size() < 2)
                continue;
            const bool all_continuous = std::all_of(
                symbol_writers.begin(), symbol_writers.end(), [](const auto& writer) {
                    return writer.kind ==
                           VerilogProcessKind::ContinuousAssignment;
                });
            // Multiple continuous drivers on nets have resolution semantics and
            // are left for the net-kind/type policy; procedural sharing conflicts.
            if (all_continuous &&
                (symbol->kind == VerilogSemanticSymbolKind::Net ||
                 is_net_port(symbol, module)))
                continue;
            for (size_t index = 0; index < symbol_writers.size(); ++index) {
                const auto& writer = symbol_writers[index];
                const auto related = index == 0
                    ? std::optional<VerilogSyntaxLocation>{}
                    : std::optional<VerilogSyntaxLocation>{symbol_writers[0].location};
                add_diagnostic(
                    module_result.processes[writer.process_index].diagnostics,
                    "MultipleProcessWriters",
                    "multiple process records write the same module signal",
                    writer.location, related);
                module_result.processes[writer.process_index].synthesis_eligibility =
                    VerilogSynthesisEligibility::Unsupported;
            }
        }

        for (const auto& process : module_result.processes) {
            module_result.diagnostics.insert(module_result.diagnostics.end(),
                                             process.diagnostics.begin(),
                                             process.diagnostics.end());
        }
        result.diagnostics.insert(result.diagnostics.end(),
                                  module_result.diagnostics.begin(),
                                  module_result.diagnostics.end());
        result.modules.push_back(std::move(module_result));
    }
    return result;
}

} // namespace labneura::eda
