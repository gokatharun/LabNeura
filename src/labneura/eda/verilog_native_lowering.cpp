#include "labneura/eda/verilog_native_lowering_internal.h"

#include "labneura/eda/verilog_constant.h"

#include <algorithm>
#include <utility>

namespace labneura::eda::detail {

void VerilogNativeLoweringSession::diagnose(
    std::string code, std::string message,
    const VerilogSyntaxLocation& location,
    std::optional<VerilogSyntaxLocation> related,
    bool unsupported_construct) {
    diagnostics.push_back(
        {std::move(code), std::move(message), location, std::move(related)});
    if (unsupported_construct)
        unsupported = true;
    else
        invalid = true;
}

std::string VerilogNativeLoweringSession::make_net(std::string_view purpose) {
    for (;;) {
        auto name = "__native_" + std::string(purpose) +
                    std::to_string(next_net_id++);
        if (used_net_names.insert(name).second)
            return name;
    }
}

void VerilogNativeLoweringSession::add_net(Net net) {
    const auto& name = net.get_name();
    const auto found = net_types.find(name);
    if (found != net_types.end()) {
        if (found->second != net.get_kind())
            diagnose("ConflictingNativeNetDeclaration",
                     "native lowering produced incompatible declarations for net '" +
                         name + "'",
                     module.location);
        return;
    }
    used_net_names.insert(name);
    net_types.emplace(name, net.get_kind());
    nets.push_back(std::move(net));
}

std::string VerilogNativeLoweringSession::emit_cell(
    std::string type, std::vector<PinConnection> connections,
    std::string_view purpose) {
    const auto name = make_net(purpose);
    add_net(Net(name, NetType::Internal));

    if (!used_instance_names.insert("__native_cell_" +
                                    std::to_string(next_instance_id)).second) {
        diagnose("NativeInstanceNameCollision",
                 "deterministic native cell instance naming collided",
                 module.location, std::nullopt, false);
        return name;
    }
    const auto instance_name = "__native_cell_" +
                               std::to_string(next_instance_id++);
    connections.emplace_back("Y", name);
    instances.emplace_back(instance_name, std::move(type),
                           std::move(connections));
    driven_net_names.insert(name);
    return name;
}

std::string VerilogNativeLoweringSession::emit_not(const std::string& input) {
    return emit_cell("INV", {{"A", input}}, "not");
}

std::string VerilogNativeLoweringSession::emit_binary(
    std::string_view cell, const std::string& lhs, const std::string& rhs) {
    return emit_cell(std::string(cell), {{"A", lhs}, {"B", rhs}}, "gate");
}

std::string VerilogNativeLoweringSession::emit_mux(
    const std::string& when_false, const std::string& when_true,
    const std::string& select) {
    return emit_cell("MUX2", {{"A0", when_false}, {"A1", when_true},
                               {"S", select}}, "mux");
}

std::string VerilogNativeLoweringSession::constant(bool value) {
    if (const auto found = constant_nets.find(value);
        found != constant_nets.end())
        return found->second;

    const auto name = make_net(value ? "one" : "zero");
    add_net(Net(name, NetType::Internal));
    if (!used_instance_names.insert("__native_cell_" +
                                    std::to_string(next_instance_id)).second) {
        diagnose("NativeInstanceNameCollision",
                 "deterministic native cell instance naming collided",
                 module.location, std::nullopt, false);
        return name;
    }
    const auto instance_name = "__native_cell_" +
                               std::to_string(next_instance_id++);
    instances.emplace_back(instance_name, value ? "TIEHI" : "TIELO",
                           std::vector<PinConnection>{{"Y", name}});
    driven_net_names.insert(name);
    constant_nets.emplace(value, name);
    return name;
}

void VerilogNativeLoweringSession::emit_buffer(
    const std::string& source, const std::string& destination) {
    if (source == destination)
        return;
    if (!net_types.contains(destination)) {
        diagnose("UnresolvedNativeAssignmentDestination",
                 "native assignment destination net '" + destination +
                     "' is not declared",
                 module.location);
        return;
    }
    if (!used_instance_names.insert("__native_cell_" +
                                    std::to_string(next_instance_id)).second) {
        diagnose("NativeInstanceNameCollision",
                 "deterministic native cell instance naming collided",
                 module.location, std::nullopt, false);
        return;
    }
    const auto instance_name = "__native_cell_" +
                               std::to_string(next_instance_id++);
    instances.emplace_back(instance_name, "BUF",
                           std::vector<PinConnection>{{"A", source},
                                                      {"Y", destination}});
    if (!driven_net_names.insert(destination).second)
        diagnose("MultipleNativeDrivers",
                 "native lowering attempted to drive net '" + destination +
                     "' more than once",
                 module.location);
}

const VerilogNativeLoweringSignal* VerilogNativeLoweringSession::signal(
    const VerilogSemanticSymbol* symbol) const {
    const auto found = signal_indices.find(symbol);
    return found == signal_indices.end() ? nullptr : &signals[found->second];
}

VerilogNativeLoweringSignal* VerilogNativeLoweringSession::signal(
    const VerilogSemanticSymbol* symbol) {
    const auto found = signal_indices.find(symbol);
    return found == signal_indices.end() ? nullptr : &signals[found->second];
}

const VerilogSemanticSymbol* VerilogNativeLoweringSession::resolve_symbol(
    const VerilogExpression& expression) const {
    for (const auto& entry : semantic.references_at(expression.location)) {
        const auto& reference = semantic.references[entry.reference_ordinal];
        if (reference.declaration)
            return reference.declaration;
    }
    return nullptr;
}

std::optional<VerilogConstantValue>
VerilogNativeLoweringSession::evaluate_constant(
    const VerilogExpression& expression) const {
    const auto resolver = [&](std::string_view name,
                              const VerilogSyntaxLocation& location)
        -> std::optional<VerilogConstantValue> {
        for (const auto& entry : semantic.references_at(location)) {
            const auto& reference = semantic.references[entry.reference_ordinal];
            if (reference.name != name || !reference.declaration)
                continue;
            if (const auto constant_value =
                    types.symbol_constants.find(reference.declaration);
                constant_value != types.symbol_constants.end())
                return constant_value->second;
            if (const auto parameter =
                    elaborated_top.parameters.find(reference.declaration->name);
                parameter != elaborated_top.parameters.end())
                return parameter->second;
        }
        const auto parameter = elaborated_top.parameters.find(std::string(name));
        if (parameter != elaborated_top.parameters.end())
            return parameter->second;
        return std::nullopt;
    };
    const auto result = evaluate_verilog_constant(expression, {}, resolver);
    return result.value;
}

VerilogLvalueResolutionResult VerilogNativeLoweringSession::resolve_lvalue(
    const VerilogExpression& expression) const {
    return resolve_verilog_lvalue(
        expression, semantic, types,
        [&](const VerilogExpression& index) { return evaluate_constant(index); });
}

std::optional<VerilogNativeLoweringValue>
VerilogNativeLoweringSession::resolve_value(
    const VerilogSemanticSymbol* symbol,
    const VerilogNativeValueEnvironment* environment) const {
    if (environment) {
        const auto value = environment->find(symbol);
        if (value != environment->end())
            return value->second;
    }
    const auto* declared = signal(symbol);
    if (!declared)
        return std::nullopt;
    return VerilogNativeLoweringValue{declared->bits,
                                      declared->type.is_signed};
}

Module VerilogNativeLoweringSession::finalize() && {
    return Module(module.name, std::move(ports), std::move(nets),
                  std::move(instances), {}, false, {}, {}, std::move(memories));
}

} // namespace labneura::eda::detail

namespace labneura::eda {
namespace {

VerilogNativeLoweringDiagnostic invalid_diagnostic(std::string code,
                                                  std::string message,
                                                  VerilogSyntaxLocation location) {
    return {std::move(code), std::move(message), std::move(location),
            std::nullopt};
}

const VerilogModuleDeclaration* find_module(
    const VerilogCompilationUnit& syntax, std::string_view name) {
    const auto found = std::find_if(syntax.modules.begin(), syntax.modules.end(),
                                    [&](const auto& module) {
                                        return module.name == name;
                                    });
    return found == syntax.modules.end() ? nullptr : &*found;
}

const VerilogModuleProcessAnalysis* find_process_analysis(
    const VerilogSynthesisEligibilityResult& analysis, std::string_view name) {
    const auto found = std::find_if(analysis.modules.begin(), analysis.modules.end(),
                                    [&](const auto& module) {
                                        return module.module_name == name;
                                    });
    return found == analysis.modules.end() ? nullptr : &*found;
}

} // namespace

VerilogNativeLoweringResult lower_verilog_native(
    const VerilogCompilationUnit& syntax,
    const VerilogElaborationResult& elaboration,
    const VerilogTypeAnalysisResult& type_analysis,
    const VerilogSynthesisEligibilityResult& process_analysis,
    std::string top_module) {
    VerilogNativeLoweringResult result;
    if (!elaboration.top || !elaboration.top->module_definition) {
        result.diagnostics.push_back(invalid_diagnostic(
            "MissingElaboratedTop",
            "native RTL lowering requires a successful native elaboration top",
            syntax.location));
        result.status = VerilogNativeLoweringStatus::InvalidInput;
        return result;
    }
    if (top_module.empty())
        top_module = elaboration.top->module_definition->name;
    if (elaboration.top->module_definition->name != top_module) {
        result.diagnostics.push_back(invalid_diagnostic(
            "TopModuleMismatch",
            "requested top module does not match the supplied elaboration result",
            elaboration.top->location));
        result.status = VerilogNativeLoweringStatus::InvalidInput;
        return result;
    }
    const auto* module = find_module(syntax, top_module);
    const auto* process_result = find_process_analysis(process_analysis, top_module);
    if (!module || !process_result) {
        result.diagnostics.push_back(invalid_diagnostic(
            "MissingNativeAnalysisInput",
            "syntax or process analysis does not contain the elaborated top module",
            elaboration.top->location));
        result.status = VerilogNativeLoweringStatus::InvalidInput;
        return result;
    }
    if (!elaboration.diagnostics.empty() || type_analysis.has_errors()) {
        result.diagnostics.push_back(invalid_diagnostic(
            "InvalidNativeAnalysisInput",
            "native lowering requires error-free elaboration and type analysis",
            elaboration.top->location));
        result.status = VerilogNativeLoweringStatus::InvalidInput;
        return result;
    }
    if (process_result->module_symbol &&
        process_result->module_symbol != elaboration.top->module_symbol) {
        result.diagnostics.push_back(invalid_diagnostic(
            "MismatchedSemanticModel",
            "process and elaboration inputs do not share semantic symbol identity",
            elaboration.top->location));
        result.status = VerilogNativeLoweringStatus::InvalidInput;
        return result;
    }

    detail::VerilogNativeLoweringSession session(
        *module, elaboration.semantic, type_analysis, *elaboration.top,
        *process_result);
    if (!elaboration.top->children.empty() ||
        !elaboration.top->generate_scopes.empty() ||
        !module->instantiations.empty() ||
        !module->generate_constructs.empty() ||
        !elaboration.top->generated_assignments.empty() ||
        !elaboration.top->generated_declarations.empty()) {
        session.diagnose(
            "UnsupportedNativeHierarchy",
            "native lowering currently accepts one flat elaborated module without instances or generate scopes",
            module->location);
    }
    if (process_result->processes.size() == 0 &&
        module->assignments.empty() && module->procedural_blocks.empty()) {
        // A declaration-only module is still useful, but outputs will be
        // checked for drivers after declarations are initialized.
    }

    if (!session.unsupported && !session.invalid) {
        (void)detail::initialize_verilog_native_module(session);
        if (!session.unsupported && !session.invalid)
            (void)detail::lower_verilog_native_continuous_assignments(session);
        if (!session.unsupported && !session.invalid) {
            for (const auto& process : process_result->processes) {
                if (process.process.kind ==
                    VerilogProcessKind::ContinuousAssignment)
                    continue;
                if (const auto memory =
                        detail::lower_verilog_native_memory_process(session, process)) {
                    if (!*memory)
                        break;
                    continue;
                }
                if (process.synthesis_eligibility !=
                    VerilogSynthesisEligibility::Eligible) {
                    session.diagnose(
                        "ProcessNotEligibleForNativeLowering",
                        "M7.15 process eligibility is required before native lowering",
                        process.process.location);
                    for (const auto& diagnostic : process.diagnostics)
                        session.diagnostics.push_back(
                            {diagnostic.code, diagnostic.message,
                             diagnostic.location,
                             diagnostic.related_location});
                    break;
                }
                if (process.process.kind == VerilogProcessKind::AlwaysFF ||
                    (process.process.kind == VerilogProcessKind::Always &&
                     !process.process.events.empty() &&
                     std::all_of(process.process.events.begin(),
                                 process.process.events.end(),
                                 [](const auto& event) {
                                     return event.edge != VerilogEventEdge::Any;
                                 }))) {
                    (void)detail::lower_verilog_native_sequential_process(
                        session, process);
                } else {
                    (void)detail::lower_verilog_native_combinational_process(
                        session, process);
                }
                if (session.unsupported || session.invalid)
                    break;
            }
        }
    }

    if (!session.unsupported && !session.invalid) {
        for (const auto& [net_name, net_type] : session.net_types) {
            if (net_type == NetType::Input &&
                session.driven_net_names.contains(net_name)) {
                session.diagnose("NativeInputPortDriven",
                                 "native lowering attempted to drive input port bit '" +
                                     net_name + "'",
                                 module->location);
            }
            if (net_type == NetType::Output &&
                !session.driven_net_names.contains(net_name)) {
                session.diagnose("UndrivenNativeOutput",
                                 "native lowering found undriven output bit '" +
                                     net_name + "'",
                                 module->location);
            }
        }
        for (const auto& instance : session.instances) {
            for (const auto& connection : instance.get_connections()) {
                if (connection.get_pin() == "Y")
                    continue;
                const auto net = session.net_types.find(connection.get_net());
                if (net == session.net_types.end()) {
                    session.diagnose("UnresolvedNativeCellInput",
                                     "cell input references undeclared net '" +
                                         connection.get_net() + "'",
                                     module->location);
                    continue;
                }
                if (net->second != NetType::Input &&
                    !session.driven_net_names.contains(connection.get_net()))
                    session.diagnose("UndrivenNativeCellInput",
                                     "cell input references undriven net '" +
                                         connection.get_net() + "'",
                                     module->location);
            }
        }
    }

    result.diagnostics = std::move(session.diagnostics);
    if (session.invalid) {
        result.status = VerilogNativeLoweringStatus::InvalidInput;
        return result;
    }
    if (session.unsupported || !result.diagnostics.empty()) {
        result.status = VerilogNativeLoweringStatus::Unsupported;
        return result;
    }
    result.module.emplace(std::move(session).finalize());
    result.status = VerilogNativeLoweringStatus::Succeeded;
    return result;
}

} // namespace labneura::eda
