#pragma once

#include "labneura/eda/verilog_assignment.h"
#include "labneura/eda/verilog_native_lowering.h"

#include <functional>
#include <optional>
#include <unordered_map>
#include <unordered_set>

namespace labneura::eda::detail {

// Bit vectors are least-significant bit first and contain net names in the
// existing Module representation. Unknown/high-impedance bits are never
// represented here: expressions containing X/Z fail with a diagnostic.
struct VerilogNativeLoweringValue {
    std::vector<std::string> bits;
    bool is_signed = false;
};

using VerilogNativeValueEnvironment =
    std::unordered_map<const VerilogSemanticSymbol*,
                       VerilogNativeLoweringValue>;
using VerilogNativeSymbolResolver =
    std::function<std::optional<VerilogNativeLoweringValue>(
        const VerilogSemanticSymbol*)>;

struct VerilogNativeLoweringSignal {
    const VerilogSemanticSymbol* symbol = nullptr;
    VerilogIntegralType type;
    std::string name;
    std::vector<std::string> bits; // LSB first, respecting declared range.
    std::optional<PortDirection> direction;
};

// Shared lowering contract. It owns only construction data for the existing
// Module/Net/Instance model; it is not a second graph representation.
struct VerilogNativeLoweringSession {
    const VerilogModuleDeclaration& module;
    const VerilogSemanticAnalysisResult& semantic;
    const VerilogTypeAnalysisResult& types;
    const VerilogElaboratedInstance& elaborated_top;
    const VerilogModuleProcessAnalysis& process_analysis;

    std::vector<Port> ports;
    std::vector<Net> nets;
    std::vector<Instance> instances;
    std::vector<VerilogNativeLoweringDiagnostic> diagnostics;
    std::vector<VerilogNativeLoweringSignal> signals;
    std::unordered_map<const VerilogSemanticSymbol*, size_t> signal_indices;
    std::unordered_set<std::string> used_net_names;
    std::unordered_map<std::string, NetType> net_types;
    std::unordered_set<std::string> driven_net_names;
    std::unordered_set<std::string> used_instance_names;
    std::unordered_map<bool, std::string> constant_nets;
    size_t next_net_id = 0;
    size_t next_instance_id = 0;
    bool unsupported = false;
    bool invalid = false;

    VerilogNativeLoweringSession(
        const VerilogModuleDeclaration& module_,
        const VerilogSemanticAnalysisResult& semantic_,
        const VerilogTypeAnalysisResult& types_,
        const VerilogElaboratedInstance& elaborated_top_,
        const VerilogModuleProcessAnalysis& process_analysis_)
        : module(module_), semantic(semantic_), types(types_),
          elaborated_top(elaborated_top_), process_analysis(process_analysis_) {}

    void diagnose(std::string code, std::string message,
                  const VerilogSyntaxLocation& location,
                  std::optional<VerilogSyntaxLocation> related = std::nullopt,
                  bool unsupported_construct = true);
    std::string make_net(std::string_view purpose);
    void add_net(Net net);
    std::string emit_cell(std::string type,
                          std::vector<PinConnection> connections,
                          std::string_view purpose = "value");
    std::string emit_not(const std::string& input);
    std::string emit_binary(std::string_view cell, const std::string& lhs,
                            const std::string& rhs);
    std::string emit_mux(const std::string& when_false,
                         const std::string& when_true,
                         const std::string& select);
    std::string constant(bool value);
    void emit_buffer(const std::string& source, const std::string& destination);
    const VerilogNativeLoweringSignal* signal(
        const VerilogSemanticSymbol* symbol) const;
    VerilogNativeLoweringSignal* signal(
        const VerilogSemanticSymbol* symbol);
    const VerilogSemanticSymbol* resolve_symbol(
        const VerilogExpression& expression) const;
    std::optional<VerilogConstantValue> evaluate_constant(
        const VerilogExpression& expression) const;
    VerilogLvalueResolutionResult resolve_lvalue(
        const VerilogExpression& expression) const;
    std::optional<VerilogNativeLoweringValue> resolve_value(
        const VerilogSemanticSymbol* symbol,
        const VerilogNativeValueEnvironment* environment = nullptr) const;
    Module finalize() &&;
};

std::optional<VerilogNativeLoweringValue> lower_verilog_native_expression(
    VerilogNativeLoweringSession& session,
    const VerilogExpression& expression,
    const VerilogNativeSymbolResolver& resolve_symbol = {},
    std::optional<size_t> context_width = std::nullopt);

// Module-level declaration and continuous-assignment lowering (Agent D).
bool initialize_verilog_native_module(VerilogNativeLoweringSession& session);
bool lower_verilog_native_continuous_assignments(
    VerilogNativeLoweringSession& session);

// Process lowering (Agents B/C). Each routine consumes M7.15 results and
// appends cells/nets directly to the shared existing-netlist builder.
bool lower_verilog_native_combinational_process(
    VerilogNativeLoweringSession& session,
    const VerilogProcessAnalysis& process);
bool lower_verilog_native_sequential_process(
    VerilogNativeLoweringSession& session,
    const VerilogProcessAnalysis& process);

} // namespace labneura::eda::detail
