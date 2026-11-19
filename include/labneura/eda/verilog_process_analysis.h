#pragma once

#include "labneura/eda/verilog_syntax.h"
#include "labneura/eda/verilog_semantic.h"
#include "labneura/eda/verilog_type.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace labneura::eda {

enum class VerilogProcessKind : uint8_t {
    ContinuousAssignment,
    Always,
    AlwaysComb,
    AlwaysFF,
    AlwaysLatch,
    Initial
};

enum class VerilogProcessClassification : uint8_t {
    ContinuousDriver,
    CombinationalCandidate,
    SequentialCandidate,
    LatchCandidate,
    InitializationProcess,
    GenericAlways,
    Unknown
};

enum class VerilogAnalysisKnowledge : uint8_t {
    Known,
    Unsupported,
    Unknown
};

enum class VerilogSynthesisEligibility : uint8_t {
    Eligible,
    Unsupported,
    Indeterminate
};

enum class VerilogAssignmentCoverageState : uint8_t {
    DefinitelyAssigned,
    PossiblyAssigned,
    NotAssigned,
    Unknown
};

struct VerilogProcessDiagnostic {
    std::string code;
    std::string message;
    VerilogSyntaxLocation location;
    std::optional<VerilogSyntaxLocation> related_location;
};

struct VerilogProcessEvent {
    VerilogEventEdge edge = VerilogEventEdge::Any;
    const VerilogExpression* expression = nullptr;
    const VerilogSemanticSymbol* symbol = nullptr;
    VerilogSyntaxLocation location;
    VerilogAnalysisKnowledge resolution = VerilogAnalysisKnowledge::Unknown;
};

enum class VerilogProcessAssignmentKind : uint8_t {
    Continuous,
    Blocking,
    Nonblocking
};

struct VerilogProcessAssignment {
    VerilogProcessAssignmentKind kind = VerilogProcessAssignmentKind::Blocking;
    const VerilogExpression* lvalue = nullptr;
    const VerilogExpression* value = nullptr;
    const VerilogStatement* statement = nullptr;
    VerilogSyntaxLocation location;
};

// Source-order process record. The AST pointers refer to syntax owned by the
// compilation unit and remain valid for the duration of the analysis.
struct VerilogProcess {
    size_t index = 0;
    std::string module_name;
    const VerilogSemanticSymbol* module_symbol = nullptr;
    VerilogProcessKind kind = VerilogProcessKind::Always;
    VerilogProcessClassification classification =
        VerilogProcessClassification::Unknown;
    VerilogSyntaxLocation location;
    const VerilogProceduralBlock* procedural_block = nullptr;
    const VerilogContinuousAssignment* continuous_assignment = nullptr;
    bool has_explicit_event_control = false;
    bool implicit_sensitivity = false;
    bool wildcard_sensitivity = false;
    bool contains_timing_control = false;
    bool contains_unsupported_statement = false;
    std::vector<VerilogProcessEvent> events;
    std::vector<VerilogProcessAssignment> assignments;
    std::vector<VerilogProcessDiagnostic> diagnostics;
};

struct VerilogProcessDiscoveryResult {
    std::vector<VerilogProcess> processes;
    std::vector<VerilogProcessDiagnostic> diagnostics;
};

struct VerilogProcessSymbolAccess {
    const VerilogSemanticSymbol* symbol = nullptr;
    bool read = false;
    bool write = false;
    std::optional<size_t> lsb;
    std::optional<size_t> width;
    bool dynamic_select = false;
    VerilogAnalysisKnowledge resolution = VerilogAnalysisKnowledge::Known;
    VerilogSyntaxLocation location;
};

struct VerilogProcessDependencyResult {
    std::vector<VerilogProcessSymbolAccess> accesses;
    std::vector<const VerilogSemanticSymbol*> raw_reads;
    std::vector<const VerilogSemanticSymbol*> writes;
    std::vector<const VerilogSemanticSymbol*> implicit_sensitivity;
    VerilogAnalysisKnowledge knowledge = VerilogAnalysisKnowledge::Known;
    bool unresolved_callable_side_effects = false;
    std::vector<VerilogProcessDiagnostic> diagnostics;
};

struct VerilogSymbolAssignmentCoverage {
    const VerilogSemanticSymbol* symbol = nullptr;
    size_t bit_width = 0;
    std::vector<VerilogAssignmentCoverageState> bits;
    VerilogAssignmentCoverageState whole_signal =
        VerilogAssignmentCoverageState::Unknown;
};

struct VerilogProcessAssignmentCoverage {
    VerilogAnalysisKnowledge knowledge = VerilogAnalysisKnowledge::Known;
    std::vector<VerilogSymbolAssignmentCoverage> symbols;
    bool may_require_storage = false;
    std::vector<VerilogProcessDiagnostic> diagnostics;
};

struct VerilogSequentialClock {
    const VerilogSemanticSymbol* symbol = nullptr;
    VerilogEventEdge edge = VerilogEventEdge::Any;
    VerilogSyntaxLocation location;
};

struct VerilogSequentialReset {
    const VerilogSemanticSymbol* symbol = nullptr;
    bool active_high = true;
    bool asynchronous = false;
    bool synchronous = false;
    VerilogSyntaxLocation event_location;
    VerilogSyntaxLocation condition_location;
    bool reset_value_is_constant = false;
};

struct VerilogSequentialProcessAnalysis {
    VerilogAnalysisKnowledge knowledge = VerilogAnalysisKnowledge::Unknown;
    std::vector<VerilogSequentialClock> clocks;
    std::vector<VerilogSequentialReset> resets;
    bool has_blocking_assignments = false;
    bool has_nonblocking_assignments = false;
    bool has_ambiguous_clock_or_reset = false;
    bool has_enable_or_hold_pattern = false;
    std::vector<VerilogProcessDiagnostic> diagnostics;
};

struct VerilogProcessAnalysis {
    VerilogProcess process;
    VerilogProcessDependencyResult dependencies;
    VerilogProcessAssignmentCoverage assignment_coverage;
    VerilogSequentialProcessAnalysis sequential;
    VerilogSynthesisEligibility synthesis_eligibility =
        VerilogSynthesisEligibility::Indeterminate;
    std::vector<VerilogProcessDiagnostic> diagnostics;
};

struct VerilogModuleProcessAnalysis {
    std::string module_name;
    const VerilogSemanticSymbol* module_symbol = nullptr;
    std::vector<VerilogProcessAnalysis> processes;
    std::vector<VerilogProcessDiagnostic> diagnostics;
};

struct VerilogSynthesisEligibilityResult {
    std::vector<VerilogModuleProcessAnalysis> modules;
    std::vector<VerilogProcessDiagnostic> diagnostics;
};

VerilogProcessDiscoveryResult discover_verilog_processes(
    const VerilogModuleDeclaration& module,
    const VerilogSemanticAnalysisResult& semantic);

VerilogProcessDependencyResult analyze_verilog_process_dependencies(
    const VerilogProcess& process,
    const VerilogCompilationUnit& syntax,
    const VerilogSemanticAnalysisResult& semantic,
    const VerilogTypeAnalysisResult& type_analysis);

VerilogProcessAssignmentCoverage analyze_verilog_assignment_coverage(
    const VerilogProcess& process,
    const VerilogSemanticAnalysisResult& semantic,
    const VerilogTypeAnalysisResult& type_analysis);

VerilogSequentialProcessAnalysis analyze_verilog_sequential_process(
    const VerilogProcess& process,
    const VerilogSemanticAnalysisResult& semantic,
    const VerilogTypeAnalysisResult& type_analysis,
    const VerilogProcessDependencyResult& dependencies);

VerilogSynthesisEligibilityResult analyze_verilog_synthesis_eligibility(
    const VerilogCompilationUnit& syntax,
    const VerilogSemanticAnalysisResult& semantic,
    const VerilogTypeAnalysisResult& type_analysis);

} // namespace labneura::eda
