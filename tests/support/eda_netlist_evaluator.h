#pragma once

#include "labneura/eda/netlist.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

// Test-only, two-state evaluation contract for the existing generic netlist.
// Every bit vector is least-significant bit first. Port vector bit zero maps
// to the port's declared right bound (or bit index zero for an implicit range).
namespace labneura::eda::test {

using VerificationBits = std::vector<std::uint8_t>;
using VerificationPortValues = std::map<std::string, VerificationBits>;
using VerificationNetValues = std::map<std::string, std::uint8_t>;

enum class NetlistVerificationStatus : std::uint8_t {
    Succeeded,
    InvalidStimulus,
    InvalidNetlist,
    MultipleDrivers,
    UndrivenNet,
    CombinationalCycle,
    UnsupportedCell,
    ResourceLimit
};

struct NetlistVerificationDiagnostic {
    std::string code;
    std::string message;
    std::optional<std::string> net;
    std::optional<std::string> instance;
};

struct CombinationalEvaluationOptions {
    // DFF/DFFR/DFFS are accepted only as explicit state cutpoints. Their Y
    // outputs must have a supplied value here; all combinational cells remain
    // fully evaluated and validated.
    bool allow_sequential_cutpoints = false;
    VerificationNetValues cutpoint_values;
    std::size_t max_cell_evaluations = 100000;
};

struct NetlistEvaluationResult {
    NetlistVerificationStatus status = NetlistVerificationStatus::InvalidNetlist;
    VerificationPortValues outputs;
    VerificationNetValues net_values;
    std::vector<NetlistVerificationDiagnostic> diagnostics;

    bool succeeded() const noexcept {
        return status == NetlistVerificationStatus::Succeeded;
    }
};

enum class VerificationKind : std::uint8_t {
    ExhaustiveCombinational,
    SampledAgreement,
    BoundedSequential,
    FormalCombinational,
    FormalSequential,
    Structural
};

enum class VerificationOutcome : std::uint8_t {
    Proven,
    Agreed,
    Disagreed,
    Unsupported,
    Unavailable,
    TimedOut,
    Invalid
};

struct VerificationCounterexample {
    std::size_t vector_or_frame = 0;
    VerificationPortValues inputs;
    VerificationPortValues expected_outputs;
    VerificationPortValues actual_outputs;
};

// Shared evidence vocabulary: Proven is reserved for a successful formal
// proof; finite exhaustive, sampled, and bounded results use Agreed.
struct VerificationEvidence {
    VerificationKind kind = VerificationKind::Structural;
    VerificationOutcome outcome = VerificationOutcome::Invalid;
    std::size_t checked_vectors = 0;
    std::size_t checked_frames = 0;
    std::uint64_t random_seed = 0;
    std::optional<VerificationCounterexample> counterexample;
    std::vector<NetlistVerificationDiagnostic> diagnostics;
};

// Returns the port's scalar net names in LSB-first order using its declared
// packed range and the bit-net naming already present in the Module.
std::vector<std::string> module_port_bit_nets(const Module& module,
                                             const Port& port);

// Evaluates only the documented two-state combinational cell subset. Missing
// inputs, malformed pins, multiple drivers, undriven reads/outputs, cycles,
// unsupported cells, and resource exhaustion are explicit outcomes.
NetlistEvaluationResult evaluate_combinational_netlist(
    const Module& module, const VerificationPortValues& inputs,
    const CombinationalEvaluationOptions& options = {});

} // namespace labneura::eda::test
