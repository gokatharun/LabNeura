#include "support/eda_sequential_evaluator.h"

#include <algorithm>
#include <map>
#include <set>
#include <string_view>

namespace labneura::eda::test {
namespace {

using RegisterCells = std::map<std::string, const Instance*>;

bool sequential_type(std::string_view type) {
    return type == "DFF" || type == "DFFR" || type == "DFFS";
}

void add_diagnostic(SequentialEvaluationResult& result,
                    NetlistVerificationStatus status, std::string code,
                    std::string message,
                    std::optional<std::string> net = std::nullopt,
                    std::optional<std::string> instance = std::nullopt) {
    result.status = status;
    result.diagnostics.push_back({std::move(code), std::move(message),
                                  std::move(net), std::move(instance)});
}

std::optional<std::string> pin_net(const Instance& instance,
                                   std::string_view pin) {
    const PinConnection* found = nullptr;
    for (const auto& connection : instance.get_connections()) {
        if (connection.get_pin() != pin)
            continue;
        if (found)
            return std::nullopt;
        found = &connection;
    }
    if (!found || found->get_net().empty())
        return std::nullopt;
    return found->get_net();
}

bool valid_register_pins(const Instance& instance) {
    std::set<std::string> expected{"D", "CLK", "Y"};
    if (instance.get_cell_type() == "DFFR")
        expected.insert("RESET_B");
    else if (instance.get_cell_type() == "DFFS")
        expected.insert("SET_B");

    std::set<std::string> actual;
    for (const auto& connection : instance.get_connections()) {
        if (!actual.insert(connection.get_pin()).second)
            return false;
    }
    if (actual != expected)
        return false;
    for (const auto& pin : expected)
        if (!pin_net(instance, pin))
            return false;
    return true;
}

std::optional<std::uint8_t> net_bit(const NetlistEvaluationResult& evaluation,
                                    const std::string& net) {
    const auto found = evaluation.net_values.find(net);
    if (found == evaluation.net_values.end())
        return std::nullopt;
    return found->second;
}

NetlistEvaluationResult evaluate_frame(
    const Module& module, const VerificationPortValues& inputs,
    const RegisterCells& registers,
    const std::map<std::string, std::uint8_t>& state,
    std::size_t max_cell_evaluations) {
    CombinationalEvaluationOptions options;
    options.allow_sequential_cutpoints = true;
    options.max_cell_evaluations = max_cell_evaluations;
    for (const auto& [name, instance] : registers) {
        const auto state_value = state.find(name);
        const auto output = pin_net(*instance, "Y");
        if (state_value == state.end() || !output) {
            NetlistEvaluationResult result;
            result.status = NetlistVerificationStatus::InvalidNetlist;
            result.diagnostics.push_back(
                {"InvalidSequentialState", "register state or Y pin is missing",
                 output, name});
            return result;
        }
        options.cutpoint_values.emplace(*output, state_value->second);
    }
    return evaluate_combinational_netlist(module, inputs, options);
}

SequentialEvaluationResult failed_from(
    const NetlistEvaluationResult& evaluation) {
    SequentialEvaluationResult result;
    result.status = evaluation.status;
    result.diagnostics = evaluation.diagnostics;
    return result;
}

} // namespace

SequentialEvaluationResult evaluate_sequential_netlist(
    const Module& module, const std::vector<SequentialStimulusFrame>& frames,
    const SequentialEvaluationOptions& options) {
    SequentialEvaluationResult result;
    if (frames.empty()) {
        add_diagnostic(result, NetlistVerificationStatus::InvalidStimulus,
                       "EmptySequentialStimulus",
                       "sequential evaluation requires an initial input frame");
        return result;
    }
    if (options.max_frames == 0 || frames.size() > options.max_frames) {
        add_diagnostic(result, NetlistVerificationStatus::ResourceLimit,
                       "SequentialFrameLimit",
                       "sequential stimulus exceeds the configured frame limit");
        return result;
    }
    if (options.max_cell_evaluations_per_frame == 0) {
        add_diagnostic(result, NetlistVerificationStatus::ResourceLimit,
                       "SequentialCellLimit",
                       "sequential cell evaluation limit must be positive");
        return result;
    }

    RegisterCells registers;
    std::set<std::string> all_instance_names;
    for (const auto& instance : module.get_instances()) {
        if (!all_instance_names.insert(instance.get_name()).second) {
            add_diagnostic(result, NetlistVerificationStatus::InvalidNetlist,
                           "DuplicateInstanceName",
                           "netlist contains duplicate instance names",
                           std::nullopt, instance.get_name());
            return result;
        }
        const auto& type = instance.get_cell_type();
        if (sequential_type(type)) {
            if (!valid_register_pins(instance)) {
                add_diagnostic(result, NetlistVerificationStatus::InvalidNetlist,
                               "MalformedRegisterPins",
                               "register cell has missing, duplicate, or unsupported pins",
                               std::nullopt, instance.get_name());
                return result;
            }
            registers.emplace(instance.get_name(), &instance);
        } else if (type == "DFFSR" || instance.is_sequential_primitive()) {
            add_diagnostic(result, NetlistVerificationStatus::UnsupportedCell,
                           "UnsupportedSequentialCell",
                           "sequential evaluator supports DFF, DFFR, and DFFS only",
                           std::nullopt, instance.get_name());
            return result;
        }
    }
    if (registers.empty()) {
        add_diagnostic(result, NetlistVerificationStatus::InvalidNetlist,
                       "NoRegisterCells",
                       "sequential evaluation requires at least one supported register");
        return result;
    }
    if (options.initial_state.size() != registers.size()) {
        add_diagnostic(result, NetlistVerificationStatus::InvalidStimulus,
                       "IncompleteInitialState",
                       "initial state must name every register instance exactly once");
        return result;
    }

    auto state = options.initial_state;
    for (const auto& [name, instance] : registers) {
        (void)instance;
        const auto value = state.find(name);
        if (value == state.end() || value->second > 1) {
            add_diagnostic(result, NetlistVerificationStatus::InvalidStimulus,
                           "InvalidInitialState",
                           "initial state must provide a two-state value for every register",
                           std::nullopt, name);
            return result;
        }
    }
    for (const auto& [name, value] : state) {
        (void)value;
        if (!registers.contains(name)) {
            add_diagnostic(result, NetlistVerificationStatus::InvalidStimulus,
                           "UnknownInitialStateRegister",
                           "initial state names an unknown register instance",
                           std::nullopt, name);
            return result;
        }
    }

    auto previous = evaluate_frame(module, frames.front().inputs, registers,
                                   state,
                                   options.max_cell_evaluations_per_frame);
    if (!previous.succeeded())
        return failed_from(previous);

    // The first frame establishes a stable input level. An asynchronous
    // reset/set already asserted at that level forces the corresponding
    // registers before the initial trace observation; no clock edge is
    // required for this level-sensitive behavior.
    auto initial_state = state;
    for (const auto& [name, instance] : registers) {
        std::optional<std::string> reset_net;
        if (instance->get_cell_type() == "DFFR")
            reset_net = pin_net(*instance, "RESET_B");
        else if (instance->get_cell_type() == "DFFS")
            reset_net = pin_net(*instance, "SET_B");
        if (!reset_net)
            continue;
        const auto reset_value = net_bit(previous, *reset_net);
        if (!reset_value) {
            add_diagnostic(result, NetlistVerificationStatus::UndrivenNet,
                           "UnresolvedRegisterReset",
                           "asynchronous reset or set input was not evaluated",
                           reset_net, name);
            return result;
        }
        if (*reset_value == 0)
            initial_state[name] = instance->get_cell_type() == "DFFR" ? 0 : 1;
    }
    if (initial_state != state) {
        state = std::move(initial_state);
        previous = evaluate_frame(module, frames.front().inputs, registers,
                                  state,
                                  options.max_cell_evaluations_per_frame);
        if (!previous.succeeded())
            return failed_from(previous);
    }
    result.trace.push_back({previous.outputs, state});

    for (std::size_t frame_index = 1; frame_index < frames.size(); ++frame_index) {
        // Evaluate every D/reset/clock cone from the same old state. Pending
        // register updates are collected before any state value is changed.
        auto current = evaluate_frame(module, frames[frame_index].inputs,
                                      registers, state,
                                      options.max_cell_evaluations_per_frame);
        if (!current.succeeded())
            return failed_from(current);

        auto next_state = state;
        for (const auto& [name, instance] : registers) {
            const auto d_net = pin_net(*instance, "D");
            const auto clock_net = pin_net(*instance, "CLK");
            if (!d_net || !clock_net) {
                add_diagnostic(result, NetlistVerificationStatus::InvalidNetlist,
                               "MalformedRegisterPins",
                               "register D or CLK pin is unresolved",
                               std::nullopt, name);
                return result;
            }
            const auto clock_before = net_bit(previous, *clock_net);
            const auto clock_after = net_bit(current, *clock_net);
            const auto data = net_bit(current, *d_net);
            if (!clock_before || !clock_after || !data) {
                add_diagnostic(result, NetlistVerificationStatus::UndrivenNet,
                               "UnresolvedRegisterInput",
                               "register clock or data input was not evaluated",
                               !data ? d_net : clock_net, name);
                return result;
            }

            bool reset_active = false;
            std::optional<std::string> reset_net;
            if (instance->get_cell_type() == "DFFR")
                reset_net = pin_net(*instance, "RESET_B");
            else if (instance->get_cell_type() == "DFFS")
                reset_net = pin_net(*instance, "SET_B");
            if (reset_net) {
                const auto reset_after = net_bit(current, *reset_net);
                if (!reset_after) {
                    add_diagnostic(result, NetlistVerificationStatus::UndrivenNet,
                                   "UnresolvedRegisterReset",
                                   "asynchronous reset or set input was not evaluated",
                                   reset_net, name);
                    return result;
                }
                reset_active = *reset_after == 0;
            }

            // DFFR/DFFS asynchronously force their state whenever asserted.
            // Deassertion by itself does not alter state; a later active clock
            // edge samples D.
            if (reset_active) {
                next_state[name] = instance->get_cell_type() == "DFFR" ? 0 : 1;
            } else if (*clock_before == 0 && *clock_after == 1) {
                next_state[name] = *data;
            }
        }
        state = std::move(next_state);

        // The reported post-transition outputs reflect all simultaneously
        // committed register changes, not the pre-edge cutpoint values.
        auto after = evaluate_frame(module, frames[frame_index].inputs,
                                    registers, state,
                                    options.max_cell_evaluations_per_frame);
        if (!after.succeeded())
            return failed_from(after);
        result.trace.push_back({after.outputs, state});
        previous = std::move(after);
    }

    result.status = NetlistVerificationStatus::Succeeded;
    return result;
}

} // namespace labneura::eda::test
