#include "support/eda_netlist_evaluator.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace labneura::eda::test {
namespace {

struct CellPlan {
    const Instance* instance = nullptr;
    std::string type;
    std::map<std::string, std::string> pins;
    std::string output;
    std::vector<std::string> inputs;
    bool sequential_cutpoint = false;
};

void diagnose(NetlistEvaluationResult& result,
              NetlistVerificationStatus status, std::string code,
              std::string message,
              std::optional<std::string> net = std::nullopt,
              std::optional<std::string> instance = std::nullopt) {
    result.diagnostics.push_back(
        {std::move(code), std::move(message), std::move(net),
         std::move(instance)});
    if (result.status == NetlistVerificationStatus::InvalidNetlist ||
        result.status == NetlistVerificationStatus::Succeeded)
        result.status = status;
}

std::optional<std::size_t> packed_range_width(std::int64_t left,
                                              std::int64_t right) {
    const auto distance = left >= right
        ? static_cast<std::uint64_t>(left) - static_cast<std::uint64_t>(right)
        : static_cast<std::uint64_t>(right) - static_cast<std::uint64_t>(left);
    if (distance == std::numeric_limits<std::uint64_t>::max())
        return std::nullopt;
    const auto width = distance + 1;
    if (width > std::numeric_limits<std::size_t>::max()) return std::nullopt;
    return static_cast<std::size_t>(width);
}

std::optional<std::int64_t> packed_index_at(std::int64_t right, int step,
                                           std::size_t bit) {
    if (bit > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()))
        return std::nullopt;
    const auto offset = static_cast<std::int64_t>(bit);
    if (step > 0) {
        if (right > std::numeric_limits<std::int64_t>::max() - offset)
            return std::nullopt;
        return right + offset;
    }
    if (right < std::numeric_limits<std::int64_t>::min() + offset)
        return std::nullopt;
    return right - offset;
}

std::set<std::string> declared_net_names(const Module& module) {
    std::set<std::string> declared;
    for (const auto& net : module.get_nets()) {
        if (net.get_width() <= 1) {
            declared.insert(net.get_name());
            continue;
        }
        const auto left = net.get_left_index().value_or(
            static_cast<std::int64_t>(net.get_width() - 1));
        const auto right = net.get_right_index().value_or(0);
        const auto width = packed_range_width(left, right);
        if (!width || net.get_width() != *width) {
            declared.insert(net.get_name());
            continue;
        }
        const auto step = left >= right ? 1 : -1;
        for (std::size_t bit = 0; bit < net.get_width(); ++bit) {
            const auto index = packed_index_at(right, step, bit);
            if (!index) break;
            declared.insert(net.get_name() + "[" + std::to_string(*index) + "]");
        }
    }
    for (const auto& port : module.get_ports()) {
        const auto bits = module_port_bit_nets(module, port);
        declared.insert(bits.begin(), bits.end());
    }
    return declared;
}

bool is_sequential_cell(std::string_view type) {
    return type == "DFF" || type == "DFFR" || type == "DFFS";
}

bool has_exact_pins(const CellPlan& cell,
                    const std::vector<std::string>& expected) {
    if (cell.pins.size() != expected.size()) return false;
    return std::all_of(expected.begin(), expected.end(), [&](const auto& pin) {
        return cell.pins.contains(pin);
    });
}

bool make_cell_plan(const Instance& instance, bool allow_sequential,
                    CellPlan& plan, NetlistEvaluationResult& result) {
    plan.instance = &instance;
    plan.type = instance.get_cell_type();
    if (instance.is_user_defined_primitive()) {
        diagnose(result, NetlistVerificationStatus::UnsupportedCell,
                 "UnsupportedUserPrimitive",
                 "test evaluator does not model user-defined primitives",
                 std::nullopt, instance.get_name());
        return false;
    }
    if (!instance.get_parameters().empty()) {
        diagnose(result, NetlistVerificationStatus::InvalidNetlist,
                 "UnexpectedCellParameters",
                 "supported generic verification cells do not accept parameters",
                 std::nullopt, instance.get_name());
        return false;
    }

    for (const auto& connection : instance.get_connections()) {
        if (connection.get_pin().empty() || connection.get_net().empty()) {
            diagnose(result, NetlistVerificationStatus::InvalidNetlist,
                     "EmptyPinConnection",
                     "cell pin and net names must be nonempty", std::nullopt,
                     instance.get_name());
            return false;
        }
        if (!plan.pins.emplace(connection.get_pin(), connection.get_net()).second) {
            diagnose(result, NetlistVerificationStatus::InvalidNetlist,
                     "DuplicateCellPin",
                     "cell has duplicate connection for pin '" +
                         connection.get_pin() + "'",
                     connection.get_net(), instance.get_name());
            return false;
        }
    }

    std::vector<std::string> expected;
    if (plan.type == "AND2" || plan.type == "OR2" || plan.type == "XOR2") {
        expected = {"A", "B", "Y"};
        plan.inputs = {"A", "B"};
    } else if (plan.type == "INV" || plan.type == "BUF") {
        expected = {"A", "Y"};
        plan.inputs = {"A"};
    } else if (plan.type == "MUX2") {
        expected = {"A0", "A1", "S", "Y"};
        plan.inputs = {"A0", "A1", "S"};
    } else if (plan.type == "TIEHI" || plan.type == "TIELO") {
        expected = {"Y"};
    } else if (is_sequential_cell(plan.type)) {
        if (!allow_sequential) {
            diagnose(result, NetlistVerificationStatus::UnsupportedCell,
                     "SequentialCellUnsupported",
                     "combinational evaluator requires explicit sequential cutpoints",
                     std::nullopt, instance.get_name());
            return false;
        }
        expected = plan.type == "DFF"
                       ? std::vector<std::string>{"D", "CLK", "Y"}
                       : std::vector<std::string>{
                             "D", "CLK",
                             plan.type == "DFFR" ? "RESET_B" : "SET_B", "Y"};
        plan.inputs = {"D", "CLK"};
        if (plan.type == "DFFR") plan.inputs.push_back("RESET_B");
        if (plan.type == "DFFS") plan.inputs.push_back("SET_B");
        plan.sequential_cutpoint = true;
    } else {
        diagnose(result, NetlistVerificationStatus::UnsupportedCell,
                 "UnsupportedCell",
                 "test evaluator does not support generic cell '" + plan.type +
                     "'",
                 std::nullopt, instance.get_name());
        return false;
    }

    if (!has_exact_pins(plan, expected)) {
        diagnose(result, NetlistVerificationStatus::InvalidNetlist,
                 "MalformedCellPins",
                 "cell '" + plan.type + "' must have exactly its documented pins",
                 std::nullopt, instance.get_name());
        return false;
    }
    plan.output = plan.pins.at("Y");
    return true;
}

} // namespace

std::vector<std::string> module_port_bit_nets(const Module& module,
                                              const Port& port) {
    (void)module;
    std::vector<std::string> result;
    result.reserve(port.get_width());
    if (port.get_width() == 1 && !port.get_left_index() &&
        !port.get_right_index()) {
        result.push_back(port.get_name());
        return result;
    }

    const auto left = port.get_left_index().value_or(
        static_cast<std::int64_t>(port.get_width() - 1));
    const auto right = port.get_right_index().value_or(0);
    const auto range_width = packed_range_width(left, right);
    if (!range_width || *range_width != port.get_width()) return result;

    const auto step = left >= right ? 1 : -1;
    for (std::size_t bit = 0; bit < port.get_width(); ++bit) {
        const auto index = packed_index_at(right, step, bit);
        if (!index) return {};
        result.push_back(port.get_name() + "[" + std::to_string(*index) + "]");
    }
    return result;
}

NetlistEvaluationResult evaluate_combinational_netlist(
    const Module& module, const VerificationPortValues& inputs,
    const CombinationalEvaluationOptions& options) {
    NetlistEvaluationResult result;
    result.status = NetlistVerificationStatus::Succeeded;
    if (module.is_behavioral() || !module.get_assignments().empty() ||
        !module.get_child_modules().empty()) {
        diagnose(result, NetlistVerificationStatus::InvalidNetlist,
                 "UnloweredModuleContent",
                 "test evaluator accepts only flat structural modules");
        return result;
    }

    const auto declared = declared_net_names(module);
    std::set<std::string> declared_names;
    for (const auto& net : module.get_nets()) {
        if (net.get_name().empty() || !declared_names.insert(net.get_name()).second) {
            diagnose(result, NetlistVerificationStatus::InvalidNetlist,
                     "DuplicateNetDeclaration",
                     "module has an empty or duplicate net declaration '" +
                         net.get_name() + "'",
                     net.get_name());
            return result;
        }
    }
    std::set<std::string> input_nets;
    std::set<std::string> output_nets;
    std::set<std::string> seen_port_names;
    for (const auto& port : module.get_ports()) {
        if (port.get_name().empty() || !seen_port_names.insert(port.get_name()).second) {
            diagnose(result, NetlistVerificationStatus::InvalidNetlist,
                     "InvalidPortDeclaration",
                     "module has an empty or duplicate port name", std::nullopt,
                     std::nullopt);
            return result;
        }
        if (port.get_direction() == PortDirection::InOut ||
            port.get_unpacked_left_index() || port.get_unpacked_right_index()) {
            diagnose(result, NetlistVerificationStatus::UnsupportedCell,
                     "UnsupportedPortShape",
                     "test evaluator supports only input and output integral ports",
                     std::nullopt, std::nullopt);
            return result;
        }
        const auto bits = module_port_bit_nets(module, port);
        if (bits.size() != port.get_width() || bits.empty()) {
            diagnose(result, NetlistVerificationStatus::InvalidNetlist,
                     "InvalidPortRange",
                     "port range does not match its declared bit width",
                     std::nullopt, std::nullopt);
            return result;
        }
        for (const auto& bit : bits) {
            if (!declared.contains(bit)) {
                diagnose(result, NetlistVerificationStatus::UndrivenNet,
                         "UndeclaredPortBit",
                         "port bit '" + bit + "' has no declared net", bit);
                return result;
            }
            auto& destinations = port.get_direction() == PortDirection::Input
                                     ? input_nets
                                     : output_nets;
            if (!destinations.insert(bit).second) {
                diagnose(result, NetlistVerificationStatus::InvalidNetlist,
                         "AliasedPortBit",
                         "multiple port bits map to net '" + bit + "'", bit);
                return result;
            }
        }
        if (port.get_direction() == PortDirection::Input) {
            const auto stimulus = inputs.find(port.get_name());
            if (stimulus == inputs.end() || stimulus->second.size() != bits.size()) {
                diagnose(result, NetlistVerificationStatus::InvalidStimulus,
                         "MissingOrWrongWidthInput",
                         "input port '" + port.get_name() +
                             "' requires exactly " + std::to_string(bits.size()) +
                             " two-state bits");
                return result;
            }
            for (size_t bit = 0; bit < bits.size(); ++bit) {
                if (stimulus->second[bit] > 1) {
                    diagnose(result, NetlistVerificationStatus::InvalidStimulus,
                             "NonTwoStateInput",
                             "input port stimulus must contain only zero or one",
                             bits[bit]);
                    return result;
                }
                result.net_values.emplace(bits[bit], stimulus->second[bit]);
            }
        }
    }

    for (const auto& [name, bits] : inputs) {
        (void)bits;
        const auto port = std::find_if(module.get_ports().begin(),
                                      module.get_ports().end(),
            [&](const auto& candidate) {
                return candidate.get_name() == name &&
                       candidate.get_direction() == PortDirection::Input;
            });
        if (port == module.get_ports().end()) {
            diagnose(result, NetlistVerificationStatus::InvalidStimulus,
                     "UnexpectedInputPort",
                     "stimulus names undeclared or non-input port '" + name + "'");
            return result;
        }
    }

    std::map<std::string, size_t> driver_counts;
    for (const auto& input : input_nets) driver_counts[input] = 1;
    std::set<std::string> instance_names;
    std::vector<CellPlan> cells;
    cells.reserve(module.get_instances().size());
    for (const auto& instance : module.get_instances()) {
        if (instance.get_name().empty() ||
            !instance_names.insert(instance.get_name()).second) {
            diagnose(result, NetlistVerificationStatus::InvalidNetlist,
                     "DuplicateInstanceName",
                     "module has an empty or duplicate cell instance name '" +
                         instance.get_name() + "'",
                     std::nullopt, instance.get_name());
            return result;
        }
        CellPlan plan;
        if (!make_cell_plan(instance, options.allow_sequential_cutpoints, plan,
                            result))
            return result;
        for (const auto& [pin, net] : plan.pins) {
            (void)pin;
            if (!declared.contains(net)) {
                diagnose(result, NetlistVerificationStatus::UndrivenNet,
                         "UndeclaredCellNet",
                         "cell connection references undeclared net '" + net + "'",
                         net, instance.get_name());
                return result;
            }
        }
        ++driver_counts[plan.output];
        if (driver_counts[plan.output] > 1) {
            diagnose(result, NetlistVerificationStatus::MultipleDrivers,
                     "MultipleNetDrivers",
                     "net '" + plan.output + "' has multiple drivers",
                     plan.output, instance.get_name());
            return result;
        }
        if (plan.sequential_cutpoint) {
            const auto value = options.cutpoint_values.find(plan.output);
            if (value == options.cutpoint_values.end() || value->second > 1) {
                diagnose(result, NetlistVerificationStatus::InvalidStimulus,
                         "MissingSequentialCutpointValue",
                         "sequential output requires an explicit two-state cutpoint value",
                         plan.output, instance.get_name());
                return result;
            }
            result.net_values.emplace(plan.output, value->second);
        }
        cells.push_back(std::move(plan));
    }

    std::set<std::string> sequential_outputs;
    for (const auto& cell : cells)
        if (cell.sequential_cutpoint) sequential_outputs.insert(cell.output);
    for (const auto& [name, value] : options.cutpoint_values) {
        if (!declared.contains(name) || value > 1 ||
            !sequential_outputs.contains(name)) {
            diagnose(result, NetlistVerificationStatus::InvalidStimulus,
                     "InvalidSequentialCutpointValue",
                     "cutpoint values must name declared nets and be zero or one",
                     name);
            return result;
        }
    }

    for (const auto& cell : cells) {
        for (const auto& input : cell.inputs) {
            const auto& net = cell.pins.at(input);
            if (driver_counts[net] == 0) {
                diagnose(result, NetlistVerificationStatus::UndrivenNet,
                         "UndrivenCellInput",
                         "cell input pin '" + input + "' reads undriven net '" +
                             net + "'",
                         net, cell.instance->get_name());
                return result;
            }
        }
    }
    for (const auto& output : output_nets) {
        if (driver_counts[output] == 0) {
            diagnose(result, NetlistVerificationStatus::UndrivenNet,
                     "UndrivenOutputPort",
                     "output port bit '" + output + "' has no driver", output);
            return result;
        }
    }

    std::vector<bool> complete(cells.size(), false);
    size_t completed = 0;
    size_t cell_evaluations = 0;
    for (size_t index = 0; index < cells.size(); ++index) {
        if (!cells[index].sequential_cutpoint) continue;
        complete[index] = true;
        ++completed;
    }

    while (completed < cells.size()) {
        bool progress = false;
        for (size_t index = 0; index < cells.size(); ++index) {
            if (complete[index]) continue;
            const auto& cell = cells[index];
            if (!std::all_of(cell.inputs.begin(), cell.inputs.end(),
                             [&](const auto& pin) {
                                 return result.net_values.contains(cell.pins.at(pin));
                             }))
                continue;
            if (cell_evaluations >= options.max_cell_evaluations) {
                diagnose(result, NetlistVerificationStatus::ResourceLimit,
                         "EvaluationResourceLimit",
                         "combinational evaluation exceeded its configured cell limit",
                         std::nullopt, cell.instance->get_name());
                return result;
            }
            ++cell_evaluations;
            const auto value = [&](std::string_view pin) {
                return result.net_values.at(cell.pins.at(std::string(pin)));
            };
            std::uint8_t output = 0;
            if (cell.type == "AND2") output = value("A") & value("B");
            else if (cell.type == "OR2") output = value("A") | value("B");
            else if (cell.type == "XOR2") output = value("A") ^ value("B");
            else if (cell.type == "INV") output = static_cast<std::uint8_t>(!value("A"));
            else if (cell.type == "BUF") output = value("A");
            else if (cell.type == "MUX2") output = value("S") ? value("A1") : value("A0");
            else if (cell.type == "TIEHI") output = 1;
            else if (cell.type == "TIELO") output = 0;
            result.net_values.emplace(cell.output, output);
            complete[index] = true;
            ++completed;
            progress = true;
        }
        if (!progress) {
            diagnose(result, NetlistVerificationStatus::CombinationalCycle,
                     "CombinationalCycle",
                     "remaining combinational cells form a cycle or depend on an unresolved net");
            return result;
        }
    }

    for (const auto& port : module.get_ports()) {
        if (port.get_direction() != PortDirection::Output) continue;
        const auto bits = module_port_bit_nets(module, port);
        VerificationBits values;
        values.reserve(bits.size());
        for (const auto& bit : bits) {
            const auto found = result.net_values.find(bit);
            if (found == result.net_values.end()) {
                diagnose(result, NetlistVerificationStatus::UndrivenNet,
                         "UnresolvedOutputValue",
                         "output port bit '" + bit + "' has no evaluated value", bit);
                return result;
            }
            values.push_back(found->second);
        }
        result.outputs.emplace(port.get_name(), std::move(values));
    }
    result.status = NetlistVerificationStatus::Succeeded;
    return result;
}

} // namespace labneura::eda::test
