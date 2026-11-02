#include "labneura/eda/netlist.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <map>
#include <ostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace labneura::eda {
namespace {

bool simple_identifier(const std::string& value) {
    if (value.empty() ||
        !(std::isalpha(static_cast<unsigned char>(value.front())) ||
          value.front() == '_' || value.front() == '$')) {
        return false;
    }
    return std::all_of(value.begin() + 1, value.end(), [](unsigned char ch) {
        return std::isalnum(ch) || ch == '_' || ch == '$';
    });
}

std::string identifier(const std::string& value) {
    if (simple_identifier(value)) return value;
    if (value.find_first_of(" \t\r\n") != std::string::npos) {
        throw std::runtime_error("cannot emit Verilog identifier containing whitespace: " +
                                 value);
    }
    return "\\" + value + " ";
}

std::string signal_reference(const std::string& value) {
    if (value.empty() || value.front() == '\\') return value;
    const auto dot = value.rfind('.');
    const auto select = value.find('[', dot == std::string::npos ? 0 : dot + 1);
    const auto base = value.substr(0, select);
    const bool scoped_identifier = !base.empty() &&
        std::all_of(base.begin(), base.end(), [](unsigned char ch) {
            return std::isalnum(ch) || ch == '_' || ch == '$' || ch == '.' ||
                   ch == '[' || ch == ']';
        });
    if (!scoped_identifier) return value;
    return identifier(base) + (select == std::string::npos
                                   ? std::string{}
                                   : value.substr(select));
}

std::string range(std::size_t width,
                  const std::optional<std::int64_t>& left = std::nullopt,
                  const std::optional<std::int64_t>& right = std::nullopt) {
    if (left && right) {
        return " [" + std::to_string(*left) + ":" +
               std::to_string(*right) + "]";
    }
    return width > 1 ? " [" + std::to_string(width - 1) + ":0]" : "";
}

struct NetGroup {
    std::string base;
    std::vector<std::set<std::int64_t>> indices;
    bool has_scalar = false;
    std::size_t width = 1;
    std::size_t packed_width = 1;
    std::optional<std::int64_t> packed_left;
    std::optional<std::int64_t> packed_right;
    std::optional<std::int64_t> unpacked_left;
    std::optional<std::int64_t> unpacked_right;
};

bool parse_index(const std::string& value, std::int64_t& index) {
    try {
        std::size_t consumed = 0;
        index = std::stoll(value, &consumed);
        return consumed == value.size();
    } catch (const std::exception&) {
        return false;
    }
}

std::pair<std::string, std::vector<std::int64_t>> split_net_indices(
    const std::string& name) {
    std::string base = name;
    std::vector<std::int64_t> indices;
    while (!base.empty() && base.back() == ']') {
        const auto open = base.rfind('[');
        if (open == std::string::npos) break;
        std::int64_t index = 0;
        if (!parse_index(base.substr(open + 1, base.size() - open - 2), index)) {
            break;
        }
        indices.push_back(index);
        base.resize(open);
    }
    if (indices.empty()) return {name, {}};
    std::reverse(indices.begin(), indices.end());
    return {std::move(base), std::move(indices)};
}

std::vector<std::string> internal_declarations(const Module& module) {
    std::unordered_set<std::string> ports;
    for (const auto& port : module.get_ports()) ports.insert(port.get_name());

    std::map<std::string, NetGroup> groups;
    for (const auto& net : module.get_nets()) {
        if (net.get_kind() != NetType::Internal) continue;
        auto [base, indices] = split_net_indices(net.get_name());
        auto& group = groups[base];
        group.base = base;
        if (!indices.empty()) {
            if (group.indices.empty()) {
                group.indices.resize(indices.size());
            } else if (group.indices.size() != indices.size()) {
                throw std::runtime_error(
                    "inconsistent dimensionality for net '" + base + "'");
            }
            for (std::size_t dimension = 0; dimension < indices.size(); ++dimension) {
                group.indices[dimension].insert(indices[dimension]);
            }
        } else {
            group.has_scalar = true;
            group.width = net.get_width();
            group.packed_width = net.get_packed_width();
            group.packed_left = net.get_left_index();
            group.packed_right = net.get_right_index();
            group.unpacked_left = net.get_unpacked_left_index();
            group.unpacked_right = net.get_unpacked_right_index();
        }
    }

    std::vector<std::string> declarations;
    declarations.reserve(groups.size());
    for (const auto& [key, group] : groups) {
        if (ports.count(group.base)) continue;
        std::string declaration = "wire";
        if (group.unpacked_left && group.unpacked_right) {
            declaration += range(group.packed_width, group.packed_left,
                                 group.packed_right);
            declaration += " " + identifier(group.base);
            declaration += " [" + std::to_string(*group.unpacked_left) +
                           ":" + std::to_string(*group.unpacked_right) + "]";
        } else if (group.packed_left && group.packed_right) {
            declaration += range(group.packed_width, group.packed_left,
                                 group.packed_right);
            declaration += " " + identifier(group.base);
        } else if (!group.indices.empty()) {
            const auto& packed = group.indices.back();
            declaration += " [" + std::to_string(*packed.rbegin()) + ":" +
                           std::to_string(*packed.begin()) + "]";
            declaration += " " + identifier(group.base);
            for (std::size_t dimension = 0;
                 dimension + 1 < group.indices.size(); ++dimension) {
                const auto& values = group.indices[dimension];
                declaration += " [" + std::to_string(*values.rbegin()) + ":" +
                               std::to_string(*values.begin()) + "]";
            }
        } else {
            declaration += range(group.packed_width, group.packed_left,
                                 group.packed_right);
            declaration += " " + identifier(group.base);
            if (group.unpacked_left && group.unpacked_right) {
                declaration += " [" + std::to_string(*group.unpacked_left) +
                               ":" + std::to_string(*group.unpacked_right) + "]";
            }
        }
        declaration += ";";
        declarations.push_back(std::move(declaration));
    }
    return declarations;
}

bool is_gate_primitive(const std::string& type) {
    static const std::unordered_set<std::string> types = {
        "and", "nand", "or", "nor", "xor", "xnor", "buf", "not",
        "bufif0", "bufif1", "notif0", "notif1", "pullup", "pulldown",
        "nmos", "pmos", "rnmos", "rpmos", "cmos", "rcmos", "tran",
        "rtran", "tranif0", "tranif1", "rtranif0", "rtranif1"};
    return types.count(type) != 0;
}

bool output_pin_name(const std::string& pin) {
    static const std::unordered_set<std::string> outputs = {
        "Y", "Q", "X", "Z", "ZN", "QN", "Q_N", "HI", "LO",
        "SUM", "COUT", "O"};
    return outputs.count(pin) != 0;
}

std::string cell_module_alias(const std::string& type) {
    static constexpr char hex[] = "0123456789abcdef";
    std::string alias = "__labneura_cell__";
    for (const unsigned char ch : type) {
        alias.push_back(hex[ch >> 4]);
        alias.push_back(hex[ch & 0x0f]);
    }
    return alias;
}

void emit_instance(std::ostream& out, const Instance& instance,
                   const std::string& cell_type) {
    if (instance.is_user_defined_primitive()) {
        throw std::runtime_error(
            "Verilog writer cannot reconstruct user-defined primitive declarations");
    }
    out << "  " << identifier(cell_type);
    if (!instance.get_parameters().empty()) {
        out << " #(";
        bool first = true;
        for (const auto& [name, value] : instance.get_parameters()) {
            out << (first ? "" : ", ") << "." << identifier(name) << "("
                << value << ")";
            first = false;
        }
        out << ")";
    }
    out << " " << identifier(instance.get_name()) << " (";

    if (is_gate_primitive(cell_type)) {
        std::vector<const PinConnection*> outputs;
        std::vector<const PinConnection*> inputs;
        for (const auto& connection : instance.get_connections()) {
            (output_pin_name(connection.get_pin()) ? outputs : inputs)
                .push_back(&connection);
        }
        std::sort(inputs.begin(), inputs.end(),
                  [](const auto* lhs, const auto* rhs) {
                      return lhs->get_pin() < rhs->get_pin();
                  });
        bool first = true;
        for (const auto* connection : outputs) {
            out << (first ? "" : ", ")
                << signal_reference(connection->get_net());
            first = false;
        }
        for (const auto* connection : inputs) {
            out << (first ? "" : ", ")
                << signal_reference(connection->get_net());
            first = false;
        }
    } else {
        bool first = true;
        for (const auto& connection : instance.get_connections()) {
            out << (first ? "" : ", ") << "." << identifier(connection.get_pin())
                << "(" << signal_reference(connection.get_net()) << ")";
            first = false;
        }
    }
    out << ");\n";
}

struct EmissionNode {
    const Module* module;
    std::string emitted_name;
    std::unordered_map<std::string, std::string> child_types;
};

void validate_structural(const Module& module) {
    if (module.is_behavioral()) {
        throw std::runtime_error(
            "Verilog writer requires structural RTL; generic_map behavioral "
            "modules before writing");
    }
    for (const auto& instance : module.get_instances()) {
        if (instance.is_user_defined_primitive()) {
            throw std::runtime_error(
                "Verilog writer cannot reconstruct user-defined primitive declarations");
        }
    }
}

void emit_module(std::ostream& out, const EmissionNode& node,
                 const std::map<std::string, std::string>& cell_aliases) {
    const auto& module = *node.module;
    validate_structural(module);
    out << "module " << identifier(node.emitted_name) << " (";
    for (std::size_t i = 0; i < module.get_ports().size(); ++i) {
        if (i) out << ", ";
        out << identifier(module.get_ports()[i].get_name());
    }
    out << ");\n";
    for (const auto& port : module.get_ports()) {
        const char* direction = "inout";
        switch (port.get_direction()) {
            case PortDirection::Input: direction = "input"; break;
            case PortDirection::Output: direction = "output"; break;
            case PortDirection::InOut: break;
        }
        out << "  " << direction << " wire"
            << range(port.get_packed_width(), port.get_left_index(),
                     port.get_right_index()) << " "
            << identifier(port.get_name());
        if (port.get_unpacked_left_index() &&
            port.get_unpacked_right_index()) {
            out << " [" << *port.get_unpacked_left_index() << ":"
                << *port.get_unpacked_right_index() << "]";
        }
        out << ";\n";
    }
    for (const auto& declaration : internal_declarations(module)) {
        out << "  " << declaration << "\n";
    }
    for (const auto& assignment : module.get_assignments()) {
        out << "  assign " << assignment.get_lhs() << " = "
            << assignment.get_rhs() << ";\n";
    }
    for (const auto& instance : module.get_instances()) {
        auto type = node.child_types.find(instance.get_name());
        std::string cell_type = type == node.child_types.end()
                                    ? instance.get_cell_type()
                                    : type->second;
        if (type == node.child_types.end()) {
            const auto alias = cell_aliases.find(cell_type);
            if (alias != cell_aliases.end()) cell_type = alias->second;
        }
        emit_instance(out, instance, cell_type);
    }
    out << "endmodule\n\n";
}

std::vector<EmissionNode> collect_nodes(const Module& root) {
    std::vector<EmissionNode> nodes;
    std::unordered_set<const Module*> active;
    std::size_t serial = 0;
    const auto visit = [&](const auto& self, const Module& module,
                           const std::string& emitted_name) -> void {
        if (!active.insert(&module).second) {
            throw std::runtime_error("recursive hierarchy cannot be written as Verilog");
        }
        EmissionNode node{&module, emitted_name, {}};
        for (const auto& child : module.get_child_modules()) {
            std::string child_name = child.get_name() + "__labneura_inst_" +
                                     std::to_string(serial++);
            node.child_types.emplace(child.get_hierarchy_instance_name(),
                                     child_name);
            self(self, child, child_name);
        }
        nodes.push_back(std::move(node));
        active.erase(&module);
    };
    visit(visit, root, root.get_name());
    return nodes;
}

void emit_verilog(std::ostream& output, const Module& module) {
    const auto nodes = collect_nodes(module);
    std::unordered_set<std::string> hierarchy_types;
    for (const auto& node : nodes) {
        for (const auto& [instance, type] : node.child_types) {
            (void)instance;
            hierarchy_types.insert(type);
        }
    }
    std::map<std::string, std::string> cell_aliases;
    std::map<std::string, std::vector<std::string>> cell_pins;
    for (const auto& node : nodes) {
        for (const auto& instance : node.module->get_instances()) {
            if (node.child_types.count(instance.get_name()) ||
                is_gate_primitive(instance.get_cell_type()) ||
                hierarchy_types.count(instance.get_cell_type())) {
                continue;
            }
            auto [alias, inserted] = cell_aliases.emplace(
                instance.get_cell_type(), cell_module_alias(instance.get_cell_type()));
            (void)inserted;
            auto& pins = cell_pins[alias->second];
            for (const auto& connection : instance.get_connections()) {
                if (std::find(pins.begin(), pins.end(), connection.get_pin()) ==
                    pins.end()) {
                    pins.push_back(connection.get_pin());
                }
            }
        }
    }
    for (const auto& node : nodes) emit_module(output, node, cell_aliases);
    for (const auto& [type, alias] : cell_aliases) {
        const auto& pins = cell_pins[alias];
        output << "(* blackbox *) module " << identifier(alias) << " (";
        for (std::size_t i = 0; i < pins.size(); ++i) {
            if (i) output << ", ";
            output << identifier(pins[i]);
        }
        output << ");\n";
        for (const auto& pin : pins) {
            output << "  " << (output_pin_name(pin) ? "output" : "input")
                   << " wire " << identifier(pin) << ";\n";
        }
        output << "endmodule\n\n";
        (void)type;
    }
    if (!output) throw std::runtime_error("failed while writing Verilog netlist");
}

void validate_verilog_tree(const Module& module) {
    validate_structural(module);
    for (const auto& child : module.get_child_modules()) {
        validate_verilog_tree(child);
    }
}

}  // namespace

std::string to_verilog(const Module& module) {
    validate_verilog_tree(module);
    std::ostringstream output;
    emit_verilog(output, module);
    return output.str();
}

void write_verilog(const Module& module, const std::string& path) {
    validate_verilog_tree(module);
    std::ofstream output(path);
    if (!output) throw std::runtime_error("cannot open Verilog output: " + path);
    emit_verilog(output, module);
}

}  // namespace labneura::eda
