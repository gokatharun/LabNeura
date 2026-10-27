#include "labneura/eda/netlist.h"

#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace labneura::eda {
namespace {

std::string escaped(const std::string& text) {
    std::string out;
    for (const char c : text) {
        if (c == '"' || c == '\\') out += '\\';
        out += c;
    }
    return out;
}

std::string quoted(const std::string& text) { return "\"" + escaped(text) + "\""; }

bool is_output_pin(const Instance& instance, const std::string& pin) {
    if (instance.get_cell()) {
        if (const auto* found = instance.get_cell()->find_pin(pin)) {
            return found->get_direction() == "output";
        }
    }
    static const std::set<std::string> outputs = {
        "Y", "Q", "X", "Z", "ZN", "QN", "Q_N", "HI", "LO", "SUM", "COUT"};
    return outputs.count(pin) > 0;
}

// Strip a bit select so "bus[3]" and "bus" refer to one label group.
std::string base_net(const std::string& net) {
    const auto bracket = net.find('[');
    return bracket == std::string::npos ? net : net.substr(0, bracket);
}

}  // namespace

std::string to_dot(const Module& module) {
    std::ostringstream out;
    out << "digraph " << quoted(module.get_name()) << " {\n  rankdir=LR;\n";

    std::set<std::string> input_ports;
    std::set<std::string> output_ports;
    for (const auto& port : module.get_ports()) {
        const bool input = port.get_direction() == PortDirection::Input;
        (input ? input_ports : output_ports).insert(port.get_name());
        out << "  " << quoted("port:" + port.get_name())
            << " [shape=" << (input ? "invhouse" : "house") << ", label="
            << quoted(port.get_name()) << "];\n";
    }
    for (const auto& instance : module.get_instances()) {
        out << "  " << quoted(instance.get_name()) << " [shape=box, label=\""
            << escaped(instance.get_name()) << "\\n"
            << escaped(instance.get_cell_type()) << "\"];\n";
    }

    struct Endpoint {
        std::string node;
        std::string pin;
    };
    std::map<std::string, std::vector<Endpoint>> drivers;
    std::map<std::string, std::vector<Endpoint>> sinks;
    // Nets are matched per bit ("count[2]"), so bits never cross-connect; a port
    // node stands in for every bit of its port.
    std::set<std::string> port_nets_seen;
    for (const auto& instance : module.get_instances()) {
        for (const auto& connection : instance.get_connections()) {
            const auto& net = connection.get_net();
            auto& side = is_output_pin(instance, connection.get_pin()) ? drivers : sinks;
            side[net].push_back({instance.get_name(), connection.get_pin()});
            if (port_nets_seen.insert(net).second) {
                const auto base = base_net(net);
                if (input_ports.count(base)) drivers[net].push_back({"port:" + base, ""});
                if (output_ports.count(base)) sinks[net].push_back({"port:" + base, ""});
            }
        }
    }
    for (const auto& [net, from] : drivers) {
        const auto found = sinks.find(net);
        if (found == sinks.end()) continue;
        for (const auto& source : from) {
            for (const auto& target : found->second) {
                out << "  " << quoted(source.node) << " -> " << quoted(target.node)
                    << " [label=" << quoted(net) << ", headlabel="
                    << quoted(target.pin) << "];\n";
            }
        }
    }
    out << "}\n";
    return out.str();
}

void write_dot(const Module& module, const std::string& path) {
    std::ofstream file(path);
    if (!file) throw std::runtime_error("cannot open graph output: " + path);
    file << to_dot(module);
}

}  // namespace labneura::eda
