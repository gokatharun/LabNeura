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
    for (const auto& name : input_ports) drivers[name].push_back({"port:" + name, ""});
    for (const auto& name : output_ports) sinks[name].push_back({"port:" + name, ""});
    for (const auto& instance : module.get_instances()) {
        for (const auto& connection : instance.get_connections()) {
            auto& side = is_output_pin(instance, connection.get_pin()) ? drivers : sinks;
            // Bit selects of a port ("count[2]") belong to that port's node.
            const auto base = base_net(connection.get_net());
            const bool is_port = input_ports.count(base) || output_ports.count(base);
            side[is_port ? base : connection.get_net()].push_back(
                {instance.get_name(), connection.get_pin()});
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
