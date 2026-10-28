#include "labneura/eda/netlist.h"

#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace labneura::eda {
namespace {

std::string base_net(const std::string& net) {
    const auto bracket = net.find('[');
    return bracket == std::string::npos ? net : net.substr(0, bracket);
}

// All bit-level nets of a port, in a stable order.
std::vector<std::string> port_bits(const Module& module, const std::string& port) {
    std::set<std::string> bits;
    for (const auto& instance : module.get_instances()) {
        for (const auto& connection : instance.get_connections()) {
            if (base_net(connection.get_net()) == port) bits.insert(connection.get_net());
        }
    }
    if (bits.empty()) bits.insert(port);
    return {bits.begin(), bits.end()};
}

const std::string* find_net(const Instance& instance, const std::string& pin) {
    for (const auto& connection : instance.get_connections()) {
        if (connection.get_pin() == pin) return &connection.get_net();
    }
    return nullptr;
}

int hex_digit(char character) {
    if (character >= '0' && character <= '9') return character - '0';
    if (character >= 'A' && character <= 'F') return character - 'A' + 10;
    if (character >= 'a' && character <= 'f') return character - 'a' + 10;
    throw std::invalid_argument("invalid hex digit in LUT INIT");
}

}  // namespace

std::string to_blif(const Module& module) {
    std::ostringstream out;
    out << ".model " << module.get_name() << "\n";
    std::ostringstream inputs, outputs;
    for (const auto& port : module.get_ports()) {
        auto& target = port.get_direction() == PortDirection::Input ? inputs : outputs;
        for (const auto& bit : port_bits(module, port.get_name())) target << ' ' << bit;
    }
    out << ".inputs" << inputs.str() << "\n.outputs" << outputs.str() << "\n";

    for (const auto& instance : module.get_instances()) {
        const auto& type = instance.get_cell_type();
        if (type.rfind("LUT", 0) == 0 && type.size() == 4) {
            const std::size_t width = static_cast<std::size_t>(type[3] - '0');
            const auto init = instance.get_parameters().find("INIT");
            if (init == instance.get_parameters().end()) {
                throw std::runtime_error("LUT '" + instance.get_name() + "' has no INIT");
            }
            out << ".names";
            for (std::size_t i = 0; i < width; ++i) {
                out << ' ' << *find_net(instance, "I" + std::to_string(i));
            }
            out << ' ' << *find_net(instance, "O") << "\n";
            const std::string& text = init->second;
            for (std::size_t row = 0; row < (1U << width); ++row) {
                const std::size_t digit_index = text.size() - 1 - row / 4;
                if (!((hex_digit(text.at(digit_index)) >> (row % 4)) & 1)) continue;
                for (std::size_t i = 0; i < width; ++i) out << ((row >> i) & 1U ? '1' : '0');
                out << " 1\n";
            }
        } else if (type == "TIEHI" || type == "TIELO") {
            out << ".names " << *find_net(instance, "Y") << "\n";
            if (type == "TIEHI") out << "1\n";
        } else if (type == "DFF") {
            out << ".latch " << *find_net(instance, "D") << ' ' << *find_net(instance, "Y")
                << " re " << *find_net(instance, "CLK") << " 3\n";
        } else {
            out << ".subckt " << type;
            for (const auto& connection : instance.get_connections()) {
                out << ' ' << connection.get_pin() << '=' << connection.get_net();
            }
            out << "\n";
        }
    }
    out << ".end\n";
    return out.str();
}

void write_blif(const Module& module, const std::string& path) {
    std::ofstream file(path);
    if (!file) throw std::runtime_error("cannot open BLIF output: " + path);
    file << to_blif(module);
}

}  // namespace labneura::eda
