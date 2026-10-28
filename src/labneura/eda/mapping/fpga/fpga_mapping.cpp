#include "labneura/eda/netlist.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <iomanip>
#include <iterator>
#include <map>
#include <optional>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace labneura::eda {
namespace {

constexpr std::size_t kCutsPerNode = 8;

enum class Gate { And, Or, Xor, Nand, Nor, Xnor, Maj, Not, Buf, Mux };

std::string upper(std::string text) {
    for (auto& character : text) {
        character = static_cast<char>(std::toupper(static_cast<unsigned char>(character)));
    }
    return text;
}

bool is_output_pin(const std::string& pin) {
    return pin == "Y" || pin == "Q" || pin == "X" || pin == "Z" || pin == "ZN" ||
           pin == "QN" || pin == "Q_N";
}

bool combinational_gate(const std::string& type, Gate& gate) {
    static const std::regex wide(R"(^(AND|OR|XOR|NAND|NOR|XNOR|MAJ)([2-9][0-9]*)?$)");
    const auto name = upper(type);
    std::smatch match;
    if (std::regex_match(name, match, wide)) {
        const auto& kind = match[1].str();
        gate = kind == "AND" ? Gate::And : kind == "OR" ? Gate::Or
             : kind == "XOR" ? Gate::Xor : kind == "NAND" ? Gate::Nand
             : kind == "NOR" ? Gate::Nor : kind == "XNOR" ? Gate::Xnor : Gate::Maj;
        return true;
    }
    if (name == "INV" || name == "NOT") { gate = Gate::Not; return true; }
    if (name == "BUF") { gate = Gate::Buf; return true; }
    if (name == "MUX2") { gate = Gate::Mux; return true; }
    return false;
}

// One combinational gate of the input netlist.
struct Node {
    Gate gate;
    std::string output;
    std::vector<int> inputs;  // node ids; constants and primary inputs are leaves
    int depth = 0;
    double flow = 0.0;
    std::vector<std::vector<int>> cuts;
    std::vector<int> best;
    bool required = false;
    int fanout = 0;
};

class Mapper {
public:
    Mapper(const Module& module, std::size_t k, const StandardCellLibrary& library)
        : module_(module), k_(k), library_(library) {}

    Module run() {
        classify();
        order();
        enumerate_cuts();
        select();
        return emit();
    }

private:
    const Module& module_;
    std::size_t k_;
    const StandardCellLibrary& library_;

    std::optional<StandardCell> cell_for(const std::string& type) const {
        if (const auto* cell = library_.find_cell(type)) return *cell;
        return std::nullopt;
    }

    std::vector<Node> nodes_;
    std::unordered_map<std::string, int> node_of_net_;   // comb output net -> node
    std::unordered_map<std::string, int> constant_;      // net -> 0/1
    std::vector<const Instance*> passthrough_;
    std::vector<std::string> sink_nets_;                 // nets that must exist
    std::vector<std::vector<std::string>> leaf_nets_;    // per node: net names for leaves
    std::vector<int> topo_;
    // A leaf is identified by its net name; interior nodes by node id.
    std::unordered_map<std::string, int> leaf_id_;
    std::vector<std::string> leaf_name_;

    int leaf(const std::string& net) {
        const auto found = leaf_id_.find(net);
        if (found != leaf_id_.end()) return found->second;
        const int id = -1 - static_cast<int>(leaf_name_.size());
        leaf_id_.emplace(net, id);
        leaf_name_.push_back(net);
        return id;
    }

    void classify() {
        std::vector<std::pair<const Instance*, Gate>> gates;
        for (const auto& instance : module_.get_instances()) {
            Gate gate;
            const auto type = upper(instance.get_cell_type());
            if (type == "TIEHI" || type == "TIELO") {
                for (const auto& connection : instance.get_connections()) {
                    if (is_output_pin(connection.get_pin())) {
                        constant_[connection.get_net()] = type == "TIEHI" ? 1 : 0;
                    }
                }
                passthrough_.push_back(&instance);
            } else if (combinational_gate(instance.get_cell_type(), gate)) {
                gates.emplace_back(&instance, gate);
            } else {
                passthrough_.push_back(&instance);
                for (const auto& connection : instance.get_connections()) {
                    if (!is_output_pin(connection.get_pin())) {
                        sink_nets_.push_back(connection.get_net());
                    }
                }
            }
        }
        for (const auto& port : module_.get_ports()) {
            if (port.get_direction() == PortDirection::Input) continue;
            output_ports_.insert(port.get_name());
        }
        nodes_.reserve(gates.size());
        for (const auto& [instance, gate] : gates) {
            Node node;
            node.gate = gate;
            for (const auto& connection : instance->get_connections()) {
                if (is_output_pin(connection.get_pin())) node.output = connection.get_net();
            }
            if (node.output.empty()) {
                throw std::runtime_error("gate '" + instance->get_name() + "' has no output");
            }
            if (node_of_net_.count(node.output)) {
                throw std::runtime_error("net '" + node.output + "' has several drivers");
            }
            node_of_net_.emplace(node.output, static_cast<int>(nodes_.size()));
            nodes_.push_back(std::move(node));
        }
        for (std::size_t i = 0; i < gates.size(); ++i) {
            const auto& instance = *gates[i].first;
            std::vector<std::pair<std::string, int>> pins;
            for (const auto& connection : instance.get_connections()) {
                if (is_output_pin(connection.get_pin())) continue;
                const auto& net = connection.get_net();
                const auto driver = node_of_net_.find(net);
                const int source = driver != node_of_net_.end() ? driver->second : leaf(net);
                pins.emplace_back(connection.get_pin(), source);
            }
            if (nodes_[i].gate == Gate::Mux) {
                // Fixed order: S, A1, A0.
                std::sort(pins.begin(), pins.end(), [](const auto& a, const auto& b) {
                    const auto rank = [](const std::string& pin) {
                        return pin == "S" ? 0 : pin == "A1" ? 1 : 2;
                    };
                    return rank(a.first) < rank(b.first);
                });
            }
            for (const auto& pin : pins) nodes_[i].inputs.push_back(pin.second);
        }
        const std::size_t roots = gates.size();
        for (std::size_t i = 0; i < roots; ++i) split_wide(i);
    }

    // Rewrites an n-input AND/OR/XOR family gate as a balanced tree of 2-input gates
    // so that the cut enumeration can repack it into LUTs of any size.
    void split_wide(std::size_t index) {
        const Gate gate = nodes_[index].gate;
        const bool family = gate == Gate::And || gate == Gate::Or || gate == Gate::Xor ||
                            gate == Gate::Nand || gate == Gate::Nor || gate == Gate::Xnor;
        if (!family || nodes_[index].inputs.size() <= 2) return;
        const Gate inner = (gate == Gate::And || gate == Gate::Nand) ? Gate::And
                         : (gate == Gate::Or || gate == Gate::Nor) ? Gate::Or : Gate::Xor;
        std::vector<int> level = nodes_[index].inputs;
        while (level.size() > 2) {
            std::vector<int> next;
            for (std::size_t i = 0; i + 1 < level.size(); i += 2) {
                Node node;
                node.gate = inner;
                node.output = "__fpga_t" + std::to_string(nodes_.size());
                node.inputs = {level[i], level[i + 1]};
                nodes_.push_back(std::move(node));
                next.push_back(static_cast<int>(nodes_.size()) - 1);
            }
            if (level.size() % 2) next.push_back(level.back());
            level = std::move(next);
        }
        nodes_[index].inputs = level;
    }

    void order() {
        // Kahn's algorithm so that cones of any depth are handled iteratively.
        std::vector<int> pending(nodes_.size(), 0);
        std::vector<std::vector<int>> users(nodes_.size());
        for (std::size_t i = 0; i < nodes_.size(); ++i) {
            for (const int input : nodes_[i].inputs) {
                if (input >= 0) {
                    ++pending[i];
                    users[static_cast<std::size_t>(input)].push_back(static_cast<int>(i));
                }
                if (input >= 0) ++nodes_[static_cast<std::size_t>(input)].fanout;
            }
        }
        std::vector<int> ready;
        for (std::size_t i = 0; i < nodes_.size(); ++i) {
            if (pending[i] == 0) ready.push_back(static_cast<int>(i));
        }
        while (!ready.empty()) {
            const int id = ready.back();
            ready.pop_back();
            topo_.push_back(id);
            for (const int user : users[static_cast<std::size_t>(id)]) {
                if (--pending[static_cast<std::size_t>(user)] == 0) ready.push_back(user);
            }
        }
        if (topo_.size() != nodes_.size()) {
            throw std::runtime_error("fpga_map: combinational loop in the netlist");
        }
    }

    std::set<std::string> output_ports_;

    bool is_constant_leaf(int id) const {
        return id < 0 && constant_.count(leaf_name_[static_cast<std::size_t>(-1 - id)]);
    }

    double leaf_flow(int id) const {
        return id >= 0 ? nodes_[static_cast<std::size_t>(id)].flow /
                             std::max(1, nodes_[static_cast<std::size_t>(id)].fanout)
                       : 0.0;
    }
    int leaf_depth(int id) const {
        return id >= 0 ? nodes_[static_cast<std::size_t>(id)].depth : 0;
    }

    static bool merge(const std::vector<int>& a, const std::vector<int>& b,
                      std::size_t limit, std::vector<int>& out) {
        out.clear();
        std::set_union(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(out));
        return out.size() <= limit;
    }

    void enumerate_cuts() {
        for (const int id : topo_) {
            auto& node = nodes_[static_cast<std::size_t>(id)];
            std::vector<std::vector<int>> partial = {{}};
            for (const int input : node.inputs) {
                std::vector<std::vector<int>> choices;
                if (is_constant_leaf(input)) {
                    choices.push_back({});
                } else {
                    choices.push_back({input});
                    if (input >= 0) {
                        for (const auto& cut : nodes_[static_cast<std::size_t>(input)].cuts) {
                            choices.push_back(cut);
                        }
                    }
                }
                std::vector<std::vector<int>> next;
                std::vector<int> merged;
                for (const auto& base : partial) {
                    for (const auto& choice : choices) {
                        if (!merge(base, choice, k_, merged)) continue;
                        next.push_back(merged);
                    }
                }
                std::sort(next.begin(), next.end());
                next.erase(std::unique(next.begin(), next.end()), next.end());
                partial = rank(std::move(next), 4 * kCutsPerNode);
            }
            node.cuts = rank(std::move(partial), kCutsPerNode);
            if (node.cuts.empty()) {
                throw std::runtime_error("fpga_map: no feasible cut for net " + node.output);
            }
            node.best = node.cuts.front();
            node.depth = depth_of(node.best);
            node.flow = flow_of(node.best);
        }
    }

    int depth_of(const std::vector<int>& cut) const {
        int depth = 0;
        for (const int leaf_id : cut) depth = std::max(depth, leaf_depth(leaf_id));
        return depth + 1;
    }
    double flow_of(const std::vector<int>& cut) const {
        double flow = 1.0;
        for (const int leaf_id : cut) flow += leaf_flow(leaf_id);
        return flow;
    }

    // Best first: lowest depth, then area flow, then fewest leaves.
    std::vector<std::vector<int>> rank(std::vector<std::vector<int>> cuts,
                                       std::size_t keep) const {
        std::stable_sort(cuts.begin(), cuts.end(), [&](const auto& a, const auto& b) {
            const int da = depth_of(a), db = depth_of(b);
            if (da != db) return da < db;
            const double fa = flow_of(a), fb = flow_of(b);
            if (fa != fb) return fa < fb;
            return a.size() < b.size();
        });
        if (cuts.size() > keep) cuts.resize(keep);
        return cuts;
    }

    void require(const std::string& net, std::vector<int>& stack) {
        const auto found = node_of_net_.find(net);
        if (found == node_of_net_.end()) return;
        auto& node = nodes_[static_cast<std::size_t>(found->second)];
        if (!node.required) {
            node.required = true;
            stack.push_back(found->second);
        }
    }

    void select() {
        std::vector<int> stack;
        for (const auto& net : sink_nets_) require(net, stack);
        for (const auto& [net, id] : node_of_net_) {
            const auto bracket = net.find('[');
            if (output_ports_.count(bracket == std::string::npos ? net : net.substr(0, bracket))) {
                require(net, stack);
            }
        }
        while (!stack.empty()) {
            const int id = stack.back();
            stack.pop_back();
            for (const int cut_leaf : nodes_[static_cast<std::size_t>(id)].best) {
                if (cut_leaf >= 0) {
                    auto& inner = nodes_[static_cast<std::size_t>(cut_leaf)];
                    if (!inner.required) {
                        inner.required = true;
                        stack.push_back(cut_leaf);
                    }
                }
            }
        }
    }

    // 64-bit parallel evaluation: bit p of a word is the value for input pattern p.
    static constexpr std::uint64_t kPattern[6] = {
        0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL, 0xF0F0F0F0F0F0F0F0ULL,
        0xFF00FF00FF00FF00ULL, 0xFFFF0000FFFF0000ULL, 0xFFFFFFFF00000000ULL};

    std::uint64_t apply(const Node& node, const std::vector<std::uint64_t>& in) const {
        std::uint64_t value = 0;
        switch (node.gate) {
            case Gate::And: case Gate::Nand:
                value = ~0ULL;
                for (const auto word : in) value &= word;
                return node.gate == Gate::And ? value : ~value;
            case Gate::Or: case Gate::Nor:
                for (const auto word : in) value |= word;
                return node.gate == Gate::Or ? value : ~value;
            case Gate::Xor: case Gate::Xnor:
                for (const auto word : in) value ^= word;
                return node.gate == Gate::Xor ? value : ~value;
            case Gate::Maj: {
                std::uint64_t result = 0;
                for (std::size_t bit = 0; bit < 64; ++bit) {
                    std::size_t ones = 0;
                    for (const auto word : in) ones += (word >> bit) & 1U;
                    if (ones * 2 > in.size()) result |= 1ULL << bit;
                }
                return result;
            }
            case Gate::Not: return ~in.at(0);
            case Gate::Buf: return in.at(0);
            case Gate::Mux: return (in.at(0) & in.at(1)) | (~in.at(0) & in.at(2));
        }
        return 0;
    }

    std::uint64_t truth_table(int root, const std::vector<int>& cut) const {
        std::unordered_map<int, std::uint64_t> value;
        for (std::size_t i = 0; i < cut.size(); ++i) value[cut[i]] = kPattern[i];
        // Iterative post-order over the cone.
        std::vector<std::pair<int, bool>> stack = {{root, false}};
        while (!stack.empty()) {
            const auto [id, expanded] = stack.back();
            stack.pop_back();
            if (value.count(id)) continue;
            const auto& node = nodes_[static_cast<std::size_t>(id)];
            if (!expanded) {
                stack.emplace_back(id, true);
                for (const int input : node.inputs) {
                    if (!value.count(input) && !is_constant_leaf(input)) {
                        stack.emplace_back(input, false);
                    }
                }
                continue;
            }
            std::vector<std::uint64_t> in;
            for (const int input : node.inputs) {
                if (is_constant_leaf(input)) {
                    in.push_back(constant_.at(leaf_name_[static_cast<std::size_t>(-1 - input)])
                                     ? ~0ULL : 0ULL);
                } else {
                    in.push_back(value.at(input));
                }
            }
            value[id] = apply(node, in);
        }
        return value.at(root);
    }

    std::string net_name(int id) const {
        return id >= 0 ? nodes_[static_cast<std::size_t>(id)].output
                       : leaf_name_[static_cast<std::size_t>(-1 - id)];
    }

    Module emit() {
        std::vector<Instance> instances;
        for (const auto* instance : passthrough_) {
            instances.emplace_back(instance->get_name(), instance->get_cell_type(),
                                   instance->get_connections(),
                                   cell_for(instance->get_cell_type()),
                                   instance->get_parameters());
        }
        std::size_t counter = 0;
        for (const int id : topo_) {
            const auto& node = nodes_[static_cast<std::size_t>(id)];
            if (!node.required) continue;
            const auto& cut = node.best;
            std::uint64_t table = truth_table(id, cut);
            const std::size_t width = std::max<std::size_t>(cut.size(), 1);
            const std::uint64_t rows = 1ULL << width;
            const std::uint64_t mask = rows == 64 ? ~0ULL : (1ULL << rows) - 1;
            table &= mask;
            if (cut.empty()) {
                const std::string tie = table & 1 ? "TIEHI" : "TIELO";
                instances.emplace_back("__lut" + std::to_string(counter++), tie,
                                       std::vector<PinConnection>{{"Y", node.output}},
                                       cell_for(tie));
                continue;
            }
            std::vector<PinConnection> pins;
            for (std::size_t i = 0; i < cut.size(); ++i) {
                pins.emplace_back("I" + std::to_string(i), net_name(cut[i]));
            }
            pins.emplace_back("O", node.output);
            std::ostringstream init;
            init << std::hex << std::uppercase << std::setfill('0')
                 << std::setw(static_cast<int>((rows + 3) / 4)) << table;
            instances.emplace_back("__lut" + std::to_string(counter++),
                                   "LUT" + std::to_string(cut.size()), std::move(pins),
                                   cell_for("LUT" + std::to_string(cut.size())),
                                   std::map<std::string, std::string>{{"INIT", init.str()}});
        }
        std::vector<Net> nets;
        std::set<std::string> declared;
        for (const auto& net : module_.get_nets()) {
            nets.push_back(net);
            declared.insert(net.get_name());
        }
        for (const auto& instance : instances) {
            for (const auto& connection : instance.get_connections()) {
                const auto& name = connection.get_net();
                if (declared.insert(name).second) nets.emplace_back(name, NetType::Internal);
            }
        }
        return Module(module_.get_name(), module_.get_ports(), std::move(nets),
                      std::move(instances));
    }
};

}  // namespace

Module fpga_map(const Module& generic_module, std::size_t lut_size,
                const StandardCellLibrary& library) {
    if (lut_size < 2 || lut_size > 6) {
        throw std::invalid_argument("lut_size must be between 2 and 6");
    }
    if (generic_module.is_behavioral()) {
        throw std::invalid_argument(
            "fpga_map needs a generic netlist; run generic_map first");
    }
    return Mapper(generic_module, lut_size, library).run();
}

Module fpga_map(const Project& project, const std::string& top,
                const std::map<std::string, std::string>& parameters,
                std::size_t lut_size, const StandardCellLibrary& library) {
    return fpga_map(generic_map(project, top, parameters), lut_size, library);
}

}  // namespace labneura::eda
