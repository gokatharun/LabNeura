#include "labneura/eda/netlist.h"

#include <algorithm>
#include <cctype>
#include <optional>
#include <map>
#include <regex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace labneura::eda {
namespace {

class LibertyFunctionParser {
public:
    LibertyFunctionParser(
        std::string_view expression,
        const std::unordered_map<std::string, bool>& values)
        : expression_(expression), values_(values) {}

    std::optional<bool> parse() {
        const auto value = parse_or();
        skip_space();
        if (!value || position_ != expression_.size()) return std::nullopt;
        return value;
    }

private:
    void skip_space() {
        while (position_ < expression_.size() &&
               std::isspace(static_cast<unsigned char>(expression_[position_]))) {
            ++position_;
        }
    }

    std::optional<bool> parse_or() {
        auto value = parse_xor();
        while (value) {
            skip_space();
            if (position_ == expression_.size() ||
                (expression_[position_] != '|' &&
                 expression_[position_] != '+')) {
                break;
            }
            ++position_;
            const auto rhs = parse_xor();
            if (!rhs) return std::nullopt;
            value = *value || *rhs;
        }
        return value;
    }

    std::optional<bool> parse_xor() {
        auto value = parse_and();
        while (value) {
            skip_space();
            if (position_ == expression_.size() ||
                expression_[position_] != '^') {
                break;
            }
            ++position_;
            const auto rhs = parse_and();
            if (!rhs) return std::nullopt;
            value = *value != *rhs;
        }
        return value;
    }

    std::optional<bool> parse_and() {
        auto value = parse_unary();
        while (value) {
            skip_space();
            if (position_ == expression_.size() ||
                (expression_[position_] != '&' &&
                 expression_[position_] != '*')) {
                break;
            }
            ++position_;
            const auto rhs = parse_unary();
            if (!rhs) return std::nullopt;
            value = *value && *rhs;
        }
        return value;
    }

    std::optional<bool> parse_unary() {
        skip_space();
        if (position_ == expression_.size()) return std::nullopt;
        if (expression_[position_] == '!' || expression_[position_] == '~') {
            ++position_;
            const auto value = parse_unary();
            if (!value) return std::nullopt;
            return !*value;
        }
        auto value = parse_primary();
        while (value) {
            skip_space();
            if (position_ == expression_.size() ||
                expression_[position_] != '\'') {
                break;
            }
            ++position_;
            value = !*value;
        }
        return value;
    }

    std::optional<bool> parse_primary() {
        skip_space();
        if (position_ == expression_.size()) return std::nullopt;
        if (expression_[position_] == '(') {
            ++position_;
            auto value = parse_or();
            skip_space();
            if (!value || position_ == expression_.size() ||
                expression_[position_] != ')') {
                return std::nullopt;
            }
            ++position_;
            return value;
        }
        if (expression_[position_] == '0' || expression_[position_] == '1') {
            return expression_[position_++] == '1';
        }
        const auto start = position_;
        while (position_ < expression_.size()) {
            const auto character =
                static_cast<unsigned char>(expression_[position_]);
            if (!std::isalnum(character) && expression_[position_] != '_') break;
            ++position_;
        }
        if (start == position_) return std::nullopt;
        const auto value = values_.find(
            std::string(expression_.substr(start, position_ - start)));
        if (value == values_.end()) return std::nullopt;
        return value->second;
    }

    std::string_view expression_;
    const std::unordered_map<std::string, bool>& values_;
    std::size_t position_ = 0;
};

struct LogicGate {
    std::string family;
    std::size_t input_count;
};

std::optional<LogicGate> logic_gate(const std::string& type) {
    static const std::regex gate(
        R"(^(AND|NAND|OR|NOR|XOR|XNOR)([2-9][0-9]*)$)");
    std::smatch match;
    if (std::regex_match(type, match, gate)) {
        return LogicGate{match[1].str(),
                         static_cast<std::size_t>(std::stoul(match[2].str()))};
    }
    if (type == "INV") return LogicGate{"INV", 1};
    if (type == "BUF") return LogicGate{"BUF", 1};
    return std::nullopt;
}

bool matches_logic_function(const CellPin& output, const LogicGate& gate,
                            const std::vector<const CellPin*>& inputs) {
    if (inputs.size() != gate.input_count || gate.input_count > 16 ||
        output.get_function().empty()) {
        return false;
    }
    const std::size_t combinations = std::size_t{1} << gate.input_count;
    for (std::size_t combination = 0; combination < combinations;
         ++combination) {
        std::unordered_map<std::string, bool> values;
        for (std::size_t index = 0; index < inputs.size(); ++index) {
            values.emplace(inputs[index]->get_name(),
                           (combination & (std::size_t{1} << index)) != 0);
        }
        const auto actual =
            LibertyFunctionParser(output.get_function(), values).parse();
        if (!actual) return false;

        bool expected = false;
        if (gate.family == "AND" || gate.family == "NAND") {
            expected = combination == combinations - 1;
            if (gate.family == "NAND") expected = !expected;
        } else if (gate.family == "OR" || gate.family == "NOR") {
            expected = combination != 0;
            if (gate.family == "NOR") expected = !expected;
        } else if (gate.family == "XOR" || gate.family == "XNOR") {
            auto bits = combination;
            while (bits != 0) {
                expected = !expected;
                bits &= bits - 1;
            }
            if (gate.family == "XNOR") expected = !expected;
        } else if (gate.family == "INV") {
            expected = combination == 0;
        } else {
            expected = combination != 0;
        }
        if (*actual != expected) return false;
    }
    return true;
}

std::vector<const CellPin*> input_pins(const StandardCell& cell) {
    std::vector<const CellPin*> inputs;
    for (const auto& pin : cell.get_pins()) {
        if (pin.get_direction() == "input") inputs.push_back(&pin);
    }
    return inputs;
}

std::string normalized(std::string value);

bool conventional_logic_cell_name(const StandardCell& cell,
                                  const LogicGate& gate) {
    auto fragment = normalized(gate.family);
    if (gate.input_count > 1) fragment += std::to_string(gate.input_count);
    const auto name = normalized(cell.get_name());
    const auto position = name.find(fragment);
    if (position == std::string::npos) return false;
    if (position > 0 && name[position - 1] != '_') return false;
    const auto end = position + fragment.size();
    return end == name.size() || name[end] == '_';
}

std::string normalized(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char ch) {
                       return static_cast<char>(std::tolower(ch));
                   });
    return value;
}

std::string cell_family(const std::string& type) {
    const auto lower_type = normalized(type);
    if (lower_type == "inv") return "inv";
    if (lower_type == "buf") return "buf";
    if (lower_type == "mux2") return "mux2";
    if (lower_type == "dff") return "dfxtp";
    if (lower_type == "dffr") return "dfrtp";
    if (lower_type == "dffs") return "dfstp";
    if (lower_type == "dffsr") return "dfbbp";
    if (lower_type == "tribuf") return "ebufn";
    if (lower_type == "latch") return "dlxtp";
    if (lower_type == "tiehi" || lower_type == "tielo") return "conb";
    for (const auto* family : {"and", "nand", "or", "nor", "xor", "xnor",
                               "maj"}) {
        if (lower_type.rfind(family, 0) == 0) return family;
    }
    return {};
}

std::optional<StandardCell> find_cell_for_generic_type(
    const StandardCellLibrary& library, const std::string& type) {
    if (const auto* exact = library.find_cell(type)) return *exact;
    if (const auto gate = logic_gate(type)) {
        const StandardCell* best = nullptr;
        bool best_has_conventional_name = false;
        bool best_has_single_output = false;
        bool best_is_unit_drive = false;
        for (const auto& cell : library.get_cells()) {
            const auto inputs = input_pins(cell);
            if (inputs.size() != gate->input_count) continue;
            const bool has_matching_output = std::any_of(
                cell.get_pins().begin(), cell.get_pins().end(),
                [&](const CellPin& pin) {
                    return pin.get_direction() == "output" &&
                           matches_logic_function(pin, *gate, inputs);
                });
            if (!has_matching_output) continue;

            const auto candidate_name = normalized(cell.get_name());
            const bool has_conventional_name =
                conventional_logic_cell_name(cell, *gate);
            const auto output_count = static_cast<std::size_t>(std::count_if(
                cell.get_pins().begin(), cell.get_pins().end(),
                [](const CellPin& pin) {
                    return pin.get_direction() == "output";
                }));
            const bool has_single_output = output_count == 1;
            const bool unit_drive =
                candidate_name.size() >= 2 &&
                candidate_name.compare(candidate_name.size() - 2, 2, "_1") == 0;
            const bool better =
                !best ||
                (has_conventional_name && !best_has_conventional_name) ||
                (has_conventional_name == best_has_conventional_name &&
                 has_single_output && !best_has_single_output) ||
                (has_conventional_name == best_has_conventional_name &&
                 has_single_output == best_has_single_output && unit_drive &&
                 !best_is_unit_drive) ||
                (has_conventional_name == best_has_conventional_name &&
                 has_single_output == best_has_single_output &&
                 unit_drive == best_is_unit_drive && cell.get_area() &&
                 (!best->get_area() ||
                  *cell.get_area() < *best->get_area()));
            if (better) {
                best = &cell;
                best_has_conventional_name = has_conventional_name;
                best_has_single_output = has_single_output;
                best_is_unit_drive = unit_drive;
            }
        }
        if (best) return *best;
        return std::nullopt;
    }
    static const std::regex generic_gate(
        R"(^(AND|NAND|OR|NOR|XOR|XNOR|MAJ)([2-9][0-9]*)$)");
    static const std::regex generic_unary(R"(^(INV|BUF|MUX2|DFF|DFFR|DFFS|DFFSR|LATCH|TRIBUF|TIEHI|TIELO)$)");
    std::smatch generic_match;
    if (!std::regex_match(type, generic_match, generic_gate) &&
        !std::regex_match(type, generic_unary)) {
        return std::nullopt;
    }
    const auto family = cell_family(type);
    if (family.empty()) return std::nullopt;

    std::string digit_count;
    for (const auto character : type) {
        if (std::isdigit(static_cast<unsigned char>(character))) {
            digit_count.push_back(character);
        }
    }
    const auto name_fragment =
        type == "MUX2" || type.rfind("DFF", 0) == 0 || type == "LATCH" ||
            type == "TRIBUF"
            ? family
            : family + digit_count;
    const StandardCell* best = nullptr;
    bool best_is_unit_drive = false;
    for (const auto& cell : library.get_cells()) {
        const auto candidate_name = normalized(cell.get_name());
        const auto family_position = candidate_name.find(name_fragment);
        if (family_position == std::string::npos) continue;
        if (family_position > 0 &&
            candidate_name[family_position - 1] != '_') {
            continue;
        }
        const auto family_end = family_position + name_fragment.size();
        if (family_end < candidate_name.size() &&
            candidate_name[family_end] != '_') {
            continue;
        }

        std::size_t input_count = 0;
        std::size_t output_count = 0;
        for (const auto& pin : cell.get_pins()) {
            if (pin.get_direction() == "input") ++input_count;
            if (pin.get_direction() == "output") ++output_count;
        }
        std::size_t expected_inputs = 0;
        if (type.rfind("AND", 0) == 0 || type.rfind("OR", 0) == 0 ||
            type.rfind("NAND", 0) == 0 || type.rfind("NOR", 0) == 0 ||
            type.rfind("XOR", 0) == 0 || type.rfind("XNOR", 0) == 0 ||
            type.rfind("MAJ", 0) == 0) {
            expected_inputs = static_cast<std::size_t>(std::stoul(digit_count));
        } else if (type == "MUX2") {
            expected_inputs = 3;
        } else if (type == "DFF" || type == "LATCH" || type == "TRIBUF") {
            expected_inputs = 2;
        } else if (type == "DFFR" || type == "DFFS") {
            expected_inputs = 3;
        } else if (type == "DFFSR") {
            expected_inputs = 4;
        } else if (type == "INV" || type == "BUF") {
            expected_inputs = 1;
        }
        if (input_count != expected_inputs || output_count == 0) continue;
        const bool unit_drive = candidate_name.size() >= 2 &&
                                candidate_name.compare(
                                    candidate_name.size() - 2, 2, "_1") == 0;
        if (!best || (unit_drive && !best_is_unit_drive) ||
            (unit_drive == best_is_unit_drive && cell.get_area() &&
             (!best->get_area() ||
              *cell.get_area() < *best->get_area()))) {
            best = &cell;
            best_is_unit_drive = unit_drive;
        }
    }
    if (best) return *best;
    return std::nullopt;
}

std::string generic_alias(const std::string& type) {
    if (type == "AND2_X1") return "AND2";
    if (type == "OR2_X1") return "OR2";
    if (type == "XOR2_X1") return "XOR2";
    if (type == "INV_X1") return "INV";
    if (type == "BUF_X1") return "BUF";
    if (type == "MUX2_X1") return "MUX2";
    if (type == "TIEHI_X1") return "TIEHI";
    if (type == "TIELO_X1") return "TIELO";
    return type;
}

bool is_generic_type(const std::string& type) {
    static const std::regex generic_gate(
        R"(^(AND|NAND|OR|NOR|XOR|XNOR|MAJ)([2-9][0-9]*)$)");
    static const std::regex generic_unary(R"(^(INV|BUF|MUX2|DFF|DFFR|DFFS|DFFSR|LATCH|TRIBUF|TIEHI|TIELO)$)");
    return std::regex_match(type, generic_gate) ||
           std::regex_match(type, generic_unary);
}

std::string output_pin(const StandardCell& cell, const std::string& type) {
    if (const auto gate = logic_gate(type)) {
        const auto inputs = input_pins(cell);
        for (const auto& pin : cell.get_pins()) {
            if (pin.get_direction() == "output" &&
                matches_logic_function(pin, *gate, inputs)) {
                return pin.get_name();
            }
        }
        throw std::runtime_error(
            "standard cell '" + cell.get_name() +
            "' has no output matching generic cell '" + type + "'");
    }
    const std::string expected_function =
        type == "TIEHI" ? "1" : type == "TIELO" ? "0" : "";
    for (const auto& pin : cell.get_pins()) {
        if (pin.get_direction() == "output" &&
            (expected_function.empty() ||
             pin.get_function() == expected_function)) {
            return pin.get_name();
        }
    }
    throw std::runtime_error("standard cell '" + cell.get_name() +
                             "' has no matching output pin");
}

std::optional<std::size_t> generic_input_index(const std::string& pin) {
    if (pin.size() == 1 && pin[0] >= 'A' && pin[0] <= 'Z') {
        return static_cast<std::size_t>(pin[0] - 'A');
    }
    if (pin.rfind("IN", 0) == 0 && pin.size() > 2) {
        try {
            const auto one_based = std::stoul(pin.substr(2));
            if (one_based > 26) return one_based - 1;
        } catch (const std::invalid_argument&) {
            return std::nullopt;
        } catch (const std::out_of_range&) {
            return std::nullopt;
        }
    }
    return std::nullopt;
}

}  // namespace

Module tech_map(const Module& generic_module,
                const StandardCellLibrary& cell_library) {
    std::vector<Instance> instances;
    instances.reserve(generic_module.get_instances().size());
    std::map<std::string, std::optional<StandardCell>> cell_cache;

    for (const auto& instance : generic_module.get_instances()) {
        const auto generic_type = instance.get_cell_type();
        const auto mapped_type = generic_alias(generic_type);
        if (mapped_type != generic_type) {
            if (const auto* exact = cell_library.find_cell(generic_type)) {
                instances.emplace_back(instance.get_name(), generic_type,
                                       instance.get_connections(), *exact);
                continue;
            }
        }
        const bool generic_cell = is_generic_type(mapped_type);
        if (!generic_cell) {
            if (const auto* exact = cell_library.find_cell(generic_type)) {
                instances.emplace_back(instance.get_name(), generic_type,
                                       instance.get_connections(), *exact);
                continue;
            }
        }
        if (!generic_cell) {
            instances.emplace_back(instance.get_name(), generic_type,
                                   instance.get_connections(), instance.get_cell());
            continue;
        }
        auto cached = cell_cache.find(mapped_type);
        if (cached == cell_cache.end()) {
            cached = cell_cache
                         .emplace(mapped_type,
                                  find_cell_for_generic_type(cell_library, mapped_type))
                         .first;
        }
        const auto& cell = cached->second;
        if (!cell) {
            throw std::runtime_error(
                "no standard cell matching generic cell '" + generic_type +
                "' was found in Liberty library '" + cell_library.get_name() +
                "'");
        }

        std::vector<PinConnection> connections;
        const auto gate = logic_gate(mapped_type);
        const auto candidate_inputs = gate ? input_pins(*cell)
                                           : std::vector<const CellPin*>{};
        for (const auto& connection : instance.get_connections()) {
            if (connection.get_pin() == "Y") {
                connections.emplace_back(
                    output_pin(*cell, mapped_type), connection.get_net());
            } else if (gate) {
                const auto index =
                    generic_input_index(connection.get_pin());
                if (!index || *index >= candidate_inputs.size()) {
                    throw std::runtime_error(
                        "generic cell '" + generic_type +
                        "' has invalid input pin '" +
                        connection.get_pin() + "'");
                }
                connections.emplace_back(
                    candidate_inputs[*index]->get_name(),
                    connection.get_net());
            } else {
                const auto* pin = cell->find_pin(connection.get_pin());
                if (!pin || pin->get_direction() != "input") {
                    throw std::runtime_error(
                        "cell '" + cell->get_name() +
                        "' has no input pin corresponding to generic pin '" +
                        connection.get_pin() + "'");
                }
                connections.emplace_back(pin->get_name(), connection.get_net());
            }
        }
        instances.emplace_back(instance.get_name(), cell->get_name(),
                               std::move(connections), *cell);
    }
    return Module(generic_module.get_name(), generic_module.get_ports(),
                  generic_module.get_nets(), std::move(instances),
                  generic_module.get_assignments());
}

Module tech_map(const Project& project, const StandardCellLibrary& cell_library,
                const std::string& top,
                const std::map<std::string, std::string>& parameters) {
    return tech_map(generic_map(project, top, parameters), cell_library);
}

}  // namespace labneura::eda
