#include "labneura/eda/netlist.h"

#include <algorithm>
#include <cctype>
#include <optional>
#include <regex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace labneura::eda {
namespace {

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
    if (lower_type == "tiehi" || lower_type == "tielo") return "conb";
    for (const auto* family : {"and", "or", "xor", "xnor", "maj"}) {
        if (lower_type.rfind(family, 0) == 0) return family;
    }
    return {};
}

std::optional<StandardCell> find_cell_for_generic_type(
    const StandardCellLibrary& library, const std::string& type) {
    if (const auto* exact = library.find_cell(type)) return *exact;
    static const std::regex generic_gate(
        R"(^(AND|OR|XOR|XNOR|MAJ)([2-9][0-9]*)$)");
    static const std::regex generic_unary(R"(^(INV|BUF|MUX2|TIEHI|TIELO)$)");
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
        type == "MUX2" ? family : family + digit_count;
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
            type.rfind("XOR", 0) == 0 || type.rfind("XNOR", 0) == 0 ||
            type.rfind("MAJ", 0) == 0) {
            expected_inputs = static_cast<std::size_t>(std::stoul(digit_count));
        } else if (type == "MUX2") {
            expected_inputs = 3;
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
        R"(^(AND|OR|XOR|XNOR|MAJ)([2-9][0-9]*)$)");
    static const std::regex generic_unary(R"(^(INV|BUF|MUX2|TIEHI|TIELO)$)");
    return std::regex_match(type, generic_gate) ||
           std::regex_match(type, generic_unary);
}

std::string output_pin(const StandardCell& cell, const std::string& type) {
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

}  // namespace

Module tech_map(const Module& generic_module,
                const StandardCellLibrary& cell_library) {
    std::vector<Instance> instances;
    instances.reserve(generic_module.get_instances().size());

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
        const auto cell = find_cell_for_generic_type(cell_library, mapped_type);
        if (!cell) {
            throw std::runtime_error(
                "no standard cell matching generic cell '" + generic_type +
                "' was found in Liberty library '" + cell_library.get_name() +
                "'");
        }

        std::vector<PinConnection> connections;
        for (const auto& connection : instance.get_connections()) {
            if (connection.get_pin() == "Y") {
                connections.emplace_back(
                    output_pin(*cell, mapped_type), connection.get_net());
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

}  // namespace labneura::eda
