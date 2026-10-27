#include "labneura/eda/netlist.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <iterator>
#include <optional>
#include <regex>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace labneura::eda {
namespace {

bool is_identifier(const Expression& expression) {
    return expression.get_kind() == ExpressionKind::Identifier;
}

std::optional<bool> literal_value(const Expression& expression) {
    if (expression.get_kind() != ExpressionKind::Literal &&
        expression.get_kind() != ExpressionKind::Opaque) {
        return std::nullopt;
    }
    const auto source = expression.get_source();
    static const std::regex based_literal(
        R"(^\s*[0-9]+\s*'\s*[sS]?[bBoOdDhH]\s*[0-9a-fA-F_xXzZ?]+\s*$)");
    static const std::regex decimal(R"(^\s*([0-9]+)\s*$)");
    std::smatch decimal_match;
    const bool is_based = std::regex_match(source, based_literal);
    const bool is_decimal = std::regex_match(source, decimal_match, decimal);
    if (!is_based && !is_decimal) return std::nullopt;
    if (source.find_first_of("xXzZ?") != std::string::npos) {
        throw std::runtime_error(
            "scalar X/Z constants cannot be mapped to generic tie cells");
    }
    if (is_decimal) {
        return (std::stoull(decimal_match[1].str()) & 1U) != 0;
    }
    const auto apostrophe = source.find('\'');
    const auto base_position = source.find_first_of("bBoOdDhH", apostrophe);
    const auto digit_position = source.find_first_not_of(" \t", base_position + 1);
    const auto digits = source.substr(digit_position);
    const char base = static_cast<char>(
        std::tolower(static_cast<unsigned char>(source[base_position])));
    if (digits.empty()) return std::nullopt;
    if (base == 'd') return (std::stoull(digits) & 1U) != 0;
    const char last_digit = digits.back();
    return (std::isdigit(static_cast<unsigned char>(last_digit))
                ? (last_digit - '0') & 1
                : (std::tolower(static_cast<unsigned char>(last_digit)) - 'a' + 10) &
                      1) != 0;
}

std::string pin_name(std::size_t index) {
    if (index >= 26) {
        throw std::runtime_error(
            "generic Boolean operator has more than 26 operands");
    }
    return std::string(1, static_cast<char>('A' + index));
}

std::string generic_type_for_operator(const std::string& op,
                                      std::size_t operand_count) {
    if (op == "&" || op == "&&") return "AND" + std::to_string(operand_count);
    if (op == "|" || op == "||") return "OR" + std::to_string(operand_count);
    if (op == "^") return "XOR" + std::to_string(operand_count);
    throw std::runtime_error("unsupported Boolean operator in generic mapping: " +
                             op);
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
        } else {
            expected_inputs = 0;
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

struct GenericMapper {
    std::vector<Net> nets;
    std::vector<Instance> instances;
    std::unordered_set<std::string> used_names;
    std::size_t next_index = 0;

    explicit GenericMapper(const Module& module) {
        for (const auto& net : module.get_nets()) used_names.insert(net.get_name());
        for (const auto& instance : module.get_instances()) {
            used_names.insert(instance.get_name());
        }
    }

    std::string unique_name(const std::string& prefix) {
        std::string name;
        do {
            name = prefix + std::to_string(next_index++);
        } while (!used_names.insert(name).second);
        return name;
    }

    void add_cell(std::string type, std::vector<PinConnection> connections,
                  const std::string& output) {
        connections.emplace_back("Y", output);
        instances.emplace_back(unique_name("__generic_inst"), std::move(type),
                               std::move(connections));
    }

    bool collect_associative(const Expression& expression,
                             const std::string& op,
                             std::vector<const Expression*>& operands) {
        if (expression.get_kind() == ExpressionKind::Binary &&
            expression.get_value() == op &&
            expression.get_operands().size() == 2) {
            return collect_associative(expression.get_operands()[0], op,
                                        operands) &&
                   collect_associative(expression.get_operands()[1], op,
                                        operands);
        }
        operands.push_back(&expression);
        return true;
    }

    bool is_supported_operator(const std::string& op) const {
        return op == "&" || op == "&&" || op == "|" || op == "||" ||
               op == "^";
    }

    std::optional<std::string> lower(const Expression& expression,
                                     const std::string& destination) {
        if (is_identifier(expression)) return expression.get_source();
        if (expression.get_kind() == ExpressionKind::BitSelect &&
            expression.get_operands().size() == 2) {
            const auto& base = expression.get_operands()[0];
            const auto& selector = expression.get_operands()[1];
            if (selector.get_kind() == ExpressionKind::Literal) {
                return expression.get_source();
            }
            const auto data_width = base.get_width();
            const auto selector_width = selector.get_width();
            if (!is_identifier(base) || !is_identifier(selector) ||
                data_width < 2 || selector_width == 0 ||
                selector_width >= sizeof(std::size_t) * 8 ||
                data_width != (std::size_t{1} << selector_width)) {
                return std::nullopt;
            }

            std::vector<std::string> values;
            values.reserve(data_width);
            for (std::size_t i = 0; i < data_width; ++i) {
                values.push_back(base.get_source() + "[" +
                                 std::to_string(i) + "]");
            }
            for (std::size_t level = 0; level < selector_width; ++level) {
                std::vector<std::string> next_values;
                next_values.reserve(values.size() / 2);
                for (std::size_t i = 0; i < values.size(); i += 2) {
                    const bool final_cell = values.size() == 2;
                    const auto output =
                        final_cell ? destination : unique_name("__generic_net");
                    if (!final_cell) nets.emplace_back(output, NetType::Internal);
                    add_cell(
                        "MUX2",
                        {{"A0", values[i]},
                         {"A1", values[i + 1]},
                         {"S", selector.get_source() + "[" +
                                   std::to_string(level) + "]"}},
                        output);
                    next_values.push_back(output);
                }
                values = std::move(next_values);
            }
            return destination;
        }
        if (const auto value = literal_value(expression)) {
            add_cell(*value ? "TIEHI" : "TIELO", {}, destination);
            return destination;
        }
        if (expression.get_kind() == ExpressionKind::Binary &&
            is_supported_operator(expression.get_value())) {
            std::vector<const Expression*> operands;
            if (!collect_associative(expression, expression.get_value(),
                                     operands)) {
                return std::nullopt;
            }
            if (operands.size() < 2) return std::nullopt;
            std::vector<PinConnection> connections;
            for (std::size_t i = 0; i < operands.size(); ++i) {
                const auto signal = lower(*operands[i], unique_name("__generic_net"));
                if (!signal) return std::nullopt;
                if (*signal != operands[i]->get_source() ||
                    !is_identifier(*operands[i])) {
                    nets.emplace_back(*signal, NetType::Internal);
                }
                connections.emplace_back(pin_name(i), *signal);
            }
            add_cell(generic_type_for_operator(expression.get_value(),
                                               operands.size()),
                     std::move(connections), destination);
            return destination;
        }
        if (expression.get_kind() == ExpressionKind::Unary &&
            (expression.get_value() == "~" || expression.get_value() == "!") &&
            expression.get_operands().size() == 1) {
            const auto& operand = expression.get_operands().front();
            const auto signal = lower(operand, unique_name("__generic_net"));
            if (!signal) return std::nullopt;
            if (!is_identifier(operand)) nets.emplace_back(*signal, NetType::Internal);
            add_cell("INV", {{"A", *signal}}, destination);
            return destination;
        }
        if (expression.get_kind() == ExpressionKind::Conditional &&
            expression.get_operands().size() == 3) {
            const auto& operands = expression.get_operands();
            std::vector<std::string> signals;
            for (const auto& operand : operands) {
                const auto signal = lower(operand, unique_name("__generic_net"));
                if (!signal) return std::nullopt;
                if (!is_identifier(operand)) nets.emplace_back(*signal, NetType::Internal);
                signals.push_back(*signal);
            }
            add_cell("MUX2",
                     {{"S", signals[0]}, {"A1", signals[1]},
                      {"A0", signals[2]}},
                     destination);
            return destination;
        }
        return std::nullopt;
    }
};

}  // namespace

Module generic_map(const Module& rtl_module) {
    GenericMapper mapper(rtl_module);
    std::vector<ContinuousAssignment> remaining_assignments;

    for (const auto& assignment : rtl_module.get_assignments()) {
        const auto& lhs = assignment.get_lhs_expression();
        if (!is_identifier(lhs)) {
            throw std::runtime_error(
                "generic mapping currently requires scalar identifier assignment targets: " +
                assignment.get_lhs());
        }
        const auto result =
            mapper.lower(assignment.get_rhs_expression(), lhs.get_source());
        if (!result) {
            throw std::runtime_error("continuous assignment '" +
                                     assignment.get_lhs() + " = " +
                                     assignment.get_rhs() +
                                     "' has no supported generic mapping");
        }
        if (*result != lhs.get_source()) {
            mapper.add_cell("BUF", {{"A", *result}}, lhs.get_source());
        }
    }

    auto nets = rtl_module.get_nets();
    nets.insert(nets.end(), std::make_move_iterator(mapper.nets.begin()),
                std::make_move_iterator(mapper.nets.end()));
    auto instances = rtl_module.get_instances();
    for (auto& instance : instances) {
        std::string generic_type;
        const auto& cell_type = instance.get_cell_type();
        if (cell_type == "and" || cell_type == "or" || cell_type == "xor") {
            std::size_t input_count = 0;
            for (const auto& connection : instance.get_connections()) {
                if (connection.get_pin() != "Y") ++input_count;
            }
            if (input_count < 2) {
                throw std::runtime_error(
                    "primitive gate '" + instance.get_name() +
                    "' must have at least two inputs");
            }
            generic_type = (cell_type == "and" ? "AND" :
                            cell_type == "or"  ? "OR"
                                                : "XOR") +
                           std::to_string(input_count);
        }
        if (!generic_type.empty()) {
            instance = Instance(instance.get_name(), std::move(generic_type),
                                instance.get_connections());
        }
    }
    instances.insert(instances.end(),
                     std::make_move_iterator(mapper.instances.begin()),
                     std::make_move_iterator(mapper.instances.end()));
    return Module(rtl_module.get_name(), rtl_module.get_ports(), std::move(nets),
                  std::move(instances), std::move(remaining_assignments));
}

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
