#include "labneura/eda/netlist.h"

#include <fstream>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace labneura::eda {
namespace {

void append_json_string(std::ostream& output, const std::string& text) {
    output << '"';
    for (const unsigned char character : text) {
        switch (character) {
            case '"': output << "\\\""; break;
            case '\\': output << "\\\\"; break;
            case '\b': output << "\\b"; break;
            case '\f': output << "\\f"; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default:
                if (character < 0x20) {
                    static constexpr char digits[] = "0123456789abcdef";
                    output << "\\u00" << digits[character >> 4]
                           << digits[character & 0x0f];
                } else {
                    output << character;
                }
        }
    }
    output << '"';
}

const char* expression_kind_name(ExpressionKind kind) {
    switch (kind) {
        case ExpressionKind::Identifier: return "identifier";
        case ExpressionKind::Literal: return "literal";
        case ExpressionKind::Unary: return "unary";
        case ExpressionKind::Binary: return "binary";
        case ExpressionKind::Conditional: return "conditional";
        case ExpressionKind::BitSelect: return "bit_select";
        case ExpressionKind::RangeSelect: return "range_select";
        case ExpressionKind::Concatenation: return "concatenation";
        case ExpressionKind::Opaque: return "opaque";
    }
    return "opaque";
}

void append_expression(std::ostream& output, const Expression& expression) {
    output << "{\"kind\": ";
    append_json_string(output, expression_kind_name(expression.get_kind()));
    output << ", \"value\": ";
    append_json_string(output, expression.get_value());
    output << ", \"source\": ";
    append_json_string(output, expression.get_source());
    output << ", \"width\": " << expression.get_width();
    output << ", \"operands\": [";
    for (std::size_t i = 0; i < expression.get_operands().size(); ++i) {
        if (i != 0) output << ", ";
        append_expression(output, expression.get_operands()[i]);
    }
    output << "]}";
}

}  // namespace

std::string to_json(const Module& module) {
    std::ostringstream output;
    output << "{\n  \"format\": \"eda-netlist\",\n  \"version\": 1,\n"
              "  \"modules\": [\n    {\n      \"name\": ";
    append_json_string(output, module.get_name());
    output << ",\n      \"ports\": [";
    for (std::size_t i = 0; i < module.get_ports().size(); ++i) {
        const auto& port = module.get_ports()[i];
        output << (i == 0 ? "\n" : ",\n") << "        {\"name\": ";
        append_json_string(output, port.get_name());
        output << ", \"direction\": \""
               << (port.get_direction() == PortDirection::Input ? "input" : "output")
               << "\"}";
    }
    if (!module.get_ports().empty()) output << '\n' << "      ";
    output << "],\n      \"nets\": [";
    bool first_net = true;
    for (const auto& net : module.get_nets()) {
        if (net.get_kind() == NetType::Internal) {
            output << (first_net ? "\n" : ",\n") << "        ";
            append_json_string(output, net.get_name());
            first_net = false;
        }
    }
    if (!first_net) output << '\n' << "      ";
    output << "],\n      \"instances\": [";
    for (std::size_t i = 0; i < module.get_instances().size(); ++i) {
        const auto& instance = module.get_instances()[i];
        output << (i == 0 ? "\n" : ",\n")
               << "        {\n          \"name\": ";
        append_json_string(output, instance.get_name());
        output << ",\n          \"cell_type\": ";
        append_json_string(output, instance.get_cell_type());
        output << ",\n          \"connections\": {";
        for (std::size_t pin = 0; pin < instance.get_connections().size(); ++pin) {
            const auto& connection = instance.get_connections()[pin];
            output << (pin == 0 ? "\n" : ",\n") << "            ";
            append_json_string(output, connection.get_pin());
            output << ": ";
            append_json_string(output, connection.get_net());
        }
        if (!instance.get_connections().empty()) output << '\n' << "          ";
        output << "}\n        }";
    }
    if (!module.get_instances().empty()) output << '\n' << "      ";
    output << "],\n      \"assignments\": [";
    for (std::size_t i = 0; i < module.get_assignments().size(); ++i) {
        const auto& assignment = module.get_assignments()[i];
        output << (i == 0 ? "\n" : ",\n") << "        {\"lhs\": ";
        append_json_string(output, assignment.get_lhs());
        output << ", \"rhs\": ";
        append_json_string(output, assignment.get_rhs());
        output << ", \"lhs_expression\": ";
        append_expression(output, assignment.get_lhs_expression());
        output << ", \"rhs_expression\": ";
        append_expression(output, assignment.get_rhs_expression());
        output << "}";
    }
    if (!module.get_assignments().empty()) output << '\n' << "      ";
    output << "]\n    }\n  ]\n}\n";
    return output.str();
}

void write_json(const Module& module, const std::string& path) {
    std::ofstream output(path);
    if (!output) throw std::runtime_error("could not open JSON output: " + path);
    output << to_json(module);
    if (!output) throw std::runtime_error("failed writing JSON output: " + path);
}

}  // namespace labneura::eda
