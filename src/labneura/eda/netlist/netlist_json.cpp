#include "labneura/eda/netlist.h"

#include <fstream>
#include <cctype>
#include <istream>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace labneura::eda {
namespace {

void append_string(std::ostream& out, const std::string& text) {
    static constexpr char hex[] = "0123456789abcdef";
    out.put('"');
    for (const unsigned char ch : text) {
        switch (ch) {
            case '"': out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\b': out << "\\b"; break;
            case '\f': out << "\\f"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (ch < 0x20) {
                    out << "\\u00" << hex[ch >> 4] << hex[ch & 0x0f];
                } else {
                    out.put(static_cast<char>(ch));
                }
        }
    }
    out.put('"');
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
    throw std::runtime_error("invalid expression kind in netlist");
}

void write_expression(std::ostream& out, const Expression& expression) {
    out << "{\"kind\":";
    append_string(out, expression_kind_name(expression.get_kind()));
    out << ",\"value\":";
    append_string(out, expression.get_value());
    out << ",\"source\":";
    append_string(out, expression.get_source());
    out << ",\"width\":" << expression.get_width() << ",\"operands\":[";
    bool first = true;
    for (const auto& operand : expression.get_operands()) {
        if (!first) out.put(',');
        write_expression(out, operand);
        first = false;
    }
    out << "]}";
}

const char* port_direction_name(PortDirection direction) {
    switch (direction) {
        case PortDirection::Input: return "input";
        case PortDirection::Output: return "output";
        case PortDirection::InOut: return "inout";
    }
    throw std::runtime_error("invalid port direction in netlist");
}

const char* net_type_name(NetType type) {
    switch (type) {
        case NetType::Input: return "input";
        case NetType::Output: return "output";
        case NetType::InOut: return "inout";
        case NetType::Internal: return "internal";
    }
    throw std::runtime_error("invalid net type in netlist");
}

void write_index(std::ostream& out,
                 const std::optional<std::int64_t>& index) {
    if (index) out << *index;
    else out << "null";
}

void write_module(std::ostream& out, const Module& module) {
    out << "{\"name\":";
    append_string(out, module.get_name());
    out << ",\"hierarchy_instance_name\":";
    append_string(out, module.get_hierarchy_instance_name());
    out << ",\"is_behavioral\":" << (module.is_behavioral() ? "true" : "false")
        << ",\"ports\":[";
    bool first = true;
    for (const auto& port : module.get_ports()) {
        if (!first) out.put(',');
        out << "{\"name\":";
        append_string(out, port.get_name());
        out << ",\"direction\":";
        append_string(out, port_direction_name(port.get_direction()));
        out << ",\"width\":" << port.get_width() << ",\"left_index\":";
        write_index(out, port.get_left_index());
        out << ",\"right_index\":";
        write_index(out, port.get_right_index());
        out << ",\"unpacked_left_index\":";
        write_index(out, port.get_unpacked_left_index());
        out << ",\"unpacked_right_index\":";
        write_index(out, port.get_unpacked_right_index());
        out << ",\"packed_width\":" << port.get_packed_width();
        out.put('}');
        first = false;
    }
    out << "],\"nets\":[";
    first = true;
    for (const auto& net : module.get_nets()) {
        if (!first) out.put(',');
        out << "{\"name\":";
        append_string(out, net.get_name());
        out << ",\"kind\":";
        append_string(out, net_type_name(net.get_kind()));
        out << ",\"width\":" << net.get_width() << ",\"left_index\":";
        write_index(out, net.get_left_index());
        out << ",\"right_index\":";
        write_index(out, net.get_right_index());
        out << ",\"unpacked_left_index\":";
        write_index(out, net.get_unpacked_left_index());
        out << ",\"unpacked_right_index\":";
        write_index(out, net.get_unpacked_right_index());
        out << ",\"packed_width\":" << net.get_packed_width();
        out.put('}');
        first = false;
    }
    out << "],\"instances\":[";
    first = true;
    for (const auto& instance : module.get_instances()) {
        if (instance.get_cell()) {
            throw std::runtime_error(
                "JSON netlist I/O currently supports parsed and generic-mapped "
                "netlists, not technology-mapped cell metadata");
        }
        if (!first) out.put(',');
        out << "{\"name\":";
        append_string(out, instance.get_name());
        out << ",\"cell_type\":";
        append_string(out, instance.get_cell_type());
        out << ",\"connections\":[";
        bool first_connection = true;
        for (const auto& connection : instance.get_connections()) {
            if (!first_connection) out.put(',');
            out << "{\"pin\":";
            append_string(out, connection.get_pin());
            out << ",\"net\":";
            append_string(out, connection.get_net());
            out.put('}');
            first_connection = false;
        }
        out << "],\"parameters\":{";
        bool first_parameter = true;
        for (const auto& [name, value] : instance.get_parameters()) {
            if (!first_parameter) out.put(',');
            append_string(out, name);
            out.put(':');
            append_string(out, value);
            first_parameter = false;
        }
        out << "},\"user_defined_primitive\":"
            << (instance.is_user_defined_primitive() ? "true" : "false")
            << ",\"sequential_primitive\":"
            << (instance.is_sequential_primitive() ? "true" : "false")
            << ",\"primitive_table\":[";
        bool first_row = true;
        for (const auto& row : instance.get_primitive_table()) {
            if (!first_row) out.put(',');
            out << "{\"inputs\":";
            append_string(out, row.inputs);
            out << ",\"output\":";
            append_string(out, std::string(1, row.output));
            out.put('}');
            first_row = false;
        }
        out << "]}";
        first = false;
    }
    out << "],\"assignments\":[";
    first = true;
    for (const auto& assignment : module.get_assignments()) {
        if (!first) out.put(',');
        out << "{\"lhs_expression\":";
        write_expression(out, assignment.get_lhs_expression());
        out << ",\"rhs_expression\":";
        write_expression(out, assignment.get_rhs_expression());
        out.put('}');
        first = false;
    }
    out << "],\"child_modules\":[";
    first = true;
    for (const auto& child : module.get_child_modules()) {
        if (!first) out.put(',');
        write_module(out, child);
        first = false;
    }
    out << "]}";
}

void validate_json_module(const Module& module) {
    for (const auto& instance : module.get_instances()) {
        if (instance.get_cell()) {
            throw std::runtime_error(
                "JSON netlist I/O currently supports parsed and generic-mapped "
                "netlists, not technology-mapped cell metadata");
        }
    }
    for (const auto& child : module.get_child_modules()) {
        validate_json_module(child);
    }
}

class JsonReader {
public:
    explicit JsonReader(std::istream& input) : input_(input) {}

    Module read() {
        expect('{');
        std::string format;
        unsigned version = 0;
        bool has_modules = false;
        bool has_root_module = false;
        Module result;
        while (!consume('}')) {
            const auto key = string();
            expect(':');
            if (key == "format") {
                format = string();
            } else if (key == "version") {
                version = unsigned_number();
            } else if (key == "modules") {
                expect('[');
                if (!consume(']')) {
                    result = module();
                    has_root_module = true;
                    if (!consume(']')) {
                        expect(',');
                        skip_value();
                        while (consume(',')) skip_value();
                        expect(']');
                        fail("module JSON reader accepts one root module at a time");
                    }
                }
                has_modules = true;
            } else {
                skip_value();
            }
            if (!consume(',')) {
                expect('}');
                break;
            }
        }
        whitespace();
        if (peek() != std::char_traits<char>::eof()) {
            fail("trailing data after JSON object");
        }
        if (format != "eda-netlist") fail("unsupported JSON netlist format");
        if (version != 1 && version != 2) fail("unsupported JSON netlist version");
        if (!has_modules) fail("JSON netlist is missing modules");
        if (!has_root_module) fail("modules must contain exactly one module");
        return result;
    }

private:
    std::istream& input_;
    std::size_t position_ = 0;

    [[noreturn]] void fail(const std::string& message) const {
        throw std::runtime_error("invalid JSON netlist at byte " +
                                 std::to_string(position_) + ": " + message);
    }

    void whitespace() {
        while (peek() == ' ' || peek() == '\t' || peek() == '\r' ||
               peek() == '\n') {
            (void)get();
        }
    }

    int peek() {
        return input_.peek();
    }

    char get() {
        const auto value = input_.get();
        if (value == std::char_traits<char>::eof()) fail("unexpected end of file");
        ++position_;
        return static_cast<char>(value);
    }

    bool consume(char ch) {
        whitespace();
        if (peek() == ch) {
            (void)get();
            return true;
        }
        return false;
    }

    void expect(char ch) {
        if (!consume(ch)) fail(std::string("expected '") + ch + "'");
    }

    static void append_utf8(std::string& out, unsigned codepoint) {
        if (codepoint <= 0x7f) {
            out.push_back(static_cast<char>(codepoint));
        } else if (codepoint <= 0x7ff) {
            out.push_back(static_cast<char>(0xc0 | (codepoint >> 6)));
            out.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
        } else if (codepoint <= 0xffff) {
            out.push_back(static_cast<char>(0xe0 | (codepoint >> 12)));
            out.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
            out.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
        } else {
            out.push_back(static_cast<char>(0xf0 | (codepoint >> 18)));
            out.push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3f)));
            out.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
            out.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
        }
    }

    unsigned hex4() {
        unsigned value = 0;
        for (int i = 0; i < 4; ++i) {
            const char ch = get();
            value <<= 4;
            if (ch >= '0' && ch <= '9') value |= static_cast<unsigned>(ch - '0');
            else if (ch >= 'a' && ch <= 'f') value |= static_cast<unsigned>(ch - 'a' + 10);
            else if (ch >= 'A' && ch <= 'F') value |= static_cast<unsigned>(ch - 'A' + 10);
            else fail("invalid unicode escape");
        }
        return value;
    }

    std::string string() {
        expect('"');
        std::string result;
        while (peek() != std::char_traits<char>::eof()) {
            const unsigned char ch = static_cast<unsigned char>(get());
            if (ch == '"') return result;
            if (ch < 0x20) fail("unescaped control character in string");
            if (ch != '\\') {
                result.push_back(static_cast<char>(ch));
                continue;
            }
            switch (get()) {
                case '"': result.push_back('"'); break;
                case '\\': result.push_back('\\'); break;
                case '/': result.push_back('/'); break;
                case 'b': result.push_back('\b'); break;
                case 'f': result.push_back('\f'); break;
                case 'n': result.push_back('\n'); break;
                case 'r': result.push_back('\r'); break;
                case 't': result.push_back('\t'); break;
                case 'u': {
                    unsigned codepoint = hex4();
                    if (codepoint >= 0xd800 && codepoint <= 0xdbff) {
                        if (get() != '\\' || get() != 'u') {
                            fail("missing low surrogate");
                        }
                        const unsigned low = hex4();
                        if (low < 0xdc00 || low > 0xdfff) fail("invalid low surrogate");
                        codepoint = 0x10000 + ((codepoint - 0xd800) << 10) +
                                    (low - 0xdc00);
                    } else if (codepoint >= 0xdc00 && codepoint <= 0xdfff) {
                        fail("unexpected low surrogate");
                    }
                    append_utf8(result, codepoint);
                    break;
                }
                default: fail("invalid string escape");
            }
        }
        fail("unterminated string");
    }

    void literal(const char* value) {
        while (*value) {
            if (peek() == std::char_traits<char>::eof() || get() != *value++) {
                fail("invalid literal");
            }
        }
    }

    bool boolean() {
        whitespace();
        if (peek() == 't') {
            literal("true");
            return true;
        }
        if (peek() == 'f') {
            literal("false");
            return false;
        }
        fail("expected boolean");
    }

    std::size_t unsigned_number() {
        whitespace();
        std::string digits;
        if (peek() == '0') {
            digits.push_back(get());
            if (peek() >= '0' && peek() <= '9') {
                fail("leading zero in integer");
            }
        } else {
            while (peek() >= '0' && peek() <= '9') digits.push_back(get());
        }
        if (digits.empty()) fail("expected non-negative integer");
        try {
            return static_cast<std::size_t>(std::stoull(digits));
        } catch (const std::exception&) {
            fail("integer is out of range");
        }
    }

    std::optional<std::int64_t> optional_index() {
        whitespace();
        if (peek() == 'n') {
            literal("null");
            return std::nullopt;
        }
        std::string number;
        if (peek() == '-') number.push_back(get());
        if (peek() == '0') {
            number.push_back(get());
            if (peek() >= '0' && peek() <= '9') {
                fail("leading zero in range index");
            }
        } else {
            while (peek() >= '0' && peek() <= '9') number.push_back(get());
        }
        if (number.empty() || number == "-") fail("expected integer or null");
        try {
            return std::stoll(number);
        } catch (const std::exception&) {
            fail("range index is out of range");
        }
    }

    void skip_number() {
        whitespace();
        if (peek() == '-') (void)get();
        if (peek() == '0') {
            (void)get();
            if (peek() >= '0' && peek() <= '9') fail("invalid JSON number");
        } else {
            if (peek() < '1' || peek() > '9') fail("expected JSON value");
            while (peek() >= '0' && peek() <= '9') (void)get();
        }
        if (peek() == '.') {
            (void)get();
            if (peek() < '0' || peek() > '9') fail("invalid JSON fraction");
            while (peek() >= '0' && peek() <= '9') (void)get();
        }
        if (peek() == 'e' || peek() == 'E') {
            (void)get();
            if (peek() == '+' || peek() == '-') (void)get();
            if (peek() < '0' || peek() > '9') fail("invalid JSON exponent");
            while (peek() >= '0' && peek() <= '9') (void)get();
        }
    }

    void skip_value() {
        whitespace();
        if (peek() == std::char_traits<char>::eof()) fail("expected JSON value");
        if (peek() == '"') {
            (void)string();
        } else if (peek() == '{') {
            expect('{');
            if (consume('}')) return;
            do {
                (void)string();
                expect(':');
                skip_value();
            } while (consume(','));
            expect('}');
        } else if (peek() == '[') {
            expect('[');
            if (consume(']')) return;
            do { skip_value(); } while (consume(','));
            expect(']');
        } else if (peek() == 't') {
            literal("true");
        } else if (peek() == 'f') {
            literal("false");
        } else if (peek() == 'n') {
            literal("null");
        } else {
            skip_number();
        }
    }

    template <typename Callback>
    void array(Callback&& callback) {
        expect('[');
        if (consume(']')) return;
        do { callback(); } while (consume(','));
        expect(']');
    }

    ExpressionKind expression_kind(const std::string& name) {
        if (name == "identifier") return ExpressionKind::Identifier;
        if (name == "literal") return ExpressionKind::Literal;
        if (name == "unary") return ExpressionKind::Unary;
        if (name == "binary") return ExpressionKind::Binary;
        if (name == "conditional") return ExpressionKind::Conditional;
        if (name == "bit_select") return ExpressionKind::BitSelect;
        if (name == "range_select") return ExpressionKind::RangeSelect;
        if (name == "concatenation") return ExpressionKind::Concatenation;
        if (name == "opaque") return ExpressionKind::Opaque;
        fail("unknown expression kind");
    }

    Expression expression() {
        expect('{');
        ExpressionKind kind = ExpressionKind::Opaque;
        std::string value;
        std::string source;
        std::size_t width = 0;
        std::vector<Expression> operands;
        while (!consume('}')) {
            const auto key = string();
            expect(':');
            if (key == "kind") kind = expression_kind(string());
            else if (key == "value") value = string();
            else if (key == "source") source = string();
            else if (key == "width") width = unsigned_number();
            else if (key == "operands") {
                array([&] { operands.push_back(expression()); });
            } else skip_value();
            if (!consume(',')) { expect('}'); break; }
        }
        return Expression(kind, std::move(value), std::move(source),
                          std::move(operands), width);
    }

    PortDirection port_direction() {
        const auto value = string();
        if (value == "input") return PortDirection::Input;
        if (value == "output") return PortDirection::Output;
        if (value == "inout") return PortDirection::InOut;
        fail("unknown port direction");
    }

    NetType net_type() {
        const auto value = string();
        if (value == "input") return NetType::Input;
        if (value == "output") return NetType::Output;
        if (value == "inout") return NetType::InOut;
        if (value == "internal") return NetType::Internal;
        fail("unknown net type");
    }

    Port port() {
        expect('{');
        std::string name;
        PortDirection direction = PortDirection::Input;
        std::size_t width = 1;
        std::optional<std::int64_t> left;
        std::optional<std::int64_t> right;
        std::optional<std::int64_t> unpacked_left;
        std::optional<std::int64_t> unpacked_right;
        std::size_t packed_width = 0;
        while (!consume('}')) {
            const auto key = string();
            expect(':');
            if (key == "name") name = string();
            else if (key == "direction") direction = port_direction();
            else if (key == "width") width = unsigned_number();
            else if (key == "left_index") left = optional_index();
            else if (key == "right_index") right = optional_index();
            else if (key == "unpacked_left_index") unpacked_left = optional_index();
            else if (key == "unpacked_right_index") unpacked_right = optional_index();
            else if (key == "packed_width") packed_width = unsigned_number();
            else skip_value();
            if (!consume(',')) { expect('}'); break; }
        }
        if (name.empty() || width == 0) fail("invalid port");
        return Port(std::move(name), direction, width, left, right,
                    unpacked_left, unpacked_right, packed_width);
    }

    Net net() {
        whitespace();
        if (peek() == '"') return Net(string(), NetType::Internal);
        expect('{');
        std::string name;
        NetType kind = NetType::Internal;
        std::size_t width = 1;
        std::optional<std::int64_t> left;
        std::optional<std::int64_t> right;
        std::optional<std::int64_t> unpacked_left;
        std::optional<std::int64_t> unpacked_right;
        std::size_t packed_width = 0;
        while (!consume('}')) {
            const auto key = string();
            expect(':');
            if (key == "name") name = string();
            else if (key == "kind") kind = net_type();
            else if (key == "width") width = unsigned_number();
            else if (key == "left_index") left = optional_index();
            else if (key == "right_index") right = optional_index();
            else if (key == "unpacked_left_index") unpacked_left = optional_index();
            else if (key == "unpacked_right_index") unpacked_right = optional_index();
            else if (key == "packed_width") packed_width = unsigned_number();
            else skip_value();
            if (!consume(',')) { expect('}'); break; }
        }
        if (name.empty() || width == 0) fail("invalid net");
        return Net(std::move(name), kind, width, left, right,
                   unpacked_left, unpacked_right, packed_width);
    }

    PinConnection connection() {
        expect('{');
        std::string pin;
        std::string net_name;
        while (!consume('}')) {
            const auto key = string();
            expect(':');
            if (key == "pin") pin = string();
            else if (key == "net") net_name = string();
            else skip_value();
            if (!consume(',')) { expect('}'); break; }
        }
        if (pin.empty()) fail("connection has no pin");
        return PinConnection(std::move(pin), std::move(net_name));
    }

    Instance instance() {
        expect('{');
        std::string name;
        std::string type;
        std::vector<PinConnection> connections;
        std::map<std::string, std::string> parameters;
        std::vector<UserDefinedPrimitiveRow> table;
        bool udp = false;
        bool sequential = false;
        while (!consume('}')) {
            const auto key = string();
            expect(':');
            if (key == "name") name = string();
            else if (key == "cell_type") type = string();
            else if (key == "connections") {
                whitespace();
                if (peek() == '[') {
                    array([&] { connections.push_back(connection()); });
                } else {
                    expect('{');
                    if (!consume('}')) {
                        do {
                            auto pin = string();
                            expect(':');
                            connections.emplace_back(std::move(pin), string());
                        } while (consume(','));
                        expect('}');
                    }
                }
            } else if (key == "parameters") {
                expect('{');
                if (!consume('}')) {
                    do {
                        const auto parameter = string();
                        expect(':');
                        parameters[parameter] = string();
                    } while (consume(','));
                    expect('}');
                }
            } else if (key == "user_defined_primitive") udp = boolean();
            else if (key == "sequential_primitive") sequential = boolean();
            else if (key == "primitive_table") {
                array([&] {
                    expect('{');
                    std::string inputs;
                    std::string output;
                    while (!consume('}')) {
                        const auto field = string();
                        expect(':');
                        if (field == "inputs") inputs = string();
                        else if (field == "output") output = string();
                        else skip_value();
                        if (!consume(',')) { expect('}'); break; }
                    }
                    if (output.size() != 1) fail("invalid UDP row");
                    table.push_back({std::move(inputs), output.front()});
                });
            } else skip_value();
            if (!consume(',')) { expect('}'); break; }
        }
        if (name.empty() || type.empty()) fail("invalid instance");
        return Instance(std::move(name), std::move(type), std::move(connections),
                        std::nullopt, std::move(parameters), udp, std::move(table),
                        sequential);
    }

    ContinuousAssignment assignment() {
        expect('{');
        Expression lhs;
        Expression rhs;
        bool has_lhs = false;
        bool has_rhs = false;
        while (!consume('}')) {
            const auto key = string();
            expect(':');
            if (key == "lhs_expression") {
                lhs = expression();
                has_lhs = true;
            } else if (key == "rhs_expression") {
                rhs = expression();
                has_rhs = true;
            } else {
                skip_value();
            }
            if (!consume(',')) { expect('}'); break; }
        }
        if (!has_lhs || !has_rhs) fail("assignment lacks expression tree");
        return ContinuousAssignment(std::move(lhs), std::move(rhs));
    }

    Module module() {
        expect('{');
        std::string name;
        std::string hierarchy_name;
        bool behavioral = false;
        std::vector<Port> ports;
        std::vector<Net> nets;
        std::vector<Instance> instances;
        std::vector<ContinuousAssignment> assignments;
        std::vector<Module> children;
        while (!consume('}')) {
            const auto key = string();
            expect(':');
            if (key == "name") name = string();
            else if (key == "hierarchy_instance_name") hierarchy_name = string();
            else if (key == "is_behavioral") behavioral = boolean();
            else if (key == "ports") array([&] { ports.push_back(port()); });
            else if (key == "nets") array([&] { nets.push_back(net()); });
            else if (key == "instances") array([&] { instances.push_back(instance()); });
            else if (key == "assignments") array([&] { assignments.push_back(assignment()); });
            else if (key == "child_modules") {
                array([&] { children.push_back(module()); });
            } else {
                skip_value();
            }
            if (!consume(',')) { expect('}'); break; }
        }
        if (name.empty()) fail("module has no name");
        return Module(std::move(name), std::move(ports), std::move(nets),
                      std::move(instances), std::move(assignments), behavioral,
                      std::move(hierarchy_name), std::move(children));
    }
};

void write_json_stream(std::ostream& out, const Module& module) {
    validate_json_module(module);
    out << "{\"format\":\"eda-netlist\",\"version\":2,\"modules\":[";
    write_module(out, module);
    out << "]}\n";
    if (!out) throw std::runtime_error("failed writing JSON netlist");
}

}  // namespace

std::string to_json(const Module& module) {
    std::ostringstream output;
    write_json_stream(output, module);
    return output.str();
}

void write_json(const Module& module, const std::string& path) {
    validate_json_module(module);
    std::ofstream output(path);
    if (!output) throw std::runtime_error("could not open JSON output: " + path);
    write_json_stream(output, module);
}

Module read_json(const std::string& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("could not open JSON netlist: " + path);
    auto result = JsonReader(input).read();
    if (input.bad()) throw std::runtime_error("failed reading JSON netlist: " + path);
    return result;
}

}  // namespace labneura::eda
