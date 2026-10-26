#include "labneura/eda/netlist.h"

#include "slang/ast/ASTSerializer.h"
#include "slang/ast/Compilation.h"
#include "slang/ast/expressions/AssignmentExpressions.h"
#include "slang/ast/expressions/LiteralExpressions.h"
#include "slang/ast/expressions/MiscExpressions.h"
#include "slang/ast/expressions/OperatorExpressions.h"
#include "slang/ast/expressions/SelectExpressions.h"
#include "slang/ast/symbols/CompilationUnitSymbols.h"
#include "slang/ast/symbols/InstanceSymbols.h"
#include "slang/ast/symbols/BlockSymbols.h"
#include "slang/ast/symbols/MemberSymbols.h"
#include "slang/ast/symbols/PortSymbols.h"
#include "slang/ast/types/Type.h"
#include "slang/diagnostics/TextDiagnosticClient.h"
#include "slang/driver/Driver.h"
#include "slang/ast/SemanticFacts.h"
#include "slang/syntax/SyntaxTree.h"
#include "slang/text/Json.h"
#include "slang/text/SourceManager.h"

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

namespace labneura::eda {
namespace {

struct Declaration {
    std::string name;
    std::string direction;
};

struct ParsedInstance {
    Instance instance;
};

std::string trim(const std::string& text) {
    const auto first = text.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    const auto last = text.find_last_not_of(" \t\r\n");
    return text.substr(first, last - first + 1);
}

std::string remove_comments(const std::string& source) {
    std::string result;
    result.reserve(source.size());
    bool line_comment = false;
    bool block_comment = false;
    for (std::size_t i = 0; i < source.size(); ++i) {
        const char current = source[i];
        const char next = i + 1 < source.size() ? source[i + 1] : '\0';
        if (line_comment) {
            if (current == '\n') {
                line_comment = false;
                result += current;
            }
            continue;
        }
        if (block_comment) {
            if (current == '*' && next == '/') {
                block_comment = false;
                ++i;
            } else if (current == '\n') {
                result += current;
            }
            continue;
        }
        if (current == '/' && next == '/') {
            line_comment = true;
            ++i;
        } else if (current == '/' && next == '*') {
            block_comment = true;
            ++i;
        } else {
            result += current;
        }
    }
    if (block_comment) throw std::runtime_error("unterminated block comment");
    return result;
}

std::vector<std::string> split_names(const std::string& text) {
    static const std::regex identifier(R"([A-Za-z_][A-Za-z0-9_$]*)");
    std::vector<std::string> names;
    std::stringstream stream(text);
    std::string name;
    while (std::getline(stream, name, ',')) {
        name = trim(name);
        if (!std::regex_match(name, identifier)) {
            throw std::runtime_error(
                "expected comma-separated scalar identifiers, got: " + text);
        }
        names.push_back(name);
    }
    if (names.empty()) throw std::runtime_error("expected at least one identifier");
    return names;
}

std::string source_range_text(const slang::SourceRange& range,
                              const slang::SourceManager& source_manager) {
    const auto start = range.start();
    const auto end = range.end();
    if (!start.valid() || !end.valid() || start.buffer() != end.buffer() ||
        end.offset() < start.offset()) {
        throw std::runtime_error(
            "could not determine source text for Verilog assignment expression");
    }

    const auto source = source_manager.getSourceText(start.buffer());
    if (end.offset() > source.size()) {
        throw std::runtime_error(
            "Verilog assignment expression source range is out of bounds");
    }
    return trim(std::string(source.substr(start.offset(),
                                          end.offset() - start.offset())));
}

std::string expression_source(const slang::ast::Expression& expression,
                              const slang::SourceManager& source_manager) {
    return source_range_text(expression.sourceRange, source_manager);
}

std::string liberty_cell_name(const std::string& verilog_type) {
    if (verilog_type == "and" || verilog_type == "AND" ||
        verilog_type == "AND2_X1") {
        return "sky130_fd_sc_hd__and2_1";
    }
    if (verilog_type == "or" || verilog_type == "OR" ||
        verilog_type == "OR2_X1") {
        return "sky130_fd_sc_hd__or2_1";
    }
    if (verilog_type == "xor" || verilog_type == "XOR" ||
        verilog_type == "XOR2_X1") {
        return "sky130_fd_sc_hd__xor2_1";
    }
    if (verilog_type == "INV_X1") return "sky130_fd_sc_hd__inv_1";
    if (verilog_type == "BUF_X1") return "sky130_fd_sc_hd__buf_1";
    if (verilog_type == "TIELO_X1" || verilog_type == "TIEHI_X1") {
        return "sky130_fd_sc_hd__conb_1";
    }
    if (verilog_type == "MUX2_X1") return "sky130_fd_sc_hd__mux2_1";
    return verilog_type;
}

std::string physical_pin_name(const StandardCell& cell,
                              const std::string& logical_cell_type,
                              const std::string& pin_name) {
    for (const auto& pin : cell.get_pins()) {
        if (pin.get_name() == pin_name) return pin_name;
    }

    if (pin_name == "Y") {
        const std::string expected_function =
            logical_cell_type == "TIELO_X1"
                ? "0"
                : logical_cell_type == "TIEHI_X1" ? "1" : "";
        for (const auto& pin : cell.get_pins()) {
            if (pin.get_direction() == "output" &&
                (expected_function.empty() ||
                 pin.get_function() == expected_function)) {
                return pin.get_name();
            }
        }
    }

    throw std::runtime_error("standard cell '" + cell.get_name() +
                             "' has no pin corresponding to '" + pin_name +
                             "'");
}

std::vector<PinConnection> physical_connections(
    const StandardCell& cell, const std::string& logical_cell_type,
    std::vector<PinConnection> connections) {
    for (auto& connection : connections) {
        connection = PinConnection(
            physical_pin_name(cell, logical_cell_type, connection.get_pin()),
            connection.get_net());
    }
    return connections;
}

bool is_cell_type(const StandardCellLibrary& library,
                  const std::string& verilog_type) {
    return library.find_cell(verilog_type) ||
           library.find_cell(liberty_cell_name(verilog_type));
}

const StandardCell& resolve_cell(const StandardCellLibrary& library,
                                 const std::string& verilog_type) {
    const auto liberty_name = liberty_cell_name(verilog_type);
    const auto* cell = library.find_cell(liberty_name);
    if (!cell) cell = library.find_cell(verilog_type);
    if (!cell) {
        throw std::runtime_error("standard cell '" + verilog_type +
                                 "' was not found in Liberty library '" +
                                 library.get_name() + "'");
    }
    return *cell;
}

std::string make_cell_stub_source(const StandardCellLibrary& library) {
    std::ostringstream source;
    for (const auto& cell : library.get_cells()) {
        source << "module " << cell.get_name() << " (";
        bool first = true;
        for (const auto& pin : cell.get_pins()) {
            if (pin.get_direction() != "input" &&
                pin.get_direction() != "output" &&
                pin.get_direction() != "inout")
                continue;
            if (!first) source << ", ";
            source << pin.get_direction() << ' ' << pin.get_name();
            first = false;
        }
        source << "); endmodule\n";
    }

    const std::vector<std::pair<std::string, std::string>> aliases = {
        {"AND2_X1", "input A, input B, output Y"},
        {"OR2_X1", "input A, input B, output Y"},
        {"XOR2_X1", "input A, input B, output Y"},
        {"INV_X1", "input A, output Y"},
        {"BUF_X1", "input A, output Y"},
        {"TIELO_X1", "output Y"},
        {"TIEHI_X1", "output Y"},
        {"MUX2_X1", "input A0, input A1, input S, output Y"},
    };
    for (const auto& [alias, pins] : aliases) {
        if (!library.find_cell(alias) &&
            library.find_cell(liberty_cell_name(alias))) {
            source << "module " << alias << " (" << pins << "); endmodule\n";
        }
    }
    return source.str();
}

std::vector<PinConnection> module_pin_connections(
    const slang::ast::InstanceSymbol& instance,
    const slang::SourceManager& source_manager, const StandardCell& cell,
    const std::string& logical_cell_type) {
    std::vector<PinConnection> connections;
    for (const auto* connection : instance.getPortConnections()) {
        if (!connection->getExpression()) continue;
        const std::string pin_name = physical_pin_name(
            cell, logical_cell_type, std::string(connection->port.name));
        if (!cell.find_pin(pin_name)) {
            throw std::runtime_error("cell '" + cell.get_name() +
                                     "' has no pin named '" + pin_name + "'");
        }
        connections.emplace_back(
            pin_name, expression_source(*connection->getExpression(),
                                        source_manager));
    }
    return connections;
}

std::vector<PinConnection> module_port_connections(
    const slang::ast::InstanceSymbol& instance,
    const slang::SourceManager& source_manager) {
    std::vector<PinConnection> connections;
    for (const auto* connection : instance.getPortConnections()) {
        if (!connection->getExpression()) continue;
        connections.emplace_back(
            std::string(connection->port.name),
            expression_source(*connection->getExpression(), source_manager));
    }
    return connections;
}

std::vector<PinConnection> primitive_pin_connections(
    const slang::ast::PrimitiveInstanceSymbol& instance,
    const slang::SourceManager& source_manager, const StandardCell& cell) {
    std::vector<const CellPin*> outputs;
    std::vector<const CellPin*> inputs;
    for (const auto& pin : cell.get_pins()) {
        if (pin.get_direction() == "output") {
            outputs.push_back(&pin);
        } else if (pin.get_direction() == "input") {
            inputs.push_back(&pin);
        }
    }

    const auto expressions = instance.getPortConnections();
    if (expressions.size() != outputs.size() + inputs.size()) {
        throw std::runtime_error("primitive '" + std::string(instance.primitiveType.name) +
                                 "' does not match the pin count of cell '" +
                                 cell.get_name() + "'");
    }

    std::vector<PinConnection> connections;
    connections.reserve(expressions.size());
    std::size_t expression_index = 0;
    for (const auto* pin : outputs) {
        connections.emplace_back(pin->get_name(),
                                 expression_source(*expressions[expression_index++],
                                                   source_manager));
    }
    for (const auto* pin : inputs) {
        connections.emplace_back(pin->get_name(),
                                 expression_source(*expressions[expression_index++],
                                                   source_manager));
    }
    return connections;
}

struct LoweredAssignment {
    std::vector<Instance> instances;
    std::vector<Net> nets;
};

std::optional<LoweredAssignment> lower_mux4_assignment(
    const slang::ast::AssignmentExpression& assignment,
    const slang::SourceManager& source_manager,
    const StandardCellLibrary& library, std::size_t generated_index) {
    using slang::ast::ExpressionKind;
    if (assignment.left().kind != ExpressionKind::NamedValue ||
        assignment.left().type->getBitWidth() != 1 ||
        assignment.right().kind != ExpressionKind::ElementSelect) {
        return std::nullopt;
    }

    const auto& select =
        assignment.right().as<slang::ast::ElementSelectExpression>();
    if (select.value().kind != ExpressionKind::NamedValue ||
        !select.value().type->hasFixedRange() ||
        select.value().type->getFixedRange().lower() != 0 ||
        select.value().type->getFixedRange().upper() != 3 ||
        select.selector().type->getBitWidth() != 2 ||
        !select.selector().type->hasFixedRange() ||
        select.selector().type->getFixedRange().lower() != 0 ||
        select.selector().type->getFixedRange().upper() != 1 ||
        select.selector().type->getFixedRange().width() != 2) {
        return std::nullopt;
    }

    const std::string data_name =
        std::string(select.value().as<slang::ast::NamedValueExpression>()
                        .symbol.name);
    const std::string select_name =
        expression_source(select.selector(), source_manager);
    const std::string output =
        expression_source(assignment.left(), source_manager);

    if (const auto* mux4 = library.find_cell("MUX4_X1")) {
        std::vector<PinConnection> connections;
        for (int bit = 0; bit < 4; ++bit) {
            connections.emplace_back(
                "A" + std::to_string(bit),
                data_name + "[" + std::to_string(bit) + "]");
        }
        for (int bit = 0; bit < 2; ++bit) {
            connections.emplace_back(
                "S" + std::to_string(bit),
                select_name + "[" + std::to_string(bit) + "]");
        }
        connections.emplace_back("Y", output);
        return LoweredAssignment{
            {Instance("__labneura_mux" + std::to_string(generated_index),
                      mux4->get_name(), std::move(connections), *mux4)},
            {}};
    }

    const auto& mux2 = resolve_cell(library, "MUX2_X1");
    const std::string low_net =
        "__labneura_muxnet" + std::to_string(generated_index);
    const std::string high_net =
        "__labneura_muxnet" + std::to_string(generated_index + 1);
    const auto make_mux = [&](std::size_t index, const std::string& a0,
                              const std::string& a1, const std::string& sel,
                              const std::string& y) {
        std::vector<PinConnection> connections = {
            {"A0", a0}, {"A1", a1}, {"S", sel}, {"Y", y}};
        connections =
            physical_connections(mux2, "MUX2_X1", std::move(connections));
        return Instance("__labneura_mux" + std::to_string(index),
                        mux2.get_name(), std::move(connections), mux2);
    };

    std::vector<Instance> instances;
    instances.reserve(3);
    instances.push_back(make_mux(generated_index, data_name + "[0]",
                                 data_name + "[1]", select_name + "[0]",
                                 low_net));
    instances.push_back(make_mux(generated_index + 1, data_name + "[2]",
                                 data_name + "[3]", select_name + "[0]",
                                 high_net));
    instances.push_back(make_mux(generated_index + 2, low_net, high_net,
                                 select_name + "[1]", output));
    return LoweredAssignment{
        std::move(instances),
        {Net(low_net, NetType::Internal), Net(high_net, NetType::Internal)}};
}

std::optional<std::string> scalar_signal(
    const slang::ast::Expression& expression,
    const slang::SourceManager& source_manager) {
    using slang::ast::ExpressionKind;
    if (expression.type->getBitWidth() != 1) return std::nullopt;
    if (expression.kind == ExpressionKind::NamedValue) {
        return std::string(
            expression.as<slang::ast::NamedValueExpression>().symbol.name);
    }
    if (expression.kind == ExpressionKind::ElementSelect) {
        return expression_source(expression, source_manager);
    }
    return std::nullopt;
}

std::optional<bool> scalar_constant(const slang::ast::Expression& expression) {
    using slang::ast::ExpressionKind;
    if (expression.kind == ExpressionKind::IntegerLiteral) {
        const auto bit =
            expression.as<slang::ast::IntegerLiteral>().getValue()[0];
        if (bit.isUnknown()) {
            throw std::runtime_error(
                "scalar X/Z constants cannot be mapped to tie cells");
        }
        return static_cast<bool>(bit);
    }
    if (expression.kind == ExpressionKind::UnbasedUnsizedIntegerLiteral) {
        const auto bit = expression.as<
            slang::ast::UnbasedUnsizedIntegerLiteral>().getLiteralValue();
        if (bit.isUnknown()) {
            throw std::runtime_error(
                "scalar X/Z constants cannot be mapped to tie cells");
        }
        return static_cast<bool>(bit);
    }

    const auto* constant = expression.getConstant();
    if (!constant || !constant->isInteger()) return std::nullopt;

    const auto bit = constant->integer()[0];
    if (bit.isUnknown()) {
        throw std::runtime_error(
            "scalar X/Z constants cannot be mapped to tie cells");
    }
    return static_cast<bool>(bit);
}

std::optional<LoweredAssignment> lower_scalar_boolean_assignment(
    const slang::ast::AssignmentExpression& assignment,
    const slang::SourceManager& source_manager,
    const StandardCellLibrary& library, std::size_t generated_index) {
    using slang::ast::ExpressionKind;
    const auto output = scalar_signal(assignment.left(), source_manager);
    if (!output) return std::nullopt;

    LoweredAssignment lowered;
    std::size_t next_index = generated_index;
    const auto make_net = [&]() {
        const auto name = "__labneura_logicnet" + std::to_string(next_index++);
        lowered.nets.emplace_back(name, NetType::Internal);
        return name;
    };
    const auto add_cell = [&](const std::string& cell_name,
                              std::vector<PinConnection> connections,
                              const std::string& destination) {
        const auto& cell = resolve_cell(library, cell_name);
        connections.emplace_back("Y", destination);
        connections =
            physical_connections(cell, cell_name, std::move(connections));
        lowered.instances.emplace_back(
            "__labneura_logic" + std::to_string(next_index++),
            cell.get_name(), std::move(connections), cell);
    };

    const auto lower_into = [&](auto&& self, const slang::ast::Expression& expr,
                                const std::string& destination) -> bool {
        if (expr.type->getBitWidth() != 1) return false;
        const auto lower_value = [&](const slang::ast::Expression& child)
            -> std::optional<std::string> {
            if (const auto signal = scalar_signal(child, source_manager)) {
                return signal;
            }
            const auto net = make_net();
            if (!self(self, child, net)) return std::nullopt;
            return net;
        };

        if (expr.kind == ExpressionKind::BinaryOp) {
            const auto& binary = expr.as<slang::ast::BinaryExpression>();
            const std::string op =
                source_range_text(binary.opRange, source_manager);
            std::string cell_name;
            if (op == "&" || op == "&&") {
                cell_name = "AND2_X1";
            } else if (op == "|" || op == "||") {
                cell_name = "OR2_X1";
            } else if (op == "^") {
                cell_name = "XOR2_X1";
            } else {
                return false;
            }
            const auto left = lower_value(binary.left());
            const auto right = lower_value(binary.right());
            if (!left || !right) return false;
            add_cell(cell_name, {{"A", *left}, {"B", *right}}, destination);
            return true;
        }

        if (expr.kind == ExpressionKind::UnaryOp) {
            const auto& unary = expr.as<slang::ast::UnaryExpression>();
            const auto op = source_range_text(unary.opRange, source_manager);
            if (op != "~" && op != "!") return false;
            const auto operand = lower_value(unary.operand());
            if (!operand) return false;
            add_cell("INV_X1", {{"A", *operand}}, destination);
            return true;
        }

        if (expr.kind == ExpressionKind::ConditionalOp) {
            const auto& conditional =
                expr.as<slang::ast::ConditionalExpression>();
            if (conditional.conditions.size() != 1) return false;
            const auto select =
                lower_value(*conditional.conditions.front().expr);
            const auto when_true = lower_value(conditional.left());
            const auto when_false = lower_value(conditional.right());
            if (!select || !when_true || !when_false) return false;
            add_cell("MUX2_X1",
                     {{"A0", *when_false}, {"A1", *when_true}, {"S", *select}},
                     destination);
            return true;
        }

        if (const auto constant = scalar_constant(expr)) {
            add_cell(*constant ? "TIEHI_X1" : "TIELO_X1", {}, destination);
            return true;
        }
        if (const auto signal = scalar_signal(expr, source_manager)) {
            add_cell("BUF_X1", {{"A", *signal}}, destination);
            return true;
        }
        return false;
    };

    if (!lower_into(lower_into, assignment.right(), *output)) {
        return std::nullopt;
    }
    return lowered;
}

Expression lower_expression(const slang::ast::Expression& expression,
                            const slang::SourceManager& source_manager) {
    using slang::ast::ExpressionKind;
    const std::string source = expression_source(expression, source_manager);
    switch (expression.kind) {
        case ExpressionKind::NamedValue: {
            const auto& named = expression.as<slang::ast::NamedValueExpression>();
            return {labneura::eda::ExpressionKind::Identifier,
                    std::string(named.symbol.name), source};
        }
        case ExpressionKind::IntegerLiteral:
        case ExpressionKind::RealLiteral:
        case ExpressionKind::TimeLiteral:
        case ExpressionKind::UnbasedUnsizedIntegerLiteral:
        case ExpressionKind::NullLiteral:
        case ExpressionKind::StringLiteral:
            return {labneura::eda::ExpressionKind::Literal, source, source};
        case ExpressionKind::ElementSelect: {
            const auto& select =
                expression.as<slang::ast::ElementSelectExpression>();
            return {labneura::eda::ExpressionKind::BitSelect, "", source,
                    {lower_expression(select.value(), source_manager),
                     lower_expression(select.selector(), source_manager)}};
        }
        case ExpressionKind::RangeSelect: {
            const auto& select =
                expression.as<slang::ast::RangeSelectExpression>();
            return {labneura::eda::ExpressionKind::RangeSelect, "", source,
                    {lower_expression(select.value(), source_manager),
                     lower_expression(select.left(), source_manager),
                     lower_expression(select.right(), source_manager)}};
        }
        case ExpressionKind::UnaryOp: {
            const auto& unary = expression.as<slang::ast::UnaryExpression>();
            return {labneura::eda::ExpressionKind::Unary,
                    source_range_text(unary.opRange, source_manager), source,
                    {lower_expression(unary.operand(), source_manager)}};
        }
        case ExpressionKind::BinaryOp: {
            const auto& binary = expression.as<slang::ast::BinaryExpression>();
            const auto operator_text =
                source_range_text(binary.opRange, source_manager);
            return {labneura::eda::ExpressionKind::Binary, operator_text, source,
                    {lower_expression(binary.left(), source_manager),
                     lower_expression(binary.right(), source_manager)}};
        }
        case ExpressionKind::ConditionalOp: {
            const auto& conditional =
                expression.as<slang::ast::ConditionalExpression>();
            std::vector<labneura::eda::Expression> operands;
            for (const auto& condition : conditional.conditions) {
                operands.push_back(
                    lower_expression(*condition.expr, source_manager));
            }
            operands.push_back(
                lower_expression(conditional.left(), source_manager));
            operands.push_back(
                lower_expression(conditional.right(), source_manager));
            return {labneura::eda::ExpressionKind::Conditional, "?:", source,
                    std::move(operands)};
        }
        case ExpressionKind::Concatenation: {
            const auto& concat =
                expression.as<slang::ast::ConcatenationExpression>();
            std::vector<labneura::eda::Expression> operands;
            for (const auto* operand : concat.operands()) {
                operands.push_back(lower_expression(*operand, source_manager));
            }
            return {labneura::eda::ExpressionKind::Concatenation, "", source,
                    std::move(operands)};
        }
        default:
            return {labneura::eda::ExpressionKind::Opaque, "", source};
    }
}

Module summarize_module(const slang::ast::InstanceSymbol& instance,
                       const slang::SourceManager& source_manager,
                       const StandardCellLibrary* cell_library = nullptr) {
    std::vector<Port> ports;
    std::vector<Net> nets;
    std::vector<Instance> instances;
    std::vector<ContinuousAssignment> assignments;
    std::unordered_set<std::string> port_names;
    std::unordered_set<std::string> instance_names;
    std::unordered_set<std::string> net_names;
    std::size_t generated_mux_index = 0;

    for (const auto* port_symbol : instance.body.getPortList()) {
        if (port_symbol->kind != slang::ast::SymbolKind::Port) continue;
        const auto& port = port_symbol->as<slang::ast::PortSymbol>();
        const std::string port_name(port.name);
        port_names.insert(port_name);
        net_names.insert(port_name);
        PortDirection direction = PortDirection::InOut;
        if (port.direction == slang::ast::ArgumentDirection::In) {
            direction = PortDirection::Input;
        } else if (port.direction == slang::ast::ArgumentDirection::Out) {
            direction = PortDirection::Output;
        }
        ports.emplace_back(port_name, direction);
        const auto net_type =
            direction == PortDirection::Input
                ? NetType::Input
                : direction == PortDirection::Output ? NetType::Output
                                                     : NetType::InOut;
        nets.emplace_back(port_name, net_type);
    }

    for (const auto& member : instance.body.members()) {
        if ((member.kind == slang::ast::SymbolKind::Net ||
             member.kind == slang::ast::SymbolKind::Variable) &&
            port_names.count(std::string(member.name)) == 0) {
            nets.emplace_back(std::string(member.name), NetType::Internal);
            net_names.insert(std::string(member.name));
        } else if (member.kind == slang::ast::SymbolKind::Instance) {
            const auto& child = member.as<slang::ast::InstanceSymbol>();
            instance_names.insert(std::string(child.name));
            const std::string cell_type(child.getDefinition().name);
            std::optional<StandardCell> cell;
            std::vector<PinConnection> connections;
            if (cell_library) {
                const auto& definition = resolve_cell(*cell_library, cell_type);
                cell = definition;
                connections = module_pin_connections(child, source_manager,
                                                     definition, cell_type);
            } else {
                connections = module_port_connections(child, source_manager);
            }
            instances.emplace_back(
                std::string(child.name), cell_type, std::move(connections),
                std::move(cell));
        } else if (member.kind == slang::ast::SymbolKind::PrimitiveInstance) {
            const auto& child =
                member.as<slang::ast::PrimitiveInstanceSymbol>();
            instance_names.insert(std::string(child.name));
            const std::string cell_type(child.primitiveType.name);
            std::optional<StandardCell> cell;
            std::vector<PinConnection> connections;
            if (cell_library) {
                const auto& definition = resolve_cell(*cell_library, cell_type);
                cell = definition;
                connections = primitive_pin_connections(child, source_manager,
                                                        definition);
            }
            instances.emplace_back(
                std::string(child.name), cell_type, std::move(connections),
                std::move(cell));
        } else if (member.kind == slang::ast::SymbolKind::ContinuousAssign) {
            const auto& assignment =
                member.as<slang::ast::ContinuousAssignSymbol>().getAssignment()
                    .as<slang::ast::AssignmentExpression>();
            if (cell_library) {
                auto mux = lower_mux4_assignment(assignment, source_manager,
                                                 *cell_library,
                                                 generated_mux_index);
                if (mux) {
                    const auto has_name_collision = [&]() {
                        for (const auto& generated : mux->instances) {
                            if (instance_names.count(generated.get_name()) !=
                                    0 ||
                                net_names.count(generated.get_name()) != 0) {
                                return true;
                            }
                        }
                        for (const auto& generated : mux->nets) {
                            if (net_names.count(generated.get_name()) != 0 ||
                                instance_names.count(generated.get_name()) !=
                                    0) {
                                return true;
                            }
                        }
                        return false;
                    };
                    while (has_name_collision()) {
                        ++generated_mux_index;
                        mux = lower_mux4_assignment(
                            assignment, source_manager, *cell_library,
                            generated_mux_index);
                        if (!mux) {
                            throw std::runtime_error(
                                "failed to generate unique mux instance and net names");
                        }
                    }
                    for (auto& generated_net : mux->nets) {
                        net_names.insert(generated_net.get_name());
                        nets.push_back(std::move(generated_net));
                    }
                    for (auto& generated_instance : mux->instances) {
                        instance_names.insert(generated_instance.get_name());
                        instances.push_back(std::move(generated_instance));
                    }
                    generated_mux_index += mux->instances.size();
                    continue;
                }
                auto logic = lower_scalar_boolean_assignment(
                    assignment, source_manager, *cell_library,
                    generated_mux_index);
                if (logic) {
                    const auto has_name_collision = [&]() {
                        for (const auto& generated : logic->instances) {
                            if (instance_names.count(generated.get_name()) !=
                                    0 ||
                                net_names.count(generated.get_name()) != 0) {
                                return true;
                            }
                        }
                        for (const auto& generated : logic->nets) {
                            if (net_names.count(generated.get_name()) != 0 ||
                                instance_names.count(generated.get_name()) !=
                                    0) {
                                return true;
                            }
                        }
                        return false;
                    };
                    while (has_name_collision()) {
                        logic = lower_scalar_boolean_assignment(
                            assignment, source_manager, *cell_library,
                            ++generated_mux_index);
                        if (!logic) {
                            throw std::runtime_error(
                                "failed to generate unique logic-cell names");
                        }
                    }
                    const auto generated_count =
                        logic->instances.size() + logic->nets.size();
                    for (auto& generated_net : logic->nets) {
                        net_names.insert(generated_net.get_name());
                        nets.push_back(std::move(generated_net));
                    }
                    for (auto& generated_instance : logic->instances) {
                        instance_names.insert(generated_instance.get_name());
                        instances.push_back(std::move(generated_instance));
                    }
                    generated_mux_index += generated_count;
                    continue;
                }
                throw std::runtime_error(
                    "continuous assignment '" +
                    expression_source(assignment.left(), source_manager) +
                    " = " +
                    expression_source(assignment.right(), source_manager) +
                    "' has no supported standard-cell mapping in library '" +
                    cell_library->get_name() + "'");
            }
            assignments.emplace_back(
                lower_expression(assignment.left(), source_manager),
                lower_expression(assignment.right(), source_manager));
        } else if (member.kind == slang::ast::SymbolKind::ProceduralBlock &&
                   cell_library) {
            const auto& block =
                member.as<slang::ast::ProceduralBlockSymbol>();
            throw std::runtime_error(
                "procedural block '" +
                std::string(slang::ast::SemanticFacts::getProcedureKindStr(
                    block.procedureKind)) +
                "' is not yet supported by the in-house cell mapper");
        }
    }

    return Module(std::string(instance.getDefinition().name), std::move(ports),
                  std::move(nets), std::move(instances),
                  std::move(assignments));
}

}  // namespace

Module parse_verilog_source(const std::string& source) {
    const std::string uncommented = remove_comments(source);
    static const std::regex module_end(R"(\bendmodule\b)");
    const std::string delimited =
        std::regex_replace(uncommented, module_end, ";endmodule;");

    std::vector<std::string> statements;
    std::stringstream stream(delimited);
    std::string statement;
    while (std::getline(stream, statement, ';')) {
        statement = trim(statement);
        if (!statement.empty()) statements.push_back(statement);
    }

    static const std::regex header(
        R"(^module\s+([A-Za-z_][A-Za-z0-9_$]*)\s*\((.*?)\)$)");
    static const std::regex declaration(
        R"(^(input|output|wire)\s+(?:wire\s+)?(.+)$)");
    static const std::regex gate(
        R"(^(and|or|xor)\s+([A-Za-z_][A-Za-z0-9_$]*)\s*\((.*?)\)$)");

    Module module;
    std::vector<std::string> header_ports;
    std::vector<Declaration> declarations;
    std::vector<ParsedInstance> parsed_instances;
    std::unordered_set<std::string> declared_nets;
    std::unordered_set<std::string> declared_ports;
    std::unordered_set<std::string> instance_names;
    bool saw_header = false;
    bool saw_endmodule = false;

    for (std::size_t i = 0; i < statements.size(); ++i) {
        const auto& current = statements[i];
        std::smatch match;
        if (!saw_header) {
            if (!std::regex_match(current, match, header)) {
                throw std::runtime_error("expected module header with a port list");
            }
            module.name_ = match[1].str();
            header_ports = split_names(match[2].str());
            saw_header = true;
            continue;
        }
        if (current == "endmodule") {
            if (i + 1 != statements.size()) {
                throw std::runtime_error("unexpected content after endmodule");
            }
            saw_endmodule = true;
            break;
        }
        if (std::regex_match(current, match, declaration)) {
            const auto direction = match[1].str();
            const auto names = split_names(match[2].str());
            for (const auto& name : names) {
                if (!declared_nets.insert(name).second) {
                    throw std::runtime_error("net declared more than once: " + name);
                }
                if (direction != "wire" && !declared_ports.insert(name).second) {
                    throw std::runtime_error("port declared more than once: " + name);
                }
                declarations.push_back({name, direction});
            }
            continue;
        }
        if (std::regex_match(current, match, gate)) {
            const auto gate_type = match[1].str();
            const auto instance_name = match[2].str();
            if (!instance_names.insert(instance_name).second) {
                throw std::runtime_error("duplicate instance name: " + instance_name);
            }
            const auto pins = split_names(match[3].str());
            if (pins.size() != 3) {
                throw std::runtime_error(gate_type + " instance " + instance_name +
                                         " must have output and two inputs");
            }
            const std::string cell_type =
                gate_type == "and" ? "AND" : gate_type == "or" ? "OR" : "XOR";
            const auto& cell = resolve_cell(get_default_cell_library(), cell_type);
            auto connections = physical_connections(
                cell, cell_type,
                {{"Y", pins[0]}, {"A", pins[1]}, {"B", pins[2]}});
            ParsedInstance instance{Instance{
                instance_name, cell_type, std::move(connections), cell}};
            parsed_instances.push_back(std::move(instance));
            continue;
        }
        throw std::runtime_error("unsupported Verilog statement: " + current);
    }

    if (!saw_header || !saw_endmodule) {
        throw std::runtime_error("incomplete Verilog module");
    }

    std::unordered_set<std::string> header_port_set;
    for (const auto& port : header_ports) {
        if (!header_port_set.insert(port).second) {
            throw std::runtime_error("duplicate module port: " + port);
        }
        if (declared_ports.count(port) == 0) {
            throw std::runtime_error("module port lacks input/output declaration: " +
                                     port);
        }
    }
    if (header_port_set != declared_ports) {
        throw std::runtime_error(
            "module header ports and input/output declarations do not match");
    }

    for (const auto& port_name : header_ports) {
        const auto declaration = std::find_if(
            declarations.begin(), declarations.end(),
            [&](const Declaration& candidate) {
                return candidate.name == port_name && candidate.direction != "wire";
            });
        const auto direction = declaration->direction == "input"
                                   ? PortDirection::Input
                                   : PortDirection::Output;
        module.ports_.emplace_back(port_name, direction);
        module.nets_.emplace_back(
            port_name, direction == PortDirection::Input ? NetType::Input
                                                          : NetType::Output);
    }

    for (const auto& declaration : declarations) {
        if (declaration.direction == "wire") {
            module.nets_.emplace_back(declaration.name, NetType::Internal);
        }
    }

    for (auto& parsed : parsed_instances) {
        for (const auto& connection : parsed.instance.get_connections()) {
            if (declared_nets.count(connection.get_net()) == 0) {
                throw std::runtime_error("instance " +
                                         parsed.instance.get_name() +
                                         " uses undeclared net: " +
                                         connection.get_net());
            }
        }
        module.instances_.push_back(std::move(parsed.instance));
    }

    return module;
}

Module parse_verilog(const std::string& path) {
    const std::filesystem::path source_path(path);
    if (!std::filesystem::is_regular_file(source_path)) {
        throw std::runtime_error("could not open Verilog file: " + path);
    }

    const auto absolute_path =
        std::filesystem::absolute(source_path).lexically_normal().string();
    const auto source_directory =
        source_path.has_parent_path() ? source_path.parent_path()
                                      : std::filesystem::path(".");
    const auto include_directory =
        std::filesystem::absolute(source_directory)
            .lexically_normal()
            .string();
    std::vector<std::string> arguments = {
        "labneura-eda", "--std=1364-2005",
        "--include-directory=" + include_directory, absolute_path};

    slang::driver::Driver driver;
    driver.addStandardArgs();
    std::vector<char*> argv;
    argv.reserve(arguments.size());
    for (auto& argument : arguments) argv.push_back(argument.data());

    if (!driver.parseCommandLine(static_cast<int>(argv.size()), argv.data())) {
        (void)driver.reportDiagnostics(false);
        throw std::runtime_error("slang could not parse the Verilog file options");
    }
    if (!driver.processOptions()) {
        (void)driver.reportDiagnostics(false);
        throw std::runtime_error("slang failed to process the Verilog file options");
    }
    const bool parsed_sources = driver.parseAllSources();
    const bool parse_diagnostics_ok = driver.reportParseDiags();
    if (!parsed_sources || !parse_diagnostics_ok) {
        throw std::runtime_error("slang failed to parse the Verilog file");
    }

    const auto& cell_library = get_default_cell_library();
    driver.syntaxTrees.push_back(slang::syntax::SyntaxTree::fromText(
        make_cell_stub_source(cell_library), driver.sourceManager,
        "labneura_cells.v", "labneura_cells.v"));
    auto compilation = driver.createCompilation();
    const auto& root_symbol = compilation->getRoot();
    driver.reportCompilation(*compilation, true);
    if (!driver.reportDiagnostics(true)) {
        throw std::runtime_error(
            "slang failed to compile the Verilog file against the bundled "
            "standard-cell library:\n" +
            driver.textDiagClient->getString());
    }
    const slang::ast::InstanceSymbol* top_module = nullptr;
    for (const auto* candidate : root_symbol.topInstances) {
        if (is_cell_type(
                cell_library,
                std::string(candidate->getDefinition().name))) {
            continue;
        }
        if (top_module) {
            throw std::runtime_error(
                "parse_verilog requires exactly one top-level module in the file");
        }
        top_module = candidate;
    }
    if (!top_module) {
        throw std::runtime_error(
            "parse_verilog found no top-level Verilog module in the file");
    }
    return summarize_module(*top_module, driver.sourceManager, &cell_library);
}

Module parse_verilog_file(const std::string& path) {
    return parse_verilog(path);
}

Project parse_verilog_project(const std::string& directory) {
    const std::filesystem::path root(directory);
    if (!std::filesystem::is_directory(root)) {
        throw std::runtime_error("Verilog project path is not a directory: " +
                                 directory);
    }

    std::vector<std::filesystem::path> verilog_files;
    std::unordered_set<std::string> include_directories;
    include_directories.insert(
        std::filesystem::absolute(root).lexically_normal().string());
    for (const auto& entry : std::filesystem::recursive_directory_iterator(root)) {
        if (entry.is_directory()) {
            include_directories.insert(
                std::filesystem::absolute(entry.path()).lexically_normal().string());
        }
        if (entry.is_regular_file() && entry.path().extension() == ".v") {
            verilog_files.push_back(entry.path());
        }
    }
    std::sort(verilog_files.begin(), verilog_files.end());
    if (verilog_files.empty()) {
        throw std::runtime_error("no .v Verilog files found in project: " +
                                 directory);
    }

    Project project;
    std::vector<std::string> arguments = {"labneura-eda", "--std=1364-2005"};
    for (const auto& path : verilog_files) {
        project.source_files_.push_back(
            std::filesystem::absolute(path).lexically_normal().string());
    }
    std::vector<std::string> sorted_include_directories(include_directories.begin(),
                                                        include_directories.end());
    std::sort(sorted_include_directories.begin(), sorted_include_directories.end());
    for (const auto& include_directory : sorted_include_directories) {
        arguments.push_back("--include-directory=" + include_directory);
    }
    for (const auto& source_file : project.source_files_) {
        arguments.push_back(source_file);
    }

    slang::driver::Driver driver;
    driver.addStandardArgs();
    std::vector<char*> argv;
    argv.reserve(arguments.size());
    for (auto& argument : arguments) argv.push_back(argument.data());

    if (!driver.parseCommandLine(static_cast<int>(argv.size()), argv.data())) {
        (void)        driver.reportDiagnostics(false);
        throw std::runtime_error("slang could not parse the Verilog project options");
    }
    if (!driver.processOptions()) {
        (void)driver.reportDiagnostics(false);
        throw std::runtime_error("slang failed to process the Verilog project options");
    }
    const bool parsed_sources = driver.parseAllSources();
    const bool parse_diagnostics_ok = driver.reportParseDiags();
    if (!parsed_sources || !parse_diagnostics_ok) {
        throw std::runtime_error("slang failed to parse the Verilog project");
    }

    auto compilation = driver.createCompilation();
    const auto& root_symbol = compilation->getRoot();
    driver.reportCompilation(*compilation, true);
    if (!driver.reportDiagnostics(true)) {
        throw std::runtime_error("slang failed to compile the Verilog project");
    }

    slang::JsonWriter writer;
    writer.setPrettyPrint(true);
    slang::ast::ASTSerializer serializer(*compilation, writer);
    serializer.setIncludeAddresses(false);
    serializer.setIncludeSourceInfo(true);
    writer.startObject();
    writer.writeProperty("definitions");
    writer.startArray();
    for (const auto* definition : compilation->getDefinitions()) {
        serializer.serialize(*definition);
    }
    writer.endArray();
    writer.writeProperty("elaborated_root");
    serializer.serialize(root_symbol);
    writer.endObject();
    project.ast_json_ = std::string(writer.view());

    std::unordered_set<std::string> module_names;
    std::vector<Module> modules;
    const auto collect_module = [&](const auto& self,
                                    const slang::ast::InstanceSymbol& instance) -> void {
        if (!instance.isModule()) return;

        const std::string module_name(instance.getDefinition().name);
        if (module_names.insert(module_name).second) {
            modules.push_back(summarize_module(instance, driver.sourceManager));

            for (const auto& member : instance.body.members()) {
                if (member.kind == slang::ast::SymbolKind::Instance) {
                    self(self, member.as<slang::ast::InstanceSymbol>());
                }
            }
        }
    };
    for (const auto* top : root_symbol.topInstances) {
        collect_module(collect_module, *top);
    }
    if (modules.empty()) {
        throw std::runtime_error("no Verilog modules found in project: " + directory);
    }
    project.modules_ = std::move(modules);

    return project;
}

}  // namespace labneura::eda
