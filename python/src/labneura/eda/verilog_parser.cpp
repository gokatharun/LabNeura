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

bool is_cell_type(const StandardCellLibrary& library,
                  const std::string& verilog_type) {
    return library.find_cell(verilog_type) ||
           library.find_cell(liberty_cell_name(verilog_type));
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

std::vector<PinConnection> parsed_primitive_connections(
    const slang::ast::PrimitiveInstanceSymbol& instance,
    const slang::SourceManager& source_manager) {
    const auto expressions = instance.getPortConnections();
    if (expressions.empty()) return {};

    std::vector<PinConnection> connections;
    connections.reserve(expressions.size());
    connections.emplace_back(
        "Y", expression_source(*expressions.front(), source_manager));
    for (std::size_t i = 1; i < expressions.size(); ++i) {
        const auto input_name =
            i <= 26 ? std::string(1, static_cast<char>('A' + i - 1))
                    : "IN" + std::to_string(i);
        connections.emplace_back(
            input_name, expression_source(*expressions[i], source_manager));
    }
    return connections;
}

Expression lower_expression(const slang::ast::Expression& expression,
                            const slang::SourceManager& source_manager) {
    using slang::ast::ExpressionKind;
    const std::string source = expression_source(expression, source_manager);
    switch (expression.kind) {
        case ExpressionKind::NamedValue: {
            const auto& named = expression.as<slang::ast::NamedValueExpression>();
            return {labneura::eda::ExpressionKind::Identifier,
                    std::string(named.symbol.name), source, {},
                    expression.type->getBitWidth()};
        }
        case ExpressionKind::IntegerLiteral:
        case ExpressionKind::RealLiteral:
        case ExpressionKind::TimeLiteral:
        case ExpressionKind::UnbasedUnsizedIntegerLiteral:
        case ExpressionKind::NullLiteral:
        case ExpressionKind::StringLiteral:
            return {labneura::eda::ExpressionKind::Literal, source, source, {},
                    expression.type->getBitWidth()};
        case ExpressionKind::ElementSelect: {
            const auto& select =
                expression.as<slang::ast::ElementSelectExpression>();
            return {labneura::eda::ExpressionKind::BitSelect, "", source,
                    {lower_expression(select.value(), source_manager),
                     lower_expression(select.selector(), source_manager)},
                    expression.type->getBitWidth()};
        }
        case ExpressionKind::RangeSelect: {
            const auto& select =
                expression.as<slang::ast::RangeSelectExpression>();
            return {labneura::eda::ExpressionKind::RangeSelect, "", source,
                    {lower_expression(select.value(), source_manager),
                     lower_expression(select.left(), source_manager),
                    lower_expression(select.right(), source_manager)},
                    expression.type->getBitWidth()};
        }
        case ExpressionKind::UnaryOp: {
            const auto& unary = expression.as<slang::ast::UnaryExpression>();
            return {labneura::eda::ExpressionKind::Unary,
                    source_range_text(unary.opRange, source_manager), source,
                    {lower_expression(unary.operand(), source_manager)},
                    expression.type->getBitWidth()};
        }
        case ExpressionKind::BinaryOp: {
            const auto& binary = expression.as<slang::ast::BinaryExpression>();
            const auto operator_text =
                source_range_text(binary.opRange, source_manager);
            return {labneura::eda::ExpressionKind::Binary, operator_text, source,
                    {lower_expression(binary.left(), source_manager),
                     lower_expression(binary.right(), source_manager)},
                    expression.type->getBitWidth()};
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
                    std::move(operands), expression.type->getBitWidth()};
        }
        case ExpressionKind::Concatenation: {
            const auto& concat =
                expression.as<slang::ast::ConcatenationExpression>();
            std::vector<labneura::eda::Expression> operands;
            for (const auto* operand : concat.operands()) {
                operands.push_back(lower_expression(*operand, source_manager));
            }
            return {labneura::eda::ExpressionKind::Concatenation, "", source,
                    std::move(operands), expression.type->getBitWidth()};
        }
        default:
            return {labneura::eda::ExpressionKind::Opaque, "", source, {},
                    expression.type->getBitWidth()};
    }
}

Module summarize_module(const slang::ast::InstanceSymbol& instance,
                       const slang::SourceManager& source_manager,
                       bool project_summary = false) {
    std::vector<Port> ports;
    std::vector<Net> nets;
    std::vector<Instance> instances;
    std::vector<ContinuousAssignment> assignments;
    std::unordered_set<std::string> port_names;

    for (const auto* port_symbol : instance.body.getPortList()) {
        if (port_symbol->kind != slang::ast::SymbolKind::Port) continue;
        const auto& port = port_symbol->as<slang::ast::PortSymbol>();
        const std::string port_name(port.name);
        port_names.insert(port_name);
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
        } else if (member.kind == slang::ast::SymbolKind::Instance) {
            const auto& child = member.as<slang::ast::InstanceSymbol>();
            const std::string cell_type(child.getDefinition().name);
            instances.emplace_back(
                std::string(child.name), cell_type,
                module_port_connections(child, source_manager));
        } else if (member.kind == slang::ast::SymbolKind::PrimitiveInstance) {
            const auto& child =
                member.as<slang::ast::PrimitiveInstanceSymbol>();
            const std::string cell_type(child.primitiveType.name);
            instances.emplace_back(
                std::string(child.name), cell_type,
                parsed_primitive_connections(child, source_manager));
        } else if (member.kind == slang::ast::SymbolKind::ContinuousAssign) {
            const auto& assignment =
                member.as<slang::ast::ContinuousAssignSymbol>().getAssignment()
                    .as<slang::ast::AssignmentExpression>();
            assignments.emplace_back(
                lower_expression(assignment.left(), source_manager),
                lower_expression(assignment.right(), source_manager));
        } else if (member.kind == slang::ast::SymbolKind::ProceduralBlock) {
            if (project_summary) continue;
            const auto& block =
                member.as<slang::ast::ProceduralBlockSymbol>();
            throw std::runtime_error(
                "procedural block '" +
                std::string(slang::ast::SemanticFacts::getProcedureKindStr(
                    block.procedureKind)) +
                "' is not yet representable in the RTL netlist model");
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
            ParsedInstance instance{Instance{
                instance_name, gate_type,
                {{"Y", pins[0]}, {"A", pins[1]}, {"B", pins[2]}}}};
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
    return summarize_module(*top_module, driver.sourceManager);
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
            modules.push_back(summarize_module(instance, driver.sourceManager, true));

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
