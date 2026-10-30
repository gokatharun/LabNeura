#include "labneura/eda/netlist.h"
#include "labneura/eda/parsed_verilog_context.h"

#include "slang/ast/ASTSerializer.h"
#include "slang/ast/Compilation.h"
#include "slang/ast/expressions/AssignmentExpressions.h"
#include "slang/ast/expressions/LiteralExpressions.h"
#include "slang/ast/expressions/MiscExpressions.h"
#include "slang/ast/expressions/OperatorExpressions.h"
#include "slang/ast/expressions/SelectExpressions.h"
#include "slang/ast/symbols/CompilationUnitSymbols.h"
#include "slang/ast/symbols/BlockSymbols.h"
#include "slang/ast/symbols/InstanceSymbols.h"
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
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

namespace labneura::eda {
namespace {

std::shared_ptr<detail::ParsedVerilogContext> parse_sources(
    std::vector<std::string>& arguments, const std::string& context_name) {
    auto context = std::make_shared<detail::ParsedVerilogContext>();
    auto& driver = context->driver;
    driver.addStandardArgs();
    std::vector<char*> argv;
    argv.reserve(arguments.size());
    for (auto& argument : arguments) argv.push_back(argument.data());

    if (!driver.parseCommandLine(static_cast<int>(argv.size()), argv.data())) {
        (void)driver.reportDiagnostics(false);
        throw std::runtime_error("slang could not parse the " + context_name +
                                 " options");
    }
    if (!driver.processOptions()) {
        (void)driver.reportDiagnostics(false);
        throw std::runtime_error("slang failed to process the " + context_name +
                                 " options");
    }
    const bool parsed_sources = driver.parseAllSources();
    const bool parse_diagnostics_ok = driver.reportParseDiags();
    if (!parsed_sources || !parse_diagnostics_ok) {
        throw std::runtime_error("slang failed to parse the " + context_name);
    }

    context->compilation = driver.createCompilation();
    driver.reportCompilation(*context->compilation, true);
    if (!driver.reportDiagnostics(true)) {
        throw std::runtime_error("slang failed to compile the " + context_name +
                                 ":\n" + driver.textDiagClient->getString());
    }
    return context;
}

bool is_systemverilog(const std::filesystem::path& path) {
    return path.extension() == ".sv";
}

std::size_t declared_width(const slang::ast::Symbol& symbol) {
    const auto* type = symbol.getDeclaredType();
    return type ? static_cast<std::size_t>(type->getType().getBitWidth()) : 1;
}

std::string trim(const std::string& text) {
    const auto first = text.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    const auto last = text.find_last_not_of(" \t\r\n");
    return text.substr(first, last - first + 1);
}

std::string source_range_text(const slang::SourceRange& macro_range,
                              const slang::SourceManager& source_manager) {
    const auto range = source_manager.getFullyOriginalRange(macro_range);
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
                       bool project_summary = false,
                       bool preserve_hierarchy = false,
                       std::string hierarchy_instance_name = {}) {
    bool behavioral = false;
    std::vector<Port> ports;
    std::vector<Net> nets;
    std::vector<Instance> instances;
    std::vector<Module> child_modules;
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
        const auto width = static_cast<std::size_t>(port.getType().getBitWidth());
        ports.emplace_back(port_name, direction, width);
        const auto net_type =
            direction == PortDirection::Input
                ? NetType::Input
                : direction == PortDirection::Output ? NetType::Output
                                                     : NetType::InOut;
        nets.emplace_back(port_name, net_type, width);
    }

    for (const auto& member : instance.body.members()) {
        if ((member.kind == slang::ast::SymbolKind::Net ||
             member.kind == slang::ast::SymbolKind::Variable) &&
            port_names.count(std::string(member.name)) == 0) {
             nets.emplace_back(std::string(member.name), NetType::Internal,
                               declared_width(member));
        } else if (member.kind == slang::ast::SymbolKind::Instance) {
            const auto& child = member.as<slang::ast::InstanceSymbol>();
            const std::string cell_type(child.getDefinition().name);
            instances.emplace_back(
                std::string(child.name), cell_type,
                module_port_connections(child, source_manager), std::nullopt,
                std::map<std::string, std::string>{});
            if (preserve_hierarchy && child.isModule()) {
                child_modules.push_back(summarize_module(
                    child, source_manager, false, true,
                    std::string(child.name)));
            }
        } else if (member.kind == slang::ast::SymbolKind::PrimitiveInstance) {
            const auto& child =
                member.as<slang::ast::PrimitiveInstanceSymbol>();
            const std::string cell_type(child.primitiveType.name);
            std::vector<UserDefinedPrimitiveRow> primitive_table;
            if (child.primitiveType.primitiveKind ==
                slang::ast::PrimitiveSymbol::UserDefined) {
                primitive_table.reserve(child.primitiveType.table.size());
                for (const auto& row : child.primitiveType.table) {
                    primitive_table.push_back(
                        {std::string(row.inputs), row.output});
                }
            }
            instances.emplace_back(
                std::string(child.name), cell_type,
                parsed_primitive_connections(child, source_manager),
                std::nullopt, std::map<std::string, std::string>{},
                child.primitiveType.primitiveKind ==
                    slang::ast::PrimitiveSymbol::UserDefined,
                std::move(primitive_table),
                child.primitiveType.isSequential ||
                    child.primitiveType.isEdgeSensitive);
        } else if (member.kind == slang::ast::SymbolKind::ContinuousAssign) {
            const auto& assignment =
                member.as<slang::ast::ContinuousAssignSymbol>().getAssignment()
                    .as<slang::ast::AssignmentExpression>();
            assignments.emplace_back(
                lower_expression(assignment.left(), source_manager),
                lower_expression(assignment.right(), source_manager));
        } else if (member.kind == slang::ast::SymbolKind::ProceduralBlock) {
            const auto& block =
                member.as<slang::ast::ProceduralBlockSymbol>();
            if (block.procedureKind == slang::ast::ProceduralBlockKind::Initial ||
                block.procedureKind == slang::ast::ProceduralBlockKind::Final) {
                if (project_summary) continue;
                throw std::runtime_error(
                    "procedural block '" +
                    std::string(slang::ast::SemanticFacts::getProcedureKindStr(
                        block.procedureKind)) +
                    "' is simulation-only and has no netlist form");
            }
            behavioral = true;
        }
    }

    return Module(std::string(instance.getDefinition().name), std::move(ports),
                  std::move(nets), std::move(instances),
                  std::move(assignments), behavioral,
                  std::move(hierarchy_instance_name),
                  std::move(child_modules));
}

}  // namespace

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
        "labneura-eda",
        is_systemverilog(source_path) ? "--std=1800-2017" : "--std=1364-2005",
        "--include-directory=" + include_directory, absolute_path};

    auto parsed_context = parse_sources(arguments, "Verilog file");
    auto& driver = parsed_context->driver;
    const auto& root_symbol = parsed_context->compilation->getRoot();
    const slang::ast::InstanceSymbol* top_module = nullptr;
    for (const auto* candidate : root_symbol.topInstances) {
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
    auto module = summarize_module(*top_module, driver.sourceManager, false,
                                   true);
    module.parsed_context_ = parsed_context;
    const auto retain_behavioral_context =
        [&](const auto& self, Module& current) -> void {
        if (current.is_behavioral()) current.parsed_context_ = parsed_context;
        for (auto& child : current.child_modules_) self(self, child);
    };
    retain_behavioral_context(retain_behavioral_context, module);

    return module;
}

Project parse_verilog_directory(const std::string& directory) {
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
        if (entry.is_regular_file() && (entry.path().extension() == ".v" || is_systemverilog(entry.path()))) {
            verilog_files.push_back(entry.path());
        }
    }
    std::sort(verilog_files.begin(), verilog_files.end());
    if (verilog_files.empty()) {
        throw std::runtime_error("no .v or .sv files found in project: " +
                                 directory);
    }

    Project project;
    const bool any_systemverilog =
        std::any_of(verilog_files.begin(), verilog_files.end(), is_systemverilog);
    std::vector<std::string> arguments = {
        "labneura-eda", any_systemverilog ? "--std=1800-2017" : "--std=1364-2005"};
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

    project.parsed_context_ = parse_sources(arguments, "Verilog project");
    auto& driver = project.parsed_context_->driver;
    const auto& root_symbol = project.parsed_context_->compilation->getRoot();

    slang::JsonWriter writer;
    writer.setPrettyPrint(true);
    slang::ast::ASTSerializer serializer(*project.parsed_context_->compilation,
                                         writer);
    serializer.setIncludeAddresses(false);
    serializer.setIncludeSourceInfo(true);
    writer.startObject();
    writer.writeProperty("definitions");
    writer.startArray();
    for (const auto* definition :
         project.parsed_context_->compilation->getDefinitions()) {
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
            auto module = summarize_module(instance, driver.sourceManager, true);
            if (module.is_behavioral()) {
                module.parsed_context_ = project.parsed_context_;
            }
            modules.push_back(std::move(module));

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
