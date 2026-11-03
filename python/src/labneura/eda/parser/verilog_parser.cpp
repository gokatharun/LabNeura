#include "labneura/eda/netlist.h"
#include "labneura/eda/parsed_verilog_context.h"

#include "slang/ast/ASTSerializer.h"
#include "slang/ast/Compilation.h"
#include "slang/ast/expressions/AssignmentExpressions.h"
#include "slang/ast/expressions/ConversionExpression.h"
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
#include <string_view>
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

std::pair<std::optional<std::int64_t>, std::optional<std::int64_t>>
declared_bounds(const slang::ast::Type& type) {
    if (!type.isPackedArray()) return {};
    const auto range = type.getFixedRange();
    if (range.width() != static_cast<std::size_t>(type.getBitWidth())) {
        const auto width = static_cast<std::int64_t>(type.getBitWidth());
        return {width - 1, 0};
    }
    return {range.left, range.right};
}

struct DeclaredShape {
    std::size_t width = 1;
    std::size_t packed_width = 1;
    std::optional<std::int64_t> packed_left;
    std::optional<std::int64_t> packed_right;
    std::optional<std::int64_t> unpacked_left;
    std::optional<std::int64_t> unpacked_right;
};

DeclaredShape declared_shape(const slang::ast::Type& type) {
    DeclaredShape shape;
    if (type.isUnpackedArray()) {
        if (!type.hasFixedRange() || !type.getArrayElementType()) {
            throw std::runtime_error(
                "netlist I/O requires fixed-size unpacked arrays");
        }
        const auto* element = type.getArrayElementType();
        if (element->isUnpackedArray()) {
            throw std::runtime_error(
                "netlist I/O currently supports one unpacked array dimension");
        }
        const auto array_range = type.getFixedRange();
        shape.packed_width = static_cast<std::size_t>(element->getBitWidth());
        shape.width = array_range.width() * shape.packed_width;
        const auto [left, right] = declared_bounds(*element);
        shape.packed_left = left;
        shape.packed_right = right;
        shape.unpacked_left = array_range.left;
        shape.unpacked_right = array_range.right;
    } else {
        shape.width = static_cast<std::size_t>(type.getBitWidth());
        shape.packed_width = shape.width;
        const auto [left, right] = declared_bounds(type);
        shape.packed_left = left;
        shape.packed_right = right;
    }
    return shape;
}

void collect_constant_selects(
    const slang::ast::Expression& expression,
    const slang::SourceManager& source_manager,
    const slang::BufferID source_buffer, std::size_t source_offset,
    const std::string& scope_prefix,
    std::vector<std::pair<std::size_t, std::pair<std::size_t, std::string>>>&
        replacements) {
    using slang::ast::ExpressionKind;
    const auto replace_source = [&](const slang::ast::Expression& source,
                                    const std::string& value) {
        const auto range = source_manager.getFullyOriginalRange(source.sourceRange);
        const auto start = range.start();
        const auto end = range.end();
        if (!start.valid() || !end.valid() ||
            start.buffer() != source_buffer || end.buffer() != source_buffer ||
            end.offset() < start.offset()) {
            return;
        }
        replacements.push_back(
            {start.offset() - source_offset,
             {end.offset() - source_offset, value}});
    };
    const auto replace_constant = [&](const slang::ast::Expression& selector) {
        const auto range =
            source_manager.getFullyOriginalRange(selector.sourceRange);
        const auto start = range.start();
        const auto end = range.end();
        const auto* constant = selector.getConstant();
        if (!start.valid() || !end.valid() || start.buffer() != source_buffer ||
            end.buffer() != source_buffer || end.offset() < start.offset() ||
            !constant || !constant->isInteger()) {
            return false;
        }
        const auto value = constant->integer().as<std::int64_t>();
        if (!value) return false;
        replace_source(selector, std::to_string(*value));
        return true;
    };

    switch (expression.kind) {
        case ExpressionKind::NamedValue:
        case ExpressionKind::HierarchicalValue: {
            const auto& symbol =
                expression.as<slang::ast::ValueExpressionBase>().symbol;
            if (scope_prefix.empty() ||
                symbol.kind == slang::ast::SymbolKind::Parameter ||
                symbol.kind == slang::ast::SymbolKind::EnumValue) {
                return;
            }
            std::string path = symbol.getHierarchicalPath();
            if (path.rfind(scope_prefix, 0) == 0) {
                path.erase(0, scope_prefix.size());
                if (path != symbol.name) replace_source(expression, path);
            }
            return;
        }
        case ExpressionKind::ElementSelect: {
            const auto& select =
                expression.as<slang::ast::ElementSelectExpression>();
            collect_constant_selects(select.value(), source_manager,
                                     source_buffer, source_offset, scope_prefix,
                                     replacements);
            if (!replace_constant(select.selector())) {
                collect_constant_selects(select.selector(), source_manager,
                                         source_buffer, source_offset, scope_prefix,
                                         replacements);
            }
            return;
        }
        case ExpressionKind::RangeSelect: {
            const auto& select =
                expression.as<slang::ast::RangeSelectExpression>();
            collect_constant_selects(select.value(), source_manager,
                                     source_buffer, source_offset, scope_prefix,
                                     replacements);
            if (!replace_constant(select.left())) {
                collect_constant_selects(select.left(), source_manager,
                                         source_buffer, source_offset, scope_prefix,
                                         replacements);
            }
            if (!replace_constant(select.right())) {
                collect_constant_selects(select.right(), source_manager,
                                         source_buffer, source_offset, scope_prefix,
                                         replacements);
            }
            return;
        }
        case ExpressionKind::MemberAccess:
            collect_constant_selects(
                expression.as<slang::ast::MemberAccessExpression>().value(),
                source_manager, source_buffer, source_offset, scope_prefix,
                replacements);
            return;
        case ExpressionKind::UnaryOp:
            collect_constant_selects(
                expression.as<slang::ast::UnaryExpression>().operand(),
                source_manager, source_buffer, source_offset, scope_prefix,
                replacements);
            return;
        case ExpressionKind::BinaryOp: {
            const auto& binary =
                expression.as<slang::ast::BinaryExpression>();
            collect_constant_selects(binary.left(), source_manager,
                                     source_buffer, source_offset, scope_prefix,
                                     replacements);
            collect_constant_selects(binary.right(), source_manager,
                                     source_buffer, source_offset, scope_prefix,
                                     replacements);
            return;
        }
        case ExpressionKind::ConditionalOp: {
            const auto& conditional =
                expression.as<slang::ast::ConditionalExpression>();
            for (const auto& condition : conditional.conditions) {
                collect_constant_selects(*condition.expr, source_manager,
                                         source_buffer, source_offset, scope_prefix,
                                         replacements);
            }
            collect_constant_selects(conditional.left(), source_manager,
                                     source_buffer, source_offset, scope_prefix,
                                     replacements);
            collect_constant_selects(conditional.right(), source_manager,
                                     source_buffer, source_offset, scope_prefix,
                                     replacements);
            return;
        }
        case ExpressionKind::Concatenation:
            for (const auto* operand :
                 expression.as<slang::ast::ConcatenationExpression>()
                     .operands()) {
                collect_constant_selects(*operand, source_manager,
                                         source_buffer, source_offset, scope_prefix,
                                         replacements);
            }
            return;
        case ExpressionKind::Conversion:
            collect_constant_selects(
                expression.as<slang::ast::ConversionExpression>().operand(),
                source_manager, source_buffer, source_offset, scope_prefix,
                replacements);
            return;
        default:
            return;
    }
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
                              const slang::SourceManager& source_manager,
                              const std::string& scope_prefix = {}) {
    auto source = source_range_text(expression.sourceRange, source_manager);
    const auto range =
        source_manager.getFullyOriginalRange(expression.sourceRange);
    const auto start = range.start();
    const auto end = range.end();
    if (!start.valid() || !end.valid() || start.buffer() != end.buffer() ||
        end.offset() < start.offset()) {
        return source;
    }
    std::vector<std::pair<
        std::size_t, std::pair<std::size_t, std::string>>> replacements;
    collect_constant_selects(expression, source_manager, start.buffer(),
                             start.offset(), scope_prefix, replacements);
    std::sort(replacements.rbegin(), replacements.rend());
    for (const auto& [begin, replacement] : replacements) {
        const auto& [finish, value] = replacement;
        if (finish <= source.size() && begin <= finish) {
            source.replace(begin, finish - begin, value);
        }
    }
    return trim(source);
}

std::vector<PinConnection> module_port_connections(
    const slang::ast::InstanceSymbol& instance,
    const slang::SourceManager& source_manager,
    const std::string& scope_prefix) {
    std::vector<PinConnection> connections;
    for (const auto* connection : instance.getPortConnections()) {
        if (!connection->getExpression()) continue;
        connections.emplace_back(
            std::string(connection->port.name),
            expression_source(*connection->getExpression(), source_manager,
                              scope_prefix));
    }
    return connections;
}

std::vector<PinConnection> parsed_primitive_connections(
    const slang::ast::PrimitiveInstanceSymbol& instance,
    const slang::SourceManager& source_manager,
    const std::string& scope_prefix) {
    const auto expressions = instance.getPortConnections();
    if (expressions.empty()) return {};

    std::vector<PinConnection> connections;
    connections.reserve(expressions.size());
    connections.emplace_back(
        "Y", expression_source(*expressions.front(), source_manager,
                               scope_prefix));
    for (std::size_t i = 1; i < expressions.size(); ++i) {
        const auto input_name =
            i <= 26 ? std::string(1, static_cast<char>('A' + i - 1))
                    : "IN" + std::to_string(i);
        connections.emplace_back(
            input_name,
            expression_source(*expressions[i], source_manager, scope_prefix));
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
        case ExpressionKind::Conversion: {
            auto operand = lower_expression(
                expression.as<slang::ast::ConversionExpression>().operand(),
                source_manager);
            const auto width = expression.type->getBitWidth();
            if (width == operand.get_width()) return operand;
            if (operand.get_kind() ==
                    labneura::eda::ExpressionKind::Literal &&
                width == 1) {
                return {operand.get_kind(), operand.get_value(), source, {},
                        width};
            }
            return {labneura::eda::ExpressionKind::Opaque,
                    "implicit_conversion", source, {std::move(operand)}, width};
        }
        default:
            return {labneura::eda::ExpressionKind::Opaque, "", source, {},
                    expression.type->getBitWidth()};
    }
}

template <typename Callback>
void visit_generate_instances(const slang::ast::Scope& scope,
                              Callback&& callback) {
    const auto visit_scope = [&](const auto& self,
                                 const slang::ast::Scope& current,
                                 bool inside_generate) -> void {
        for (const auto& member : current.members()) {
            if (member.kind == slang::ast::SymbolKind::Instance) {
                if (inside_generate) {
                    callback(member.as<slang::ast::InstanceSymbol>());
                }
            } else if (member.kind == slang::ast::SymbolKind::GenerateBlock) {
                const auto& block =
                    member.as<slang::ast::GenerateBlockSymbol>();
                if (!block.isUninstantiated) self(self, block, true);
            } else if (member.kind ==
                       slang::ast::SymbolKind::GenerateBlockArray) {
                const auto& array =
                    member.as<slang::ast::GenerateBlockArraySymbol>();
                for (const auto* entry : array.entries) {
                    if (!entry->isUninstantiated) self(self, *entry, true);
                }
            }
        }
    };
    visit_scope(visit_scope, scope, false);
}

template <typename Callback>
void visit_generate_assignments(const slang::ast::Scope& scope,
                                Callback&& callback) {
    const auto visit_scope = [&](const auto& self,
                                 const slang::ast::Scope& current,
                                 bool inside_generate) -> void {
        for (const auto& member : current.members()) {
            if (inside_generate &&
                member.kind == slang::ast::SymbolKind::ContinuousAssign) {
                callback(member.as<slang::ast::ContinuousAssignSymbol>());
            } else if (member.kind == slang::ast::SymbolKind::GenerateBlock) {
                const auto& block =
                    member.as<slang::ast::GenerateBlockSymbol>();
                if (!block.isUninstantiated) self(self, block, true);
            } else if (member.kind ==
                       slang::ast::SymbolKind::GenerateBlockArray) {
                const auto& array =
                    member.as<slang::ast::GenerateBlockArraySymbol>();
                for (const auto* entry : array.entries) {
                    if (!entry->isUninstantiated) self(self, *entry, true);
                }
            }
        }
    };
    visit_scope(visit_scope, scope, false);
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
    const std::string scope_prefix =
        std::string(instance.getHierarchicalPath()) + ".";

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
        const auto shape = declared_shape(port.getType());
        ports.emplace_back(port_name, direction, shape.width,
                           shape.packed_left, shape.packed_right,
                           shape.unpacked_left, shape.unpacked_right,
                           shape.packed_width);
        const auto net_type =
            direction == PortDirection::Input
                ? NetType::Input
                : direction == PortDirection::Output ? NetType::Output
                                                     : NetType::InOut;
        nets.emplace_back(port_name, net_type, shape.width,
                          shape.packed_left, shape.packed_right,
                          shape.unpacked_left, shape.unpacked_right,
                          shape.packed_width);
    }

    for (const auto& member : instance.body.members()) {
        if ((member.kind == slang::ast::SymbolKind::Net ||
             member.kind == slang::ast::SymbolKind::Variable) &&
            port_names.count(std::string(member.name)) == 0) {
             const auto* declared_type = member.getDeclaredType();
             const auto shape = declared_type
                                    ? declared_shape(declared_type->getType())
                                    : DeclaredShape{};
             nets.emplace_back(std::string(member.name), NetType::Internal,
                               shape.width, shape.packed_left,
                               shape.packed_right, shape.unpacked_left,
                               shape.unpacked_right, shape.packed_width);
        } else if (member.kind == slang::ast::SymbolKind::Instance) {
            const auto& child = member.as<slang::ast::InstanceSymbol>();
            const std::string cell_type(child.getDefinition().name);
            instances.emplace_back(
                std::string(child.name), cell_type,
                module_port_connections(child, source_manager, scope_prefix),
                std::nullopt,
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
                parsed_primitive_connections(child, source_manager,
                                             scope_prefix),
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

    visit_generate_instances(instance.body, [&](const auto& child) {
        const std::string cell_type(child.getDefinition().name);
        instances.emplace_back(
            std::string(child.name), cell_type,
            module_port_connections(child, source_manager, scope_prefix));
        if (preserve_hierarchy && child.isModule()) {
            child_modules.push_back(summarize_module(
                child, source_manager, false, true, std::string(child.name)));
        }
    });

    visit_generate_assignments(instance.body, [&](const auto& symbol) {
        const auto& assignment =
            symbol.getAssignment()
                .template as<slang::ast::AssignmentExpression>();
        assignments.emplace_back(
            lower_expression(assignment.left(), source_manager),
            lower_expression(assignment.right(), source_manager));
    });

    return Module(std::string(instance.getDefinition().name), std::move(ports),
                  std::move(nets), std::move(instances),
                  std::move(assignments), behavioral,
                  std::move(hierarchy_instance_name),
                  std::move(child_modules));
}

}  // namespace

std::string detail::verilog_expression_source(
    const slang::ast::Expression& expression,
    const slang::SourceManager& source_manager,
    const std::string& scope_prefix) {
    return expression_source(expression, source_manager, scope_prefix);
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
        current.parsed_context_ = parsed_context;
        for (auto& child : current.child_modules_) self(self, child);
    };
    retain_behavioral_context(retain_behavioral_context, module);

    return module;
}

Module read_verilog(const std::string& path) {
    auto module = parse_verilog(path);
    const auto normalize = [&](const auto& self, Module& current) -> void {
        std::unordered_map<std::string, std::string> cell_types;
        std::vector<Module> children;
        children.reserve(current.child_modules_.size());
        for (auto& child : current.child_modules_) {
            if (child.get_name().rfind("__labneura_cell__", 0) == 0) {
                static constexpr char digits[] = "0123456789abcdef";
                constexpr std::string_view prefix = "__labneura_cell__";
                const std::string_view encoded(child.get_name().data() +
                                                   prefix.size(),
                                               child.get_name().size() -
                                                   prefix.size());
                std::string type;
                if (encoded.size() % 2 != 0) {
                    throw std::runtime_error(
                        "invalid internal black-box name in Verilog netlist");
                }
                for (std::size_t i = 0; i < encoded.size(); i += 2) {
                    const auto high = std::string_view(digits).find(encoded[i]);
                    const auto low = std::string_view(digits).find(encoded[i + 1]);
                    if (high == std::string_view::npos ||
                        low == std::string_view::npos) {
                        throw std::runtime_error(
                            "invalid internal black-box name in Verilog netlist");
                    }
                    type.push_back(static_cast<char>((high << 4) | low));
                }
                cell_types[child.get_hierarchy_instance_name()] = std::move(type);
                continue;
            }
            self(self, child);
            children.push_back(child);
        }

        std::vector<Instance> instances;
        instances.reserve(current.instances_.size());
        for (const auto& instance : current.instances_) {
            const auto cell = cell_types.find(instance.get_name());
            instances.emplace_back(
                instance.get_name(),
                cell == cell_types.end() ? instance.get_cell_type() : cell->second,
                instance.get_connections(), instance.get_cell(),
                instance.get_parameters(), instance.is_user_defined_primitive(),
                instance.get_primitive_table(), instance.is_sequential_primitive());
        }
        Module normalized(
            current.get_name(), current.get_ports(), current.get_nets(),
            std::move(instances), current.get_assignments(),
            current.is_behavioral(), current.get_hierarchy_instance_name(),
            std::move(children));
        normalized.parsed_context_ = current.parsed_context_;
        current = std::move(normalized);
    };
    normalize(normalize, module);
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
            module.parsed_context_ = project.parsed_context_;
            modules.push_back(std::move(module));

            for (const auto& member : instance.body.members()) {
                if (member.kind == slang::ast::SymbolKind::Instance) {
                    self(self, member.as<slang::ast::InstanceSymbol>());
                }
            }
            visit_generate_instances(instance.body, [&](const auto& child) {
                self(self, child);
            });
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
