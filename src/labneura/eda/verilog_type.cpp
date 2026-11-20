#include "labneura/eda/verilog_type.h"
#include "labneura/eda/verilog_callable_binding.h"

#include <algorithm>
#include <cctype>
#include <functional>
#include <limits>
#include <string>
#include <unordered_set>
#include <utility>

namespace labneura::eda {
namespace {

bool contains_word(std::string_view text, std::string_view word) {
    size_t position = 0;
    while ((position = text.find(word, position)) != std::string_view::npos) {
        const bool left = position == 0 ||
                          !std::isalnum(static_cast<unsigned char>(text[position - 1]));
        const size_t end = position + word.size();
        const bool right = end == text.size() ||
                           !std::isalnum(static_cast<unsigned char>(text[end]));
        if (left && right)
            return true;
        position = end;
    }
    return false;
}

VerilogIntegralType unknown_type() { return {}; }

VerilogIntegralType integral_type(size_t width, bool is_signed, bool four_state) {
    VerilogIntegralType type;
    type.category = VerilogTypeCategory::Integral;
    type.bit_width = width;
    type.is_signed = is_signed;
    type.four_state = four_state;
    return type;
}

std::optional<int64_t> constant_integer(const VerilogExpression* expression,
                                        const VerilogConstantEnvironment& constants) {
    if (!expression)
        return std::nullopt;
    const auto result = evaluate_verilog_constant(*expression, constants);
    if (!result.value || !result.value->is_fully_known())
        return std::nullopt;
    return result.value->to_int64();
}

std::optional<size_t> range_width(const VerilogExpressionRange& range,
                                  const VerilogConstantEnvironment& constants) {
    if (!range.right)
        return 1;
    const auto left = constant_integer(range.left.get(), constants);
    const auto right = constant_integer(range.right.get(), constants);
    if (!left || !right)
        return std::nullopt;
    const uint64_t distance = *left >= *right
                                  ? static_cast<uint64_t>(*left) -
                                        static_cast<uint64_t>(*right)
                                  : static_cast<uint64_t>(*right) -
                                        static_cast<uint64_t>(*left);
    if (distance >= std::numeric_limits<size_t>::max())
        return std::nullopt;
    return static_cast<size_t>(distance + 1);
}

std::optional<size_t> range_width(const VerilogExpression* left_expression,
                                  const VerilogExpression* right_expression,
                                  const VerilogConstantEnvironment& constants) {
    if (!right_expression)
        return 1;
    const auto left = constant_integer(left_expression, constants);
    const auto right = constant_integer(right_expression, constants);
    if (!left || !right)
        return std::nullopt;
    const uint64_t distance = *left >= *right
                                  ? static_cast<uint64_t>(*left) -
                                        static_cast<uint64_t>(*right)
                                  : static_cast<uint64_t>(*right) -
                                        static_cast<uint64_t>(*left);
    if (distance >= std::numeric_limits<size_t>::max())
        return std::nullopt;
    return static_cast<size_t>(distance + 1);
}

class TypeAnalyzer {
public:
    TypeAnalyzer(const VerilogCompilationUnit& syntax,
                 const VerilogSemanticAnalysisResult& semantic) :
        syntax_(syntax), semantic_(semantic) {}

    VerilogTypeAnalysisResult run() {
        collect_package_types();
        for (const auto& module : syntax_.modules)
            analyze_module(module);
        for (const auto& interface : syntax_.interfaces)
            analyze_nonmodule(interface.name, interface.parameters, interface.ports,
                              interface.declarations, interface.type_definitions,
                              interface.callables);
        for (const auto& package : syntax_.packages)
            analyze_nonmodule(package.name, package.parameters, {},
                              package.declarations, package.type_definitions,
                              package.callables);
        return std::move(result_);
    }

    VerilogExpressionTypeResult infer_one(
        const VerilogExpression& expression, const VerilogTypeEnvironment& types,
        const VerilogConstantEnvironment& constants) {
        VerilogExpressionTypeResult inferred;
        inferred.type = infer_expression(expression, types, constants);
        inferred.diagnostics = std::move(result_.diagnostics);
        return inferred;
    }

private:
    using ParameterList = std::vector<VerilogParameterDeclaration>;
    using TypeDefinitionMap =
        std::unordered_map<std::string, const VerilogTypeDefinition*>;
    using CallableTypeMap =
        std::unordered_map<const VerilogCallableDeclaration*, VerilogTypeEnvironment>;

    enum class TypeResolutionState : uint8_t { Unvisited, Resolving, Resolved, Failed };

    void diagnose(std::string code, std::string message, const VerilogSyntaxLocation& location,
                  VerilogTypeDiagnosticSeverity severity = VerilogTypeDiagnosticSeverity::Error) {
        result_.diagnostics.push_back(
            {severity, std::move(code), std::move(message), location});
    }

    VerilogConstantEnvironment evaluate_parameters(
        const ParameterList& parameters, const VerilogConstantEnvironment& inherited) {
        VerilogConstantEnvironment values = inherited;
        std::unordered_map<std::string, const VerilogParameterDeclaration*> declarations;
        for (const auto& parameter : parameters)
            if (!parameter.named_override)
                declarations.try_emplace(parameter.name, &parameter);

        enum class State : uint8_t { Unvisited, Visiting, Complete, Failed };
        std::unordered_map<std::string, State> states;
        std::function<std::optional<VerilogConstantValue>(std::string_view)> resolve;
        resolve = [&](std::string_view name) -> std::optional<VerilogConstantValue> {
            const auto found = declarations.find(std::string(name));
            if (found == declarations.end()) {
                const auto inherited_value = values.find(std::string(name));
                if (inherited_value != values.end())
                    return inherited_value->second;
                return std::nullopt;
            }
            auto& state = states[found->first];
            if (state == State::Complete)
                return values.at(found->first);
            if (state == State::Visiting || state == State::Failed)
                return std::nullopt;
            state = State::Visiting;
            const auto* expression = found->second->value.get();
            if (!expression) {
                state = State::Failed;
                return std::nullopt;
            }
            auto evaluated = evaluate_verilog_constant(
                *expression, values,
                [&](std::string_view referenced, const VerilogSyntaxLocation&)
                    -> std::optional<VerilogConstantValue> { return resolve(referenced); });
            if (!evaluated.value) {
                state = State::Failed;
                return std::nullopt;
            }
            values[found->first] = *evaluated.value;
            state = State::Complete;
            return evaluated.value;
        };
        for (const auto& parameter : parameters)
            if (!parameter.named_override)
                (void)resolve(parameter.name);
        return values;
    }

    static std::string normalize_qualified_name(std::string_view name) {
        std::string normalized(name);
        size_t position = 0;
        while ((position = normalized.find("::", position)) != std::string::npos) {
            normalized.replace(position, 2, ".");
            ++position;
        }
        return normalized;
    }

    static VerilogPackedDimensionType resolve_dimension(
        const VerilogExpressionRange& dimension,
        const VerilogConstantEnvironment& constants) {
        VerilogPackedDimensionType resolved;
        resolved.width = range_width(dimension, constants);
        if (dimension.left)
            resolved.left = constant_integer(dimension.left.get(), constants);
        if (dimension.right)
            resolved.right = constant_integer(dimension.right.get(), constants);
        return resolved;
    }

    void apply_packed_dimensions(
        VerilogIntegralType& type, const std::vector<VerilogExpressionRange>& dimensions,
        const VerilogConstantEnvironment& constants,
        const VerilogSyntaxLocation& location) {
        if (dimensions.empty())
            return;
        size_t width = type.bit_width.value_or(1);
        bool known = type.bit_width.has_value() || type.category == VerilogTypeCategory::Integral;
        for (const auto& dimension : dimensions) {
            const auto resolved = resolve_dimension(dimension, constants);
            type.packed_dimensions.push_back(resolved);
            if (!resolved.width || !known ||
                width > std::numeric_limits<size_t>::max() / *resolved.width) {
                known = false;
                continue;
            }
            width *= *resolved.width;
        }
        if (type.category == VerilogTypeCategory::Integral ||
            type.category == VerilogTypeCategory::Enumeration ||
            (type.packed && (type.category == VerilogTypeCategory::Structure ||
                             type.category == VerilogTypeCategory::Union)))
            type.bit_width = known ? std::optional<size_t>(width) : std::nullopt;
        if (!known)
            diagnose("UnknownPackedDimension",
                     "packed type width could not be resolved from constant dimensions",
                     location, VerilogTypeDiagnosticSeverity::Warning);
    }

    VerilogIntegralType apply_unpacked_dimensions(
        VerilogIntegralType element, const std::vector<VerilogExpressionRange>& dimensions,
        const VerilogConstantEnvironment& constants) {
        for (auto dimension = dimensions.rbegin(); dimension != dimensions.rend(); ++dimension) {
            auto array = unknown_type();
            array.category = VerilogTypeCategory::Array;
            array.four_state = element.four_state;
            array.alias_name = element.alias_name;
            array.declared_type = element.declared_type;
            array.declaration_location = element.declaration_location;
            array.identity_name = element.identity_name;
            array.element_type =
                std::make_shared<const VerilogIntegralType>(std::move(element));
            const auto resolved = resolve_dimension(*dimension, constants);
            array.unpacked_dimensions.push_back(resolved);
            array.unpacked_element_count = resolved.width;
            element = std::move(array);
        }
        return element;
    }

    VerilogIntegralType resolve_type_name(
        std::string_view name, const TypeDefinitionMap& definitions,
        const VerilogTypeEnvironment& inherited_types, VerilogTypeEnvironment& resolved_types,
        VerilogConstantEnvironment& constants,
        std::unordered_map<std::string, TypeResolutionState>& states,
        const VerilogSemanticScope* semantic_scope) {
        const auto normalized_name = normalize_qualified_name(name);
        if (const auto found = resolved_types.find(normalized_name);
            found != resolved_types.end())
            return found->second;
        if (const auto found = inherited_types.find(normalized_name);
            found != inherited_types.end())
            return found->second;
        const auto definition = definitions.find(normalized_name);
        if (definition == definitions.end()) {
            const auto builtin = infer_verilog_declared_type(name, {}, constants);
            return builtin.category == VerilogTypeCategory::Unknown ? unknown_type() : builtin;
        }

        auto& state = states[normalized_name];
        if (state == TypeResolutionState::Resolving ||
            state == TypeResolutionState::Failed)
            return unknown_type();
        if (state == TypeResolutionState::Resolved)
            return resolved_types.at(normalized_name);
        state = TypeResolutionState::Resolving;
        auto type = resolve_type_definition(*definition->second, definitions, inherited_types,
                                            resolved_types, constants, states, semantic_scope);
        if (type.category == VerilogTypeCategory::Unknown) {
            state = TypeResolutionState::Failed;
            return type;
        }
        resolved_types[normalized_name] = type;
        state = TypeResolutionState::Resolved;
        if (semantic_scope && definition->second->kind == VerilogTypeDefinitionKind::Alias) {
            if (const auto* symbol =
                    lookup_verilog_symbol(*semantic_scope, definition->second->name, false))
                result_.symbol_types[symbol] = type;
        }
        return type;
    }

    VerilogIntegralType resolve_type_definition(
        const VerilogTypeDefinition& definition, const TypeDefinitionMap& definitions,
        const VerilogTypeEnvironment& inherited_types, VerilogTypeEnvironment& resolved_types,
        VerilogConstantEnvironment& constants,
        std::unordered_map<std::string, TypeResolutionState>& states,
        const VerilogSemanticScope* semantic_scope) {
        if (definition.kind == VerilogTypeDefinitionKind::Alias) {
            auto type = resolve_type_name(definition.target_type, definitions, inherited_types,
                                          resolved_types, constants, states, semantic_scope);
            if (type.category == VerilogTypeCategory::Unknown) {
                diagnose("UnresolvedTypeAlias",
                         "typedef '" + definition.name + "' refers to unknown type '" +
                             definition.target_type + "'",
                         definition.location);
                return type;
            }
            type.alias_name = definition.name;
            type.declared_type = definition.target_type;
            if (type.identity_name.starts_with("$native_type_"))
                type.identity_name = definition.name;
            type.declaration_location = definition.location;
            apply_packed_dimensions(type, definition.packed_dimensions, constants,
                                    definition.location);
            return apply_unpacked_dimensions(std::move(type),
                                             definition.unpacked_dimensions, constants);
        }

        if (definition.kind == VerilogTypeDefinitionKind::Enumeration) {
            auto type = resolve_type_name(definition.base_type, definitions, inherited_types,
                                          resolved_types, constants, states, semantic_scope);
            if (!type.is_known_integral()) {
                diagnose("InvalidEnumBaseType",
                         "enum base type must have a known integral width",
                         definition.location);
                return unknown_type();
            }
            apply_packed_dimensions(type, definition.packed_dimensions, constants,
                                    definition.location);
            type.category = VerilogTypeCategory::Enumeration;
            type.identity_name = definition.name;
            type.declaration_location = definition.location;
            type.packed = true;
            std::optional<VerilogConstantValue> previous;
            for (const auto& enumerator : definition.enumerators) {
                std::optional<VerilogConstantValue> value;
                if (enumerator.value) {
                    auto evaluated = evaluate_verilog_constant(*enumerator.value, constants);
                    for (const auto& diagnostic : evaluated.diagnostics)
                        diagnose(diagnostic.code, diagnostic.message, diagnostic.location);
                    value = std::move(evaluated.value);
                }
                else if (previous) {
                    auto next = *previous;
                    bool carry = true;
                    for (auto& bit : next.bits) {
                        if (!carry)
                            break;
                        if (bit == VerilogLogicBit::Zero) {
                            bit = VerilogLogicBit::One;
                            carry = false;
                        }
                        else if (bit == VerilogLogicBit::One)
                            bit = VerilogLogicBit::Zero;
                        else {
                            carry = false;
                            bit = VerilogLogicBit::X;
                        }
                    }
                    next.location = enumerator.location;
                    value = std::move(next);
                }
                if (value) {
                    value->bits.resize(*type.bit_width,
                                       type.is_signed && !value->bits.empty()
                                           ? value->bits.back()
                                           : VerilogLogicBit::Zero);
                    value->bit_width = *type.bit_width;
                    value->is_signed = type.is_signed;
                    constants[enumerator.name] = *value;
                    previous = *value;
                }
                else {
                    diagnose("InvalidEnumValue",
                             "enum value is not a supported constant expression",
                             enumerator.location);
                }
                type.enumerators.push_back(
                    {enumerator.name, enumerator.location, std::move(value)});
            }
            if (semantic_scope)
                for (const auto& enumerator : type.enumerators)
                    if (const auto* symbol = lookup_verilog_symbol(
                            *semantic_scope, enumerator.name, false)) {
                        result_.symbol_types[symbol] = type;
                        if (enumerator.value)
                            result_.symbol_constants[symbol] = *enumerator.value;
                    }
            return type;
        }

        VerilogIntegralType aggregate;
        aggregate.category = definition.kind == VerilogTypeDefinitionKind::Structure
                                 ? VerilogTypeCategory::Structure
                                 : VerilogTypeCategory::Union;
        aggregate.identity_name = definition.name;
        aggregate.declaration_location = definition.location;
        aggregate.packed = definition.is_packed;
        aggregate.is_aggregate_signed = definition.is_signed;
        aggregate.is_signed = definition.is_signed;
        aggregate.four_state = false;
        bool complete_layout = true;
        size_t total_width = 0;
        std::optional<size_t> union_width;
        for (const auto& member : definition.members) {
            auto member_type = resolve_type_name(member.type, definitions, inherited_types,
                                                 resolved_types, constants, states,
                                                 semantic_scope);
            apply_packed_dimensions(member_type, member.packed_dimensions, constants,
                                    member.location);
            member_type = apply_unpacked_dimensions(std::move(member_type),
                                                    member.unpacked_dimensions, constants);
            if (member_type.category == VerilogTypeCategory::Unknown) {
                diagnose("UnknownAggregateMemberType",
                         "aggregate member '" + member.name + "' has unresolved type '" +
                             member.type + "'",
                         member.location);
                complete_layout = false;
            }
            if (definition.is_packed) {
                if (!member_type.is_known_integral()) {
                    diagnose("UnsupportedPackedMemberType",
                             "packed aggregate members must have known integral widths",
                             member.location);
                    complete_layout = false;
                }
                else if (definition.kind == VerilogTypeDefinitionKind::Structure) {
                    if (total_width > std::numeric_limits<size_t>::max() -
                                          *member_type.bit_width) {
                        complete_layout = false;
                    }
                    else
                        total_width += *member_type.bit_width;
                }
                else {
                    if (union_width && *union_width != *member_type.bit_width) {
                        diagnose("PackedUnionWidthMismatch",
                                 "packed union members must have identical widths",
                                 member.location);
                        complete_layout = false;
                    }
                    union_width = union_width
                                      ? std::max(*union_width, *member_type.bit_width)
                                      : member_type.bit_width;
                }
            }
            aggregate.four_state |= member_type.four_state;
            aggregate.members.push_back(
                {member.name,
                 std::make_shared<const VerilogIntegralType>(std::move(member_type)),
                 std::nullopt, std::nullopt, member.location});
        }
        if (definition.is_packed && complete_layout) {
            const size_t width = definition.kind == VerilogTypeDefinitionKind::Structure
                                     ? total_width
                                     : union_width.value_or(0);
            aggregate.bit_width = width;
            if (definition.kind == VerilogTypeDefinitionKind::Structure) {
                size_t offset = 0;
                for (auto member = aggregate.members.rbegin();
                     member != aggregate.members.rend(); ++member) {
                    member->bit_offset = offset;
                    member->bit_width = member->type->bit_width;
                    offset += *member->type->bit_width;
                }
            }
            else {
                for (auto& member : aggregate.members) {
                    member.bit_offset = 0;
                    member.bit_width = member.type->bit_width;
                }
            }
        }
        else if (definition.is_packed) {
            diagnose("UnknownAggregateLayout",
                     "packed aggregate layout is unknown because a member width is unresolved",
                     definition.location);
        }
        return aggregate;
    }

    VerilogIntegralType resolve_declared_type(
        std::string_view name, const std::vector<VerilogExpressionRange>& packed_dimensions,
        const std::vector<VerilogExpressionRange>& unpacked_dimensions,
        const VerilogTypeEnvironment& types, const VerilogConstantEnvironment& constants) {
        VerilogIntegralType type;
        const auto normalized = normalize_qualified_name(name);
        bool dimensions_applied = false;
        if (const auto found = types.find(normalized); found != types.end())
            type = found->second;
        else {
            type = infer_verilog_declared_type(name, packed_dimensions, constants);
            dimensions_applied = true;
        }
        if (!dimensions_applied)
            apply_packed_dimensions(type, packed_dimensions, constants, {});
        return apply_unpacked_dimensions(std::move(type), unpacked_dimensions, constants);
    }

    VerilogTypeEnvironment resolve_scope_type_definitions(
        const std::vector<VerilogTypeDefinition>& definitions,
        const VerilogTypeEnvironment& inherited_types,
        VerilogConstantEnvironment& constants, const VerilogSemanticScope* semantic_scope) {
        TypeDefinitionMap definition_map;
        for (const auto& definition : definitions)
            definition_map.try_emplace(definition.name, &definition);
        VerilogTypeEnvironment resolved_types;
        std::unordered_map<std::string, TypeResolutionState> states;
        for (const auto& definition : definitions)
            (void)resolve_type_name(definition.name, definition_map, inherited_types,
                                    resolved_types, constants, states, semantic_scope);
        for (const auto& definition : definitions) {
            if (definition.kind != VerilogTypeDefinitionKind::Enumeration)
                continue;
            const auto type = resolved_types.find(definition.name);
            if (type == resolved_types.end())
                continue;
            for (const auto& enumerator : definition.enumerators)
                resolved_types[enumerator.name] = type->second;
        }
        auto all_types = inherited_types;
        for (const auto& [name, type] : resolved_types)
            all_types[name] = type;
        return all_types;
    }

    VerilogTypeEnvironment declared_types(
        const ParameterList& parameters, const std::vector<VerilogPortDeclaration>& ports,
        const std::vector<VerilogVariableDeclaration>& declarations,
        const VerilogTypeEnvironment& inherited_types,
        const VerilogConstantEnvironment& constants) {
        auto types = inherited_types;
        for (const auto& parameter : parameters) {
            if (parameter.named_override)
                continue;
            types[parameter.name] = resolve_declared_type(
                parameter.type, parameter.packed_dimensions, {}, inherited_types, constants);
            if (!parameter.type.empty() &&
                types[parameter.name].category == VerilogTypeCategory::Unknown)
                diagnose("UnknownTypeReference",
                         "parameter '" + parameter.name + "' has unresolved type '" +
                             parameter.type + "'",
                         parameter.location);
        }
        for (size_t pass = 0; pass < parameters.size(); ++pass) {
            bool changed = false;
            for (const auto& parameter : parameters) {
                if (parameter.named_override || !parameter.type.empty() || !parameter.value ||
                    types[parameter.name].category != VerilogTypeCategory::Unknown)
                    continue;
                auto inferred = infer_expression(*parameter.value, types, constants);
                if (inferred.category != VerilogTypeCategory::Unknown) {
                    types[parameter.name] = std::move(inferred);
                    changed = true;
                }
            }
            if (!changed)
                break;
        }
        for (const auto& port : ports) {
            types[port.name] = resolve_declared_type(
                port.type, port.packed_dimensions, port.unpacked_dimensions,
                inherited_types, constants);
            if (!port.type.empty() &&
                types[port.name].category == VerilogTypeCategory::Unknown)
                diagnose("UnknownTypeReference",
                         "port '" + port.name + "' has unresolved type '" + port.type + "'",
                         port.location);
        }
        for (const auto& declaration : declarations) {
            types[declaration.name] = resolve_declared_type(
                declaration.type, declaration.packed_dimensions,
                declaration.unpacked_dimensions, inherited_types, constants);
            if (!declaration.type.empty() &&
                types[declaration.name].category == VerilogTypeCategory::Unknown)
                diagnose("UnknownTypeReference",
                         "declaration '" + declaration.name + "' has unresolved type '" +
                             declaration.type + "'",
                         declaration.location);
        }
        return types;
    }

    const VerilogSemanticScope* design_scope(std::string_view unit_name) const {
        if (!semantic_.model.compilation_unit_scope)
            return nullptr;
        const auto* unit_symbol = lookup_verilog_symbol(
            *semantic_.model.compilation_unit_scope, unit_name, false);
        return unit_symbol ? unit_symbol->member_scope : nullptr;
    }

    void collect_symbols(const VerilogSemanticScope* scope,
                         const VerilogTypeEnvironment& types) {
        if (!scope)
            return;
        for (const auto& [name, type] : types) {
            const auto* symbol = lookup_verilog_symbol(*scope, name, false);
            if (symbol)
                result_.symbol_types[symbol] = type;
        }
    }

    const VerilogSemanticSymbol* find_callable_symbol(
        const VerilogSemanticScope* design_scope,
        const VerilogCallableDeclaration& callable) const {
        if (!design_scope)
            return nullptr;
        const auto* symbol =
            lookup_verilog_symbol(*design_scope, callable.name, false);
        if (!symbol)
            return nullptr;
        const auto expected_kind = callable.kind == VerilogCallableKind::Function
                                       ? VerilogSemanticSymbolKind::Function
                                       : VerilogSemanticSymbolKind::Task;
        return symbol->kind == expected_kind ? symbol : nullptr;
    }

    VerilogIntegralType callable_return_type(
        const VerilogCallableDeclaration& callable,
        const VerilogTypeEnvironment& types,
        const VerilogConstantEnvironment& constants,
        const VerilogSemanticSymbol* callable_symbol) {
        if (callable.kind != VerilogCallableKind::Function)
            return unknown_type();

        // An omitted SystemVerilog function return type defaults to 1-bit logic.
        const std::string_view semantic_return_spelling =
            callable_symbol && callable_symbol->callable
                ? std::string_view(callable_symbol->callable->return_type)
                : std::string_view{};
        const std::string_view return_spelling =
            !semantic_return_spelling.empty()
                ? semantic_return_spelling
                : callable.return_type && !callable.return_type->empty()
                      ? std::string_view(*callable.return_type)
                      : std::string_view("logic");
        const auto& packed_dimensions =
            callable_symbol && callable_symbol->callable
                ? callable_symbol->callable->return_packed_dimensions
                : callable.return_packed_dimensions;
        auto type = resolve_declared_type(
            return_spelling, packed_dimensions, {}, types, constants);
        if (type.category == VerilogTypeCategory::Unknown) {
            diagnose("UnknownCallableReturnType",
                     "function '" + callable.name + "' has unresolved return type '" +
                         std::string(return_spelling) + "'",
                     callable.location, VerilogTypeDiagnosticSeverity::Warning);
        }
        return type;
    }

    CallableTypeMap prepare_callable_types(
        const std::vector<VerilogCallableDeclaration>& callables,
        const VerilogTypeEnvironment& inherited_types,
        const VerilogConstantEnvironment& constants,
        const VerilogSemanticScope* design_scope) {
        CallableTypeMap callable_types;

        // Resolve all return types before building callable environments so
        // recursive and forward function calls can use their declared result.
        for (const auto& callable : callables) {
            const auto* symbol = find_callable_symbol(design_scope, callable);
            if (callable.kind != VerilogCallableKind::Function)
                continue;
            auto type =
                callable_return_type(callable, inherited_types, constants, symbol);
            if (symbol) {
                result_.symbol_types[symbol] = type;
            }
            infer_dimensions(callable.return_packed_dimensions, inherited_types, constants);
        }

        for (const auto& callable : callables) {
            auto types = inherited_types;
            const auto* symbol = find_callable_symbol(design_scope, callable);
            const VerilogSemanticCallableMetadata* metadata =
                symbol && symbol->callable ? &*symbol->callable : nullptr;

            if (callable.kind == VerilogCallableKind::Function) {
                if (symbol) {
                    if (const auto found = result_.symbol_types.find(symbol);
                        found != result_.symbol_types.end())
                        types[callable.name] = found->second;
                }
            }

            for (size_t i = 0; i < callable.formal_arguments.size(); ++i) {
                const auto& formal = callable.formal_arguments[i];
                const VerilogSemanticSymbol* formal_symbol =
                    metadata && i < metadata->formal_arguments.size()
                        ? metadata->formal_arguments[i]
                        : nullptr;
                const std::string_view declared_type =
                    formal_symbol && !formal_symbol->declared_type.empty()
                        ? std::string_view(formal_symbol->declared_type)
                        : std::string_view(formal.type);
                const auto& packed_dimensions =
                    formal_symbol ? formal_symbol->packed_dimensions
                                  : formal.packed_dimensions;
                const auto& unpacked_dimensions =
                    formal_symbol ? formal_symbol->unpacked_dimensions
                                  : formal.unpacked_dimensions;
                auto type = resolve_declared_type(declared_type, packed_dimensions,
                                                  unpacked_dimensions, types, constants);
                if (formal_symbol) {
                    type.declaration_location = formal_symbol->declaration_location;
                    result_.symbol_types[formal_symbol] = type;
                }
                if (!declared_type.empty() &&
                    type.category == VerilogTypeCategory::Unknown) {
                    diagnose("UnknownCallableArgumentType",
                             "formal argument '" + formal.name + "' of callable '" +
                                 callable.name + "' has unresolved type '" +
                                 std::string(declared_type) + "'",
                             formal.location, VerilogTypeDiagnosticSeverity::Warning);
                }
                types[formal.name] = std::move(type);
                infer_dimensions(formal.packed_dimensions, types, constants);
                infer_dimensions(formal.unpacked_dimensions, types, constants);
            }

            types = declared_types({}, {}, callable.local_declarations, types, constants);
            for (const auto& declaration : callable.local_declarations) {
                infer_dimensions(declaration.packed_dimensions, types, constants);
                infer_dimensions(declaration.unpacked_dimensions, types, constants);
            }
            collect_symbols(symbol ? symbol->member_scope : nullptr, types);
            callable_types.emplace(&callable, std::move(types));
        }
        return callable_types;
    }

    void analyze_callable_bodies(
        const std::vector<VerilogCallableDeclaration>& callables,
        const CallableTypeMap& callable_types,
        const VerilogConstantEnvironment& constants,
        const VerilogSemanticScope* design_scope) {
        for (const auto& callable : callables) {
            const auto found_types = callable_types.find(&callable);
            const auto* symbol = find_callable_symbol(design_scope, callable);
            const auto* scope = symbol ? symbol->member_scope : nullptr;
            if (found_types == callable_types.end())
                continue;
            const auto& types = found_types->second;
            for (const auto& declaration : callable.local_declarations)
                infer_and_store(declaration.initializer.get(), types, constants);
            analyze_statement(callable.body.get(), types, constants, scope);
        }
    }

    void analyze_nonmodule(std::string_view unit_name, const ParameterList& parameters,
                           const std::vector<VerilogPortDeclaration>& ports,
                           const std::vector<VerilogVariableDeclaration>& declarations,
                           const std::vector<VerilogTypeDefinition>& type_definitions,
                           const std::vector<VerilogCallableDeclaration>& callables) {
        auto constants = package_constants_;
        auto* semantic_scope = design_scope(unit_name);
        constants = evaluate_parameters(parameters, constants);
        auto types = resolve_scope_type_definitions(type_definitions, package_types_, constants,
                                                    semantic_scope);
        constants = evaluate_parameters(parameters, constants);
        types = declared_types(parameters, ports, declarations, types, constants);
        collect_symbols(semantic_scope, types);
        const auto callable_types =
            prepare_callable_types(callables, types, constants, semantic_scope);
        analyze_callable_bodies(callables, callable_types, constants, semantic_scope);
        for (const auto& parameter : parameters)
            infer_and_store(parameter.value.get(), types, constants);
        for (const auto& port : ports)
            infer_dimensions(port.packed_dimensions, types, constants);
        for (const auto& declaration : declarations) {
            infer_dimensions(declaration.packed_dimensions, types, constants);
            infer_dimensions(declaration.unpacked_dimensions, types, constants);
            infer_and_store(declaration.initializer.get(), types, constants);
        }
    }

    void collect_package_types() {
        for (const auto& package : syntax_.packages) {
            auto constants = evaluate_parameters(package.parameters, {});
            const auto* semantic_scope = design_scope(package.name);
            auto local_types = resolve_scope_type_definitions(
                package.type_definitions, {}, constants, semantic_scope);
            constants = evaluate_parameters(package.parameters, constants);
            auto types = declared_types(package.parameters, {}, package.declarations,
                                        local_types, constants);
            for (const auto& definition : package.type_definitions) {
                const auto found = local_types.find(definition.name);
                if (found != local_types.end()) {
                    package_types_[package.name + "." + definition.name] = found->second;
                    if (definition.kind == VerilogTypeDefinitionKind::Enumeration)
                        for (const auto& enumerator : definition.enumerators)
                            package_types_[package.name + "." + enumerator.name] =
                                found->second;
                }
            }
            for (const auto& [name, type] : types) {
                if (name.find('.') == std::string::npos &&
                    std::any_of(package.type_definitions.begin(),
                                package.type_definitions.end(), [&](const auto& definition) {
                                    return definition.name == name;
                                }))
                    package_types_[package.name + "." + name] = type;
            }
            for (const auto& [name, value] : constants)
                if (name.find('.') == std::string::npos)
                    package_constants_[package.name + "." + name] = value;
        }
    }

    void analyze_module(const VerilogModuleDeclaration& module) {
        auto constants = evaluate_parameters(module.parameters, package_constants_);
        const auto* semantic_scope = design_scope(module.name);
        auto types = resolve_scope_type_definitions(module.type_definitions, package_types_,
                                                    constants, semantic_scope);
        constants = evaluate_parameters(module.parameters, constants);
        types = declared_types(module.parameters, module.ports, module.declarations,
                               types, constants);
        collect_symbols(semantic_scope, types);
        const auto callable_types =
            prepare_callable_types(module.callables, types, constants, semantic_scope);
        analyze_callable_bodies(module.callables, callable_types, constants, semantic_scope);

        for (const auto& parameter : module.parameters)
            infer_and_store(parameter.value.get(), types, constants);
        for (const auto& port : module.ports) {
            infer_dimensions(port.packed_dimensions, types, constants);
            infer_dimensions(port.unpacked_dimensions, types, constants);
        }
        for (const auto& declaration : module.declarations) {
            infer_dimensions(declaration.packed_dimensions, types, constants);
            infer_dimensions(declaration.unpacked_dimensions, types, constants);
            infer_and_store(declaration.initializer.get(), types, constants);
        }
        for (const auto& instantiation : module.instantiations)
            analyze_instantiation(instantiation, types, constants);
        for (const auto& assignment : module.assignments) {
            const auto left = infer_and_store(assignment.left.get(), types, constants);
            const auto right = infer_and_store(assignment.right.get(), types, constants);
            check_assignment(left, right, assignment.location);
        }
        for (const auto& block : module.procedural_blocks)
            analyze_statement(block.body.get(), types, constants, semantic_scope);
        for (const auto& construct : module.generate_constructs)
            analyze_generate(*construct, types, constants, semantic_scope);
    }

    void infer_dimensions(const std::vector<VerilogExpressionRange>& dimensions,
                          const VerilogTypeEnvironment& types,
                          const VerilogConstantEnvironment& constants) {
        for (const auto& dimension : dimensions) {
            infer_and_store(dimension.left.get(), types, constants);
            infer_and_store(dimension.right.get(), types, constants);
        }
    }

    void analyze_instantiation(const VerilogModuleInstantiation& instantiation,
                               const VerilogTypeEnvironment& types,
                               const VerilogConstantEnvironment& constants) {
        for (const auto& override : instantiation.parameter_overrides)
            infer_and_store(override.value.get(), types, constants);
        for (const auto& instance : instantiation.instances) {
            for (const auto& dimension : instance.unpacked_dimensions) {
                infer_and_store(dimension.left.get(), types, constants);
                infer_and_store(dimension.right.get(), types, constants);
            }
            for (const auto& connection : instance.connections)
                infer_and_store(connection.expression.get(), types, constants);
        }
    }

    VerilogIntegralType infer_and_store(const VerilogExpression* expression,
                                        const VerilogTypeEnvironment& types,
                                        const VerilogConstantEnvironment& constants) {
        if (!expression)
            return unknown_type();
        auto type = infer_expression(*expression, types, constants);
        result_.expression_types.insert_or_assign(expression, type);
        return type;
    }

    const VerilogSemanticSymbol* resolved_callable(
        const VerilogExpression& call) const {
        if (!call.first)
            return nullptr;
        const auto& callee_location = call.first->location;
        const auto references = semantic_.references_at(callee_location);
        if (!references.empty())
            return semantic_.references[references.begin()->reference_ordinal]
                .declaration;
        return nullptr;
    }

    VerilogIntegralType infer_expression(const VerilogExpression& expression,
                                         const VerilogTypeEnvironment& types,
                                         const VerilogConstantEnvironment& constants) {
        auto child_type = [&](const VerilogExpression* child) {
            return child ? infer_expression(*child, types, constants) : unknown_type();
        };
        VerilogIntegralType result;
        switch (expression.kind) {
            case VerilogExpressionKind::Literal:
                result = literal_type(expression.text);
                break;
            case VerilogExpressionKind::Identifier:
                if (const auto found = types.find(expression.text); found != types.end())
                    result = found->second;
                break;
            case VerilogExpressionKind::MemberAccess:
                if (expression.first) {
                    const auto base = child_type(expression.first.get());
                    const std::string qualified =
                        qualified_member_name(expression);
                    if (const auto found = types.find(qualified); found != types.end()) {
                        result = found->second;
                        break;
                    }
                    const auto member = std::find_if(
                        base.members.begin(), base.members.end(), [&](const auto& candidate) {
                            return candidate.name == expression.text;
                        });
                    if ((base.category == VerilogTypeCategory::Structure ||
                         base.category == VerilogTypeCategory::Union) &&
                        member != base.members.end() && member->type)
                        result = *member->type;
                    else if (base.category == VerilogTypeCategory::Structure ||
                             base.category == VerilogTypeCategory::Union)
                        diagnose("InvalidMemberAccess",
                                 "type '" + base.identity_name + "' has no member named '" +
                                     expression.text + "'",
                                 expression.location);
                    else if (base.category == VerilogTypeCategory::Enumeration) {
                        const auto enumerator = std::find_if(
                            base.enumerators.begin(), base.enumerators.end(),
                            [&](const auto& candidate) {
                                return candidate.name == expression.text;
                            });
                        if (enumerator != base.enumerators.end())
                            result = base;
                        else
                            diagnose("InvalidEnumMember",
                                     "enum type has no enumerator named '" + expression.text +
                                         "'",
                                     expression.location);
                    }
                }
                break;
            case VerilogExpressionKind::Unary: {
                auto operand = child_type(expression.first.get());
                if (!operand.is_known_integral())
                    break;
                if (expression.text == "!" || expression.text == "&" ||
                    expression.text == "~&" || expression.text == "|" ||
                    expression.text == "~|" || expression.text == "^" ||
                    expression.text == "~^" || expression.text == "^~") {
                    result = integral_type(1, false, operand.four_state);
                }
                else {
                    result = operand;
                    result.unbased_unsized = false;
                }
                break;
            }
            case VerilogExpressionKind::Binary: {
                auto left = child_type(expression.first.get());
                auto right = child_type(expression.second.get());
                if (!left.is_known_integral() || !right.is_known_integral())
                    break;
                const auto& op = expression.text;
                const bool four_state = left.four_state || right.four_state;
                if (op == "&&" || op == "||" || op == "==" || op == "!=" ||
                    op == "===" || op == "!==" || op == "==?" || op == "!=?" ||
                    op == "<" || op == "<=" || op == ">" || op == ">=") {
                    result = integral_type(1, false, four_state);
                }
                else if (op == "<<" || op == ">>" || op == "<<<" || op == ">>>") {
                    result = left;
                    result.unbased_unsized = false;
                }
                else if (left.bit_width && right.bit_width) {
                    result = integral_type(std::max(*left.bit_width, *right.bit_width),
                                           left.is_signed && right.is_signed, four_state);
                }
                break;
            }
            case VerilogExpressionKind::Conditional: {
                (void)child_type(expression.first.get());
                auto when_true = child_type(expression.second.get());
                auto when_false = child_type(expression.third.get());
                if (when_true.is_known_integral() && when_false.is_known_integral()) {
                    result = integral_type(std::max(*when_true.bit_width, *when_false.bit_width),
                                           when_true.is_signed && when_false.is_signed,
                                           when_true.four_state || when_false.four_state);
                }
                break;
            }
            case VerilogExpressionKind::Concatenation: {
                size_t width = 0;
                bool four_state = false;
                bool known = !expression.operands.empty();
                for (const auto& operand : expression.operands) {
                    auto type = child_type(operand.get());
                    if (!type.is_known_integral() ||
                        width > std::numeric_limits<size_t>::max() - *type.bit_width) {
                        known = false;
                        continue;
                    }
                    width += *type.bit_width;
                    four_state |= type.four_state;
                }
                if (known)
                    result = integral_type(width, false, four_state);
                else
                    diagnose("UnknownConcatenationWidth",
                             "concatenation operand width is unknown",
                             expression.location, VerilogTypeDiagnosticSeverity::Warning);
                break;
            }
            case VerilogExpressionKind::Replication: {
                auto count = constant_integer(expression.first.get(), constants);
                size_t item_width = 0;
                bool known = count && *count > 0;
                bool four_state = false;
                for (const auto& operand : expression.operands) {
                    auto type = child_type(operand.get());
                    if (!type.is_known_integral() ||
                        item_width > std::numeric_limits<size_t>::max() - *type.bit_width) {
                        known = false;
                        continue;
                    }
                    item_width += *type.bit_width;
                    four_state |= type.four_state;
                }
                if (known && item_width != 0 &&
                    static_cast<uint64_t>(*count) <=
                        std::numeric_limits<size_t>::max() / item_width)
                    result = integral_type(static_cast<size_t>(*count) * item_width, false,
                                           four_state);
                else if (count && *count > 0)
                    diagnose("UnknownReplicationWidth",
                             "replication width is unknown or exceeds the supported range",
                             expression.location, VerilogTypeDiagnosticSeverity::Warning);
                else if (count)
                    diagnose("InvalidReplicationCount",
                             "replication count must be greater than zero",
                             expression.location);
                else
                    diagnose("UnknownReplicationCount",
                             "replication count is not constant in the available environment",
                             expression.location, VerilogTypeDiagnosticSeverity::Warning);
                break;
            }
            case VerilogExpressionKind::Select: {
                const auto base = child_type(expression.first.get());
                (void)child_type(expression.second.get());
                if (expression.third) {
                    (void)child_type(expression.third.get());
                    if (base.category == VerilogTypeCategory::Array) {
                        diagnose("UnsupportedArrayPartSelect",
                                 "part-selects of unpacked arrays are not supported",
                                 expression.location, VerilogTypeDiagnosticSeverity::Warning);
                        break;
                    }
                    std::optional<size_t> width;
                    if (expression.select_kind == VerilogSelectKind::IndexedUp ||
                        expression.select_kind == VerilogSelectKind::IndexedDown) {
                        const auto indexed_width =
                            constant_integer(expression.third.get(), constants);
                        if (indexed_width && *indexed_width > 0 &&
                            static_cast<uint64_t>(*indexed_width) <=
                                std::numeric_limits<size_t>::max())
                            width = static_cast<size_t>(*indexed_width);
                    } else {
                        width = range_width(expression.second.get(),
                                            expression.third.get(), constants);
                    }
                    if (width)
                        result = integral_type(*width, false, base.four_state);
                    else
                        diagnose("UnknownSelectWidth",
                                 "part-select width is not known in the available constant environment",
                                 expression.location, VerilogTypeDiagnosticSeverity::Warning);
                }
                else if (base.category == VerilogTypeCategory::Array && base.element_type) {
                    result = *base.element_type;
                }
                else if (base.is_known_integral()) {
                    result = integral_type(1, false, base.four_state);
                }
                break;
            }
            case VerilogExpressionKind::Call: {
                (void)child_type(expression.first.get());
                for (const auto& operand : expression.operands)
                    (void)child_type(operand.get());
                const auto* callable = resolved_callable(expression);
                if (callable &&
                    (callable->kind == VerilogSemanticSymbolKind::Function ||
                     callable->kind == VerilogSemanticSymbolKind::Task)) {
                    const auto binding = bind_verilog_callable_arguments(
                        *callable, expression, result_);
                    for (const auto& diagnostic : binding.diagnostics)
                        diagnose(diagnostic.code, diagnostic.message,
                                 diagnostic.location, diagnostic.severity);
                    if (callable->kind == VerilogSemanticSymbolKind::Function) {
                        if (const auto found = result_.symbol_types.find(callable);
                            found != result_.symbol_types.end())
                            result = found->second;
                    }
                }
                else {
                    diagnose("UnsupportedExpressionType",
                             "function-call return type is not represented by the native type subset",
                             expression.location, VerilogTypeDiagnosticSeverity::Warning);
                }
                break;
            }
        }
        result_.expression_types.insert_or_assign(&expression, result);
        return result;
    }

public:
    static VerilogIntegralType literal_type(std::string_view spelling) {
        if (spelling.empty())
            return unknown_type();
        if (spelling.front() == '"') {
            VerilogIntegralType type;
            type.category = VerilogTypeCategory::String;
            return type;
        }
        const auto apostrophe = spelling.find('\'');
        if (spelling.find('.') != std::string_view::npos ||
            (apostrophe == std::string_view::npos &&
             (spelling.find('e') != std::string_view::npos ||
              spelling.find('E') != std::string_view::npos))) {
            VerilogIntegralType type;
            type.category = VerilogTypeCategory::Real;
            return type;
        }
        if (apostrophe == std::string_view::npos) {
            VerilogExpression literal;
            literal.kind = VerilogExpressionKind::Literal;
            literal.text = spelling;
            const auto evaluated = evaluate_verilog_constant(literal);
            if (!evaluated.value)
                return unknown_type();
            return integral_type(evaluated.value->bit_width,
                                 evaluated.value->is_signed, false);
        }
        if (apostrophe == 0 && spelling.size() == 2 &&
            (spelling[1] == '0' || spelling[1] == '1' || spelling[1] == 'x' ||
             spelling[1] == 'X' || spelling[1] == 'z' || spelling[1] == 'Z')) {
            const char fill = spelling.back();
            auto type = integral_type(1, false, fill == 'x' || fill == 'X' ||
                                                fill == 'z' || fill == 'Z');
            type.unbased_unsized = true;
            return type;
        }
        size_t width = 0;
        for (size_t i = 0; i < apostrophe; ++i) {
            if (spelling[i] == '_')
                continue;
            if (!std::isdigit(static_cast<unsigned char>(spelling[i])))
                return unknown_type();
            const size_t digit = static_cast<size_t>(spelling[i] - '0');
            if (width > (std::numeric_limits<size_t>::max() - digit) / 10)
                return unknown_type();
            width = width * 10 + digit;
        }
        size_t base_offset = apostrophe + 1;
        bool is_signed = false;
        if (base_offset < spelling.size() &&
            (spelling[base_offset] == 's' || spelling[base_offset] == 'S')) {
            is_signed = true;
            ++base_offset;
        }
        if (width == 0) {
            const char base = base_offset < spelling.size()
                                  ? static_cast<char>(std::tolower(
                                        static_cast<unsigned char>(spelling[base_offset])))
                                  : 'd';
            const size_t bits_per_digit =
                base == 'h' ? 4 : base == 'o' ? 3 : base == 'b' ? 1 : 4;
            size_t digits = 0;
            for (size_t i = base_offset + 1; i < spelling.size(); ++i)
                if (spelling[i] != '_')
                    ++digits;
            width = std::max<size_t>(32, digits * bits_per_digit);
        }
        return integral_type(width, is_signed, true);
    }

    static std::string qualified_member_name(const VerilogExpression& expression) {
        if (!expression.first)
            return {};
        std::string prefix;
        if (expression.first->kind == VerilogExpressionKind::Identifier)
            prefix = expression.first->text;
        else if (expression.first->kind == VerilogExpressionKind::MemberAccess)
            prefix = qualified_member_name(*expression.first);
        else
            return {};
        if (prefix.empty())
            return {};
        return prefix + "." + expression.text;
    }

private:
    void check_assignment(const VerilogIntegralType& left, const VerilogIntegralType& right,
                         const VerilogSyntaxLocation& location) {
        if (left.is_known_integral() && right.is_known_integral() &&
            left.bit_width != right.bit_width) {
            diagnose("AssignmentWidthMismatch",
                     "assignment converts " + std::to_string(*right.bit_width) +
                         "-bit expression to " + std::to_string(*left.bit_width) +
                         "-bit destination",
                     location, VerilogTypeDiagnosticSeverity::Warning);
        }
        if (left.is_known_integral() && right.is_known_integral() &&
            left.is_signed != right.is_signed) {
            diagnose("AssignmentSignednessConversion",
                     "assignment converts between signed and unsigned integral types",
                     location, VerilogTypeDiagnosticSeverity::Warning);
        }
    }

    void analyze_statement(const VerilogStatement* statement,
                           const VerilogTypeEnvironment& types,
                           const VerilogConstantEnvironment& constants,
                           const VerilogSemanticScope* semantic_scope) {
        if (!statement || statement->kind == VerilogStatementKind::Unsupported)
            return;
        if (statement->kind == VerilogStatementKind::Loop &&
            !statement->loop_variable_name.empty()) {
            auto loop_types = types;
            auto loop_type = infer_verilog_declared_type(statement->loop_variable_type);
            if (statement->loop_variable_type.empty() &&
                statement->loop_initialization &&
                statement->loop_initialization->right) {
                loop_type = infer_expression(
                    *statement->loop_initialization->right, loop_types, constants);
            }
            loop_types[statement->loop_variable_name] = loop_type;
            const std::string scope_name =
                "<loop@" + std::to_string(statement->location.location.line) + ":" +
                std::to_string(statement->location.location.column) + ">";
            const VerilogSemanticScope* loop_scope = nullptr;
            if (semantic_scope) {
                const auto found = std::find_if(
                    semantic_scope->children.begin(), semantic_scope->children.end(),
                    [&](const auto& child) {
                        return child->kind == VerilogSemanticScopeKind::Procedural &&
                               child->name == scope_name;
                    });
                if (found != semantic_scope->children.end())
                    loop_scope = found->get();
            }
            collect_symbols(loop_scope, loop_types);
            analyze_statement(statement->loop_initialization.get(), loop_types, constants,
                              loop_scope);
            infer_and_store(statement->expression.get(), loop_types, constants);
            analyze_statement(statement->loop_step.get(), loop_types, constants, loop_scope);
            analyze_statement(statement->first_statement.get(), loop_types, constants,
                              loop_scope);
            return;
        }
        const auto expression =
            statement->kind == VerilogStatementKind::TaskCall
                ? infer_and_store(statement->call_expression.get(), types, constants)
                : infer_and_store(statement->expression.get(), types, constants);
        if (statement->kind == VerilogStatementKind::ProceduralAssignment) {
            const auto left = infer_and_store(statement->left.get(), types, constants);
            const auto right = infer_and_store(statement->right.get(), types, constants);
            check_assignment(left, right, statement->location);
        }
        else if (statement->left)
            (void)infer_and_store(statement->left.get(), types, constants);
        (void)expression;
        if (statement->event_control)
            for (const auto& term : statement->event_control->terms)
                (void)infer_and_store(term.expression.get(), types, constants);
        analyze_statement(statement->first_statement.get(), types, constants, semantic_scope);
        analyze_statement(statement->second_statement.get(), types, constants, semantic_scope);
        for (const auto& child : statement->statements)
            analyze_statement(child.get(), types, constants, semantic_scope);
        for (const auto& item : statement->case_items) {
            for (const auto& item_expression : item.expressions)
                (void)infer_and_store(item_expression.get(), types, constants);
            analyze_statement(item.statement.get(), types, constants, semantic_scope);
        }
        analyze_statement(statement->loop_initialization.get(), types, constants, semantic_scope);
        analyze_statement(statement->loop_step.get(), types, constants, semantic_scope);
    }

    const VerilogSemanticScope* generate_child_scope(
        const VerilogSemanticScope* parent, std::string_view name) const {
        if (!parent)
            return nullptr;
        const auto found = std::find_if(
            parent->children.begin(), parent->children.end(), [&](const auto& child) {
                return child->kind == VerilogSemanticScopeKind::Generate &&
                       child->name == name;
            });
        return found == parent->children.end() ? nullptr : found->get();
    }

    void analyze_generate_scope(const VerilogGenerateScope& scope,
                                const VerilogTypeEnvironment& inherited_types,
                                const VerilogConstantEnvironment& inherited_constants,
                                const VerilogSemanticScope* inherited_scope,
                                bool create_child_scope,
                                std::optional<std::string_view> loop_variable = std::nullopt) {
        auto constants = evaluate_parameters(scope.parameters, inherited_constants);
        const auto* semantic_scope =
            create_child_scope ? generate_child_scope(inherited_scope, scope.name)
                               : inherited_scope;
        auto types = resolve_scope_type_definitions(scope.type_definitions,
                                                    inherited_types, constants,
                                                    semantic_scope);
        constants = evaluate_parameters(scope.parameters, constants);
        types = declared_types(scope.parameters, {}, scope.declarations, types, constants);
        if (loop_variable)
            types[std::string(*loop_variable)] = integral_type(32, true, false);
        collect_symbols(semantic_scope, types);
        for (const auto& parameter : scope.parameters)
            infer_and_store(parameter.value.get(), types, constants);
        for (const auto& declaration : scope.declarations) {
            infer_dimensions(declaration.packed_dimensions, types, constants);
            infer_dimensions(declaration.unpacked_dimensions, types, constants);
            infer_and_store(declaration.initializer.get(), types, constants);
        }
        for (const auto& assignment : scope.assignments) {
            const auto left = infer_and_store(assignment.left.get(), types, constants);
            const auto right = infer_and_store(assignment.right.get(), types, constants);
            check_assignment(left, right, assignment.location);
        }
        for (const auto& instantiation : scope.instantiations)
            analyze_instantiation(instantiation, types, constants);
        for (const auto& construct : scope.constructs)
            analyze_generate(*construct, types, constants, semantic_scope);
    }

    void analyze_generate(const VerilogGenerateConstruct& construct,
                          const VerilogTypeEnvironment& types,
                          const VerilogConstantEnvironment& constants,
                          const VerilogSemanticScope* semantic_scope) {
        infer_and_store(construct.condition.get(), types, constants);
        infer_and_store(construct.initialization.get(), types, constants);
        infer_and_store(construct.loop_condition.get(), types, constants);
        infer_and_store(construct.step.get(), types, constants);
        auto analyze_scope = [&](const std::unique_ptr<VerilogGenerateScope>& scope,
                                 bool create_child_scope,
                                 std::optional<std::string_view> loop_variable = std::nullopt) {
            if (scope)
                analyze_generate_scope(*scope, types, constants, semantic_scope,
                                       create_child_scope, loop_variable);
        };
        analyze_scope(construct.region_body, false);
        analyze_scope(construct.true_body, true);
        analyze_scope(construct.false_body, true);
        analyze_scope(construct.loop_body, true, construct.loop_variable);
        for (const auto& item : construct.case_items) {
            for (const auto& expression : item.expressions)
                infer_and_store(expression.get(), types, constants);
            analyze_scope(item.body, true);
        }
    }

    const VerilogCompilationUnit& syntax_;
    const VerilogSemanticAnalysisResult& semantic_;
    VerilogTypeAnalysisResult result_;
    VerilogTypeEnvironment package_types_;
    VerilogConstantEnvironment package_constants_;
};

} // namespace

bool VerilogTypeAnalysisResult::has_errors() const noexcept {
    return std::any_of(diagnostics.begin(), diagnostics.end(), [](const auto& diagnostic) {
        return diagnostic.severity == VerilogTypeDiagnosticSeverity::Error;
    });
}

VerilogIntegralType infer_verilog_declared_type(
    std::string_view declared_type,
    const std::vector<VerilogExpressionRange>& packed_dimensions,
    const VerilogConstantEnvironment& constant_environment,
    const VerilogIntegralType* inferred_type) {
    VerilogIntegralType type = unknown_type();
    const auto assign_integral = [&](size_t width, bool is_signed, bool four_state) {
        type = integral_type(width, is_signed, four_state);
    };
    if (contains_word(declared_type, "byte"))
        assign_integral(8, true, false);
    else if (contains_word(declared_type, "shortint"))
        assign_integral(16, true, false);
    else if (contains_word(declared_type, "int"))
        assign_integral(32, true, false);
    else if (contains_word(declared_type, "integer"))
        assign_integral(32, true, true);
    else if (contains_word(declared_type, "longint"))
        assign_integral(64, true, false);
    else if (contains_word(declared_type, "time"))
        assign_integral(64, false, true);
    else if (contains_word(declared_type, "bit"))
        assign_integral(1, false, false);
    else if (contains_word(declared_type, "logic") ||
             contains_word(declared_type, "reg") ||
             contains_word(declared_type, "wire") ||
             contains_word(declared_type, "tri") ||
             contains_word(declared_type, "wand") ||
             contains_word(declared_type, "wor") ||
             contains_word(declared_type, "uwire"))
        assign_integral(1, false, true);
    else if (contains_word(declared_type, "real") ||
             contains_word(declared_type, "realtime"))
        type.category = VerilogTypeCategory::Real;
    else if (!packed_dimensions.empty() &&
             (declared_type.empty() || contains_word(declared_type, "signed") ||
              contains_word(declared_type, "unsigned")))
        assign_integral(1, false, true);
    else if (declared_type.empty()) {
        if (inferred_type)
            type = *inferred_type;
        else
            assign_integral(1, false, true);
    }

    if (type.category == VerilogTypeCategory::Integral) {
        if (contains_word(declared_type, "signed"))
            type.is_signed = true;
        if (contains_word(declared_type, "unsigned"))
            type.is_signed = false;
        if (!packed_dimensions.empty()) {
            size_t width = type.bit_width.value_or(1);
            bool known = true;
            for (const auto& dimension : packed_dimensions) {
                VerilogPackedDimensionType resolved;
                resolved.width = range_width(dimension, constant_environment);
                if (dimension.left)
                    resolved.left = constant_integer(dimension.left.get(), constant_environment);
                if (dimension.right)
                    resolved.right = constant_integer(dimension.right.get(), constant_environment);
                type.packed_dimensions.push_back(resolved);
                if (!resolved.width ||
                    width > std::numeric_limits<size_t>::max() / *resolved.width) {
                    known = false;
                    break;
                }
                width *= *resolved.width;
            }
            type.bit_width = known ? std::optional<size_t>(width) : std::nullopt;
        }
    }
    type.declared_type = std::string(declared_type);
    return type;
}

VerilogExpressionTypeResult infer_verilog_expression_type(
    const VerilogExpression& expression, const VerilogTypeEnvironment& type_environment,
    const VerilogConstantEnvironment& constant_environment) {
    VerilogCompilationUnit syntax;
    VerilogSemanticAnalysisResult semantic;
    TypeAnalyzer analyzer(syntax, semantic);
    return analyzer.infer_one(expression, type_environment, constant_environment);
}

VerilogExpressionTypeResult infer_verilog_expression_type(
    const VerilogExpression& expression,
    const VerilogTypeAnalysisResult& type_analysis,
    const VerilogTypeEnvironment& type_environment,
    const VerilogConstantEnvironment& constant_environment) {
    // Build a name-based type environment from the type analysis result's
    // symbol type mappings so identifier expressions resolve against declared
    // types from parameters, ports, variables, and enumerators.
    VerilogTypeEnvironment merged = type_environment;
    for (const auto& [symbol, type] : type_analysis.symbol_types) {
        if (symbol)
            merged.try_emplace(symbol->name, type);
    }
    for (const auto& [symbol, value] : type_analysis.symbol_constants) {
        if (symbol && merged.find(symbol->name) == merged.end()) {
            VerilogIntegralType constant_type;
            constant_type.category = VerilogTypeCategory::Integral;
            constant_type.bit_width = value.bit_width;
            constant_type.is_signed = value.is_signed;
            constant_type.four_state = !value.is_fully_known();
            merged.try_emplace(symbol->name, constant_type);
        }
    }
    VerilogCompilationUnit syntax;
    VerilogSemanticAnalysisResult empty_semantic;
    TypeAnalyzer analyzer(syntax, empty_semantic);
    return analyzer.infer_one(expression, merged, constant_environment);
}

VerilogTypeAnalysisResult analyze_verilog_expression_types(
    const VerilogCompilationUnit& syntax, const VerilogSemanticAnalysisResult& semantic) {
    return TypeAnalyzer(syntax, semantic).run();
}

} // namespace labneura::eda
