#include "labneura/eda/verilog_elaboration.h"

#include <algorithm>
#include <bit>
#include <cctype>
#include <functional>
#include <limits>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace labneura::eda {
namespace {

struct ParameterGroup {
    std::string scope_name;
    std::vector<const VerilogParameterDeclaration*> parameters;
};

struct EvaluatedParameterGroup {
    VerilogConstantEnvironment values;
};

class Elaborator {
public:
    Elaborator(const VerilogCompilationUnit& syntax, VerilogElaborationOptions options) :
        syntax_(syntax), options_(std::move(options)) {
        result_.semantic = analyze_verilog_semantics(syntax_);
        type_analysis_ = analyze_verilog_expression_types(syntax_, result_.semantic);
        for (const auto& diagnostic : result_.semantic.diagnostics)
            result_.diagnostics.push_back({diagnostic.code, diagnostic.message, diagnostic.location,
                                           diagnostic.related_location});
        for (const auto& module : syntax_.modules)
            modules_.try_emplace(module.name, &module);
        for (const auto& package : syntax_.packages)
            packages_.try_emplace(package.name, &package);
    }

    VerilogElaborationResult run() {
        evaluate_package_parameters();
        const auto top_name = select_top();
        if (top_name.empty())
            return std::move(result_);
        const auto found = modules_.find(top_name);
        if (found == modules_.end()) {
            diagnose("MissingTopModule", "top module '" + top_name + "' was not declared",
                     syntax_.location);
            return std::move(result_);
        }
        std::vector<std::string> hierarchy;
        result_.top = elaborate_instance(top_name, found->second, found->second->location, nullptr,
                                         {}, nullptr, hierarchy, top_name);
        return std::move(result_);
    }

private:
    void diagnose(std::string code, std::string message, const VerilogSyntaxLocation& location,
                  std::optional<VerilogSyntaxLocation> related = std::nullopt) {
        result_.diagnostics.push_back(
            {std::move(code), std::move(message), location, std::move(related)});
    }

    static std::string join_path(const std::vector<std::string>& path, std::string_view tail) {
        std::string result;
        for (const auto& part : path) {
            if (!result.empty())
                result += " -> ";
            result += part;
        }
        if (!tail.empty()) {
            if (!result.empty())
                result += " -> ";
            result += tail;
        }
        return result;
    }

    std::string select_top() {
        if (options_.top_module)
            return *options_.top_module;
        std::unordered_set<std::string> instantiated;
        std::function<void(const VerilogGenerateScope&)> collect_scope;
        std::function<void(const VerilogGenerateConstruct&)> collect_construct;
        collect_scope = [&](const VerilogGenerateScope& scope) {
            for (const auto& instantiation : scope.instantiations)
                instantiated.insert(instantiation.module_type);
            for (const auto& construct : scope.constructs)
                if (construct)
                    collect_construct(*construct);
        };
        collect_construct = [&](const VerilogGenerateConstruct& construct) {
            if (construct.region_body)
                collect_scope(*construct.region_body);
            if (construct.true_body)
                collect_scope(*construct.true_body);
            if (construct.false_body)
                collect_scope(*construct.false_body);
            if (construct.loop_body)
                collect_scope(*construct.loop_body);
            for (const auto& item : construct.case_items)
                if (item.body)
                    collect_scope(*item.body);
        };
        for (const auto& module : syntax_.modules) {
            for (const auto& instantiation : module.instantiations)
                instantiated.insert(instantiation.module_type);
            for (const auto& construct : module.generate_constructs)
                if (construct)
                    collect_construct(*construct);
        }
        for (const auto& [name, module] : modules_) {
            (void)module;
            if (!instantiated.contains(name))
                result_.top_module_candidates.push_back(name);
        }
        std::sort(result_.top_module_candidates.begin(), result_.top_module_candidates.end());
        if (result_.top_module_candidates.empty()) {
            diagnose("NoTopModuleCandidate", "no uninstantiated module can be selected as top",
                     syntax_.location);
            return {};
        }
        if (result_.top_module_candidates.size() != 1) {
            std::string candidates;
            for (const auto& candidate : result_.top_module_candidates) {
                if (!candidates.empty())
                    candidates += ", ";
                candidates += candidate;
            }
            diagnose("AmbiguousTopModule", "multiple top module candidates: " + candidates,
                     syntax_.location);
            return {};
        }
        return result_.top_module_candidates.front();
    }

    void add_constant_diagnostics(const std::vector<VerilogConstantDiagnostic>& diagnostics) {
        for (const auto& diagnostic : diagnostics)
            diagnose(diagnostic.code, diagnostic.message, diagnostic.location);
    }

    static VerilogIntegralType type_from_constant(const VerilogConstantValue& value) {
        VerilogIntegralType type;
        if (value.bit_width == 0)
            return type;
        type.category = VerilogTypeCategory::Integral;
        type.bit_width = value.bit_width;
        type.is_signed = value.is_signed;
        type.unbased_unsized = value.unbased_unsized;
        type.four_state = std::any_of(value.bits.begin(), value.bits.end(), [](auto bit) {
            return bit == VerilogLogicBit::X || bit == VerilogLogicBit::Z;
        });
        return type;
    }

    static const VerilogSemanticScope* find_scope(
        const VerilogSemanticScope& scope, std::string_view name) {
        if (scope.kind != VerilogSemanticScopeKind::CompilationUnit && scope.name == name)
            return &scope;
        for (const auto& child : scope.children)
            if (const auto* found = find_scope(*child, name))
                return found;
        return nullptr;
    }

    void populate_parameter_types(
        std::string_view scope_name,
        const std::vector<VerilogParameterDeclaration>& declarations,
        const VerilogConstantEnvironment& values, VerilogTypeEnvironment& types) {
        types.clear();
        for (const auto& [name, value] : values)
            types.emplace(name, type_from_constant(value));
        const auto* semantic_scope = result_.semantic.model.compilation_unit_scope
                                         ? find_scope(
                                               *result_.semantic.model.compilation_unit_scope,
                                               scope_name)
                                         : nullptr;
        for (const auto& declaration : declarations) {
            if (declaration.named_override)
                continue;
            const auto value = values.find(declaration.name);
            if (value == values.end())
                continue;
            if (semantic_scope) {
                const auto* symbol =
                    lookup_verilog_symbol(*semantic_scope, declaration.name, false);
                if (symbol) {
                    const auto analyzed = type_analysis_.symbol_types.find(symbol);
                    if (analyzed != type_analysis_.symbol_types.end() &&
                        analyzed->second.category != VerilogTypeCategory::Unknown) {
                        types[declaration.name] = analyzed->second;
                        continue;
                    }
                }
            }
            const auto inferred = type_from_constant(value->second);
            types[declaration.name] = infer_verilog_declared_type(
                declaration.type, declaration.packed_dimensions, values, &inferred);
        }
    }

    VerilogConstantResolver package_resolver(const VerilogConstantEnvironment& base) {
        return [this, &base](std::string_view name,
                             const VerilogSyntaxLocation& location)
                   -> std::optional<VerilogConstantValue> {
            const auto found = package_values_.find(std::string(name));
            if (found != package_values_.end())
                return found->second;
            const auto direct = base.find(std::string(name));
            if (direct != base.end())
                return direct->second;
            (void)location;
            return std::nullopt;
        };
    }

    EvaluatedParameterGroup evaluate_parameters(
        std::string_view scope_name,
        const std::vector<VerilogParameterDeclaration>& declarations,
        const std::vector<VerilogParameterDeclaration>* overrides,
        const VerilogConstantEnvironment& caller_environment,
        const VerilogSyntaxLocation& fallback_location) {
        EvaluatedParameterGroup group;
        ParameterGroup parameter_group;
        parameter_group.scope_name = std::string(scope_name);
        for (const auto& declaration : declarations)
            if (!declaration.named_override)
                parameter_group.parameters.push_back(&declaration);

        std::unordered_map<std::string, const VerilogParameterDeclaration*> by_name;
        std::vector<const VerilogParameterDeclaration*> overridable;
        for (auto* parameter : parameter_group.parameters) {
            by_name.try_emplace(parameter->name, parameter);
            if (!parameter->local_parameter)
                overridable.push_back(parameter);
        }

        std::unordered_map<std::string, const VerilogExpression*> override_values;
        if (overrides) {
            size_t positional_index = 0;
            for (const auto& override : *overrides) {
                if (override.named_override) {
                    if (!by_name.contains(override.name) ||
                        by_name.at(override.name)->local_parameter) {
                        diagnose("InvalidParameterOverride",
                                 "scope '" + std::string(scope_name) +
                                     "' has no overridable parameter named '" + override.name +
                                     "'",
                                 override.location);
                        continue;
                    }
                    if (override_values.contains(override.name)) {
                        diagnose("DuplicateParameterOverride",
                                 "parameter '" + override.name +
                                     "' is overridden more than once",
                                 override.location);
                        continue;
                    }
                    override_values.emplace(override.name, override.value.get());
                }
                else {
                    if (positional_index >= overridable.size()) {
                        diagnose("TooManyParameterOverrides",
                                 "too many positional parameter overrides for '" +
                                     std::string(scope_name) + "'",
                                 override.location);
                        continue;
                    }
                    const auto& parameter_name = overridable[positional_index++]->name;
                    if (override_values.contains(parameter_name)) {
                        diagnose("DuplicateParameterOverride",
                                 "parameter '" + parameter_name +
                                     "' is overridden more than once",
                                 override.location);
                        continue;
                    }
                    override_values.emplace(parameter_name, override.value.get());
                }
            }
        }

        VerilogConstantEnvironment evaluated_overrides;
        if (overrides) {
            for (const auto& [name, expression] : override_values) {
                if (!expression)
                    continue;
                auto result = evaluate_verilog_constant(
                    *expression, caller_environment, package_resolver(caller_environment),
                    options_.constant_options);
                add_constant_diagnostics(result.diagnostics);
                if (result.value)
                    evaluated_overrides.emplace(name, std::move(*result.value));
            }
        }
        VerilogConstantEnvironment override_type_environment = caller_environment;
        for (const auto& [name, value] : evaluated_overrides)
            override_type_environment[name] = value;
        for (auto& [name, value] : evaluated_overrides) {
            const auto declaration = by_name.find(name);
            if (declaration != by_name.end())
                value = apply_declared_type(std::move(value), *declaration->second,
                                            override_type_environment, scope_name);
        }

        enum class State : uint8_t { Unvisited, Visiting, Complete, Failed };
        std::unordered_map<std::string, State> states;
        std::vector<std::string> dependency_path;
        std::function<std::optional<VerilogConstantValue>(std::string_view,
                                                          const VerilogSyntaxLocation&)>
            resolve_parameter;
        resolve_parameter = [&](std::string_view name, const VerilogSyntaxLocation& use_location)
            -> std::optional<VerilogConstantValue> {
            const auto found = by_name.find(std::string(name));
            if (found == by_name.end()) {
                const auto caller_value = caller_environment.find(std::string(name));
                if (caller_value != caller_environment.end())
                    return caller_value->second;
                const auto package = package_values_.find(std::string(name));
                if (package != package_values_.end())
                    return package->second;
                return std::nullopt;
            }
            const auto& declaration = *found->second;
            auto& state = states[declaration.name];
            if (state == State::Complete)
                return group.values.at(declaration.name);
            if (state == State::Failed)
                return std::nullopt;
            if (state == State::Visiting) {
                diagnose("CircularParameterDependency",
                         "circular parameter dependency: " +
                             join_path(dependency_path, declaration.name),
                         use_location, declaration.location);
                state = State::Failed;
                return std::nullopt;
            }
            state = State::Visiting;
            dependency_path.push_back(declaration.name);
            const auto override_value = evaluated_overrides.find(declaration.name);
            const VerilogExpression* expression =
                override_value == evaluated_overrides.end()
                    ? declaration.value.get()
                    : nullptr;
            std::optional<VerilogConstantValue> value;
            if (override_value != evaluated_overrides.end()) {
                value = override_value->second;
            }
            else if (!expression) {
                diagnose("MissingParameterValue",
                         "parameter '" + declaration.name + "' has no constant initializer",
                         declaration.location);
            }
            else {
                auto evaluation = evaluate_verilog_constant(
                    *expression, group.values,
                    [&](std::string_view referenced,
                        const VerilogSyntaxLocation& referenced_location)
                        -> std::optional<VerilogConstantValue> {
                        if (auto resolved = resolve_parameter(referenced, referenced_location))
                            return resolved;
                        if (referenced.find('.') != std::string_view::npos) {
                            const auto package = package_values_.find(std::string(referenced));
                            if (package != package_values_.end())
                                return package->second;
                        }
                        return std::nullopt;
                    },
                    options_.constant_options);
                add_constant_diagnostics(evaluation.diagnostics);
                value = std::move(evaluation.value);
            }
            dependency_path.pop_back();
            if (!value) {
                state = State::Failed;
                return std::nullopt;
            }
            VerilogConstantEnvironment type_environment = caller_environment;
            for (const auto& [parameter_name, parameter_value] : group.values)
                type_environment[parameter_name] = parameter_value;
            for (const auto& [parameter_name, parameter_value] : evaluated_overrides)
                type_environment[parameter_name] = parameter_value;
            *value = apply_declared_type(*value, declaration, type_environment, scope_name);
            group.values[declaration.name] = *value;
            state = State::Complete;
            return value;
        };

        for (auto* declaration : parameter_group.parameters) {
            if (states[declaration->name] == State::Unvisited)
                (void)resolve_parameter(declaration->name, declaration->location);
        }
        (void)fallback_location;
        return group;
    }

    VerilogConstantValue apply_declared_type(
        VerilogConstantValue value, const VerilogParameterDeclaration& declaration,
        const VerilogConstantEnvironment& environment, std::string_view scope_name) {
        const auto inferred = type_from_constant(value);
        auto type = infer_verilog_declared_type(
            declaration.type, declaration.packed_dimensions, environment, &inferred);
        const auto* semantic_scope = result_.semantic.model.compilation_unit_scope
                                         ? find_scope(
                                               *result_.semantic.model.compilation_unit_scope,
                                               scope_name)
                                         : nullptr;
        if (semantic_scope) {
            const auto* symbol =
                lookup_verilog_symbol(*semantic_scope, declaration.name, false);
            if (symbol) {
                const auto analyzed = type_analysis_.symbol_types.find(symbol);
                if (analyzed != type_analysis_.symbol_types.end() &&
                    analyzed->second.category != VerilogTypeCategory::Unknown)
                    type = analyzed->second;
            }
        }
        if (!type.is_known_integral())
            return value;
        if (!type.bit_width || *type.bit_width == 0 ||
            *type.bit_width > options_.constant_options.maximum_width) {
            diagnose("InvalidParameterWidth", "declared parameter width is invalid",
                     declaration.location);
            return value;
        }
        const auto extension = type.is_signed && !value.bits.empty()
                                   ? value.bits.back()
                                   : VerilogLogicBit::Zero;
        value.bits.resize(*type.bit_width, extension);
        value.bit_width = *type.bit_width;
        value.is_signed = type.is_signed;
        value.unbased_unsized = false;
        return value;
    }

    void evaluate_package_parameters() {
        for (const auto& package : syntax_.packages) {
            const auto group = evaluate_parameters(package.name, package.parameters, nullptr, {},
                                                   package.location);
            for (const auto& [name, value] : group.values)
                package_values_[package.name + "." + name] = value;
            const auto* package_symbol =
                lookup_verilog_symbol(*result_.semantic.model.compilation_unit_scope,
                                      package.name, false);
            if (!package_symbol || !package_symbol->member_scope)
                continue;
            for (const auto& definition : package.type_definitions) {
                if (definition.kind != VerilogTypeDefinitionKind::Enumeration)
                    continue;
                for (const auto& enumerator : definition.enumerators) {
                    const auto* symbol = lookup_verilog_symbol(
                        *package_symbol->member_scope, enumerator.name, false);
                    if (!symbol)
                        continue;
                    const auto value = type_analysis_.symbol_constants.find(symbol);
                    if (value != type_analysis_.symbol_constants.end())
                        package_values_[package.name + "." + enumerator.name] = value->second;
                }
            }
        }
    }

    void evaluate_inline_generate_parameters(
        const std::vector<std::unique_ptr<VerilogGenerateConstruct>>& constructs,
        VerilogConstantEnvironment& environment,
        const VerilogSyntaxLocation& fallback_location) {
        for (const auto& construct : constructs) {
            if (!construct || construct->kind != VerilogGenerateKind::Region ||
                !construct->region_body)
                continue;
            const auto& region = *construct->region_body;
            const auto group = evaluate_parameters(region.name, region.parameters, nullptr,
                                                   environment, fallback_location);
            for (const auto& [name, value] : group.values)
                environment[name] = value;
            evaluate_inline_generate_parameters(region.constructs, environment,
                                                fallback_location);
        }
    }

    std::optional<VerilogConstantValue> evaluate_expression(
        const VerilogExpression* expression, const VerilogConstantEnvironment& environment,
        std::string_view failure_code, std::string_view failure_message) {
        if (!expression) {
            diagnose(std::string(failure_code), std::string(failure_message), syntax_.location);
            return std::nullopt;
        }
        auto evaluated = evaluate_verilog_constant(
            *expression, environment, package_resolver(environment), options_.constant_options);
        add_constant_diagnostics(evaluated.diagnostics);
        if (!evaluated.value)
            diagnose(std::string(failure_code), std::string(failure_message),
                     expression->location);
        return std::move(evaluated.value);
    }

    static VerilogConstantValue make_genvar_value(int64_t value,
                                                  const VerilogSyntaxLocation& location) {
        VerilogConstantValue result;
        result.bit_width = 32;
        result.is_signed = true;
        result.location = location;
        result.bits.resize(32, VerilogLogicBit::Zero);
        const auto raw = std::bit_cast<uint32_t>(static_cast<int32_t>(value));
        for (size_t bit = 0; bit < result.bits.size(); ++bit)
            if ((raw >> bit) & 1U)
                result.bits[bit] = VerilogLogicBit::One;
        return result;
    }

    static std::string append_path(std::string_view base, std::string_view component) {
        if (base.empty())
            return std::string(component);
        if (component.empty())
            return std::string(base);
        return std::string(base) + "." + std::string(component);
    }

    bool generate_condition(const VerilogExpression* expression,
                            const VerilogConstantEnvironment& environment,
                            bool& selected) {
        auto value = evaluate_expression(expression, environment, "UnresolvedGenerateCondition",
                                         "generate condition is not a known constant");
        if (!value || !value->is_fully_known()) {
            if (value)
                diagnose("UnresolvedGenerateCondition",
                         "generate condition contains unknown bits", expression->location);
            return false;
        }
        const auto integer = value->to_uint64();
        if (!integer) {
            diagnose("InvalidGenerateCondition", "generate condition is too wide",
                     expression->location);
            return false;
        }
        selected = *integer != 0;
        return true;
    }

    bool case_item_matches(const VerilogConstantValue& selector,
                           const VerilogConstantValue& pattern, VerilogCaseKind case_kind) const {
        const size_t width = std::max(selector.bit_width, pattern.bit_width);
        const bool signed_values = selector.is_signed && pattern.is_signed;
        const auto extend = [width, signed_values](const VerilogConstantValue& value) {
            std::vector<VerilogLogicBit> bits(width, VerilogLogicBit::Zero);
            if (value.unbased_unsized && !value.bits.empty()) {
                std::fill(bits.begin(), bits.end(), value.bits.front());
                return bits;
            }
            const size_t copied = std::min(width, value.bits.size());
            std::copy_n(value.bits.begin(), copied, bits.begin());
            if (copied < width && signed_values && value.is_signed && !value.bits.empty())
                std::fill(bits.begin() + static_cast<std::ptrdiff_t>(copied), bits.end(),
                          value.bits.back());
            return bits;
        };
        const auto selector_bits = extend(selector);
        const auto pattern_bits = extend(pattern);
        for (size_t bit = 0; bit < width; ++bit) {
            const auto selector_bit = selector_bits[bit];
            const auto pattern_bit = pattern_bits[bit];
            const bool wildcard =
                case_kind == VerilogCaseKind::CaseX
                    ? pattern_bit == VerilogLogicBit::X || pattern_bit == VerilogLogicBit::Z ||
                          selector_bit == VerilogLogicBit::X ||
                          selector_bit == VerilogLogicBit::Z
                    : case_kind == VerilogCaseKind::CaseZ
                          ? pattern_bit == VerilogLogicBit::Z ||
                                selector_bit == VerilogLogicBit::Z
                          : false;
            if (wildcard)
                continue;
            if (selector_bit != pattern_bit)
                return false;
        }
        return true;
    }

    void elaborate_generate_scope_contents(
        const VerilogGenerateScope& syntax_scope, VerilogElaboratedGenerateScope& scope,
        const VerilogConstantEnvironment& inherited_environment,
        std::vector<std::string>& hierarchy) {
        scope.parameters = inherited_environment;
        auto parameters = evaluate_parameters(syntax_scope.name, syntax_scope.parameters,
                                              nullptr, inherited_environment,
                                              syntax_scope.location);
        for (auto& [name, value] : parameters.values)
            scope.parameters[name] = std::move(value);
        populate_parameter_types(syntax_scope.name, syntax_scope.parameters, scope.parameters,
                                 scope.parameter_types);
        for (const auto& declaration : syntax_scope.declarations)
            scope.declarations.push_back(&declaration);
        for (const auto& assignment : syntax_scope.assignments)
            scope.assignments.push_back(&assignment);
        std::unordered_set<std::string> sibling_names;
        for (const auto& instantiation : syntax_scope.instantiations)
            elaborate_instantiations(instantiation, *scope.parent_instance,
                                     &scope, scope.parameters, hierarchy);
        elaborate_generate_constructs(syntax_scope.constructs, *scope.parent_instance,
                                      &scope, scope.parameters, hierarchy, sibling_names);
    }

    VerilogElaboratedGenerateScope* create_generate_scope(
        const VerilogGenerateScope& syntax_scope, std::string generated_name,
        VerilogElaboratedInstance& parent_instance,
        VerilogElaboratedGenerateScope* parent_scope,
        const VerilogConstantEnvironment& environment,
        std::vector<std::string>& hierarchy) {
        auto scope = std::make_unique<VerilogElaboratedGenerateScope>();
        scope->name = std::move(generated_name);
        scope->location = syntax_scope.location;
        scope->parent_instance = &parent_instance;
        scope->parent_scope = parent_scope;
        const std::string_view base =
            parent_scope ? std::string_view(parent_scope->hierarchy_path)
                         : std::string_view(parent_instance.hierarchy_path);
        scope->hierarchy_path = append_path(base, scope->name);
        auto* raw = scope.get();
        if (parent_scope)
            parent_scope->generate_scopes.push_back(std::move(scope));
        else
            parent_instance.generate_scopes.push_back(std::move(scope));
        elaborate_generate_scope_contents(syntax_scope, *raw, environment, hierarchy);
        return raw;
    }

    void elaborate_inline_generate_scope(
        const VerilogGenerateScope& syntax_scope, VerilogElaboratedInstance& parent_instance,
        VerilogElaboratedGenerateScope* parent_scope,
        const VerilogConstantEnvironment& inherited_environment,
        std::vector<std::string>& hierarchy, std::unordered_set<std::string>& sibling_names) {
        VerilogConstantEnvironment environment = inherited_environment;
        auto parameters = evaluate_parameters(syntax_scope.name, syntax_scope.parameters, nullptr,
                                              inherited_environment, syntax_scope.location);
        for (auto& [name, value] : parameters.values)
            environment[name] = std::move(value);
        if (parent_scope) {
            for (const auto& declaration : syntax_scope.declarations)
                parent_scope->declarations.push_back(&declaration);
            for (const auto& assignment : syntax_scope.assignments)
                parent_scope->assignments.push_back(&assignment);
        }
        else {
            for (const auto& declaration : syntax_scope.declarations)
                parent_instance.generated_declarations.push_back(&declaration);
            for (const auto& assignment : syntax_scope.assignments)
                parent_instance.generated_assignments.push_back(&assignment);
        }
        for (const auto& instantiation : syntax_scope.instantiations)
            elaborate_instantiations(instantiation, parent_instance, parent_scope,
                                     environment, hierarchy);
        elaborate_generate_constructs(syntax_scope.constructs, parent_instance, parent_scope,
                                      environment, hierarchy, sibling_names);
    }

    bool reserve_generated_name(std::string_view name,
                                std::unordered_set<std::string>& sibling_names,
                                const VerilogSyntaxLocation& location) {
        if (sibling_names.insert(std::string(name)).second)
            return true;
        diagnose("DuplicateGeneratedName",
                 "duplicate generated block name '" + std::string(name) + "'", location);
        return false;
    }

    void elaborate_generate_constructs(
        const std::vector<std::unique_ptr<VerilogGenerateConstruct>>& constructs,
        VerilogElaboratedInstance& parent_instance,
        VerilogElaboratedGenerateScope* parent_scope,
        const VerilogConstantEnvironment& environment, std::vector<std::string>& hierarchy,
        std::unordered_set<std::string>& sibling_names) {
        for (const auto& construct : constructs)
            if (construct)
                elaborate_generate_construct(*construct, parent_instance, parent_scope,
                                             environment, hierarchy, sibling_names);
    }

    void elaborate_generate_construct(
        const VerilogGenerateConstruct& construct, VerilogElaboratedInstance& parent_instance,
        VerilogElaboratedGenerateScope* parent_scope,
        const VerilogConstantEnvironment& environment, std::vector<std::string>& hierarchy,
        std::unordered_set<std::string>& sibling_names) {
        if (construct.kind == VerilogGenerateKind::Region) {
            if (construct.region_body)
                elaborate_inline_generate_scope(*construct.region_body, parent_instance,
                                                parent_scope, environment, hierarchy,
                                                sibling_names);
            return;
        }
        if (construct.kind == VerilogGenerateKind::If) {
            bool selected = false;
            if (!generate_condition(construct.condition.get(), environment, selected))
                return;
            const auto* body = selected ? construct.true_body.get() : construct.false_body.get();
            if (!body)
                return;
            if (!reserve_generated_name(body->name, sibling_names, body->location))
                return;
            create_generate_scope(*body, body->name, parent_instance, parent_scope,
                                  environment, hierarchy);
            return;
        }
        if (construct.kind == VerilogGenerateKind::Case) {
            auto selector = evaluate_expression(construct.condition.get(), environment,
                                               "UnresolvedGenerateCondition",
                                               "case-generate selector is not constant");
            if (!selector)
                return;
            const VerilogGenerateCaseItem* selected_item = nullptr;
            const VerilogGenerateCaseItem* default_item = nullptr;
            for (const auto& item : construct.case_items) {
                if (item.is_default) {
                    if (!default_item)
                        default_item = &item;
                    continue;
                }
                for (const auto& expression : item.expressions) {
                    auto pattern = evaluate_expression(
                        expression.get(), environment, "UnresolvedGenerateCaseItem",
                        "case-generate item is not constant");
                    if (pattern && case_item_matches(*selector, *pattern, construct.case_kind)) {
                        selected_item = &item;
                        break;
                    }
                }
                if (selected_item)
                    break;
            }
            if (!selected_item)
                selected_item = default_item;
            if (!selected_item || !selected_item->body)
                return;
            const auto& body = *selected_item->body;
            if (!reserve_generated_name(body.name, sibling_names, body.location))
                return;
            create_generate_scope(body, body.name, parent_instance, parent_scope, environment,
                                  hierarchy);
            return;
        }

        if (!construct.initialization || !construct.initialization->second ||
            construct.initialization->first == nullptr ||
            construct.initialization->first->kind != VerilogExpressionKind::Identifier ||
            construct.initialization->first->text != construct.loop_variable ||
            !construct.step || !construct.step->second || !construct.step->first ||
            construct.step->first->kind != VerilogExpressionKind::Identifier ||
            construct.step->first->text != construct.loop_variable) {
            diagnose("InvalidGenerateStep",
                     "for-generate initialization and step must update the declared genvar",
                     construct.location);
            return;
        }
        auto initial_value = evaluate_expression(
            construct.initialization->second.get(), environment,
            "InvalidGenerateInitialization", "for-generate initialization is not constant");
        if (!initial_value || !initial_value->to_int64()) {
            diagnose("InvalidGenerateInitialization",
                     "for-generate initialization must fit a signed 64-bit value",
                     construct.initialization->location);
            return;
        }
        int64_t current = *initial_value->to_int64();
        if (current < std::numeric_limits<int32_t>::min() ||
            current > std::numeric_limits<int32_t>::max()) {
            diagnose("InvalidGenerateInitialization", "genvar value is outside signed 32-bit range",
                     construct.initialization->location);
            return;
        }
        if (!construct.loop_body) {
            diagnose("InvalidGenerateStep", "for-generate loop has no named block body",
                     construct.location);
            return;
        }

        std::unordered_set<int64_t> seen_values;
        size_t iteration = 0;
        for (;;) {
            auto iteration_environment = environment;
            iteration_environment[construct.loop_variable] =
                make_genvar_value(current, construct.location);
            bool condition = false;
            if (!generate_condition(construct.loop_condition.get(), iteration_environment,
                                    condition))
                return;
            if (!condition)
                return;
            if (iteration >= options_.maximum_generate_iterations) {
                diagnose("GenerateIterationLimitExceeded",
                         "for-generate exceeded the configured iteration limit of " +
                             std::to_string(options_.maximum_generate_iterations),
                         construct.location);
                return;
            }
            if (!seen_values.insert(current).second) {
                diagnose("InvalidGenerateStep",
                         "for-generate loop revisited a genvar value and cannot make progress",
                         construct.step->location);
                return;
            }
            const std::string generated_name =
                construct.loop_body->name + "[" + std::to_string(current) + "]";
            if (!reserve_generated_name(generated_name, sibling_names,
                                        construct.loop_body->location))
                return;
            create_generate_scope(*construct.loop_body, generated_name, parent_instance,
                                  parent_scope, iteration_environment, hierarchy);

            auto next_value = evaluate_expression(construct.step->second.get(),
                                                 iteration_environment, "InvalidGenerateStep",
                                                 "for-generate step is not constant");
            if (!next_value || !next_value->to_int64()) {
                diagnose("InvalidGenerateStep",
                         "for-generate step must produce a signed 64-bit value",
                         construct.step->location);
                return;
            }
            const int64_t next = *next_value->to_int64();
            if (next == current) {
                diagnose("InvalidGenerateStep", "for-generate step does not change the genvar",
                         construct.step->location);
                return;
            }
            if (next < std::numeric_limits<int32_t>::min() ||
                next > std::numeric_limits<int32_t>::max()) {
                diagnose("InvalidGenerateStep", "genvar step overflows signed 32-bit range",
                         construct.step->location);
                return;
            }
            current = next;
            ++iteration;
        }
    }

    void elaborate_instantiations(
        const VerilogModuleInstantiation& instantiation,
        VerilogElaboratedInstance& parent_instance,
        VerilogElaboratedGenerateScope* generated_scope,
        const VerilogConstantEnvironment& environment, std::vector<std::string>& hierarchy) {
        const auto target = modules_.find(instantiation.module_type);
        if (target == modules_.end()) {
            diagnose("MissingModuleDefinition",
                     "module definition for '" + instantiation.module_type +
                         "' is not available",
                     instantiation.location);
            return;
        }
        const std::string_view base =
            generated_scope ? std::string_view(generated_scope->hierarchy_path)
                            : std::string_view(parent_instance.hierarchy_path);
        for (const auto& child_ast : instantiation.instances) {
            auto child = elaborate_instance(
                child_ast.name, target->second, child_ast.location, &parent_instance,
                environment, &instantiation.parameter_overrides, hierarchy,
                append_path(base, child_ast.name), generated_scope);
            if (generated_scope)
                generated_scope->children.push_back(std::move(child));
            else
                parent_instance.children.push_back(std::move(child));
        }
    }

    std::unique_ptr<VerilogElaboratedInstance> elaborate_instance(
        std::string instance_name, const VerilogModuleDeclaration* module,
        const VerilogSyntaxLocation& location, VerilogElaboratedInstance* parent,
        const VerilogConstantEnvironment& caller_environment,
        const std::vector<VerilogParameterDeclaration>* overrides,
        std::vector<std::string>& hierarchy, std::string hierarchy_path,
        VerilogElaboratedGenerateScope* generated_scope_parent = nullptr) {
        auto instance = std::make_unique<VerilogElaboratedInstance>();
        instance->instance_name = std::move(instance_name);
        instance->hierarchy_path = std::move(hierarchy_path);
        instance->location = location;
        instance->module_definition = module;
        instance->parent = parent;
        instance->generated_scope_parent = generated_scope_parent;
        if (!module)
            return instance;
        const auto* symbol = lookup_verilog_symbol(
            *result_.semantic.model.compilation_unit_scope, module->name, false);
        instance->module_symbol = symbol;
        const std::string frame = parent ? instance->instance_name + ":" + module->name
                                         : module->name;
        const auto cycle = std::find(hierarchy.begin(), hierarchy.end(), module->name);
        if (cycle != hierarchy.end()) {
            std::vector<std::string> cycle_path(cycle, hierarchy.end());
            diagnose("RecursiveModuleInstantiation",
                     "recursive module instantiation: " +
                         join_path(cycle_path, frame),
                     location);
            return instance;
        }
        hierarchy.push_back(module->name);
        instance->parameters =
            evaluate_parameters(module->name, module->parameters, overrides, caller_environment,
                                module->location)
                .values;
        evaluate_inline_generate_parameters(module->generate_constructs, instance->parameters,
                                             module->location);
        populate_parameter_types(module->name, module->parameters, instance->parameters,
                                 instance->parameter_types);

        for (const auto& instantiation : module->instantiations)
            elaborate_instantiations(instantiation, *instance, nullptr, instance->parameters,
                                     hierarchy);
        std::unordered_set<std::string> sibling_names;
        elaborate_generate_constructs(module->generate_constructs, *instance, nullptr,
                                      instance->parameters, hierarchy, sibling_names);
        hierarchy.pop_back();
        return instance;
    }

    const VerilogCompilationUnit& syntax_;
    VerilogElaborationOptions options_;
    VerilogElaborationResult result_;
    VerilogTypeAnalysisResult type_analysis_;
    std::unordered_map<std::string, const VerilogModuleDeclaration*> modules_;
    std::unordered_map<std::string, const VerilogPackageDeclaration*> packages_;
    VerilogConstantEnvironment package_values_;
};

} // namespace

VerilogElaborationResult elaborate_verilog_design(const VerilogCompilationUnit& syntax,
                                                   VerilogElaborationOptions options) {
    return Elaborator(syntax, std::move(options)).run();
}

} // namespace labneura::eda
