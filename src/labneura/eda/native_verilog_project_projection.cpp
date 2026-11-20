#include "native_verilog_project_projection.h"

#include "labneura/eda/verilog_constant.h"
#include "labneura/eda/verilog_semantic.h"
#include "labneura/eda/verilog_type.h"

#include <algorithm>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace labneura::eda::detail {
namespace {

struct DeclaredShape {
    std::size_t width = 0;
    std::size_t packed_width = 0;
    std::optional<int64_t> packed_left;
    std::optional<int64_t> packed_right;
    std::optional<int64_t> unpacked_left;
    std::optional<int64_t> unpacked_right;
    VerilogIntegralType type;
};

struct SourceLocationKey {
    const VerilogSourceBuffer* source = nullptr;
    size_t byte_offset = 0;

    bool operator==(const SourceLocationKey&) const = default;
};

struct SourceLocationKeyHash {
    size_t operator()(const SourceLocationKey& key) const noexcept {
        return std::hash<const VerilogSourceBuffer*>{}(key.source) ^
               (std::hash<size_t>{}(key.byte_offset) << 1);
    }
};

using InstanceSyntaxMap = std::unordered_map<
    SourceLocationKey,
    std::pair<const VerilogModuleInstantiation*, const VerilogInstance*>,
    SourceLocationKeyHash>;

std::string relative_hierarchy_path(std::string_view base,
                                    std::string_view full_path) {
    if (full_path == base)
        return {};
    if (full_path.size() > base.size() &&
        full_path.substr(0, base.size()) == base &&
        full_path[base.size()] == '.') {
        return std::string(full_path.substr(base.size() + 1));
    }
    return std::string(full_path);
}

void collect_generate_syntax(const VerilogGenerateScope& scope,
                             InstanceSyntaxMap& output);

void collect_generate_construct_syntax(const VerilogGenerateConstruct& construct,
                                       InstanceSyntaxMap& output) {
    const auto collect = [&](const VerilogGenerateScope* scope) {
        if (scope)
            collect_generate_syntax(*scope, output);
    };
    collect(construct.region_body.get());
    collect(construct.true_body.get());
    collect(construct.false_body.get());
    collect(construct.loop_body.get());
    for (const auto& item : construct.case_items)
        collect(item.body.get());
}

void collect_generate_syntax(const VerilogGenerateScope& scope,
                             InstanceSyntaxMap& output) {
    for (const auto& instantiation : scope.instantiations) {
        for (const auto& instance : instantiation.instances) {
            output.insert_or_assign(
                SourceLocationKey{instance.location.source.get(),
                                  instance.location.byte_offset},
                std::pair{&instantiation, &instance});
        }
    }
    for (const auto& construct : scope.constructs)
        if (construct)
            collect_generate_construct_syntax(*construct, output);
}

void collect_instantiated_modules(const VerilogGenerateScope& scope,
                                 std::unordered_set<std::string>& output);

void collect_instantiated_modules(const VerilogGenerateConstruct& construct,
                                  std::unordered_set<std::string>& output) {
    const auto collect = [&](const VerilogGenerateScope* scope) {
        if (scope)
            collect_instantiated_modules(*scope, output);
    };
    collect(construct.region_body.get());
    collect(construct.true_body.get());
    collect(construct.false_body.get());
    collect(construct.loop_body.get());
    for (const auto& item : construct.case_items)
        collect(item.body.get());
}

void collect_instantiated_modules(const VerilogGenerateScope& scope,
                                  std::unordered_set<std::string>& output) {
    for (const auto& instantiation : scope.instantiations)
        output.insert(instantiation.module_type);
    for (const auto& construct : scope.constructs)
        if (construct)
            collect_instantiated_modules(*construct, output);
}

InstanceSyntaxMap index_instantiations(const VerilogModuleDeclaration& module) {
    InstanceSyntaxMap result;
    for (const auto& instantiation : module.instantiations) {
        for (const auto& instance : instantiation.instances) {
            result.insert_or_assign(
                SourceLocationKey{instance.location.source.get(),
                                  instance.location.byte_offset},
                std::pair{&instantiation, &instance});
        }
    }
    for (const auto& construct : module.generate_constructs) {
        if (construct)
            collect_generate_construct_syntax(*construct, result);
    }
    return result;
}

std::optional<int64_t> constant_integer(
    const VerilogExpression* expression,
    const VerilogConstantEnvironment& environment) {
    if (!expression)
        return std::nullopt;
    const auto value = evaluate_verilog_constant(*expression, environment);
    return value.value ? value.value->to_int64() : std::nullopt;
}

std::optional<size_t> range_width(
    const VerilogExpressionRange& range,
    const VerilogConstantEnvironment& environment) {
    const auto left = constant_integer(range.left.get(), environment);
    const auto right = constant_integer(range.right.get(), environment);
    if (!left || !right)
        return std::nullopt;
    const auto difference = *left >= *right
                                ? static_cast<uint64_t>(*left) -
                                      static_cast<uint64_t>(*right)
                                : static_cast<uint64_t>(*right) -
                                      static_cast<uint64_t>(*left);
    if (difference >= std::numeric_limits<size_t>::max())
        return std::nullopt;
    return static_cast<size_t>(difference + 1);
}

VerilogIntegralType type_from_constant(const VerilogConstantValue& value) {
    VerilogIntegralType type;
    type.category = VerilogTypeCategory::Integral;
    type.bit_width = value.bit_width;
    type.is_signed = value.is_signed;
    type.four_state = !value.is_fully_known();
    return type;
}

} // namespace

class NativeVerilogProjectProjector {
public:
    explicit NativeVerilogProjectProjector(
        std::shared_ptr<NativeVerilogProjectContext> context)
        : context_(std::move(context)) {}

    NativeVerilogProjectProjectionResult run() {
        if (!context_) {
            diagnose("MissingNativeContext",
                     "native project projection requires a retained context", {});
            return std::move(result_);
        }
        if (!context_->ready()) {
            add_context_diagnostics();
            return std::move(result_);
        }
        if (!context_->elaboration->top ||
            !context_->elaboration->top->module_definition) {
            diagnose("MissingElaboratedTop",
                     "native context has no elaborated top module", {});
            return std::move(result_);
        }

        result_.project.native_context_ = context_;
        result_.project.source_files_ = context_->source_files;
        result_.compatibility_gaps.push_back(
            "native projection does not yet serialize the compatible AST JSON schema");
        result_.compatibility_gaps.push_back(
            "native projection does not expose a module parameter table; evaluated "
            "defaults and overrides are reflected in specialized widths and hierarchy");
        result_.compatibility_gaps.push_back(
            "parameter-dependent typedefs and multi-dimensional unpacked arrays are "
            "not fully specialized by the public netlist model");
        const auto& syntax = context_->parsed.syntax;
        for (const auto& module : syntax.modules)
            modules_.try_emplace(module.name, &module);

        std::unordered_set<std::string> instantiated;
        for (const auto& module : syntax.modules) {
            for (const auto& instantiation : module.instantiations)
                instantiated.insert(instantiation.module_type);
            for (const auto& construct : module.generate_constructs)
                if (construct)
                    collect_instantiated_modules(*construct, instantiated);
        }
        std::vector<std::string> roots;
        for (const auto& module : syntax.modules)
            if (!instantiated.contains(module.name))
                roots.push_back(module.name);
        std::sort(roots.begin(), roots.end());
        std::unordered_set<std::string> emitted;
        const auto selected_top =
            context_->elaboration->top->module_definition->name;
        for (const auto& root : roots) {
            if (root == selected_top) {
                append_hierarchy_modules(*context_->elaboration->top, true,
                                         emitted);
            } else {
                const auto found = modules_.find(root);
                if (found != modules_.end())
                    append_definition_hierarchy(*found->second, emitted);
            }
        }
        for (const auto& module : syntax.modules) {
            if (emitted.insert(module.name).second)
                result_.project.modules_.push_back(project_definition(module));
            if (module.name !=
                    context_->elaboration->top->module_definition->name &&
                (!module.generate_constructs.empty() ||
                 !module.instantiations.empty()))
                result_.compatibility_gaps.push_back(
                    "unselected module '" + module.name +
                    "' is projected from declarations; instance-specific "
                    "hierarchy is materialized only for the selected top");
        }
        return std::move(result_);
    }

private:
    void diagnose(std::string code, std::string message,
                  const VerilogSyntaxLocation& location) {
        result_.diagnostics.push_back(
            {std::move(code), std::move(message), location});
    }

    void add_context_diagnostics() {
        for (const auto& diagnostic : context_->parsed.diagnostics) {
            if (diagnostic.severity == VerilogSyntaxDiagnosticSeverity::Error)
                diagnose(diagnostic.code, diagnostic.message, diagnostic.location);
        }
        if (context_->elaboration) {
            for (const auto& diagnostic : context_->elaboration->diagnostics)
                diagnose(diagnostic.code, diagnostic.message, diagnostic.location);
        }
        if (context_->types) {
            for (const auto& diagnostic : context_->types->diagnostics)
                diagnose(diagnostic.code, diagnostic.message, diagnostic.location);
        }
        if (result_.diagnostics.empty())
            diagnose("NativeContextNotReady",
                     "native context did not complete all required analyses", {});
    }

    void append_hierarchy_modules(
        const VerilogElaboratedInstance& instance, bool is_top,
        std::unordered_set<std::string>& emitted) {
        if (instance.module_definition &&
            emitted.insert(instance.module_definition->name).second) {
            result_.project.modules_.push_back(
                project_elaborated_module(instance, {}, is_top));
        }
        for (const auto& child : instance.children)
            append_hierarchy_modules(*child, false, emitted);
        for (const auto& scope : instance.generate_scopes)
            append_hierarchy_scope_modules(*scope, emitted);
    }

    void append_definition_hierarchy(
        const VerilogModuleDeclaration& module,
        std::unordered_set<std::string>& emitted) {
        if (!emitted.insert(module.name).second)
            return;
        result_.project.modules_.push_back(project_definition(module));
        for (const auto& instantiation : module.instantiations) {
            const auto child = modules_.find(instantiation.module_type);
            if (child != modules_.end())
                append_definition_hierarchy(*child->second, emitted);
        }
    }

    void append_hierarchy_scope_modules(
        const VerilogElaboratedGenerateScope& scope,
        std::unordered_set<std::string>& emitted) {
        for (const auto& child : scope.children)
            append_hierarchy_modules(*child, false, emitted);
        for (const auto& nested : scope.generate_scopes)
            append_hierarchy_scope_modules(*nested, emitted);
    }

    const VerilogSemanticScope* semantic_scope(std::string_view name) const {
        const auto* root =
            context_->elaboration->semantic.model.compilation_unit_scope.get();
        if (!root)
            return nullptr;
        const auto find = [&](const auto& self,
                              const VerilogSemanticScope& current)
            -> const VerilogSemanticScope* {
            if (current.kind == VerilogSemanticScopeKind::Module &&
                current.name == name)
                return &current;
            for (const auto& child : current.children) {
                if (const auto* found = self(self, *child))
                    return found;
            }
            return nullptr;
        };
        return find(find, *root);
    }

    VerilogConstantEnvironment module_parameters(
        const VerilogModuleDeclaration& module) const {
        VerilogConstantEnvironment result;
        if (const auto* scope = semantic_scope(module.name)) {
            for (const auto& parameter : module.parameters) {
                const auto* symbol =
                    lookup_verilog_symbol(*scope, parameter.name, false);
                if (!symbol)
                    continue;
                const auto value =
                    context_->types->symbol_constants.find(symbol);
                if (value != context_->types->symbol_constants.end())
                    result.emplace(parameter.name, value->second);
            }
        }
        return result;
    }

    DeclaredShape declared_shape(
        std::string_view name, std::string_view declared_type,
        const std::vector<VerilogExpressionRange>& packed_dimensions,
        const std::vector<VerilogExpressionRange>& unpacked_dimensions,
        const VerilogConstantEnvironment& parameters,
        const VerilogModuleDeclaration& module) {
        DeclaredShape shape;
        const VerilogSemanticSymbol* symbol = nullptr;
        if (const auto* scope = semantic_scope(module.name))
            symbol = lookup_verilog_symbol(*scope, name, false);
        const VerilogIntegralType* inferred = nullptr;
        if (symbol) {
            const auto type = context_->types->symbol_types.find(symbol);
            if (type != context_->types->symbol_types.end())
                inferred = &type->second;
        }
        shape.type = infer_verilog_declared_type(
            declared_type, packed_dimensions, parameters, inferred);
        if (shape.type.category == VerilogTypeCategory::Unknown && inferred)
            shape.type = *inferred;
        if (!shape.type.bit_width || *shape.type.bit_width == 0) {
            diagnose("UnsupportedNativeDeclaredType",
                     "native projection cannot determine the width of '" +
                         std::string(name) + "'",
                     symbol ? symbol->declaration_location : module.location);
            return shape;
        }

        shape.packed_width = *shape.type.bit_width;
        if (!shape.type.packed_dimensions.empty()) {
            shape.packed_left = shape.type.packed_dimensions.front().left;
            shape.packed_right = shape.type.packed_dimensions.front().right;
        }
        shape.width = shape.packed_width;
        if (unpacked_dimensions.size() > 1) {
            diagnose("UnsupportedNativeUnpackedDimensions",
                     "native project projection supports at most one unpacked "
                     "dimension on '" + std::string(name) + "'",
                     symbol ? symbol->declaration_location : module.location);
            return shape;
        }
        if (!unpacked_dimensions.empty()) {
            const auto& dimension = unpacked_dimensions.front();
            const auto count = range_width(dimension, parameters);
            if (!count || shape.width >
                              std::numeric_limits<size_t>::max() / *count) {
                diagnose("UnresolvedNativeUnpackedDimension",
                         "native projection cannot resolve the unpacked range of '" +
                             std::string(name) + "'",
                         dimension.location);
                return shape;
            }
            shape.width *= *count;
            shape.unpacked_left =
                constant_integer(dimension.left.get(), parameters);
            shape.unpacked_right =
                constant_integer(dimension.right.get(), parameters);
        }
        return shape;
    }

    VerilogTypeEnvironment type_environment(
        const VerilogModuleDeclaration& module,
        const VerilogConstantEnvironment& parameters) {
        VerilogTypeEnvironment environment;
        for (const auto& [name, value] : parameters)
            environment.insert_or_assign(name, type_from_constant(value));
        for (const auto& port : module.ports) {
            const auto shape = declared_shape(
                port.name, port.type, port.packed_dimensions,
                port.unpacked_dimensions, parameters, module);
            if (shape.type.category != VerilogTypeCategory::Unknown)
                environment.insert_or_assign(port.name, shape.type);
        }
        for (const auto& declaration : module.declarations) {
            const auto shape = declared_shape(
                declaration.name, declaration.type,
                declaration.packed_dimensions,
                declaration.unpacked_dimensions, parameters, module);
            if (shape.type.category != VerilogTypeCategory::Unknown)
                environment.insert_or_assign(declaration.name, shape.type);
        }
        return environment;
    }

    std::string render_expression(
        const VerilogExpression& expression,
        const VerilogConstantEnvironment& parameters,
        const VerilogElaboratedGenerateScope* generated_scope,
        std::string_view module_path,
        const VerilogModuleDeclaration& module) const {
        const auto child = [&](const std::unique_ptr<VerilogExpression>& value) {
            return value ? render_expression(*value, parameters, generated_scope,
                                             module_path, module)
                         : std::string{};
        };
        switch (expression.kind) {
            case VerilogExpressionKind::Identifier: {
                for (auto* scope = generated_scope; scope;
                     scope = scope->parent_scope) {
                    if (std::any_of(scope->declarations.begin(),
                                    scope->declarations.end(),
                                    [&](const auto* declaration) {
                                        return declaration &&
                                               declaration->name == expression.text;
                                    })) {
                        return relative_hierarchy_path(module_path,
                                                       scope->hierarchy_path) +
                               "." + expression.text;
                    }
                }
                return expression.text;
            }
            case VerilogExpressionKind::Literal:
                return expression.text;
            case VerilogExpressionKind::Unary:
                return expression.text + child(expression.first);
            case VerilogExpressionKind::Binary:
                return "(" + child(expression.first) + " " + expression.text +
                       " " + child(expression.second) + ")";
            case VerilogExpressionKind::Conditional:
                return "(" + child(expression.first) + " ? " +
                       child(expression.second) + " : " +
                       child(expression.third) + ")";
            case VerilogExpressionKind::Select: {
                const auto index_text = [&](const VerilogExpression* index) {
                    if (!index)
                        return std::string{};
                    const auto value =
                        evaluate_verilog_constant(*index, parameters);
                    if (value.value) {
                        if (const auto integer = value.value->to_int64())
                            return std::to_string(*integer);
                    }
                    return render_expression(*index, parameters,
                                             generated_scope, module_path,
                                             module);
                };
                const std::string left = child(expression.first) + "[" +
                                         index_text(expression.second.get());
                if (!expression.third)
                    return left + "]";
                if (expression.select_kind == VerilogSelectKind::IndexedUp)
                    return left + " +: " +
                           index_text(expression.third.get()) + "]";
                if (expression.select_kind == VerilogSelectKind::IndexedDown)
                    return left + " -: " +
                           index_text(expression.third.get()) + "]";
                return left + ":" + index_text(expression.third.get()) + "]";
            }
            case VerilogExpressionKind::MemberAccess:
                return child(expression.first) + expression.member_separator +
                       expression.text;
            case VerilogExpressionKind::Call: {
                std::string rendered = child(expression.first) + "(";
                for (size_t index = 0; index < expression.operands.size(); ++index) {
                    if (index)
                        rendered += ", ";
                    if (index < expression.operand_names.size() &&
                        expression.operand_names[index]) {
                        rendered += "." + *expression.operand_names[index] + "(";
                        rendered += render_expression(
                            *expression.operands[index], parameters,
                            generated_scope, module_path, module);
                        rendered += ")";
                    } else {
                        rendered += render_expression(
                            *expression.operands[index], parameters,
                            generated_scope, module_path, module);
                    }
                }
                return rendered + ")";
            }
            case VerilogExpressionKind::Concatenation:
            case VerilogExpressionKind::Replication: {
                std::string rendered = "{";
                if (expression.kind == VerilogExpressionKind::Replication) {
                    rendered += child(expression.first) + "{";
                }
                for (size_t index = 0; index < expression.operands.size(); ++index) {
                    if (index)
                        rendered += ", ";
                    rendered += render_expression(
                        *expression.operands[index], parameters,
                        generated_scope, module_path, module);
                }
                return rendered +
                       (expression.kind == VerilogExpressionKind::Replication
                            ? "}}"
                            : "}");
            }
        }
        return {};
    }

    Expression lower_expression(
        const VerilogExpression& expression,
        const VerilogConstantEnvironment& parameters,
        const VerilogTypeEnvironment& types,
        const VerilogElaboratedGenerateScope* generated_scope,
        std::string_view module_path,
        const VerilogModuleDeclaration& module) const {
        const auto lower = [&](const std::unique_ptr<VerilogExpression>& value) {
            return value ? lower_expression(*value, parameters, types,
                                            generated_scope, module_path, module)
                         : Expression{};
        };
        const auto inferred = infer_verilog_expression_type(
            expression, types, parameters);
        const size_t width = inferred.type.bit_width.value_or(0);
        const std::string source = render_expression(
            expression, parameters, generated_scope, module_path, module);
        switch (expression.kind) {
            case VerilogExpressionKind::Identifier:
                return {ExpressionKind::Identifier, source, source, {}, width};
            case VerilogExpressionKind::Literal:
                return {ExpressionKind::Literal, expression.text, source, {},
                        width};
            case VerilogExpressionKind::Unary:
                return {ExpressionKind::Unary, expression.text, source,
                        {lower(expression.first)}, width};
            case VerilogExpressionKind::Binary:
                return {ExpressionKind::Binary, expression.text, source,
                        {lower(expression.first), lower(expression.second)}, width};
            case VerilogExpressionKind::Conditional:
                return {ExpressionKind::Conditional, "?:", source,
                        {lower(expression.first), lower(expression.second),
                         lower(expression.third)},
                        width};
            case VerilogExpressionKind::Select:
                if (expression.select_kind == VerilogSelectKind::Bit)
                    return {ExpressionKind::BitSelect, "", source,
                            {lower(expression.first), lower(expression.second)},
                            width};
                if (expression.select_kind == VerilogSelectKind::Part)
                    return {ExpressionKind::RangeSelect, "", source,
                            {lower(expression.first), lower(expression.second),
                             lower(expression.third)},
                            width};
                return {ExpressionKind::Opaque, "", source, {}, width};
            case VerilogExpressionKind::Concatenation: {
                std::vector<Expression> operands;
                operands.reserve(expression.operands.size());
                for (const auto& operand : expression.operands)
                    operands.push_back(lower_expression(
                        *operand, parameters, types, generated_scope,
                        module_path, module));
                return {ExpressionKind::Concatenation, "", source,
                        std::move(operands), width};
            }
            case VerilogExpressionKind::MemberAccess:
            case VerilogExpressionKind::Call:
            case VerilogExpressionKind::Replication:
                return {ExpressionKind::Opaque, "", source, {}, width};
        }
        return {ExpressionKind::Opaque, "", source, {}, width};
    }

    void add_ports_and_declarations(
        Module& target, const VerilogModuleDeclaration& module,
        const VerilogConstantEnvironment& parameters,
        VerilogTypeEnvironment& types) {
        std::unordered_set<std::string> port_names;
        for (const auto& port : module.ports) {
            PortDirection direction;
            switch (port.direction) {
                case VerilogPortDirection::Input:
                    direction = PortDirection::Input;
                    break;
                case VerilogPortDirection::Output:
                    direction = PortDirection::Output;
                    break;
                case VerilogPortDirection::Inout:
                    direction = PortDirection::InOut;
                    break;
                default:
                    diagnose("UnsupportedNativePortDirection",
                             "native projection cannot determine the direction of port '" +
                                 port.name + "'",
                             port.location);
                    continue;
            }
            const auto shape = declared_shape(
                port.name, port.type, port.packed_dimensions,
                port.unpacked_dimensions, parameters, module);
            if (shape.width == 0)
                continue;
            port_names.insert(port.name);
            target.ports_.emplace_back(
                port.name, direction, shape.width, shape.packed_left,
                shape.packed_right, shape.unpacked_left,
                shape.unpacked_right, shape.packed_width);
            const auto kind =
                direction == PortDirection::Input
                    ? NetType::Input
                    : direction == PortDirection::Output ? NetType::Output
                                                         : NetType::InOut;
            target.nets_.emplace_back(
                port.name, kind, shape.width, shape.packed_left,
                shape.packed_right, shape.unpacked_left,
                shape.unpacked_right, shape.packed_width);
            types.insert_or_assign(port.name, shape.type);
        }
        for (const auto& declaration : module.declarations) {
            if (port_names.contains(declaration.name))
                continue;
            const auto shape = declared_shape(
                declaration.name, declaration.type,
                declaration.packed_dimensions,
                declaration.unpacked_dimensions, parameters, module);
            if (shape.width == 0)
                continue;
            target.nets_.emplace_back(
                declaration.name, NetType::Internal, shape.width,
                shape.packed_left, shape.packed_right,
                shape.unpacked_left, shape.unpacked_right,
                shape.packed_width);
            types.insert_or_assign(declaration.name, shape.type);
        }
    }

    void add_assignment(
        Module& target, const VerilogContinuousAssignment& assignment,
        const VerilogConstantEnvironment& parameters,
        const VerilogTypeEnvironment& types,
        const VerilogElaboratedGenerateScope* generated_scope,
        std::string_view module_path, const VerilogModuleDeclaration& module) {
        if (!assignment.left || !assignment.right)
            return;
        target.assignments_.emplace_back(
            lower_expression(*assignment.left, parameters, types,
                             generated_scope, module_path, module),
            lower_expression(*assignment.right, parameters, types,
                             generated_scope, module_path, module));
    }

    std::map<std::string, std::string> parameter_overrides(
        const VerilogModuleInstantiation& syntax,
        const VerilogModuleDeclaration* target,
        const VerilogConstantEnvironment& environment,
        const VerilogElaboratedGenerateScope* generated_scope,
        std::string_view module_path,
        const VerilogModuleDeclaration& owner) const {
        std::map<std::string, std::string> result;
        std::vector<const VerilogParameterDeclaration*> overridable;
        if (target) {
            for (const auto& parameter : target->parameters)
                if (!parameter.local_parameter)
                    overridable.push_back(&parameter);
        }
        size_t positional_index = 0;
        for (const auto& parameter : syntax.parameter_overrides) {
            const std::string name =
                parameter.named_override
                    ? parameter.name
                    : positional_index < overridable.size()
                          ? overridable[positional_index++]->name
                          : std::string{};
            if (name.empty() || !parameter.value)
                continue;
            result.insert_or_assign(
                name, render_expression(*parameter.value, environment,
                                        generated_scope, module_path, owner));
        }
        return result;
    }

    std::vector<PinConnection> connections(
        const VerilogInstance& instance,
        const VerilogModuleDeclaration* target,
        const VerilogConstantEnvironment& environment,
        const VerilogElaboratedGenerateScope* generated_scope,
        std::string_view module_path,
        const VerilogModuleDeclaration& owner) {
        std::vector<PinConnection> result;
        size_t positional_index = 0;
        for (const auto& connection : instance.connections) {
            std::string pin;
            if (connection.name) {
                pin = *connection.name;
            } else if (target && positional_index < target->ports.size()) {
                pin = target->ports[positional_index++].name;
            } else {
                diagnose("UnsupportedNativePositionalConnection",
                         "native projection cannot resolve positional port connection "
                         "for instance '" + instance.name + "'",
                         connection.location);
                continue;
            }
            if (!connection.expression)
                continue;
            result.emplace_back(
                std::move(pin),
                render_expression(*connection.expression, environment,
                                 generated_scope, module_path, owner));
        }
        return result;
    }

    const InstanceSyntaxMap& syntax_instances(
        const VerilogModuleDeclaration& module) {
        const auto found = instance_syntax_.find(&module);
        if (found != instance_syntax_.end())
            return found->second;
        return instance_syntax_.emplace(&module, index_instantiations(module))
            .first->second;
    }

    void add_actual_instance(
        Module& target, const VerilogElaboratedInstance& child,
        const VerilogElaboratedGenerateScope* generated_scope,
        std::string_view module_path, const VerilogModuleDeclaration& owner,
        const VerilogConstantEnvironment& environment) {
        const auto& instances = syntax_instances(owner);
        const auto found = instances.find(
            {child.location.source.get(), child.location.byte_offset});
        if (found == instances.end()) {
            diagnose("MissingNativeInstanceSyntax",
                     "could not match elaborated instance '" +
                         child.instance_name + "' to its syntax declaration",
                     child.location);
            return;
        }
        const auto& [instantiation, instance] = found->second;
        if (!instance->unpacked_dimensions.empty()) {
            diagnose("UnsupportedNativeInstanceArray",
                     "native project projection does not yet materialize instance "
                     "arrays",
                     instance->location);
            return;
        }
        const auto* child_definition = child.module_definition;
        const auto instance_name = relative_hierarchy_path(
            module_path, child.hierarchy_path);
        target.instances_.emplace_back(
            instance_name, instantiation->module_type,
            connections(*instance, child_definition, environment,
                        generated_scope, module_path, owner),
            std::nullopt,
            parameter_overrides(*instantiation, child_definition, environment,
                                generated_scope, module_path, owner));
        if (child_definition)
            target.child_modules_.push_back(
                project_elaborated_module(child, instance_name));
    }

    void add_generate_scope_contents(
        Module& target, const VerilogElaboratedGenerateScope& scope,
        std::string_view module_path, const VerilogModuleDeclaration& module,
        const VerilogTypeEnvironment& module_types) {
        for (const auto* declaration : scope.declarations) {
            if (!declaration)
                continue;
            const auto shape = declared_shape(
                declaration->name, declaration->type,
                declaration->packed_dimensions,
                declaration->unpacked_dimensions, scope.parameters, module);
            if (shape.width == 0)
                continue;
            const auto local_scope = relative_hierarchy_path(
                module_path, scope.hierarchy_path);
            const auto name = local_scope.empty()
                                  ? declaration->name
                                  : local_scope + "." + declaration->name;
            target.nets_.emplace_back(
                name, NetType::Internal, shape.width, shape.packed_left,
                shape.packed_right, shape.unpacked_left,
                shape.unpacked_right, shape.packed_width);
        }
        for (const auto* assignment : scope.assignments) {
            if (assignment)
                add_assignment(target, *assignment, scope.parameters,
                               module_types, &scope, module_path, module);
        }
        for (const auto& child : scope.children)
            add_actual_instance(target, *child, &scope, module_path, module,
                                scope.parameters);
        for (const auto& nested : scope.generate_scopes)
            add_generate_scope_contents(target, *nested, module_path, module,
                                        module_types);
    }

    Module project_elaborated_module(const VerilogElaboratedInstance& instance,
                                     std::string hierarchy_name,
                                     bool include_hierarchy = true) {
        const auto* definition = instance.module_definition;
        if (!definition)
            return {};
        Module result(definition->name, {}, {}, {}, {}, false,
                      std::move(hierarchy_name));
        auto types = type_environment(*definition, instance.parameters);
        add_ports_and_declarations(result, *definition, instance.parameters,
                                   types);
        for (const auto& assignment : definition->assignments)
            add_assignment(result, assignment, instance.parameters, types,
                           nullptr, instance.hierarchy_path, *definition);
        for (const auto* assignment : instance.generated_assignments) {
            if (assignment)
                add_assignment(result, *assignment, instance.parameters, types,
                               nullptr, instance.hierarchy_path, *definition);
        }
        for (const auto* declaration : instance.generated_declarations) {
            if (!declaration)
                continue;
            const auto shape = declared_shape(
                declaration->name, declaration->type,
                declaration->packed_dimensions,
                declaration->unpacked_dimensions, instance.parameters,
                *definition);
            if (shape.width == 0)
                continue;
            result.nets_.emplace_back(
                declaration->name, NetType::Internal, shape.width,
                shape.packed_left, shape.packed_right,
                shape.unpacked_left, shape.unpacked_right,
                shape.packed_width);
        }
        if (include_hierarchy) {
            for (const auto& child : instance.children)
                add_actual_instance(result, *child, nullptr,
                                    instance.hierarchy_path, *definition,
                                    instance.parameters);
            for (const auto& scope : instance.generate_scopes)
                add_generate_scope_contents(result, *scope,
                                            instance.hierarchy_path, *definition,
                                            types);
        }
        result.behavioral_ = std::any_of(
            definition->procedural_blocks.begin(),
            definition->procedural_blocks.end(), [](const auto& block) {
                return block.kind != VerilogProceduralBlockKind::Initial;
            });
        return result;
    }

    Module project_definition(const VerilogModuleDeclaration& definition) {
        const auto parameters = module_parameters(definition);
        Module result(definition.name, {}, {}, {}, {}, false);
        auto types = type_environment(definition, parameters);
        add_ports_and_declarations(result, definition, parameters, types);
        for (const auto& assignment : definition.assignments)
            add_assignment(result, assignment, parameters, types, nullptr,
                           definition.name, definition);
        for (const auto& instantiation : definition.instantiations) {
            const auto target = modules_.find(instantiation.module_type);
            const auto* child_definition =
                target == modules_.end() ? nullptr : target->second;
            for (const auto& instance : instantiation.instances) {
                if (!instance.unpacked_dimensions.empty()) {
                    diagnose("UnsupportedNativeInstanceArray",
                             "native project projection does not yet materialize "
                             "instance arrays",
                             instance.location);
                    continue;
                }
                result.instances_.emplace_back(
                    instance.name, instantiation.module_type,
                    connections(instance, child_definition,
                                parameters, nullptr, definition.name,
                                definition),
                    std::nullopt,
                    parameter_overrides(instantiation, child_definition,
                                        parameters, nullptr, definition.name,
                                        definition));
            }
        }
        result.behavioral_ = std::any_of(
            definition.procedural_blocks.begin(),
            definition.procedural_blocks.end(), [](const auto& block) {
                return block.kind != VerilogProceduralBlockKind::Initial;
            });
        return result;
    }

    std::shared_ptr<NativeVerilogProjectContext> context_;
    NativeVerilogProjectProjectionResult result_;
    std::unordered_map<std::string, const VerilogModuleDeclaration*> modules_;
    std::unordered_map<const VerilogModuleDeclaration*, InstanceSyntaxMap>
        instance_syntax_;
};

NativeVerilogProjectProjectionResult project_native_verilog(
    std::shared_ptr<NativeVerilogProjectContext> context) {
    return NativeVerilogProjectProjector(std::move(context)).run();
}

} // namespace labneura::eda::detail
