#include "labneura/eda/verilog_semantic.h"
#include "labneura/eda/verilog_callable_binding.h"

#include <algorithm>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <utility>

namespace labneura::eda {
namespace {

bool is_net_type(std::string_view type) {
    return type == "wire" || type == "tri" || type == "tri0" || type == "tri1" ||
           type == "wand" || type == "wor" || type == "uwire";
}

bool is_builtin_identifier(std::string_view name) {
    if (name.empty() || name.front() == '$')
        return true;
    static const std::unordered_set<std::string_view> builtins = {
        "signed", "unsigned", "byte", "shortint", "int", "integer", "longint", "time",
        "real", "realtime", "bit", "logic", "reg", "wire", "string", "void", "this",
        "super", "null", "new", "type"
    };
    return builtins.contains(name);
}

struct DesignUnit {
    VerilogSemanticScope* scope = nullptr;
    const VerilogModuleDeclaration* module = nullptr;
    const VerilogInterfaceDeclaration* interface = nullptr;
    const VerilogPackageDeclaration* package = nullptr;
};

std::unique_ptr<VerilogExpression> clone_expression(const VerilogExpression* expression) {
    if (!expression)
        return nullptr;
    auto copy = std::make_unique<VerilogExpression>();
    copy->kind = expression->kind;
    copy->select_kind = expression->select_kind;
    copy->location = expression->location;
    copy->text = expression->text;
    copy->first = clone_expression(expression->first.get());
    copy->second = clone_expression(expression->second.get());
    copy->third = clone_expression(expression->third.get());
    copy->member_separator = expression->member_separator;
    for (const auto& operand : expression->operands)
        copy->operands.push_back(clone_expression(operand.get()));
    return copy;
}

VerilogExpressionRange clone_range(const VerilogExpressionRange& range) {
    VerilogExpressionRange copy;
    copy.location = range.location;
    copy.left = clone_expression(range.left.get());
    copy.right = clone_expression(range.right.get());
    return copy;
}

template <typename Range>
void copy_ranges(const Range& source, std::vector<VerilogExpressionRange>& target) {
    for (const auto& range : source)
        target.push_back(clone_range(range));
}

class Analyzer {
public:
    explicit Analyzer(const VerilogCompilationUnit& syntax) : syntax_(syntax) {
        result_.model.compilation_unit_scope =
            std::make_unique<VerilogSemanticScope>();
        result_.model.compilation_unit_scope->name = "<compilation-unit>";
        result_.model.compilation_unit_scope->kind =
            VerilogSemanticScopeKind::CompilationUnit;
    }

    VerilogSemanticAnalysisResult run() {
        collect_design_units();
        collect_local_declarations();
        collect_callables();
        collect_instances();
        resolve_design_units();
        resolve_callables();
        return std::move(result_);
    }

private:
    VerilogSemanticScope& root() const {
        return *result_.model.compilation_unit_scope;
    }

    VerilogSemanticScope* create_scope(VerilogSemanticScope& parent, std::string name,
                                       VerilogSemanticScopeKind kind) {
        auto scope = std::make_unique<VerilogSemanticScope>();
        scope->name = std::move(name);
        scope->kind = kind;
        scope->parent = &parent;
        auto* raw = scope.get();
        parent.children.push_back(std::move(scope));
        return raw;
    }

    VerilogSemanticSymbol* declare(VerilogSemanticScope& scope, std::string name,
                                   VerilogSemanticSymbolKind kind,
                                   const VerilogSyntaxLocation& location) {
        const auto found = scope.symbols_by_name.find(name);
        if (found != scope.symbols_by_name.end()) {
            VerilogSemanticDiagnostic diagnostic;
            diagnostic.code = kind == VerilogSemanticSymbolKind::Instance
                                  ? "DuplicateInstanceName"
                                  : "DuplicateDeclaration";
            diagnostic.message =
                kind == VerilogSemanticSymbolKind::Instance
                    ? "duplicate instance name '" + name + "' in scope '" + scope.name + "'"
                    : "duplicate declaration of '" + name + "' in scope '" + scope.name + "'";
            diagnostic.location = location;
            diagnostic.related_location = found->second->declaration_location;
            result_.diagnostics.push_back(std::move(diagnostic));
            return found->second;
        }
        auto symbol = std::make_unique<VerilogSemanticSymbol>();
        symbol->name = std::move(name);
        symbol->kind = kind;
        symbol->declaration_location = location;
        symbol->scope = &scope;
        auto* raw = symbol.get();
        scope.symbols_by_name.emplace(raw->name, raw);
        scope.symbols.push_back(std::move(symbol));
        return raw;
    }

    VerilogSemanticScope* add_design_unit(std::string name, VerilogSemanticScopeKind scope_kind,
                                          VerilogSemanticSymbolKind symbol_kind,
                                          const VerilogSyntaxLocation& location) {
        auto* scope = create_scope(root(), name, scope_kind);
        auto* symbol = declare(root(), std::move(name), symbol_kind, location);
        if (symbol->member_scope == nullptr)
            symbol->member_scope = scope;
        return scope;
    }

    void collect_design_units() {
        for (const auto& module : syntax_.modules) {
            units_.push_back({add_design_unit(module.name, VerilogSemanticScopeKind::Module,
                                               VerilogSemanticSymbolKind::Module,
                                               module.location),
                              &module, nullptr, nullptr});
        }
        for (const auto& interface : syntax_.interfaces) {
            units_.push_back({add_design_unit(interface.name, VerilogSemanticScopeKind::Interface,
                                               VerilogSemanticSymbolKind::Interface,
                                               interface.location),
                              nullptr, &interface, nullptr});
        }
        for (const auto& package : syntax_.packages) {
            units_.push_back({add_design_unit(package.name, VerilogSemanticScopeKind::Package,
                                               VerilogSemanticSymbolKind::Package,
                                               package.location),
                              nullptr, nullptr, &package});
        }
    }

    void collect_parameters(VerilogSemanticScope& scope,
                            const std::vector<VerilogParameterDeclaration>& parameters) {
        for (const auto& parameter : parameters) {
            declare(scope, parameter.name,
                    parameter.local_parameter ? VerilogSemanticSymbolKind::LocalParameter
                                              : VerilogSemanticSymbolKind::Parameter,
                    parameter.location);
        }
    }

    void collect_ports(VerilogSemanticScope& scope,
                       const std::vector<VerilogPortDeclaration>& ports) {
        for (const auto& port : ports)
            declare(scope, port.name, VerilogSemanticSymbolKind::Port, port.location);
    }

    void collect_variables(VerilogSemanticScope& scope,
                           const std::vector<VerilogVariableDeclaration>& declarations) {
        for (const auto& declaration : declarations) {
            auto* symbol =
                declare(scope, declaration.name,
                        is_net_type(declaration.type) ? VerilogSemanticSymbolKind::Net
                                                      : VerilogSemanticSymbolKind::Variable,
                        declaration.location);
            symbol->declared_type = declaration.type;
            copy_ranges(declaration.packed_dimensions, symbol->packed_dimensions);
            copy_ranges(declaration.unpacked_dimensions, symbol->unpacked_dimensions);
        }
    }

    template <typename Callable>
    static const auto& callable_formals(const Callable& callable) {
        if constexpr (requires { callable.formal_arguments; })
            return callable.formal_arguments;
        else if constexpr (requires { callable.formals; })
            return callable.formals;
        else
            return callable.arguments;
    }

    template <typename Callable>
    static bool is_task_callable(const Callable& callable) {
        using Kind = std::remove_cvref_t<decltype(callable.kind)>;
        if constexpr (requires { Kind::Task; })
            return callable.kind == Kind::Task;
        else
            return false;
    }

    template <typename Declaration>
    static std::string declared_type(const Declaration& declaration) {
        if constexpr (requires { declaration.type; })
            return declaration.type;
        else
            return declaration.type_spelling;
    }

    template <typename Callable>
    void collect_one_callable(VerilogSemanticScope& owner, const Callable& callable) {
        const auto kind = is_task_callable(callable) ? VerilogSemanticSymbolKind::Task
                                                     : VerilogSemanticSymbolKind::Function;
        const bool duplicate = owner.symbols_by_name.contains(callable.name);
        auto* callable_symbol = declare(owner, callable.name, kind, callable.location);
        auto* callable_scope =
            create_scope(owner, callable.name, VerilogSemanticScopeKind::Callable);
        if (!duplicate && !callable_symbol->member_scope)
            callable_symbol->member_scope = callable_scope;

        VerilogSemanticCallableMetadata metadata;
        if constexpr (requires { callable.return_type; }) {
            if constexpr (requires { callable.return_type.has_value(); }) {
                if (callable.return_type)
                    metadata.return_type = *callable.return_type;
            }
            else {
                metadata.return_type = callable.return_type;
            }
        }
        if constexpr (requires { callable.return_packed_dimensions; })
            copy_ranges(callable.return_packed_dimensions, metadata.return_packed_dimensions);
        else if constexpr (requires { callable.return_dimensions; })
            copy_ranges(callable.return_dimensions, metadata.return_packed_dimensions);
        if constexpr (requires { callable.return_unpacked_dimensions; })
            copy_ranges(callable.return_unpacked_dimensions,
                        metadata.return_unpacked_dimensions);

        for (const auto& formal : callable_formals(callable)) {
            auto* argument = declare(*callable_scope, formal.name,
                                     VerilogSemanticSymbolKind::CallableArgument,
                                     formal.location);
            argument->argument_direction = formal.direction;
            argument->declared_type = declared_type(formal);
            if constexpr (requires { formal.packed_dimensions; })
                copy_ranges(formal.packed_dimensions, argument->packed_dimensions);
            if constexpr (requires { formal.unpacked_dimensions; })
                copy_ranges(formal.unpacked_dimensions, argument->unpacked_dimensions);
            metadata.formal_arguments.push_back(argument);
        }

        if constexpr (requires { callable.local_declarations; }) {
            collect_variables(*callable_scope, callable.local_declarations);
        }
        else if constexpr (requires { callable.declarations; }) {
            collect_variables(*callable_scope, callable.declarations);
        }
        if (!duplicate)
            callable_symbol->callable = std::move(metadata);
        callable_scopes_[&callable] = callable_scope;
    }

    template <typename Callables>
    void collect_callables_in(VerilogSemanticScope& scope, const Callables& callables) {
        for (const auto& callable : callables)
            collect_one_callable(scope, callable);
    }

    template <typename Unit>
    void collect_callables_from_unit(VerilogSemanticScope& scope, const Unit& unit) {
        if constexpr (requires { unit.callables; })
            collect_callables_in(scope, unit.callables);
    }

    void collect_callables() {
        for (const auto& unit : units_) {
            if (unit.module) {
                collect_callables_from_unit(*unit.scope, *unit.module);
            }
            else if (unit.interface) {
                collect_callables_from_unit(*unit.scope, *unit.interface);
            }
            else if (unit.package) {
                collect_callables_from_unit(*unit.scope, *unit.package);
            }
        }
    }

    void collect_type_definitions(VerilogSemanticScope& scope,
                                  const std::vector<VerilogTypeDefinition>& definitions) {
        for (const auto& definition : definitions) {
            if (definition.kind == VerilogTypeDefinitionKind::Alias)
                declare(scope, definition.name, VerilogSemanticSymbolKind::Type,
                        definition.location);
            else if (!definition.name.empty() &&
                     definition.name.rfind("$native_type_", 0) != 0)
                declare(scope, definition.name, VerilogSemanticSymbolKind::Type,
                        definition.location);
            for (const auto& enumerator : definition.enumerators)
                declare(scope, enumerator.name, VerilogSemanticSymbolKind::EnumValue,
                        enumerator.location);
        }
    }

    void collect_local_declarations() {
        for (const auto& unit : units_) {
            if (unit.module) {
                collect_parameters(*unit.scope, unit.module->parameters);
                collect_type_definitions(*unit.scope, unit.module->type_definitions);
                collect_ports(*unit.scope, unit.module->ports);
                collect_variables(*unit.scope, unit.module->declarations);
                for (size_t i = 0; i < unit.module->procedural_blocks.size(); ++i) {
                    const auto& block = unit.module->procedural_blocks[i];
                    const auto& location = block.location.location;
                    const std::string name = "<procedural@" + std::to_string(location.line) +
                                             ":" + std::to_string(location.column) + ":" +
                                             std::to_string(i) + ">";
                    declare(*unit.scope, name, VerilogSemanticSymbolKind::ProceduralBlock,
                            block.location);
                }
                for (const auto& construct : unit.module->generate_constructs)
                    collect_generate_construct(*unit.scope, *construct);
            }
            else if (unit.interface) {
                collect_parameters(*unit.scope, unit.interface->parameters);
                collect_type_definitions(*unit.scope, unit.interface->type_definitions);
                collect_ports(*unit.scope, unit.interface->ports);
                collect_variables(*unit.scope, unit.interface->declarations);
            }
            else if (unit.package) {
                collect_parameters(*unit.scope, unit.package->parameters);
                collect_type_definitions(*unit.scope, unit.package->type_definitions);
                collect_variables(*unit.scope, unit.package->declarations);
            }
        }
    }

    void collect_instances() {
        for (const auto& unit : units_) {
            if (!unit.module)
                continue;
            for (const auto& instantiation : unit.module->instantiations) {
                for (const auto& instance : instantiation.instances) {
                    declare(*unit.scope, instance.name, VerilogSemanticSymbolKind::Instance,
                            instance.location);
                }
            }
        }
    }

    void collect_generate_scope_contents(VerilogSemanticScope& scope,
                                         const VerilogGenerateScope& generate_scope,
                                         bool create_child_scope,
                                         std::string_view loop_variable = {}) {
        VerilogSemanticScope* local_scope = &scope;
        if (create_child_scope) {
            local_scope = create_scope(scope, generate_scope.name,
                                       VerilogSemanticScopeKind::Generate);
            generate_scopes_[&generate_scope] = local_scope;
        }
        collect_parameters(*local_scope, generate_scope.parameters);
        collect_type_definitions(*local_scope, generate_scope.type_definitions);
        collect_variables(*local_scope, generate_scope.declarations);
        if (!loop_variable.empty())
            declare(*local_scope, std::string(loop_variable), VerilogSemanticSymbolKind::Variable,
                    generate_scope.location);
        for (const auto& instantiation : generate_scope.instantiations)
            for (const auto& instance : instantiation.instances)
                declare(*local_scope, instance.name, VerilogSemanticSymbolKind::Instance,
                        instance.location);
        for (const auto& construct : generate_scope.constructs)
            collect_generate_construct(*local_scope, *construct);
    }

    void collect_generate_construct(VerilogSemanticScope& scope,
                                    const VerilogGenerateConstruct& construct) {
        switch (construct.kind) {
            case VerilogGenerateKind::Region:
                if (construct.region_body)
                    collect_generate_scope_contents(scope, *construct.region_body, false);
                break;
            case VerilogGenerateKind::If:
                if (construct.true_body)
                    collect_generate_scope_contents(scope, *construct.true_body, true);
                if (construct.false_body)
                    collect_generate_scope_contents(scope, *construct.false_body, true);
                break;
            case VerilogGenerateKind::Case:
                for (const auto& item : construct.case_items)
                    if (item.body)
                        collect_generate_scope_contents(scope, *item.body, true);
                break;
            case VerilogGenerateKind::For:
                if (construct.loop_body) {
                    collect_generate_scope_contents(scope, *construct.loop_body, true,
                                                    construct.loop_variable);
                    generate_construct_scopes_[&construct] =
                        generate_scopes_.at(construct.loop_body.get());
                }
                break;
        }
    }

    void diagnose(std::string code, std::string message,
                  const VerilogSyntaxLocation& location,
                  std::optional<VerilogSyntaxLocation> related = std::nullopt) {
        result_.diagnostics.push_back(
            {std::move(code), std::move(message), location, std::move(related)});
    }

    void add_reference(std::string name, const VerilogSyntaxLocation& location,
                       const VerilogSemanticScope& scope,
                       const VerilogSemanticSymbol* symbol) {
        result_.references.push_back({std::move(name), location, &scope, symbol});
        if (!symbol) {
            diagnose("UnresolvedIdentifier",
                     "identifier '" + result_.references.back().name +
                         "' is not declared in visible scope '" + scope.name + "'",
                     location);
        }
    }

    const VerilogSemanticSymbol* local_symbol(const VerilogSemanticScope& scope,
                                               std::string_view name) const {
        const auto found = scope.symbols_by_name.find(std::string(name));
        return found == scope.symbols_by_name.end() ? nullptr : found->second;
    }

    const VerilogSemanticSymbol* visible_symbol(const VerilogSemanticScope& scope,
                                                std::string_view name) const {
        for (auto* current = &scope; current; current = current->parent) {
            if (const auto* symbol = local_symbol(*current, name))
                return symbol;
            if (current->kind == VerilogSemanticScopeKind::CompilationUnit)
                break;
        }
        return nullptr;
    }

    const VerilogSemanticSymbol* design_unit_symbol(std::string_view name) const {
        const auto* symbol = lookup_verilog_symbol(root(), name, false);
        if (!symbol || (symbol->kind != VerilogSemanticSymbolKind::Module &&
                        symbol->kind != VerilogSemanticSymbolKind::Interface &&
                        symbol->kind != VerilogSemanticSymbolKind::Package))
            return nullptr;
        return symbol;
    }

    enum class CallContext { FunctionExpression, TaskStatement };

    const VerilogSemanticSymbol* resolve_callable_callee(
        const VerilogExpression* callee, const VerilogSemanticScope& scope) {
        if (!callee)
            return nullptr;
        if (callee->kind == VerilogExpressionKind::Identifier) {
            const auto* symbol = visible_symbol(scope, callee->text);
            if (symbol) {
                add_reference(callee->text, callee->location, scope, symbol);
                return symbol;
            }
            if (is_builtin_identifier(callee->text)) {
                result_.references.push_back(
                    {callee->text, callee->location, &scope, nullptr});
                return nullptr;
            }
            add_reference(callee->text, callee->location, scope, nullptr);
            return nullptr;
        }

        if (callee->kind == VerilogExpressionKind::MemberAccess && callee->first &&
            callee->first->kind == VerilogExpressionKind::Identifier) {
            if (const auto* owner = design_unit_symbol(callee->first->text)) {
                add_reference(callee->first->text, callee->first->location, scope, owner);
                const auto* member =
                    owner->member_scope
                        ? local_symbol(*owner->member_scope, callee->text)
                        : nullptr;
                add_reference(callee->first->text + callee->member_separator + callee->text,
                              callee->location,
                              owner->member_scope ? *owner->member_scope : scope, member);
                return member;
            }
        }
        resolve_expression(callee, scope);
        return nullptr;
    }

    void validate_call(const VerilogSemanticSymbol* callable, CallContext context,
                       const VerilogExpression& call) {
        if (!callable ||
            (callable->kind != VerilogSemanticSymbolKind::Function &&
             callable->kind != VerilogSemanticSymbolKind::Task))
            return;

        const bool expects_task = context == CallContext::TaskStatement;
        const bool is_task = callable->kind == VerilogSemanticSymbolKind::Task;
        if (expects_task != is_task) {
            diagnose("CallableKindMismatch",
                     expects_task
                         ? "function '" + callable->name + "' cannot be invoked as a task"
                         : "task '" + callable->name + "' cannot be used as a function",
                     call.location, callable->declaration_location);
        }
        VerilogTypeAnalysisResult unavailable_types;
        const auto binding =
            bind_verilog_callable_arguments(*callable, call, unavailable_types);
        for (const auto& diagnostic : binding.diagnostics) {
            if (diagnostic.severity != VerilogTypeDiagnosticSeverity::Error)
                continue;
            if (diagnostic.code == "InvalidCallableArgumentLValue") {
                std::string message = diagnostic.message;
                if (message.starts_with("actual "))
                    message.replace(0, 6, "argument");
                diagnose("InvalidCallableArgumentTarget", std::move(message),
                         diagnostic.location, diagnostic.related_location);
            } else {
                diagnose(diagnostic.code, diagnostic.message, diagnostic.location,
                         diagnostic.related_location);
            }
        }
    }

    void resolve_expression(const VerilogExpression* expression,
                            const VerilogSemanticScope& scope,
                            CallContext call_context = CallContext::FunctionExpression) {
        if (!expression)
            return;
        switch (expression->kind) {
            case VerilogExpressionKind::Identifier: {
                const auto* symbol = visible_symbol(scope, expression->text);
                if (!symbol && is_builtin_identifier(expression->text)) {
                    result_.references.push_back(
                        {expression->text, expression->location, &scope, nullptr});
                    return;
                }
                add_reference(expression->text, expression->location, scope, symbol);
                return;
            }
            case VerilogExpressionKind::MemberAccess: {
                const auto* base = expression->first.get();
                if (base && base->kind == VerilogExpressionKind::Identifier) {
                    if (const auto* owner = design_unit_symbol(base->text)) {
                        add_reference(base->text, base->location, scope, owner);
                        const auto* member =
                            owner->member_scope
                                ? local_symbol(*owner->member_scope, expression->text)
                                : nullptr;
                        add_reference(base->text + "." + expression->text,
                                      expression->location,
                                      owner->member_scope ? *owner->member_scope : scope, member);
                        return;
                    }
                }
                resolve_expression(base, scope);
                return;
            }
            case VerilogExpressionKind::Call: {
                const auto* callable =
                    resolve_callable_callee(expression->first.get(), scope);
                validate_call(callable, call_context, *expression);
                for (const auto& operand : expression->operands)
                    resolve_expression(operand.get(), scope);
                return;
            }
            default:
                break;
        }
        resolve_expression(expression->first.get(), scope);
        resolve_expression(expression->second.get(), scope);
        resolve_expression(expression->third.get(), scope);
        for (const auto& operand : expression->operands)
            resolve_expression(operand.get(), scope);
    }

    void resolve_range(const VerilogExpressionRange& range,
                      const VerilogSemanticScope& scope) {
        resolve_expression(range.left.get(), scope);
        resolve_expression(range.right.get(), scope);
    }

    void resolve_parameter(const VerilogParameterDeclaration& parameter,
                           const VerilogSemanticScope& scope) {
        if (parameter.named_override)
            return;
        for (const auto& dimension : parameter.packed_dimensions)
            resolve_range(dimension, scope);
        resolve_expression(parameter.value.get(), scope);
    }

    void resolve_port(const VerilogPortDeclaration& port,
                      const VerilogSemanticScope& scope) {
        for (const auto& dimension : port.packed_dimensions)
            resolve_range(dimension, scope);
        for (const auto& dimension : port.unpacked_dimensions)
            resolve_range(dimension, scope);
    }

    void resolve_variable(const VerilogVariableDeclaration& declaration,
                          const VerilogSemanticScope& scope) {
        for (const auto& dimension : declaration.packed_dimensions)
            resolve_range(dimension, scope);
        for (const auto& dimension : declaration.unpacked_dimensions)
            resolve_range(dimension, scope);
        resolve_expression(declaration.initializer.get(), scope);
    }

    void resolve_event_control(const VerilogEventControl* control,
                               const VerilogSemanticScope& scope) {
        if (!control)
            return;
        for (const auto& term : control->terms)
            resolve_expression(term.expression.get(), scope);
    }

    bool resolve_task_call_if_present(const VerilogStatement& statement,
                                      const VerilogSemanticScope& scope) {
        if (statement.kind != VerilogStatementKind::TaskCall)
            return false;

        const auto* call = statement.call_expression
                               ? statement.call_expression.get()
                               : statement.expression.get();
        if (call)
            resolve_expression(call, scope, CallContext::TaskStatement);
        return true;
    }

    void resolve_statement(const VerilogStatement* statement,
                           VerilogSemanticScope& scope) {
        if (!statement || statement->kind == VerilogStatementKind::Unsupported)
            return;
        if (statement->kind == VerilogStatementKind::Loop) {
            VerilogSemanticScope* loop_scope = &scope;
            if (!statement->loop_variable_name.empty()) {
                const std::string name =
                    "<loop@" + std::to_string(statement->location.location.line) + ":" +
                    std::to_string(statement->location.location.column) + ">";
                auto* local_scope =
                    create_scope(scope, name, VerilogSemanticScopeKind::Procedural);
                declare(*local_scope, statement->loop_variable_name,
                        VerilogSemanticSymbolKind::Variable,
                        statement->loop_variable_location);
                loop_scope = local_scope;
            }
            resolve_statement(statement->loop_initialization.get(), *loop_scope);
            resolve_expression(statement->expression.get(), *loop_scope);
            resolve_statement(statement->loop_step.get(), *loop_scope);
            resolve_statement(statement->first_statement.get(), *loop_scope);
            return;
        }
        if (statement->kind == VerilogStatementKind::ProceduralAssignment &&
            statement->left && !is_assignable_target(*statement->left)) {
            diagnose("InvalidAssignmentTarget",
                     "procedural assignment target is not an assignable expression",
                     statement->left->location);
        }
        const bool task_call = resolve_task_call_if_present(*statement, scope);
        if (!task_call)
            resolve_expression(statement->expression.get(), scope);
        resolve_expression(statement->left.get(), scope);
        resolve_expression(statement->right.get(), scope);
        resolve_event_control(statement->event_control.get(), scope);
        resolve_statement(statement->first_statement.get(), scope);
        resolve_statement(statement->second_statement.get(), scope);
        for (const auto& child : statement->statements)
            resolve_statement(child.get(), scope);
        for (const auto& item : statement->case_items) {
            for (const auto& expression : item.expressions)
                resolve_expression(expression.get(), scope);
            resolve_statement(item.statement.get(), scope);
        }
    }

    template <typename Callable>
    void resolve_one_callable(const Callable& callable) {
        const auto found = callable_scopes_.find(&callable);
        if (found == callable_scopes_.end())
            return;
        auto& scope = *found->second;
        for (const auto& formal : callable_formals(callable)) {
            if constexpr (requires { formal.packed_dimensions; })
                for (const auto& dimension : formal.packed_dimensions)
                    resolve_range(dimension, scope);
            if constexpr (requires { formal.unpacked_dimensions; })
                for (const auto& dimension : formal.unpacked_dimensions)
                    resolve_range(dimension, scope);
        }
        if constexpr (requires { callable.return_packed_dimensions; }) {
            for (const auto& dimension : callable.return_packed_dimensions)
                resolve_range(dimension, scope);
        }
        else if constexpr (requires { callable.return_dimensions; }) {
            for (const auto& dimension : callable.return_dimensions)
                resolve_range(dimension, scope);
        }
        if constexpr (requires { callable.return_unpacked_dimensions; }) {
            for (const auto& dimension : callable.return_unpacked_dimensions)
                resolve_range(dimension, scope);
        }
        if constexpr (requires { callable.local_declarations; }) {
            for (const auto& declaration : callable.local_declarations)
                resolve_variable(declaration, scope);
        }
        else if constexpr (requires { callable.declarations; }) {
            for (const auto& declaration : callable.declarations)
                resolve_variable(declaration, scope);
        }
        if constexpr (requires { callable.body.get(); })
            resolve_statement(callable.body.get(), scope);
        else if constexpr (requires { callable.body; })
            resolve_statement(&callable.body, scope);
    }

    template <typename Callables>
    void resolve_callable_list(const Callables& callables) {
        for (const auto& callable : callables)
            resolve_one_callable(callable);
    }

    template <typename Unit>
    void resolve_callables_from_unit(const Unit& unit) {
        if constexpr (requires { unit.callables; })
            resolve_callable_list(unit.callables);
    }

    void resolve_callables() {
        for (const auto& unit : units_) {
            if (unit.module)
                resolve_callables_from_unit(*unit.module);
            else if (unit.interface)
                resolve_callables_from_unit(*unit.interface);
            else if (unit.package)
                resolve_callables_from_unit(*unit.package);
        }
    }

    static bool is_assignable_target(const VerilogExpression& expression) {
        switch (expression.kind) {
            case VerilogExpressionKind::Identifier:
                return true;
            case VerilogExpressionKind::MemberAccess:
            case VerilogExpressionKind::Select:
                return expression.first && is_assignable_target(*expression.first);
            case VerilogExpressionKind::Concatenation:
                return !expression.operands.empty() &&
                       std::all_of(expression.operands.begin(), expression.operands.end(),
                                   [](const auto& operand) {
                                       return operand && is_assignable_target(*operand);
                                   });
            default:
                return false;
        }
    }

    void resolve_module_instantiation(const VerilogModuleInstantiation& instantiation,
                                      const VerilogSemanticScope& scope) {
        auto* target = root().symbols_by_name.contains(instantiation.module_type)
                           ? root().symbols_by_name.at(instantiation.module_type)
                           : nullptr;
        if (!target || target->kind != VerilogSemanticSymbolKind::Module ||
            !target->member_scope) {
            diagnose("UnresolvedModuleType",
                     "module type '" + instantiation.module_type + "' is not declared",
                     instantiation.location);
            for (const auto& parameter : instantiation.parameter_overrides)
                resolve_expression(parameter.value.get(), scope);
            for (const auto& instance : instantiation.instances)
                for (const auto& connection : instance.connections)
                    resolve_expression(connection.expression.get(), scope);
            return;
        }
        std::vector<const VerilogSemanticSymbol*> parameters;
        std::vector<const VerilogSemanticSymbol*> ports;
        for (const auto& symbol : target->member_scope->symbols) {
            if (symbol->kind == VerilogSemanticSymbolKind::Parameter)
                parameters.push_back(symbol.get());
            if (symbol->kind == VerilogSemanticSymbolKind::Port)
                ports.push_back(symbol.get());
        }
        size_t positional_parameter_count = 0;
        for (const auto& override : instantiation.parameter_overrides) {
            if (override.named_override) {
                const auto* named = local_symbol(*target->member_scope, override.name);
                if (!named || named->kind != VerilogSemanticSymbolKind::Parameter) {
                    diagnose("InvalidParameterOverride",
                             "module '" + instantiation.module_type +
                                 "' has no overridable parameter named '" + override.name + "'",
                             override.location);
                }
            }
            else {
                ++positional_parameter_count;
            }
            resolve_expression(override.value.get(), scope);
        }
        if (positional_parameter_count > parameters.size())
            diagnose("TooManyParameterOverrides",
                     "too many positional parameter overrides for module '" +
                         instantiation.module_type + "'",
                     instantiation.location);

        for (const auto& instance : instantiation.instances) {
            const auto* instance_symbol = local_symbol(scope, instance.name);
            if (instance_symbol && instance_symbol->kind == VerilogSemanticSymbolKind::Instance) {
                auto* mutable_instance =
                    scope.symbols_by_name.at(instance.name);
                mutable_instance->resolved_declaration = target;
            }
            for (const auto& dimension : instance.unpacked_dimensions)
                resolve_range(dimension, scope);
            for (const auto& connection : instance.connections) {
                if (connection.name) {
                    const auto found = std::find_if(
                        ports.begin(), ports.end(), [&](const auto* port) {
                            return port->name == *connection.name;
                        });
                    if (found == ports.end())
                        diagnose("InvalidPortConnection",
                                 "module '" + instantiation.module_type +
                                     "' has no port named '" + *connection.name + "'",
                                 connection.location);
                }
                resolve_expression(connection.expression.get(), scope);
            }
        }
    }

    void resolve_module(const DesignUnit& unit) {
        const auto& module = *unit.module;
        resolve_type_definitions(module.type_definitions, *unit.scope);
        for (const auto& parameter : module.parameters)
            resolve_parameter(parameter, *unit.scope);
        for (const auto& port : module.ports)
            resolve_port(port, *unit.scope);
        for (const auto& declaration : module.declarations)
            resolve_variable(declaration, *unit.scope);
        for (const auto& assignment : module.assignments) {
            resolve_expression(assignment.left.get(), *unit.scope);
            resolve_expression(assignment.right.get(), *unit.scope);
        }
        for (const auto& instantiation : module.instantiations)
            resolve_module_instantiation(instantiation, *unit.scope);
        for (const auto& block : module.procedural_blocks) {
            resolve_event_control(block.event_control.get(), *unit.scope);
            resolve_statement(block.body.get(), *unit.scope);
        }
        for (const auto& construct : module.generate_constructs)
            resolve_generate_construct(*construct, *unit.scope);
    }

    void resolve_generate_scope_contents(const VerilogGenerateScope& generate_scope,
                                         const VerilogSemanticScope& scope) {
        for (const auto& parameter : generate_scope.parameters)
            resolve_parameter(parameter, scope);
        resolve_type_definitions(generate_scope.type_definitions, scope);
        for (const auto& declaration : generate_scope.declarations)
            resolve_variable(declaration, scope);
        for (const auto& assignment : generate_scope.assignments) {
            resolve_expression(assignment.left.get(), scope);
            resolve_expression(assignment.right.get(), scope);
        }
        for (const auto& instantiation : generate_scope.instantiations)
            resolve_module_instantiation(instantiation, scope);
        for (const auto& construct : generate_scope.constructs)
            resolve_generate_construct(*construct, scope);
    }

    void resolve_type_definitions(const std::vector<VerilogTypeDefinition>& definitions,
                                  const VerilogSemanticScope& scope) {
        for (const auto& definition : definitions) {
            for (const auto& dimension : definition.packed_dimensions) {
                resolve_expression(dimension.left.get(), scope);
                resolve_expression(dimension.right.get(), scope);
            }
            for (const auto& dimension : definition.unpacked_dimensions) {
                resolve_expression(dimension.left.get(), scope);
                resolve_expression(dimension.right.get(), scope);
            }
            for (const auto& enumerator : definition.enumerators)
                resolve_expression(enumerator.value.get(), scope);
            for (const auto& member : definition.members) {
                for (const auto& dimension : member.packed_dimensions) {
                    resolve_expression(dimension.left.get(), scope);
                    resolve_expression(dimension.right.get(), scope);
                }
                for (const auto& dimension : member.unpacked_dimensions) {
                    resolve_expression(dimension.left.get(), scope);
                    resolve_expression(dimension.right.get(), scope);
                }
            }
        }
    }

    void resolve_generate_construct(const VerilogGenerateConstruct& construct,
                                    const VerilogSemanticScope& scope) {
        switch (construct.kind) {
            case VerilogGenerateKind::Region:
                if (construct.region_body)
                    resolve_generate_scope_contents(*construct.region_body, scope);
                break;
            case VerilogGenerateKind::If:
                resolve_expression(construct.condition.get(), scope);
                if (construct.true_body)
                    resolve_generate_scope_contents(
                        *construct.true_body, *generate_scopes_.at(construct.true_body.get()));
                if (construct.false_body)
                    resolve_generate_scope_contents(
                        *construct.false_body, *generate_scopes_.at(construct.false_body.get()));
                break;
            case VerilogGenerateKind::Case:
                resolve_expression(construct.condition.get(), scope);
                for (const auto& item : construct.case_items) {
                    for (const auto& expression : item.expressions)
                        resolve_expression(expression.get(), scope);
                    if (item.body)
                        resolve_generate_scope_contents(
                            *item.body, *generate_scopes_.at(item.body.get()));
                }
                break;
            case VerilogGenerateKind::For: {
                const auto found = generate_construct_scopes_.find(&construct);
                const auto& loop_scope = found == generate_construct_scopes_.end()
                                             ? scope
                                             : *found->second;
                resolve_expression(construct.initialization.get(), loop_scope);
                resolve_expression(construct.loop_condition.get(), loop_scope);
                resolve_expression(construct.step.get(), loop_scope);
                if (construct.loop_body)
                    resolve_generate_scope_contents(*construct.loop_body, loop_scope);
                break;
            }
        }
    }

    void resolve_design_units() {
        for (const auto& unit : units_) {
            if (unit.module) {
                resolve_module(unit);
            }
            else if (unit.interface) {
                resolve_type_definitions(unit.interface->type_definitions, *unit.scope);
                for (const auto& parameter : unit.interface->parameters)
                    resolve_parameter(parameter, *unit.scope);
                for (const auto& port : unit.interface->ports)
                    resolve_port(port, *unit.scope);
                for (const auto& declaration : unit.interface->declarations)
                    resolve_variable(declaration, *unit.scope);
            }
            else if (unit.package) {
                resolve_type_definitions(unit.package->type_definitions, *unit.scope);
                for (const auto& parameter : unit.package->parameters)
                    resolve_parameter(parameter, *unit.scope);
                for (const auto& declaration : unit.package->declarations)
                    resolve_variable(declaration, *unit.scope);
            }
        }
    }

    const VerilogCompilationUnit& syntax_;
    VerilogSemanticAnalysisResult result_;
    std::vector<DesignUnit> units_;
    std::unordered_map<const VerilogGenerateScope*, VerilogSemanticScope*> generate_scopes_;
    std::unordered_map<const VerilogGenerateConstruct*, VerilogSemanticScope*>
        generate_construct_scopes_;
    std::unordered_map<const void*, VerilogSemanticScope*> callable_scopes_;
};

} // namespace

const VerilogSemanticSymbol* lookup_verilog_symbol(const VerilogSemanticScope& scope,
                                                   std::string_view name,
                                                   bool include_parent_scopes) {
    for (auto* current = &scope; current; current = include_parent_scopes ? current->parent : nullptr) {
        const auto found = current->symbols_by_name.find(std::string(name));
        if (found != current->symbols_by_name.end())
            return found->second;
    }
    return nullptr;
}

VerilogSemanticAnalysisResult analyze_verilog_semantics(const VerilogCompilationUnit& syntax) {
    return Analyzer(syntax).run();
}

} // namespace labneura::eda
