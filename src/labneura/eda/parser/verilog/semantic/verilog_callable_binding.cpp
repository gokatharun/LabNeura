#include "labneura/eda/verilog_callable_binding.h"

#include <algorithm>
#include <utility>

namespace labneura::eda {
namespace {

void diagnose(VerilogCallableBindingResult& result, std::string code,
              std::string message, const VerilogSyntaxLocation& location,
              const VerilogSemanticSymbol* formal = nullptr,
              VerilogTypeDiagnosticSeverity severity =
                  VerilogTypeDiagnosticSeverity::Error) {
    result.diagnostics.push_back(
        {severity, std::move(code), std::move(message), location,
         formal ? std::optional<VerilogSyntaxLocation>(formal->declaration_location)
                : std::nullopt});
    if (severity == VerilogTypeDiagnosticSeverity::Error)
        result.is_valid = false;
}

bool is_lvalue(const VerilogExpression& expression) {
    switch (expression.kind) {
        case VerilogExpressionKind::Identifier:
            return true;
        case VerilogExpressionKind::MemberAccess:
        case VerilogExpressionKind::Select:
            return expression.first && is_lvalue(*expression.first);
        case VerilogExpressionKind::Concatenation:
            return !expression.operands.empty() &&
                   std::all_of(expression.operands.begin(), expression.operands.end(),
                               [](const auto& operand) {
                                   return operand && is_lvalue(*operand);
                               });
        default:
            return false;
    }
}

template <typename Expression>
std::vector<std::optional<std::string>> get_operand_names(const Expression& expression) {
    std::vector<std::optional<std::string>> names(expression.operands.size());
    if constexpr (requires { expression.operand_names; }) {
        const auto count = std::min(names.size(), expression.operand_names.size());
        for (size_t i = 0; i < count; ++i)
            names[i] = expression.operand_names[i];
    }
    return names;
}

bool is_function(const VerilogSemanticSymbol& callable) {
    return callable.kind == VerilogSemanticSymbolKind::Function;
}

const VerilogIntegralType* find_formal_type(
    const VerilogSemanticSymbol* formal,
    const VerilogTypeAnalysisResult& type_analysis) {
    if (!formal)
        return nullptr;
    const auto found = type_analysis.symbol_types.find(formal);
    return found == type_analysis.symbol_types.end() ? nullptr : &found->second;
}

const VerilogIntegralType* find_actual_type(
    const VerilogExpression* actual,
    const VerilogTypeAnalysisResult& type_analysis) {
    if (!actual)
        return nullptr;
    const auto found = type_analysis.expression_types.find(actual);
    return found == type_analysis.expression_types.end() ? nullptr : &found->second;
}

bool known_integral(const VerilogIntegralType* type) {
    return type && type->is_known_integral();
}

bool dimension_mismatch(const std::vector<VerilogPackedDimensionType>& lhs,
                        const std::vector<VerilogPackedDimensionType>& rhs) {
    if (lhs.size() != rhs.size())
        return true;
    for (size_t i = 0; i < lhs.size(); ++i) {
        if (lhs[i].left && rhs[i].left && lhs[i].left != rhs[i].left)
            return true;
        if (lhs[i].right && rhs[i].right && lhs[i].right != rhs[i].right)
            return true;
        if (lhs[i].width && rhs[i].width && lhs[i].width != rhs[i].width)
            return true;
    }
    return false;
}

bool ref_type_mismatch(const VerilogIntegralType& expected,
                       const VerilogIntegralType& supplied) {
    if (expected.category == VerilogTypeCategory::Unknown ||
        supplied.category == VerilogTypeCategory::Unknown)
        return false;
    if (expected.category != supplied.category)
        return true;
    if (expected.bit_width && supplied.bit_width &&
        expected.bit_width != supplied.bit_width)
        return true;
    if (expected.is_signed != supplied.is_signed ||
        expected.four_state != supplied.four_state)
        return true;
    if (dimension_mismatch(expected.packed_dimensions, supplied.packed_dimensions) ||
        dimension_mismatch(expected.unpacked_dimensions, supplied.unpacked_dimensions))
        return true;
    if (expected.unpacked_element_count && supplied.unpacked_element_count &&
        expected.unpacked_element_count != supplied.unpacked_element_count)
        return true;

    const bool identity_bearing =
        expected.category == VerilogTypeCategory::Enumeration ||
        expected.category == VerilogTypeCategory::Structure ||
        expected.category == VerilogTypeCategory::Union;
    if (identity_bearing && !expected.identity_name.empty() &&
        !supplied.identity_name.empty() &&
        expected.identity_name != supplied.identity_name)
        return true;
    if (identity_bearing && expected.packed != supplied.packed)
        return true;

    // Arrays and nested aggregate members may carry resolved element types.
    // Recurse only when both sides expose that information; one missing
    // element type remains unknown instead of being guessed incompatible.
    return expected.element_type && supplied.element_type &&
           ref_type_mismatch(*expected.element_type, *supplied.element_type);
}

void check_type_compatibility(VerilogCallableBindingResult& result,
                              const VerilogSemanticSymbol& callable,
                              const VerilogSemanticSymbol& formal,
                              const VerilogExpression* actual,
                              const VerilogTypeAnalysisResult& type_analysis) {
    const auto direction = formal.argument_direction.value_or(VerilogPortDirection::Unknown);
    const auto* expected = find_formal_type(&formal, type_analysis);
    const auto* supplied = find_actual_type(actual, type_analysis);

    if (direction == VerilogPortDirection::Ref) {
        // Unknown type information is deliberately left unknown; no guessed
        // compatibility decision or conversion is performed.
        if (!expected || !supplied ||
            expected->category == VerilogTypeCategory::Unknown ||
            supplied->category == VerilogTypeCategory::Unknown)
            return;
        if (ref_type_mismatch(*expected, *supplied)) {
            diagnose(result, "CallableRefTypeMismatch",
                     "ref actual for formal '" + formal.name +
                         "' must match resolved type identity, category, dimensions, "
                         "width, signedness, and four-state status",
                     actual ? actual->location : formal.declaration_location, &formal);
        }
        return;
    }

    if (direction != VerilogPortDirection::Input &&
        direction != VerilogPortDirection::Output &&
        direction != VerilogPortDirection::Inout)
        return;
    if (!known_integral(expected) || !known_integral(supplied))
        return;
    if (expected->bit_width != supplied->bit_width) {
        diagnose(result, "CallableArgumentWidthConversion",
                 "argument for formal '" + formal.name + "' of callable '" +
                     callable.name + "' converts " +
                     std::to_string(*supplied->bit_width) + "-bit actual to " +
                     std::to_string(*expected->bit_width) + "-bit formal",
                 actual ? actual->location : formal.declaration_location, &formal,
                 VerilogTypeDiagnosticSeverity::Warning);
    }
    if (expected->is_signed != supplied->is_signed) {
        diagnose(result, "CallableArgumentSignednessConversion",
                 "argument for formal '" + formal.name + "' of callable '" +
                     callable.name +
                     "' converts between signed and unsigned integral types",
                 actual ? actual->location : formal.declaration_location, &formal,
                 VerilogTypeDiagnosticSeverity::Warning);
    }
}

template <typename CallableRange>
const VerilogCallableDeclaration* find_in_callables(
    const CallableRange& declarations, const VerilogSemanticSymbol& callable) {
    const bool semantic_is_task = callable.kind == VerilogSemanticSymbolKind::Task;
    const auto found = std::find_if(
        declarations.begin(), declarations.end(), [&](const auto& declaration) {
            const bool syntax_is_task = declaration.kind == VerilogCallableKind::Task;
            return declaration.name == callable.name &&
                   syntax_is_task == semantic_is_task &&
                   declaration.location.source == callable.declaration_location.source &&
                   declaration.location.byte_offset ==
                       callable.declaration_location.byte_offset;
        });
    return found == declarations.end() ? nullptr : &*found;
}

} // namespace

VerilogCallableBindingResult bind_verilog_callable_arguments(
    const VerilogSemanticSymbol& callable, const VerilogExpression& call,
    const VerilogTypeAnalysisResult& type_analysis) {
    VerilogCallableBindingResult result;
    if (call.kind != VerilogExpressionKind::Call) {
        diagnose(result, "NotCallableExpression",
                 "callable argument binding requires a call expression", call.location);
        return result;
    }
    if (callable.kind != VerilogSemanticSymbolKind::Function &&
        callable.kind != VerilogSemanticSymbolKind::Task) {
        diagnose(result, "NotCallableSymbol",
                 "symbol '" + callable.name + "' is not a function or task",
                 call.location, nullptr);
        return result;
    }

    const std::vector<const VerilogSemanticSymbol*> empty_formals;
    const auto& formals = callable.callable
                              ? callable.callable->formal_arguments
                              : empty_formals;
    result.arguments.reserve(formals.size());
    for (size_t index = 0; index < formals.size(); ++index)
        result.arguments.push_back({index, formals[index], nullptr});

    const auto names = get_operand_names(call);
    const bool has_named = std::any_of(names.begin(), names.end(),
                                       [](const auto& name) { return name.has_value(); });
    const bool has_positional = std::any_of(names.begin(), names.end(),
                                            [](const auto& name) { return !name; });
    if (has_named && has_positional)
        diagnose(result, "MixedCallableArgumentStyles",
                 "call may not mix positional and named actual arguments", call.location);

    if (!has_named) {
        const size_t count = std::min(call.operands.size(), formals.size());
        for (size_t index = 0; index < count; ++index)
            result.arguments[index].actual = call.operands[index].get();
        if (call.operands.size() > formals.size()) {
            diagnose(result, "ExcessPositionalCallableArguments",
                     "call to '" + callable.name + "' has excess positional arguments",
                     call.operands[formals.size()]
                         ? call.operands[formals.size()]->location
                         : call.location);
        }
    } else {
        std::vector<bool> assigned(formals.size(), false);
        for (size_t actual_index = 0; actual_index < names.size(); ++actual_index) {
            const auto& name = names[actual_index];
            const auto* actual = call.operands[actual_index].get();
            if (!name) {
                diagnose(result, "MixedCallableArgumentStyles",
                         "call may not mix positional and named actual arguments",
                         actual ? actual->location : call.location);
                continue;
            }
            auto formal_it = std::find_if(formals.begin(), formals.end(),
                                          [&](const auto* formal) {
                                              return formal && formal->name == *name;
                                          });
            if (formal_it == formals.end()) {
                diagnose(result, "UnexpectedNamedCallableArgument",
                         "call to '" + callable.name + "' names unknown formal '" +
                             *name + "'",
                         actual ? actual->location : call.location);
                continue;
            }
            const size_t formal_index =
                static_cast<size_t>(std::distance(formals.begin(), formal_it));
            if (assigned[formal_index]) {
                diagnose(result, "DuplicateNamedCallableArgument",
                         "formal '" + *name + "' is specified more than once",
                         actual ? actual->location : call.location,
                         formals[formal_index]);
                continue;
            }
            assigned[formal_index] = true;
            result.arguments[formal_index].actual = actual;
        }
    }

    for (auto& binding : result.arguments) {
        const auto* formal = binding.formal;
        if (!formal)
            continue;
        const auto direction =
            formal->argument_direction.value_or(VerilogPortDirection::Unknown);
        if (is_function(callable) && direction != VerilogPortDirection::Input) {
            diagnose(result, "InvalidFunctionFormalDirection",
                     "function formal '" + formal->name + "' must have input direction",
                     formal->declaration_location, formal);
        }
        if (!binding.actual) {
            diagnose(result, "MissingCallableArgument",
                     "required formal '" + formal->name + "' of callable '" +
                         callable.name + "' has no actual argument",
                     call.location, formal);
            continue;
        }

        if (direction == VerilogPortDirection::Output ||
            direction == VerilogPortDirection::Inout ||
            direction == VerilogPortDirection::Ref) {
            if (!is_lvalue(*binding.actual)) {
                const std::string direction_name =
                    direction == VerilogPortDirection::Output
                        ? "output"
                        : direction == VerilogPortDirection::Inout ? "inout" : "ref";
                diagnose(result, "InvalidCallableArgumentLValue",
                         "actual for " + direction_name + " formal '" + formal->name +
                             "' must be an assignable expression",
                         binding.actual->location, formal);
            }
        }
        check_type_compatibility(result, callable, *formal, binding.actual,
                                 type_analysis);
    }
    return result;
}

const VerilogCallableDeclaration* find_verilog_callable_declaration(
    const VerilogCompilationUnit& syntax, const VerilogSemanticSymbol& callable) {
    for (const auto& module : syntax.modules)
        if (const auto* found = find_in_callables(module.callables, callable))
            return found;
    for (const auto& interface : syntax.interfaces)
        if (const auto* found = find_in_callables(interface.callables, callable))
            return found;
    for (const auto& package : syntax.packages)
        if (const auto* found = find_in_callables(package.callables, callable))
            return found;
    return nullptr;
}

} // namespace labneura::eda
