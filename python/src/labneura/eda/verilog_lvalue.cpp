#include "labneura/eda/verilog_assignment.h"

#include <algorithm>
#include <limits>
#include <string>

namespace labneura::eda {
namespace {

using Target = VerilogResolvedAssignmentTarget;
using Destination = VerilogResolvedAssignmentDestination;
using Result = VerilogLvalueResolutionResult;

void diagnose(Result& result, std::string code, std::string message,
              const VerilogSyntaxLocation& location) {
    result.diagnostics.push_back(
        {std::move(code), std::move(message), location, std::nullopt});
}

const VerilogSemanticSymbol* identifier_symbol(
    const VerilogExpression& expression,
    const VerilogSemanticAnalysisResult& semantic) {
    for (const auto& entry : semantic.references_at(expression.location)) {
        const auto& reference = semantic.references[entry.reference_ordinal];
        if (reference.declaration)
            return reference.declaration;
    }
    const auto named = std::find_if(
        semantic.references.begin(), semantic.references.end(),
        [&](const VerilogSemanticReference& reference) {
            return reference.name == expression.text && reference.declaration;
        });
    return named == semantic.references.end() ? nullptr : named->declaration;
}

bool is_constant_select_expression(
    const VerilogExpression& expression,
    const VerilogSemanticAnalysisResult& semantic) {
    switch (expression.kind) {
        case VerilogExpressionKind::Literal:
            return true;
        case VerilogExpressionKind::Identifier: {
            const auto* symbol = identifier_symbol(expression, semantic);
            return symbol &&
                   (symbol->kind == VerilogSemanticSymbolKind::Parameter ||
                    symbol->kind == VerilogSemanticSymbolKind::LocalParameter ||
                    symbol->kind == VerilogSemanticSymbolKind::EnumValue);
        }
        case VerilogExpressionKind::Unary:
            return expression.first &&
                   is_constant_select_expression(*expression.first, semantic);
        case VerilogExpressionKind::Binary:
            return expression.first && expression.second &&
                   is_constant_select_expression(*expression.first, semantic) &&
                   is_constant_select_expression(*expression.second, semantic);
        case VerilogExpressionKind::Conditional:
            return expression.first && expression.second && expression.third &&
                   is_constant_select_expression(*expression.first, semantic) &&
                   is_constant_select_expression(*expression.second, semantic) &&
                   is_constant_select_expression(*expression.third, semantic);
        case VerilogExpressionKind::Call:
            if (!expression.first ||
                expression.first->kind != VerilogExpressionKind::Identifier ||
                expression.first->text != "$clog2")
                return false;
            return std::all_of(
                expression.operands.begin(), expression.operands.end(),
                [&](const auto& operand) {
                    return operand &&
                           is_constant_select_expression(*operand, semantic);
                });
        default:
            return false;
    }
}

struct LvalueIndex {
    bool negative = false;
    uint64_t magnitude = 0;
};

std::optional<LvalueIndex> evaluate_index(
    const VerilogExpression* expression,
    const VerilogAssignmentExpressionEvaluator& evaluate,
    Result& result) {
    if (!expression || !evaluate) {
        diagnose(result, "UnsupportedLvalueIndex",
                 "lvalue index expression cannot be evaluated in this execution context",
                 expression ? expression->location : VerilogSyntaxLocation{});
        return std::nullopt;
    }
    auto value = evaluate(*expression);
    if (!value) {
        diagnose(result, "UnsupportedLvalueIndex",
                 "lvalue index expression could not be evaluated",
                 expression->location);
        return std::nullopt;
    }
    if (!value->is_fully_known()) {
        diagnose(result, "UnknownLvalueIndex",
                 "lvalue index contains X or Z and cannot select a destination",
                 expression->location);
        return std::nullopt;
    }
    if (value->is_signed) {
        const auto index = value->to_int64();
        if (!index) {
            diagnose(result, "LvalueIndexOverflow",
                     "signed lvalue index is outside the supported 64-bit range",
                     expression->location);
            return std::nullopt;
        }
        if (*index < 0)
            return LvalueIndex{true, uint64_t{0} - static_cast<uint64_t>(*index)};
        return LvalueIndex{false, static_cast<uint64_t>(*index)};
    }
    const auto index = value->to_uint64();
    if (!index) {
        diagnose(result, "LvalueIndexOverflow",
                 "unsigned lvalue index is outside the supported 64-bit range",
                 expression->location);
        return std::nullopt;
    }
    return LvalueIndex{false, *index};
}

int compare_index(const LvalueIndex& lhs, const LvalueIndex& rhs) {
    if (lhs.negative != rhs.negative)
        return lhs.negative ? -1 : 1;
    if (lhs.magnitude == rhs.magnitude)
        return 0;
    if (lhs.negative)
        return lhs.magnitude > rhs.magnitude ? -1 : 1;
    return lhs.magnitude < rhs.magnitude ? -1 : 1;
}

std::optional<uint64_t> index_distance(const LvalueIndex& lhs,
                                       const LvalueIndex& rhs) {
    if (lhs.negative == rhs.negative)
        return lhs.magnitude >= rhs.magnitude ? lhs.magnitude - rhs.magnitude
                                              : rhs.magnitude - lhs.magnitude;
    if (lhs.magnitude > std::numeric_limits<uint64_t>::max() - rhs.magnitude)
        return std::nullopt;
    return lhs.magnitude + rhs.magnitude;
}

bool step_index(LvalueIndex& index, bool upward) {
    if (upward) {
        if (index.negative) {
            if (index.magnitude == 1) {
                index.negative = false;
                index.magnitude = 0;
            } else {
                --index.magnitude;
            }
            return true;
        }
        if (index.magnitude == std::numeric_limits<uint64_t>::max())
            return false;
        ++index.magnitude;
        return true;
    }
    if (index.negative) {
        if (index.magnitude == std::numeric_limits<uint64_t>::max())
            return false;
        ++index.magnitude;
        return true;
    }
    if (index.magnitude == 0) {
        index.negative = true;
        index.magnitude = 1;
    } else {
        --index.magnitude;
    }
    return true;
}

LvalueIndex index_from_signed(int64_t value) {
    if (value < 0)
        return {true, uint64_t{0} - static_cast<uint64_t>(value)};
    return {false, static_cast<uint64_t>(value)};
}

std::optional<size_t> checked_distance(const LvalueIndex& lhs,
                                       const LvalueIndex& rhs) {
    const auto distance = index_distance(lhs, rhs);
    if (!distance || *distance >= std::numeric_limits<size_t>::max())
        return std::nullopt;
    return static_cast<size_t>(*distance + 1);
}

std::optional<size_t> physical_bit(const LvalueIndex& index, int64_t left,
                                   int64_t right, size_t width) {
    const auto left_index = index_from_signed(left);
    const auto right_index = index_from_signed(right);
    const auto lower = compare_index(left_index, right_index) < 0
                           ? left_index
                           : right_index;
    const auto upper = compare_index(left_index, right_index) > 0
                           ? left_index
                           : right_index;
    if (compare_index(index, lower) < 0 || compare_index(index, upper) > 0)
        return std::nullopt;
    const auto distance = index_distance(index, right_index);
    if (!distance || *distance >= width)
        return std::nullopt;
    return static_cast<size_t>(*distance);
}

bool set_full_target(const VerilogExpression& expression,
                     const VerilogSemanticAnalysisResult& semantic,
                     const VerilogTypeAnalysisResult& types, Target& target,
                     Result& result) {
    if (expression.kind != VerilogExpressionKind::Identifier) {
        diagnose(result, "UnsupportedLvalueTarget",
                 "lvalue base must be a simple identifier in the supported subset",
                 expression.location);
        return false;
    }
    const auto* symbol = identifier_symbol(expression, semantic);
    if (!symbol) {
        diagnose(result, "UnresolvedLvalueTarget",
                 "lvalue identifier has no resolved semantic symbol", expression.location);
        return false;
    }
    switch (symbol->kind) {
        case VerilogSemanticSymbolKind::Variable:
        case VerilogSemanticSymbolKind::Net:
        case VerilogSemanticSymbolKind::Port:
        case VerilogSemanticSymbolKind::CallableArgument:
        case VerilogSemanticSymbolKind::Function:
            break;
        default:
            diagnose(result, "NonAssignableLvalueTarget",
                     "resolved symbol is not assignable", expression.location);
            return false;
    }
    const auto type_entry = types.symbol_types.find(symbol);
    if (type_entry == types.symbol_types.end() ||
        !type_entry->second.is_known_integral()) {
        diagnose(result, "UnsupportedLvalueType",
                 "lvalue target does not have a known integral declared type",
                 expression.location);
        return false;
    }
    const auto& type = type_entry->second;
    if (type.unpacked_element_count || !type.unpacked_dimensions.empty() ||
        type.category == VerilogTypeCategory::Array) {
        diagnose(result, "UnsupportedUnpackedLvalue",
                 "unpacked array lvalue targets are not supported", expression.location);
        return false;
    }
    if (type.packed_dimensions.size() > 1) {
        diagnose(result, "UnsupportedMultidimensionalPackedLvalue",
                 "packed lvalues with multiple declared dimensions are not supported",
                 expression.location);
        return false;
    }
    const size_t width = *type.bit_width;
    if (width == 0) {
        diagnose(result, "InvalidLvalueWidth", "lvalue target width must be nonzero",
                 expression.location);
        return false;
    }
    Destination destination;
    destination.symbol = symbol;
    destination.storage_name = symbol->name;
    destination.declared_type = type;
    destination.bit_positions.reserve(width);
    for (size_t bit = 0; bit < width; ++bit)
        destination.bit_positions.push_back(bit);
    target.bit_width = width;
    target.is_integral = true;
    target.is_signed = type.is_signed;
    target.four_state = type.four_state;
    target.destinations.push_back(std::move(destination));
    return true;
}

bool dimension_bounds(const VerilogIntegralType& type, int64_t& left,
                      int64_t& right) {
    if (!type.packed_dimensions.empty()) {
        const auto& dimension = type.packed_dimensions.front();
        if (!dimension.left || !dimension.right)
            return false;
        left = *dimension.left;
        right = *dimension.right;
        return true;
    }
    if (!type.bit_width || *type.bit_width == 0 ||
        *type.bit_width - 1 > static_cast<size_t>(std::numeric_limits<int64_t>::max()))
        return false;
    left = static_cast<int64_t>(*type.bit_width - 1);
    right = 0;
    return true;
}

bool resolve_atomic(const VerilogExpression& expression,
                    const VerilogSemanticAnalysisResult& semantic,
                    const VerilogTypeAnalysisResult& types,
                    const VerilogAssignmentExpressionEvaluator& evaluate,
                    Target& target, Result& result) {
    if (expression.kind == VerilogExpressionKind::Identifier)
        return set_full_target(expression, semantic, types, target, result);
    if (expression.kind != VerilogExpressionKind::Select || !expression.first ||
        !expression.second) {
        diagnose(result, "UnsupportedLvalueTarget",
                 "lvalue must be an identifier, supported packed select, or concatenation",
                 expression.location);
        return false;
    }
    if (expression.first->kind != VerilogExpressionKind::Identifier) {
        diagnose(result, "UnsupportedSelectedLvalueBase",
                 "selected lvalue base must be a simple identifier", expression.location);
        return false;
    }
    Target base;
    if (!set_full_target(*expression.first, semantic, types, base, result))
        return false;
    auto& base_destination = base.destinations.front();
    int64_t declared_left = 0;
    int64_t declared_right = 0;
    if (!dimension_bounds(base_destination.declared_type, declared_left,
                          declared_right)) {
        diagnose(result, "UnknownLvalueDeclaredRange",
                 "packed select requires a known declared range", expression.location);
        return false;
    }

    std::vector<std::optional<size_t>> positions;
    if (expression.select_kind == VerilogSelectKind::Bit) {
        const auto index = evaluate_index(expression.second.get(), evaluate, result);
        if (!index)
            return false;
        positions.push_back(physical_bit(*index, declared_left, declared_right,
                                         base_destination.bit_positions.size()));
    } else if (expression.select_kind == VerilogSelectKind::Part) {
        if (!expression.third) {
            diagnose(result, "InvalidLvaluePartSelect",
                     "constant part-select is missing its right bound", expression.location);
            return false;
        }
        if (!is_constant_select_expression(*expression.second, semantic) ||
            !is_constant_select_expression(*expression.third, semantic)) {
            diagnose(result, "UnsupportedLvalueSelectBounds",
                     "packed part-select bounds must be constant expressions",
                     expression.location);
            return false;
        }
        const auto left = evaluate_index(expression.second.get(), evaluate, result);
        const auto right = evaluate_index(expression.third.get(), evaluate, result);
        if (!left || !right)
            return false;
        const auto count = checked_distance(*left, *right);
        if (!count || *count > 65536) {
            diagnose(result, "InvalidLvalueSelectWidth",
                     "part-select width is invalid or exceeds the supported limit",
                     expression.location);
            return false;
        }
        positions.reserve(*count);
        LvalueIndex index = *right;
        const bool upward = compare_index(*left, *right) >= 0;
        for (size_t bit = 0; bit < *count; ++bit) {
            positions.push_back(physical_bit(index, declared_left, declared_right,
                                             base_destination.bit_positions.size()));
            if (bit + 1 < *count) {
                if (!step_index(index, upward)) {
                    diagnose(result, "LvalueSelectIndexOverflow",
                             "part-select index arithmetic overflowed", expression.location);
                    return false;
                }
            }
        }
    } else {
        if (!expression.third) {
            diagnose(result, "InvalidLvalueIndexedPartSelect",
                     "indexed part-select is missing its width", expression.location);
            return false;
        }
        if (!is_constant_select_expression(*expression.third, semantic)) {
            diagnose(result, "UnsupportedLvalueSelectWidth",
                     "indexed part-select width must be a constant expression",
                     expression.third->location);
            return false;
        }
        const auto base_index = evaluate_index(expression.second.get(), evaluate, result);
        const auto width = evaluate_index(expression.third.get(), evaluate, result);
        if (!base_index || !width)
            return false;
        if (width->negative || width->magnitude == 0 || width->magnitude > 65536) {
            diagnose(result, "InvalidLvalueSelectWidth",
                     "indexed part-select width must be positive and within the supported limit",
                     expression.third->location);
            return false;
        }
        const size_t count = static_cast<size_t>(width->magnitude);
        positions.reserve(count);
        LvalueIndex index = *base_index;
        const bool upward = expression.select_kind == VerilogSelectKind::IndexedUp;
        if (!upward)
            for (size_t bit = 1; bit < count; ++bit)
                if (!step_index(index, false)) {
                    diagnose(result, "LvalueSelectIndexOverflow",
                             "indexed part-select range arithmetic overflowed",
                             expression.location);
                    return false;
                }
        for (size_t bit = 0; bit < count; ++bit) {
            positions.push_back(physical_bit(index, declared_left, declared_right,
                                             base_destination.bit_positions.size()));
            if (bit + 1 < count && !step_index(index, true)) {
                diagnose(result, "LvalueSelectIndexOverflow",
                         "indexed part-select range arithmetic overflowed",
                         expression.location);
                return false;
            }
        }
    }
    base_destination.target_lsb_offset = 0;
    base_destination.bit_positions = std::move(positions);
    target.bit_width = base_destination.bit_positions.size();
    target.is_integral = true;
    target.is_signed = false;
    target.four_state = base_destination.declared_type.four_state;
    target.destinations = std::move(base.destinations);
    return true;
}

bool resolve_into(const VerilogExpression& expression,
                  const VerilogSemanticAnalysisResult& semantic,
                  const VerilogTypeAnalysisResult& types,
                  const VerilogAssignmentExpressionEvaluator& evaluate,
                  Target& target, size_t offset, Result& result) {
    if (expression.kind == VerilogExpressionKind::Concatenation) {
        if (expression.operands.empty()) {
            diagnose(result, "EmptyLvalueConcatenation",
                     "lvalue concatenation must contain at least one operand",
                     expression.location);
            return false;
        }
        size_t total_width = 0;
        std::vector<Target> pieces;
        pieces.reserve(expression.operands.size());
        for (const auto& operand : expression.operands) {
            if (!operand) {
                diagnose(result, "InvalidLvalueConcatenation",
                         "lvalue concatenation contains a missing operand",
                         expression.location);
                return false;
            }
            Target piece;
            if (!resolve_into(*operand, semantic, types, evaluate, piece, 0, result))
                return false;
            if (piece.bit_width > std::numeric_limits<size_t>::max() - total_width) {
                diagnose(result, "LvalueWidthOverflow",
                         "lvalue concatenation width overflows size_t", expression.location);
                return false;
            }
            total_width += piece.bit_width;
            pieces.push_back(std::move(piece));
        }
        if (total_width > 65536) {
            diagnose(result, "LvalueWidthExceeded",
                     "lvalue concatenation exceeds the supported 65536-bit limit",
                     expression.location);
            return false;
        }
        size_t lsb_offset = offset;
        bool four_state = false;
        for (auto piece = pieces.rbegin(); piece != pieces.rend(); ++piece) {
            for (auto& destination : piece->destinations) {
                destination.target_lsb_offset += lsb_offset;
                four_state = four_state || destination.declared_type.four_state;
                target.destinations.push_back(std::move(destination));
            }
            lsb_offset += piece->bit_width;
        }
        target.bit_width = std::max(target.bit_width, lsb_offset);
        target.is_integral = true;
        target.is_signed = false;
        target.four_state = four_state;
        return true;
    }
    Target piece;
    if (!resolve_atomic(expression, semantic, types, evaluate, piece, result))
        return false;
    if (piece.bit_width > std::numeric_limits<size_t>::max() - offset) {
        diagnose(result, "LvalueWidthOverflow", "lvalue width overflows size_t",
                 expression.location);
        return false;
    }
    for (auto& destination : piece.destinations) {
        destination.target_lsb_offset += offset;
        target.destinations.push_back(std::move(destination));
    }
    target.bit_width = std::max(target.bit_width, offset + piece.bit_width);
    if (offset == 0 && target.destinations.size() == 1)
        target.is_integral = piece.is_integral;
    else
        target.is_integral = true;
    target.is_signed = target.is_integral && piece.is_signed;
    target.four_state = target.four_state || piece.four_state;
    return true;
}

} // namespace

VerilogLvalueResolutionResult resolve_verilog_lvalue(
    const VerilogExpression& expression,
    const VerilogSemanticAnalysisResult& semantic,
    const VerilogTypeAnalysisResult& type_analysis,
    const VerilogAssignmentExpressionEvaluator& evaluate_expression) {
    Result result;
    Target target;
    if (resolve_into(expression, semantic, type_analysis, evaluate_expression,
                     target, 0, result) && result.diagnostics.empty()) {
        target.location = expression.location;
        result.target = std::move(target);
    }
    return result;
}

} // namespace labneura::eda
