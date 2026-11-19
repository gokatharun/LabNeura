#include "verilog_native_lowering_internal.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>

namespace labneura::eda::detail {
namespace {

constexpr size_t kMaximumLoweredWidth = 4096;

const VerilogIntegralType* expression_type(
    const VerilogNativeLoweringSession& session,
    const VerilogExpression& expression) {
    const auto found = session.types.expression_types.find(&expression);
    if (found == session.types.expression_types.end())
        return nullptr;
    return &found->second;
}

bool valid_type(VerilogNativeLoweringSession& session,
                const VerilogExpression& expression,
                const VerilogIntegralType*& type) {
    type = expression_type(session, expression);
    if (!type || !type->is_known_integral() || !type->bit_width ||
        *type->bit_width == 0 || *type->bit_width > kMaximumLoweredWidth) {
        session.diagnose(
            "UnsupportedNativeExpressionType",
            "native expression lowering requires a known integral type no wider than 4096 bits",
            expression.location);
        return false;
    }
    return true;
}

std::optional<VerilogNativeLoweringValue> resize_value(
    VerilogNativeLoweringValue value, size_t width, bool is_signed,
    const VerilogSyntaxLocation& location,
    VerilogNativeLoweringSession& session) {
    if (width == 0 || width > kMaximumLoweredWidth || value.bits.empty()) {
        session.diagnose("UnsupportedNativeExpressionWidth",
                         "expression width is empty or exceeds native lowering limits",
                         location);
        return std::nullopt;
    }
    if (value.bits.size() > width) {
        value.bits.resize(width);
    } else if (value.bits.size() < width) {
        const std::string extension =
            is_signed && value.is_signed ? value.bits.back() : session.constant(false);
        value.bits.resize(width, extension);
    }
    value.is_signed = is_signed;
    return value;
}

std::optional<size_t> declared_index_to_position(
    VerilogNativeLoweringSession& session, const VerilogExpression& base,
    int64_t logical_index, size_t width) {
    int64_t left = 0;
    int64_t right = 0;
    bool has_range = false;
    if (base.kind == VerilogExpressionKind::Identifier) {
        const auto* symbol = session.resolve_symbol(base);
        if (symbol) {
            const auto found = session.types.symbol_types.find(symbol);
            if (found != session.types.symbol_types.end() &&
                !found->second.packed_dimensions.empty()) {
                const auto& dimension = found->second.packed_dimensions.front();
                if (dimension.left && dimension.right) {
                    left = *dimension.left;
                    right = *dimension.right;
                    has_range = true;
                }
            }
        }
    }
    if (!has_range) {
        if (width - 1 > static_cast<size_t>(std::numeric_limits<int64_t>::max()))
            return std::nullopt;
        left = static_cast<int64_t>(width - 1);
        right = 0;
    }

    int64_t position = 0;
    if (left >= right) {
        if (logical_index < right || logical_index > left)
            return std::nullopt;
        position = logical_index - right;
    } else {
        if (logical_index < left || logical_index > right)
            return std::nullopt;
        position = right - logical_index;
    }
    if (position < 0 || static_cast<uint64_t>(position) >= width)
        return std::nullopt;
    return static_cast<size_t>(position);
}

std::optional<int64_t> constant_integer(
    VerilogNativeLoweringSession& session,
    const VerilogExpression& expression) {
    const auto value = session.evaluate_constant(expression);
    if (!value || !value->is_fully_known())
        return std::nullopt;
    return value->to_int64();
}

std::optional<VerilogNativeLoweringValue> lower(
    VerilogNativeLoweringSession& session, const VerilogExpression& expression,
    const VerilogNativeSymbolResolver& resolver,
    std::optional<size_t> context_width = std::nullopt);

std::string reduce_bits(VerilogNativeLoweringSession& session,
                        const std::vector<std::string>& bits,
                        std::string_view operation) {
    if (bits.empty())
        return session.constant(false);
    if (operation == "&" || operation == "~&") {
        std::string result = bits.front();
        for (size_t index = 1; index < bits.size(); ++index)
            result = session.emit_binary("AND2", result, bits[index]);
        return operation == "~&" ? session.emit_not(result) : result;
    }
    if (operation == "|" || operation == "~|") {
        std::string result = bits.front();
        for (size_t index = 1; index < bits.size(); ++index)
            result = session.emit_binary("OR2", result, bits[index]);
        return operation == "~|" ? session.emit_not(result) : result;
    }
    std::string result = bits.front();
    for (size_t index = 1; index < bits.size(); ++index)
        result = session.emit_binary("XOR2", result, bits[index]);
    return operation == "~^" || operation == "^~"
               ? session.emit_not(result)
               : result;
}

std::string truth_value(VerilogNativeLoweringSession& session,
                        const VerilogNativeLoweringValue& value) {
    return reduce_bits(session, value.bits, "|");
}

std::optional<VerilogNativeLoweringValue> lower_constant(
    VerilogNativeLoweringSession& session,
    const VerilogExpression& expression, const VerilogIntegralType& type) {
    const auto evaluated = session.evaluate_constant(expression);
    if (!evaluated)
        return std::nullopt;
    if (!evaluated->is_fully_known()) {
        session.diagnose("FourStateConstantUnsupported",
                         "native Boolean netlists cannot represent X or Z constant bits",
                         expression.location);
        return std::nullopt;
    }

    VerilogNativeLoweringValue value;
    value.is_signed = evaluated->is_signed;
    value.bits.reserve(evaluated->bits.size());
    for (const auto bit : evaluated->bits)
        value.bits.push_back(session.constant(bit == VerilogLogicBit::One));
    return resize_value(std::move(value), *type.bit_width, type.is_signed,
                        expression.location, session);
}

std::optional<VerilogNativeLoweringValue> lower_select(
    VerilogNativeLoweringSession& session,
    const VerilogExpression& expression,
    const VerilogNativeSymbolResolver& resolver,
    const VerilogIntegralType& result_type) {
    if (!expression.first || !expression.second) {
        session.diagnose("MalformedNativeSelect",
                         "packed select is missing its base or index",
                         expression.location);
        return std::nullopt;
    }
    auto base = lower(session, *expression.first, resolver);
    if (!base)
        return std::nullopt;

    std::vector<std::string> selected;
    if (expression.select_kind == VerilogSelectKind::Bit) {
        const auto index = constant_integer(session, *expression.second);
        if (!index) {
            session.diagnose("DynamicNativeSelectUnsupported",
                             "dynamic bit selects cannot be represented by the current native lowering subset",
                             expression.second->location);
            return std::nullopt;
        }
        const auto position = declared_index_to_position(
            session, *expression.first, *index, base->bits.size());
        if (!position) {
            session.diagnose("OutOfRangeNativeSelect",
                             "out-of-range bit selects cannot be represented as a hardware bit",
                             expression.location);
            return std::nullopt;
        }
        selected.push_back(base->bits[*position]);
    } else {
        if (!expression.third) {
            session.diagnose("MalformedNativeSelect",
                             "part select is missing its second bound or width",
                             expression.location);
            return std::nullopt;
        }
        const auto first = constant_integer(session, *expression.second);
        const auto third = constant_integer(session, *expression.third);
        if (!first || !third) {
            session.diagnose("DynamicNativeSelectUnsupported",
                             "dynamic part-select bounds cannot be represented by the current native lowering subset",
                             expression.location);
            return std::nullopt;
        }

        int64_t low_bound = 0;
        size_t width = 0;
        if (expression.select_kind == VerilogSelectKind::Part) {
            const uint64_t distance = *first >= *third
                ? static_cast<uint64_t>(*first) - static_cast<uint64_t>(*third)
                : static_cast<uint64_t>(*third) - static_cast<uint64_t>(*first);
            if (distance >= kMaximumLoweredWidth) {
                session.diagnose("UnsupportedNativeSelectWidth",
                                 "part-select width exceeds native lowering limits",
                                 expression.location);
                return std::nullopt;
            }
            width = static_cast<size_t>(distance + 1);
            low_bound = *third;
        } else {
            if (*third <= 0 || static_cast<uint64_t>(*third) > kMaximumLoweredWidth) {
                session.diagnose("UnsupportedNativeSelectWidth",
                                 "indexed part-select width must be a positive constant within native limits",
                                 expression.third->location);
                return std::nullopt;
            }
            width = static_cast<size_t>(*third);
            if (expression.select_kind == VerilogSelectKind::IndexedUp) {
                low_bound = *first;
            } else {
                if (*first < std::numeric_limits<int64_t>::min() +
                                 static_cast<int64_t>(width) - 1) {
                    session.diagnose("NativeSelectIndexOverflow",
                                     "indexed part-select lower bound overflows signed range",
                                     expression.location);
                    return std::nullopt;
                }
                low_bound = *first - static_cast<int64_t>(width) + 1;
            }
        }
        if (width != *result_type.bit_width) {
            session.diagnose("NativeSelectTypeWidthMismatch",
                             "resolved select width disagrees with expression type analysis",
                             expression.location);
            return std::nullopt;
        }
        selected.reserve(width);
        for (size_t bit = 0; bit < width; ++bit) {
            if (bit > static_cast<size_t>(std::numeric_limits<int64_t>::max()) ||
                low_bound > std::numeric_limits<int64_t>::max() -
                                static_cast<int64_t>(bit)) {
                session.diagnose("NativeSelectIndexOverflow",
                                 "part-select index overflows signed range",
                                 expression.location);
                return std::nullopt;
            }
            const int64_t logical_index = low_bound + static_cast<int64_t>(bit);
            const auto position = declared_index_to_position(
                session, *expression.first, logical_index, base->bits.size());
            if (!position) {
                session.diagnose("OutOfRangeNativeSelect",
                                 "out-of-range part selects cannot be represented as hardware bits",
                                 expression.location);
                return std::nullopt;
            }
            selected.push_back(base->bits[*position]);
        }
    }
    return VerilogNativeLoweringValue{std::move(selected), result_type.is_signed};
}

std::optional<VerilogNativeLoweringValue> lower_concat(
    VerilogNativeLoweringSession& session,
    const VerilogExpression& expression,
    const VerilogNativeSymbolResolver& resolver,
    const VerilogIntegralType& result_type) {
    if (expression.operands.empty()) {
        session.diagnose("EmptyNativeConcatenation",
                         "native lowering does not accept an empty concatenation",
                         expression.location);
        return std::nullopt;
    }
    VerilogNativeLoweringValue result;
    for (auto operand = expression.operands.rbegin();
         operand != expression.operands.rend(); ++operand) {
        if (!*operand)
            return std::nullopt;
        auto value = lower(session, **operand, resolver);
        if (!value)
            return std::nullopt;
        if (value->bits.size() > kMaximumLoweredWidth - result.bits.size()) {
            session.diagnose("NativeConcatenationTooWide",
                             "concatenation exceeds native lowering width limits",
                             expression.location);
            return std::nullopt;
        }
        result.bits.insert(result.bits.end(), value->bits.begin(), value->bits.end());
    }
    if (result.bits.size() != *result_type.bit_width) {
        session.diagnose("NativeConcatenationTypeWidthMismatch",
                         "concatenation width disagrees with expression type analysis",
                         expression.location);
        return std::nullopt;
    }
    result.is_signed = result_type.is_signed;
    return result;
}

std::optional<VerilogNativeLoweringValue> lower_replication(
    VerilogNativeLoweringSession& session,
    const VerilogExpression& expression,
    const VerilogNativeSymbolResolver& resolver,
    const VerilogIntegralType& result_type) {
    if (!expression.first || expression.operands.empty()) {
        session.diagnose("MalformedNativeReplication",
                         "replication requires a repeat count and at least one operand",
                         expression.location);
        return std::nullopt;
    }
    const auto count = constant_integer(session, *expression.first);
    if (!count || *count < 0 ||
        static_cast<uint64_t>(*count) > kMaximumLoweredWidth) {
        session.diagnose("UnsupportedNativeReplicationCount",
                         "replication count must be a nonnegative known constant within native limits",
                         expression.first->location);
        return std::nullopt;
    }
    std::vector<std::string> unit;
    for (auto operand = expression.operands.rbegin();
         operand != expression.operands.rend(); ++operand) {
        if (!*operand)
            return std::nullopt;
        auto value = lower(session, **operand, resolver);
        if (!value)
            return std::nullopt;
        if (value->bits.size() > kMaximumLoweredWidth - unit.size()) {
            session.diagnose("NativeReplicationTooWide",
                             "replicated unit exceeds native lowering width limits",
                             expression.location);
            return std::nullopt;
        }
        unit.insert(unit.end(), value->bits.begin(), value->bits.end());
    }
    if (unit.empty() ||
        static_cast<uint64_t>(*count) >
            kMaximumLoweredWidth / unit.size()) {
        session.diagnose("NativeReplicationTooWide",
                         "replication exceeds native lowering width limits",
                         expression.location);
        return std::nullopt;
    }
    VerilogNativeLoweringValue result;
    result.bits.reserve(unit.size() * static_cast<size_t>(*count));
    for (int64_t repetition = 0; repetition < *count; ++repetition)
        result.bits.insert(result.bits.end(), unit.begin(), unit.end());
    if (result.bits.size() != *result_type.bit_width) {
        session.diagnose("NativeReplicationTypeWidthMismatch",
                         "replication width disagrees with expression type analysis",
                         expression.location);
        return std::nullopt;
    }
    result.is_signed = result_type.is_signed;
    return result;
}

std::optional<VerilogNativeLoweringValue> add_values(
    VerilogNativeLoweringSession& session,
    VerilogNativeLoweringValue lhs,
    VerilogNativeLoweringValue rhs, bool subtract, size_t width,
    bool is_signed, const VerilogSyntaxLocation& location) {
    auto left = resize_value(std::move(lhs), width, is_signed, location, session);
    auto right = resize_value(std::move(rhs), width, is_signed, location, session);
    if (!left || !right)
        return std::nullopt;
    if (subtract) {
        for (auto& bit : right->bits)
            bit = session.emit_not(bit);
    }
    std::string carry = session.constant(subtract);
    VerilogNativeLoweringValue result;
    result.is_signed = is_signed;
    result.bits.reserve(width);
    for (size_t index = 0; index < width; ++index) {
        const std::string propagate = session.emit_binary(
            "XOR2", left->bits[index], right->bits[index]);
        result.bits.push_back(session.emit_binary("XOR2", propagate, carry));
        const std::string generated = session.emit_binary(
            "AND2", left->bits[index], right->bits[index]);
        const std::string propagated = session.emit_binary(
            "AND2", propagate, carry);
        carry = session.emit_binary("OR2", generated, propagated);
    }
    return result;
}

std::string equality_value(VerilogNativeLoweringSession& session,
                           const VerilogNativeLoweringValue& lhs,
                           const VerilogNativeLoweringValue& rhs) {
    std::string equal = session.constant(true);
    for (size_t index = 0; index < lhs.bits.size(); ++index) {
        std::string bit_equal = session.emit_binary(
            "XOR2", lhs.bits[index], rhs.bits[index]);
        bit_equal = session.emit_not(bit_equal);
        equal = session.emit_binary("AND2", equal, bit_equal);
    }
    return equal;
}

std::string less_than_unsigned(VerilogNativeLoweringSession& session,
                               const VerilogNativeLoweringValue& lhs,
                               const VerilogNativeLoweringValue& rhs,
                               size_t width) {
    std::string less = session.constant(false);
    std::string equal = session.constant(true);
    for (size_t position = width; position != 0; --position) {
        const size_t index = position - 1;
        const std::string not_lhs = session.emit_not(lhs.bits[index]);
        const std::string bit_less = session.emit_binary(
            "AND2", not_lhs, rhs.bits[index]);
        const std::string newly_less = session.emit_binary(
            "AND2", equal, bit_less);
        less = session.emit_binary("OR2", less, newly_less);
        const std::string bit_equal = session.emit_not(session.emit_binary(
            "XOR2", lhs.bits[index], rhs.bits[index]));
        equal = session.emit_binary("AND2", equal, bit_equal);
    }
    return less;
}

std::optional<VerilogNativeLoweringValue> lower_binary(
    VerilogNativeLoweringSession& session, const VerilogExpression& expression,
    const VerilogNativeSymbolResolver& resolver,
    const VerilogIntegralType& result_type,
    std::optional<size_t> context_width) {
    if (!expression.first || !expression.second) {
        session.diagnose("MalformedNativeBinaryExpression",
                         "binary expression is missing an operand",
                         expression.location);
        return std::nullopt;
    }
    const auto& op = expression.text;
    const bool shift = op == "<<" || op == "<<<" ||
                       op == ">>" || op == ">>>";
    const bool context_sized =
        op == "&" || op == "|" || op == "^" || op == "~^" ||
        op == "^~" || op == "+" || op == "-" || shift;
    const size_t operation_width = context_sized && context_width
        ? std::max(*result_type.bit_width, *context_width)
        : *result_type.bit_width;
    const auto operand_context = context_sized
        ? std::optional<size_t>(operation_width) : std::nullopt;
    auto lhs = lower(session, *expression.first, resolver, operand_context);
    auto rhs = lower(session, *expression.second, resolver,
                     shift ? std::nullopt : operand_context);
    if (!lhs || !rhs)
        return std::nullopt;

    if (op == "&&" || op == "||") {
        const std::string left_truth = truth_value(session, *lhs);
        const std::string right_truth = truth_value(session, *rhs);
        const std::string result = session.emit_binary(
            op == "&&" ? "AND2" : "OR2", left_truth, right_truth);
        return VerilogNativeLoweringValue{{result}, result_type.is_signed};
    }

    if (op == "==" || op == "===" || op == "!=" || op == "!==") {
        const size_t width = std::max(lhs->bits.size(), rhs->bits.size());
        const bool comparison_signed = lhs->is_signed && rhs->is_signed;
        auto left = resize_value(std::move(*lhs), width, comparison_signed,
                                 expression.location, session);
        auto right = resize_value(std::move(*rhs), width, comparison_signed,
                                  expression.location, session);
        if (!left || !right)
            return std::nullopt;
        std::string result = equality_value(session, *left, *right);
        if (op == "!=" || op == "!==")
            result = session.emit_not(result);
        return VerilogNativeLoweringValue{{result}, result_type.is_signed};
    }

    if (op == "<" || op == "<=" || op == ">" || op == ">=") {
        const size_t width = std::max(lhs->bits.size(), rhs->bits.size());
        const bool comparison_signed = lhs->is_signed && rhs->is_signed;
        auto left = resize_value(std::move(*lhs), width, comparison_signed,
                                 expression.location, session);
        auto right = resize_value(std::move(*rhs), width, comparison_signed,
                                  expression.location, session);
        if (!left || !right)
            return std::nullopt;
        std::string less;
        std::string greater;
        if (comparison_signed && width > 0) {
            const std::string different_signs = session.emit_binary(
                "XOR2", left->bits.back(), right->bits.back());
            auto unsigned_left = *left;
            auto unsigned_right = *right;
            unsigned_left.bits.pop_back();
            unsigned_right.bits.pop_back();
            std::string low_less = unsigned_left.bits.empty()
                ? session.constant(false)
                : less_than_unsigned(session, unsigned_left, unsigned_right,
                                     unsigned_left.bits.size());
            const std::string signed_less = session.emit_mux(
                low_less, left->bits.back(), different_signs);
            less = signed_less;
            std::string low_greater = unsigned_left.bits.empty()
                ? session.constant(false)
                : less_than_unsigned(session, unsigned_right, unsigned_left,
                                     unsigned_right.bits.size());
            greater = session.emit_mux(
                low_greater, right->bits.back(), different_signs);
        } else {
            less = less_than_unsigned(session, *left, *right, width);
            greater = less_than_unsigned(session, *right, *left, width);
        }
        std::string result;
        if (op == "<") result = less;
        else if (op == ">") result = greater;
        else if (op == "<=") result = session.emit_not(greater);
        else result = session.emit_not(less);
        return VerilogNativeLoweringValue{{result}, result_type.is_signed};
    }

    if (shift) {
        const auto shift = constant_integer(session, *expression.second);
        if (!shift || *shift < 0 ||
            static_cast<uint64_t>(*shift) > kMaximumLoweredWidth) {
            session.diagnose("UnsupportedNativeShift",
                             "native shifts require a nonnegative constant shift amount",
                             expression.second->location);
            return std::nullopt;
        }
        auto left = resize_value(std::move(*lhs), operation_width,
                                 result_type.is_signed, expression.location,
                                 session);
        if (!left)
            return std::nullopt;
        VerilogNativeLoweringValue shifted;
        shifted.is_signed = result_type.is_signed;
        shifted.bits.resize(operation_width, session.constant(false));
        const size_t amount = static_cast<size_t>(*shift);
        if (op == "<<" || op == "<<<") {
            for (size_t bit = amount; bit < shifted.bits.size(); ++bit)
                shifted.bits[bit] = left->bits[bit - amount];
        } else {
            const std::string fill = op == ">>>" && left->is_signed
                ? left->bits.back()
                : session.constant(false);
            std::fill(shifted.bits.begin(), shifted.bits.end(), fill);
            for (size_t bit = 0; bit + amount < shifted.bits.size(); ++bit)
                shifted.bits[bit] = left->bits[bit + amount];
        }
        return shifted;
    }

    auto left = resize_value(std::move(*lhs), operation_width,
                             result_type.is_signed, expression.location,
                             session);
    auto right = resize_value(std::move(*rhs), operation_width,
                              result_type.is_signed, expression.location,
                              session);
    if (!left || !right)
        return std::nullopt;

    VerilogNativeLoweringValue result;
    result.is_signed = result_type.is_signed;
    result.bits.reserve(operation_width);
    if (op == "&" || op == "|" || op == "^" || op == "~^" || op == "^~") {
        for (size_t index = 0; index < operation_width; ++index) {
            if (op == "&")
                result.bits.push_back(session.emit_binary(
                    "AND2", left->bits[index], right->bits[index]));
            else if (op == "|")
                result.bits.push_back(session.emit_binary(
                    "OR2", left->bits[index], right->bits[index]));
            else {
                std::string bit = session.emit_binary(
                    "XOR2", left->bits[index], right->bits[index]);
                if (op == "~^" || op == "^~")
                    bit = session.emit_not(bit);
                result.bits.push_back(std::move(bit));
            }
        }
        return result;
    }
    if (op == "+" || op == "-")
        return add_values(session, std::move(*left), std::move(*right),
                          op == "-", operation_width,
                          result_type.is_signed, expression.location);

    session.diagnose("UnsupportedNativeBinaryOperator",
                     "native RTL lowering does not support operator '" + op + "'",
                     expression.location);
    return std::nullopt;
}

std::optional<VerilogNativeLoweringValue> lower(
    VerilogNativeLoweringSession& session, const VerilogExpression& expression,
    const VerilogNativeSymbolResolver& resolver,
    std::optional<size_t> context_width) {
    const VerilogIntegralType* type = nullptr;
    if (!valid_type(session, expression, type))
        return std::nullopt;

    const auto context_changes_operation_width =
        context_width && *context_width > *type->bit_width &&
        ((expression.kind == VerilogExpressionKind::Binary &&
          (expression.text == "&" || expression.text == "|" ||
           expression.text == "^" || expression.text == "~^" ||
           expression.text == "^~" || expression.text == "+" ||
           expression.text == "-" || expression.text == "<<" ||
           expression.text == "<<<" || expression.text == ">>" ||
           expression.text == ">>>")) ||
         expression.kind == VerilogExpressionKind::Conditional ||
         (expression.kind == VerilogExpressionKind::Unary &&
          (expression.text == "+" || expression.text == "-" ||
           expression.text == "~")));
    if (!context_changes_operation_width) {
        if (auto constant = lower_constant(session, expression, *type)) {
            if (context_width)
                return resize_value(std::move(*constant),
                                    std::max(*type->bit_width, *context_width),
                                    type->is_signed, expression.location, session);
            return constant;
        }
    }
    if (session.unsupported)
        return std::nullopt;

    switch (expression.kind) {
        case VerilogExpressionKind::Identifier: {
            const auto* symbol = session.resolve_symbol(expression);
            if (!symbol) {
                session.diagnose("UnresolvedNativeExpressionSymbol",
                                 "identifier does not resolve to a semantic symbol",
                                 expression.location);
                return std::nullopt;
            }
            auto value = resolver ? resolver(symbol) : std::nullopt;
            if (!value)
                value = session.resolve_value(symbol);
            if (!value) {
                session.diagnose("UnsupportedNativeExpressionSymbol",
                                 "identifier is not a supported lowered signal",
                                 expression.location);
                return std::nullopt;
            }
            return resize_value(std::move(*value),
                                context_width
                                    ? std::max(*type->bit_width, *context_width)
                                    : *type->bit_width,
                                type->is_signed, expression.location, session);
        }
        case VerilogExpressionKind::Literal:
            session.diagnose("UnsupportedNativeLiteral",
                             "literal could not be represented as known integral hardware bits",
                             expression.location);
            return std::nullopt;
        case VerilogExpressionKind::Unary: {
            if (!expression.first) {
                session.diagnose("MalformedNativeUnaryExpression",
                                 "unary expression is missing its operand",
                                 expression.location);
                return std::nullopt;
            }
            const auto& op = expression.text;
            const bool context_sized = op == "+" || op == "-" || op == "~";
            const size_t operation_width = context_sized && context_width
                ? std::max(*type->bit_width, *context_width)
                : *type->bit_width;
            auto operand = lower(session, *expression.first, resolver,
                                 context_sized
                                     ? std::optional<size_t>(operation_width)
                                     : std::nullopt);
            if (!operand)
                return std::nullopt;
            if (op == "+")
                return resize_value(std::move(*operand), operation_width,
                                    type->is_signed, expression.location, session);
            if (op == "!") {
                const std::string result = session.emit_not(truth_value(session, *operand));
                return VerilogNativeLoweringValue{{result}, type->is_signed};
            }
            if (op == "&" || op == "~&" || op == "|" || op == "~|" ||
                op == "^" || op == "~^" || op == "^~") {
                const std::string result = reduce_bits(session, operand->bits, op);
                return VerilogNativeLoweringValue{{result}, type->is_signed};
            }
            if (op == "~") {
                auto resized = resize_value(std::move(*operand), operation_width,
                                            type->is_signed, expression.location,
                                            session);
                if (!resized)
                    return std::nullopt;
                for (auto& bit : resized->bits)
                    bit = session.emit_not(bit);
                return resized;
            }
            if (op == "-") {
                VerilogNativeLoweringValue zero;
                zero.is_signed = type->is_signed;
                zero.bits.resize(operation_width, session.constant(false));
                return add_values(session, std::move(zero), std::move(*operand),
                                  true, operation_width, type->is_signed,
                                  expression.location);
            }
            session.diagnose("UnsupportedNativeUnaryOperator",
                             "native RTL lowering does not support unary operator '" + op + "'",
                             expression.location);
            return std::nullopt;
        }
        case VerilogExpressionKind::Binary:
            return lower_binary(session, expression, resolver, *type,
                                context_width);
        case VerilogExpressionKind::Conditional: {
            if (!expression.first || !expression.second || !expression.third) {
                session.diagnose("MalformedNativeConditionalExpression",
                                 "conditional expression is missing an operand",
                                 expression.location);
                return std::nullopt;
            }
            const size_t conditional_width = context_width
                ? std::max(*type->bit_width, *context_width)
                : *type->bit_width;
            auto condition = lower(session, *expression.first, resolver);
            auto when_true = lower(session, *expression.second, resolver,
                                   conditional_width);
            auto when_false = lower(session, *expression.third, resolver,
                                    conditional_width);
            if (!condition || !when_true || !when_false)
                return std::nullopt;
            auto yes = resize_value(std::move(*when_true), conditional_width,
                                    type->is_signed, expression.location, session);
            auto no = resize_value(std::move(*when_false), conditional_width,
                                   type->is_signed, expression.location, session);
            if (!yes || !no)
                return std::nullopt;
            const std::string select = truth_value(session, *condition);
            VerilogNativeLoweringValue result;
            result.is_signed = type->is_signed;
            result.bits.reserve(conditional_width);
            for (size_t index = 0; index < conditional_width; ++index)
                result.bits.push_back(session.emit_mux(
                    no->bits[index], yes->bits[index], select));
            return result;
        }
        case VerilogExpressionKind::Select:
            return lower_select(session, expression, resolver, *type);
        case VerilogExpressionKind::Concatenation:
            return lower_concat(session, expression, resolver, *type);
        case VerilogExpressionKind::Replication:
            return lower_replication(session, expression, resolver, *type);
        case VerilogExpressionKind::Call: {
            std::string name = expression.first ? expression.first->text : expression.text;
            if ((name == "$signed" || name == "$unsigned") &&
                expression.operands.size() == 1 && expression.operands.front()) {
                auto value = lower(session, *expression.operands.front(), resolver);
                if (!value)
                    return std::nullopt;
                value->is_signed = name == "$signed";
                return resize_value(std::move(*value), *type->bit_width,
                                    name == "$signed", expression.location,
                                    session);
            }
            session.diagnose("UnsupportedNativeCall",
                             "only constant calls and $signed/$unsigned casts are supported by native expression lowering",
                             expression.location);
            return std::nullopt;
        }
        case VerilogExpressionKind::MemberAccess:
            session.diagnose("UnsupportedNativeMemberAccess",
                             "member access is not supported by native expression lowering",
                             expression.location);
            return std::nullopt;
    }
    session.diagnose("UnsupportedNativeExpression",
                     "expression kind is not supported by native RTL lowering",
                     expression.location);
    return std::nullopt;
}

} // namespace

std::optional<VerilogNativeLoweringValue> lower_verilog_native_expression(
    VerilogNativeLoweringSession& session,
    const VerilogExpression& expression,
    const VerilogNativeSymbolResolver& resolve_symbol,
    std::optional<size_t> context_width) {
    return lower(session, expression, resolve_symbol, context_width);
}

} // namespace labneura::eda::detail
