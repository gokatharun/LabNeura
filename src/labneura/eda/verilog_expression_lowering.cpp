#include "verilog_native_lowering_internal.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <unordered_set>

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

// This is deliberately an interval proof, not value propagation.  It only
// accepts expressions whose unsigned value set is known to be one contiguous
// interval after the native type analysis has assigned widths and signedness.
// That lets dynamic select use the low selector bits without inventing a
// hardware value for an out-of-range index.
struct ProvenUnsignedRange {
    uint64_t minimum = 0;
    uint64_t maximum = 0;
    size_t width = 0;
};

constexpr size_t kMaximumRangeProofWidth = 62;

std::optional<uint64_t> unsigned_mask(size_t width) {
    if (width == 0 || width > kMaximumRangeProofWidth)
        return std::nullopt;
    return (uint64_t{1} << width) - 1;
}

bool is_power_of_two(uint64_t value) {
    return value != 0 && (value & (value - 1)) == 0;
}

bool contains_four_state_literal(VerilogNativeLoweringSession& session,
                                 const VerilogExpression& expression) {
    if (expression.kind == VerilogExpressionKind::Literal) {
        const auto value = session.evaluate_constant(expression);
        return value && !value->is_fully_known();
    }
    if (expression.first && contains_four_state_literal(session, *expression.first))
        return true;
    if (expression.second && contains_four_state_literal(session, *expression.second))
        return true;
    if (expression.third && contains_four_state_literal(session, *expression.third))
        return true;
    for (const auto& operand : expression.operands)
        if (operand && contains_four_state_literal(session, *operand))
            return true;
    return false;
}

bool collect_unique_identifier_names(
    const VerilogExpression& expression,
    std::unordered_set<std::string>& names) {
    if (expression.kind == VerilogExpressionKind::Identifier)
        return names.insert(expression.text).second;
    if (expression.first && !collect_unique_identifier_names(*expression.first, names))
        return false;
    if (expression.second && !collect_unique_identifier_names(*expression.second, names))
        return false;
    if (expression.third && !collect_unique_identifier_names(*expression.third, names))
        return false;
    for (const auto& operand : expression.operands)
        if (operand && !collect_unique_identifier_names(*operand, names))
            return false;
    return true;
}

std::optional<ProvenUnsignedRange> prove_unsigned_range(
    VerilogNativeLoweringSession& session, const VerilogExpression& expression,
    size_t effective_width, std::unordered_set<const VerilogExpression*>& visiting) {
    const auto* type = expression_type(session, expression);
    const auto effective_mask = unsigned_mask(effective_width);
    if (!type || !type->is_known_integral() || !type->bit_width ||
        type->is_signed || *type->bit_width == 0 ||
        *type->bit_width > kMaximumRangeProofWidth ||
        effective_width < *type->bit_width || !effective_mask ||
        !visiting.insert(&expression).second)
        return std::nullopt;

    std::optional<ProvenUnsignedRange> result;
    const auto finish = [&] {
        visiting.erase(&expression);
        return result;
    };

    // Constant parameters and constant expressions are exact after the
    // existing Verilog constant evaluator has applied its own sizing rules.
    if (const auto constant = session.evaluate_constant(expression);
        constant && constant->is_fully_known() && !constant->is_signed) {
        if (const auto value = constant->to_uint64(); value && *value <= *effective_mask)
            result = ProvenUnsignedRange{*value, *value, effective_width};
        return finish();
    }

    const auto full_range = [&](size_t width)
        -> std::optional<ProvenUnsignedRange> {
        const auto mask = unsigned_mask(width);
        return mask ? std::optional<ProvenUnsignedRange>(
                          ProvenUnsignedRange{0, *mask, width})
                    : std::nullopt;
    };
    const auto range_with_width = [&](const VerilogExpression& child, size_t width) {
        return prove_unsigned_range(session, child, width, visiting);
    };

    switch (expression.kind) {
        case VerilogExpressionKind::Identifier: {
            // A continuous assignment is an elaborated native signal, not a
            // copied value.  Looking through exactly one unambiguous driver
            // proves its range while lowering still reads the real signal
            // storage, so procedural state is never bypassed.
            const VerilogExpression* driver = nullptr;
            bool ambiguous = false;
            for (const auto& assignment : session.module.assignments) {
                if (!assignment.left || !assignment.right ||
                    assignment.left->kind != VerilogExpressionKind::Identifier ||
                    assignment.left->text != expression.text)
                    continue;
                if (driver) {
                    ambiguous = true;
                    break;
                }
                driver = assignment.right.get();
            }
            if (!ambiguous && driver)
                result = range_with_width(*driver, effective_width);
            else if (auto declared = full_range(*type->bit_width)) {
                // Binary lowering zero-extends an unsigned identifier when
                // the enclosing operation is wider; that does not create the
                // high values of the enclosing operation's full domain.
                declared->width = effective_width;
                result = *declared;
            }
            break;
        }
        case VerilogExpressionKind::Concatenation: {
            uint64_t minimum = 0;
            uint64_t maximum = 0;
            bool valid = !expression.operands.empty();
            size_t total_width = 0;
            std::unordered_set<std::string> identifier_names;
            for (const auto& operand : expression.operands) {
                if (!operand) {
                    valid = false;
                    break;
                }
                if (!collect_unique_identifier_names(*operand, identifier_names)) {
                    // Repeated inputs can make a concatenation sparse (for
                    // example, {index, index}); min/max bounds alone are not
                    // sufficient to lower those safely.
                    valid = false;
                    break;
                }
                const auto* operand_type = expression_type(session, *operand);
                if (!operand_type || !operand_type->bit_width || operand_type->is_signed ||
                    total_width > kMaximumRangeProofWidth - *operand_type->bit_width) {
                    valid = false;
                    break;
                }
                const auto operand_range = range_with_width(*operand, *operand_type->bit_width);
                if (!operand_range) {
                    valid = false;
                    break;
                }
                minimum = (minimum << *operand_type->bit_width) | operand_range->minimum;
                maximum = (maximum << *operand_type->bit_width) | operand_range->maximum;
                total_width += *operand_type->bit_width;
            }
            if (valid && total_width <= effective_width)
                result = ProvenUnsignedRange{minimum, maximum, effective_width};
            break;
        }
        case VerilogExpressionKind::Call: {
            const std::string name = expression.first ? expression.first->text : expression.text;
            if (name == "$unsigned" && expression.operands.size() == 1 &&
                expression.operands.front()) {
                const auto* operand_type = expression_type(session, *expression.operands.front());
                if (operand_type && !operand_type->is_signed && operand_type->bit_width &&
                    *operand_type->bit_width <= effective_width)
                    result = range_with_width(*expression.operands.front(), effective_width);
            }
            break;
        }
        case VerilogExpressionKind::Binary: {
            if (!expression.first || !expression.second)
                break;
            const auto left = range_with_width(*expression.first, effective_width);
            const auto right = range_with_width(*expression.second, effective_width);
            if (!left || !right)
                break;
            if (expression.text == "+" && right->minimum == right->maximum &&
                left->maximum <= *effective_mask - right->maximum) {
                result = ProvenUnsignedRange{left->minimum + right->minimum,
                                             left->maximum + right->maximum,
                                             effective_width};
            } else if (expression.text == "-" && right->minimum == right->maximum &&
                       left->minimum >= right->maximum) {
                result = ProvenUnsignedRange{left->minimum - right->maximum,
                                             left->maximum - right->minimum,
                                             effective_width};
            } else if (expression.text == "^" && right->minimum == right->maximum &&
                       left->minimum == 0 && left->maximum == *effective_mask) {
                // XOR by a constant permutes every value in a full unsigned
                // domain, so its exact range remains the full domain.
                result = *left;
            }
            break;
        }
        default:
            break;
    }
    return finish();
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
    std::optional<VerilogNativeLoweringValue> owned_base;
    const std::vector<std::string>* base_bits = nullptr;
    if (!resolver &&
        expression.first->kind == VerilogExpressionKind::Identifier) {
        const VerilogIntegralType* base_type = nullptr;
        if (!valid_type(session, *expression.first, base_type))
            return std::nullopt;
        if (const auto* symbol = session.resolve_symbol(*expression.first)) {
            if (const auto* signal = session.signal(symbol))
                base_bits = &signal->bits;
        }
    }
    if (!base_bits) {
        owned_base = lower(session, *expression.first, resolver);
        if (!owned_base)
            return std::nullopt;
        base_bits = &owned_base->bits;
    }

    std::vector<std::string> selected;
    if (expression.select_kind == VerilogSelectKind::Bit) {
        const auto index = constant_integer(session, *expression.second);
        if (!index) {
            if (contains_four_state_literal(session, *expression.second)) {
                session.diagnose("FourStateConstantUnsupported",
                                 "native Boolean netlists cannot represent X or Z index bits",
                                 expression.second->location);
                return std::nullopt;
            }
            const auto* index_type = expression_type(session, *expression.second);
            if (!index_type || !index_type->bit_width || index_type->is_signed ||
                *index_type->bit_width > kMaximumRangeProofWidth) {
                session.diagnose("DynamicNativeSelectUnsupported",
                                 "dynamic bit-select index must have a known unsigned width",
                                 expression.second->location);
                return std::nullopt;
            }
            const auto* base_type = expression_type(session, *expression.first);
            if (!base_type || base_type->packed_dimensions.size() != 1 ||
                !base_type->packed_dimensions.front().left ||
                !base_type->packed_dimensions.front().right) {
                session.diagnose("DynamicNativeSelectRangeUnproven",
                                 "dynamic bit-select requires one statically ranged packed dimension",
                                 expression.first->location);
                return std::nullopt;
            }
            const auto& range = base_type->packed_dimensions.front();
            const int64_t minimum_bound = std::min(*range.left, *range.right);
            const int64_t maximum_bound = std::max(*range.left, *range.right);
            std::unordered_set<const VerilogExpression*> visiting;
            const auto proven_range = prove_unsigned_range(
                session, *expression.second, *index_type->bit_width, visiting);
            if (!proven_range || maximum_bound < 0 ||
                proven_range->maximum > static_cast<uint64_t>(maximum_bound) ||
                (minimum_bound > 0 &&
                 proven_range->minimum < static_cast<uint64_t>(minimum_bound))) {
                session.diagnose("DynamicNativeSelectRangeUnproven",
                                 proven_range
                                     ? "proven dynamic bit-select index range " +
                                           std::to_string(proven_range->minimum) + ".." +
                                           std::to_string(proven_range->maximum) +
                                           " is outside the declared packed range"
                                     : "the dynamic bit-select index range could not be proven",
                                 expression.second->location);
                return std::nullopt;
            }
            const uint64_t select_count =
                proven_range->maximum - proven_range->minimum + 1;
            if (!is_power_of_two(select_count) ||
                proven_range->minimum % select_count != 0 ||
                select_count > base_bits->size()) {
                session.diagnose("DynamicNativeSelectRangeUnproven",
                                 "dynamic bit-select requires a contiguous power-of-two aligned proven index range",
                                 expression.second->location);
                return std::nullopt;
            }
            auto lowered_index = lower(session, *expression.second, resolver);
            if (!lowered_index)
                return std::nullopt;
            std::vector<std::string> mux_level;
            mux_level.reserve(static_cast<size_t>(select_count));
            for (uint64_t value = proven_range->minimum;
                 value <= proven_range->maximum; ++value) {
                const auto position = declared_index_to_position(
                    session, *expression.first, static_cast<int64_t>(value),
                    base_bits->size());
                if (!position) {
                    session.diagnose("DynamicNativeSelectRangeUnproven",
                                     "dynamic bit-select index is outside the declared packed range",
                                     expression.location);
                    return std::nullopt;
                }
                mux_level.push_back((*base_bits)[*position]);
            }
            for (size_t selector_bit = 0;
                 mux_level.size() > 1; ++selector_bit) {
                if (mux_level.size() % 2 != 0 ||
                    selector_bit >= lowered_index->bits.size()) {
                    session.diagnose("DynamicNativeSelectRangeUnproven",
                                     "dynamic bit-select index width does not cover its proven range",
                                     expression.second->location);
                    return std::nullopt;
                }
                std::vector<std::string> next_level;
                next_level.reserve(mux_level.size() / 2);
                for (size_t bit = 0; bit < mux_level.size(); bit += 2)
                    next_level.push_back(session.emit_mux(
                        mux_level[bit], mux_level[bit + 1],
                        lowered_index->bits[selector_bit]));
                mux_level = std::move(next_level);
            }
            selected.push_back(mux_level.front());
        } else {
            const auto position = declared_index_to_position(
                session, *expression.first, *index, base_bits->size());
            if (!position) {
                session.diagnose("OutOfRangeNativeSelect",
                                 "out-of-range bit selects cannot be represented as a hardware bit",
                                 expression.location);
                return std::nullopt;
            }
            selected.push_back((*base_bits)[*position]);
        }
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
                session, *expression.first, logical_index, base_bits->size());
            if (!position) {
                session.diagnose("OutOfRangeNativeSelect",
                                 "out-of-range part selects cannot be represented as hardware bits",
                                 expression.location);
                return std::nullopt;
            }
            selected.push_back((*base_bits)[*position]);
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
        if (index + 1 == width)
            continue;
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
