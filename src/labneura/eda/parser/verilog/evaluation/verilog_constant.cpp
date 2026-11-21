#include "labneura/eda/verilog_constant.h"

#include <algorithm>
#include <bit>
#include <cctype>
#include <cstdlib>
#include <limits>
#include <string>
#include <utility>

namespace labneura::eda {
namespace {

using Bits = std::vector<VerilogLogicBit>;
constexpr size_t kDefaultWidth = 32;

struct Evaluator {
    const VerilogConstantEnvironment& environment;
    VerilogConstantResolver resolver;
    VerilogConstantEvaluationOptions options;
    std::vector<VerilogConstantDiagnostic> diagnostics;

    void error(std::string code, std::string message, const VerilogSyntaxLocation& location) {
        diagnostics.push_back({std::move(code), std::move(message), location});
    }

    VerilogConstantValue make_value(size_t width, bool is_signed,
                                    VerilogLogicBit fill = VerilogLogicBit::Zero) {
        if (width == 0 || width > options.maximum_width) {
            return {};
        }
        VerilogConstantValue value;
        value.bit_width = width;
        value.is_signed = is_signed;
        value.bits.assign(width, fill);
        return value;
    }

    static bool known(VerilogLogicBit bit) {
        return bit == VerilogLogicBit::Zero || bit == VerilogLogicBit::One;
    }

    static VerilogLogicBit normalized(VerilogLogicBit bit) {
        return bit == VerilogLogicBit::Z ? VerilogLogicBit::X : bit;
    }

    static bool has_unknown(const VerilogConstantValue& value) {
        return std::any_of(value.bits.begin(), value.bits.end(),
                           [](auto bit) { return !known(bit); });
    }

    bool parse_integer(std::string_view text, const VerilogSyntaxLocation& location,
                       VerilogConstantValue& value) {
        const auto apostrophe = text.find('\'');
        if (apostrophe == std::string_view::npos) {
            Bits digits(kDefaultWidth, VerilogLogicBit::Zero);
            bool negative = !text.empty() && text.front() == '-';
            const size_t start = (!text.empty() && (text.front() == '-' || text.front() == '+'))
                                     ? 1
                                     : 0;
            for (size_t i = start; i < text.size(); ++i) {
                if (text[i] == '_')
                    continue;
                if (!std::isdigit(static_cast<unsigned char>(text[i]))) {
                    error("InvalidIntegralLiteral", "invalid decimal integral literal", location);
                    return false;
                }
                if (!multiply_add_growing(digits, 10,
                                          static_cast<unsigned>(text[i] - '0'))) {
                    error("InvalidConstantWidth", "decimal literal exceeds maximum supported width",
                          location);
                    return false;
                }
            }
            if (negative)
                twos_complement(digits);
            const size_t width = std::max(kDefaultWidth, significant_width(digits, true));
            if (width > options.maximum_width) {
                error("InvalidConstantWidth", "decimal literal exceeds maximum supported width",
                      location);
                return false;
            }
            value = make_value(width, true);
            std::copy(digits.begin(), digits.end(), value.bits.begin());
            return true;
        }

        size_t width = 0;
        if (apostrophe != 0) {
            for (size_t i = 0; i < apostrophe; ++i) {
                if (text[i] == '_')
                    continue;
                if (!std::isdigit(static_cast<unsigned char>(text[i]))) {
                    error("InvalidIntegralLiteral", "invalid size in based literal", location);
                    return false;
                }
                const size_t digit = static_cast<size_t>(text[i] - '0');
                if (width > options.maximum_width / 10 ||
                    width * 10 + digit > options.maximum_width) {
                    error("InvalidConstantWidth", "literal width exceeds maximum supported width",
                          location);
                    return false;
                }
                width = width * 10 + digit;
            }
            if (width == 0) {
                error("InvalidConstantWidth", "integral literal width must be positive", location);
                return false;
            }
        }

        size_t cursor = apostrophe + 1;
        bool is_signed = false;
        if (cursor < text.size() &&
            (text[cursor] == 's' || text[cursor] == 'S')) {
            is_signed = true;
            ++cursor;
        }
        if (cursor >= text.size()) {
            error("InvalidIntegralLiteral", "missing base specifier in literal", location);
            return false;
        }
        const char base = static_cast<char>(
            std::tolower(static_cast<unsigned char>(text[cursor++])));
        unsigned radix = 0;
        unsigned bits_per_digit = 0;
        switch (base) {
            case 'b':
                radix = 2;
                bits_per_digit = 1;
                break;
            case 'o':
                radix = 8;
                bits_per_digit = 3;
                break;
            case 'd':
                radix = 10;
                break;
            case 'h':
                radix = 16;
                bits_per_digit = 4;
                break;
            default:
                error("InvalidIntegralLiteral", "unsupported base in integral literal", location);
                return false;
        }
        const auto digit_start = cursor;
        size_t digit_count = 0;
        for (size_t i = digit_start; i < text.size(); ++i)
            digit_count += text[i] != '_';
        const size_t natural_width =
            radix == 10 ? std::max(kDefaultWidth, digit_count * 4)
                        : std::max(kDefaultWidth, digit_count * bits_per_digit);
        if (!width && natural_width > options.maximum_width) {
            error("InvalidConstantWidth", "unsized literal exceeds maximum supported width",
                  location);
            return false;
        }
        Bits bits(width ? width : natural_width, VerilogLogicBit::Zero);
        size_t processed_digits = 0;
        for (; cursor < text.size(); ++cursor) {
            const char ch = text[cursor];
            if (ch == '_')
                continue;
            if (ch == 'x' || ch == 'X' || ch == 'z' || ch == 'Z' || ch == '?') {
                if (radix == 10) {
                    std::fill(bits.begin(), bits.end(),
                              (ch == 'z' || ch == 'Z' || ch == '?') ? VerilogLogicBit::Z
                                                                     : VerilogLogicBit::X);
                }
                else {
                    const auto fill = (ch == 'z' || ch == 'Z' || ch == '?')
                                          ? VerilogLogicBit::Z
                                          : VerilogLogicBit::X;
                    const size_t digit_bits = bits_per_digit;
                    const size_t offset = (digit_count - processed_digits - 1) * digit_bits;
                    for (size_t bit = offset; bit < std::min(bits.size(), offset + digit_bits);
                         ++bit)
                        bits[bit] = fill;
                }
                ++processed_digits;
                continue;
            }
            unsigned digit = 0;
            if (ch >= '0' && ch <= '9')
                digit = static_cast<unsigned>(ch - '0');
            else if (ch >= 'a' && ch <= 'f')
                digit = static_cast<unsigned>(ch - 'a' + 10);
            else if (ch >= 'A' && ch <= 'F')
                digit = static_cast<unsigned>(ch - 'A' + 10);
            else {
                error("InvalidIntegralLiteral", "invalid digit in integral literal", location);
                return false;
            }
            if (digit >= radix) {
                error("InvalidIntegralLiteral", "digit is outside literal base", location);
                return false;
            }
            if (radix == 10)
                multiply_add(bits, 10, digit);
            else {
                const size_t digit_bits = bits_per_digit;
                const size_t offset = (digit_count - processed_digits - 1) * digit_bits;
                for (size_t bit = 0; bit < digit_bits && offset + bit < bits.size(); ++bit)
                    bits[offset + bit] = ((digit >> bit) & 1U) ? VerilogLogicBit::One
                                                               : VerilogLogicBit::Zero;
            }
            ++processed_digits;
        }
        if (cursor == digit_start) {
            error("InvalidIntegralLiteral", "integral literal has no digits", location);
            return false;
        }
        if (width && is_signed && radix != 10 && digit_count) {
            const size_t encoded = radix == 10 ? significant_width(bits, false)
                                               : digit_count * bits_per_digit;
            if (encoded > 0 && encoded < bits.size()) {
                const auto extension = bits[encoded - 1];
                if (extension == VerilogLogicBit::One || extension == VerilogLogicBit::X ||
                    extension == VerilogLogicBit::Z)
                    std::fill(bits.begin() + static_cast<std::ptrdiff_t>(encoded), bits.end(),
                              extension);
            }
        }
        if (bits.size() > options.maximum_width) {
            error("InvalidConstantWidth", "literal exceeds maximum supported width", location);
            return false;
        }
        value.bit_width = bits.size();
        value.is_signed = is_signed;
        value.bits = std::move(bits);
        return true;
    }

    bool multiply_add_growing(Bits& bits, unsigned multiplier, unsigned addend) {
        unsigned carry = addend;
        for (auto& bit : bits) {
            const unsigned product = (bit == VerilogLogicBit::One ? 1U : 0U) * multiplier + carry;
            bit = (product & 1U) ? VerilogLogicBit::One : VerilogLogicBit::Zero;
            carry = product >> 1;
        }
        while (carry && bits.size() < options.maximum_width) {
            bits.push_back((carry & 1U) ? VerilogLogicBit::One : VerilogLogicBit::Zero);
            carry >>= 1;
        }
        return carry == 0;
    }

    static void multiply_add(Bits& bits, unsigned multiplier, unsigned addend) {
        unsigned carry = addend;
        for (auto& bit : bits) {
            const unsigned product = (bit == VerilogLogicBit::One ? 1U : 0U) * multiplier + carry;
            bit = (product & 1U) ? VerilogLogicBit::One : VerilogLogicBit::Zero;
            carry = product >> 1;
        }
    }

    static size_t significant_width(const Bits& bits, bool signed_value) {
        if (bits.empty())
            return 1;
        size_t width = bits.size();
        const auto sign = bits.back();
        while (width > 1 && bits[width - 1] == sign &&
               (sign == VerilogLogicBit::Zero || (signed_value && sign == VerilogLogicBit::One)))
            --width;
        return width + (signed_value ? 1 : 0);
    }

    static void twos_complement(Bits& bits) {
        unsigned carry = 1;
        for (auto& bit : bits) {
            const unsigned inverted = bit == VerilogLogicBit::One ? 0U : 1U;
            const unsigned sum = inverted + carry;
            bit = (sum & 1U) ? VerilogLogicBit::One : VerilogLogicBit::Zero;
            carry = sum >> 1;
        }
    }

    static void sign_extend(Bits& bits, size_t width, bool is_signed) {
        const auto extension = is_signed && !bits.empty() ? bits.back() : VerilogLogicBit::Zero;
        bits.resize(width, extension);
    }

    std::optional<VerilogConstantValue> eval(const VerilogExpression& expression) {
        switch (expression.kind) {
            case VerilogExpressionKind::Literal: {
                if (expression.text == "'0" || expression.text == "'1" ||
                    expression.text == "'x" || expression.text == "'X" ||
                    expression.text == "'z" || expression.text == "'Z") {
                    const char last = expression.text.back();
                    auto value = make_value(
                        1, false,
                        last == '1' ? VerilogLogicBit::One
                                    : (last == 'x' || last == 'X')
                                          ? VerilogLogicBit::X
                                          : (last == 'z' || last == 'Z')
                                                ? VerilogLogicBit::Z
                                                : VerilogLogicBit::Zero);
                    value.unbased_unsized = true;
                    return value;
                }
                VerilogConstantValue value;
                if (parse_integer(expression.text, expression.location, value))
                    return value;
                return std::nullopt;
            }
            case VerilogExpressionKind::Identifier:
                return eval_identifier(expression.text, expression.location);
            case VerilogExpressionKind::Unary:
                return eval_unary(expression);
            case VerilogExpressionKind::Binary:
                return eval_binary(expression);
            case VerilogExpressionKind::Conditional:
                return eval_conditional(expression);
            case VerilogExpressionKind::Concatenation:
                return eval_concatenation(expression);
            case VerilogExpressionKind::Replication:
                return eval_replication(expression);
            case VerilogExpressionKind::Select:
                return eval_select(expression);
            case VerilogExpressionKind::Call:
                return eval_call(expression);
            case VerilogExpressionKind::MemberAccess: {
                const auto* base = expression.first.get();
                if (base && base->kind == VerilogExpressionKind::Identifier)
                    return eval_identifier(base->text + "." + expression.text,
                                            expression.location);
                break;
            }
        }
        error("UnsupportedConstantExpression", "expression is not supported as a constant",
              expression.location);
        return std::nullopt;
    }

    std::optional<VerilogConstantValue> eval_identifier(
        std::string_view name, const VerilogSyntaxLocation& location) {
        const auto found = environment.find(std::string(name));
        if (found != environment.end())
            return found->second;
        if (resolver) {
            if (auto value = resolver(name, location))
                return value;
        }
        error("UnresolvedParameter", "constant identifier '" + std::string(name) +
                                         "' has no parameter value",
              location);
        return std::nullopt;
    }

    VerilogConstantValue resized(const VerilogConstantValue& input, size_t width,
                                 bool sign_extend_value) {
        auto value = make_value(width, input.is_signed);
        if (input.unbased_unsized && !input.bits.empty()) {
            value.bits.assign(width, input.bits.front());
            value.unbased_unsized = true;
            return value;
        }
        const size_t copy = std::min(width, input.bits.size());
        std::copy_n(input.bits.begin(), copy, value.bits.begin());
        if (width > copy) {
            const auto extension = sign_extend_value && input.is_signed && !input.bits.empty()
                                       ? input.bits.back()
                                       : VerilogLogicBit::Zero;
            std::fill(value.bits.begin() + static_cast<std::ptrdiff_t>(copy), value.bits.end(),
                      extension);
        }
        return value;
    }

    std::optional<VerilogConstantValue> eval_unary(const VerilogExpression& expression) {
        if (!expression.first) {
            error("InvalidConstantExpression", "unary operator is missing an operand",
                  expression.location);
            return std::nullopt;
        }
        auto operand = eval(*expression.first);
        if (!operand)
            return std::nullopt;
        auto result = *operand;
        result.unbased_unsized = false;
        const auto& op = expression.text;
        if (op == "+")
            return result;
        if (op == "-") {
            if (has_unknown(result)) {
                std::fill(result.bits.begin(), result.bits.end(), VerilogLogicBit::X);
                return result;
            }
            twos_complement(result.bits);
            return result;
        }
        if (op == "~") {
            for (auto& bit : result.bits) {
                bit = normalized(bit);
                bit = bit == VerilogLogicBit::Zero ? VerilogLogicBit::One
                                                    : bit == VerilogLogicBit::One
                                                          ? VerilogLogicBit::Zero
                                                          : VerilogLogicBit::X;
            }
            return result;
        }
        if (op == "!") {
            const auto truth = logical_truth(result);
            return make_value(1, false, truth == VerilogLogicBit::One
                                           ? VerilogLogicBit::Zero
                                           : truth == VerilogLogicBit::Zero
                                                 ? VerilogLogicBit::One
                                                 : VerilogLogicBit::X);
        }
        if (op == "&" || op == "~&" || op == "|" || op == "~|" || op == "^" || op == "~^" ||
            op == "^~") {
            bool saw_unknown = false;
            bool accumulated = op == "&" || op == "~&";
            for (auto bit : result.bits) {
                bit = normalized(bit);
                if (!known(bit)) {
                    saw_unknown = true;
                    continue;
                }
                const bool value = bit == VerilogLogicBit::One;
                if (op == "&" || op == "~&")
                    accumulated = accumulated && value;
                else if (op == "|" || op == "~|")
                    accumulated = accumulated || value;
                else
                    accumulated ^= value;
            }
            if (saw_unknown) {
                const bool controlling = ((op == "&" || op == "~&") && !accumulated) ||
                                         ((op == "|" || op == "~|") && accumulated);
                if (!controlling)
                    return make_value(1, false, VerilogLogicBit::X);
            }
            if (op == "~&" || op == "~|" || op == "~^" || op == "^~")
                accumulated = !accumulated;
            return make_value(1, false, accumulated ? VerilogLogicBit::One
                                                    : VerilogLogicBit::Zero);
        }
        error("UnsupportedConstantOperator", "unsupported unary constant operator '" + op + "'",
              expression.location);
        return std::nullopt;
    }

    std::optional<VerilogConstantValue> eval_binary(const VerilogExpression& expression) {
        if (!expression.first || !expression.second) {
            error("InvalidConstantExpression", "binary operator is missing an operand",
                  expression.location);
            return std::nullopt;
        }
        auto lhs = eval(*expression.first);
        auto rhs = eval(*expression.second);
        if (!lhs || !rhs)
            return std::nullopt;
        const std::string& op = expression.text;
        const bool comparison = op == "==" || op == "!=" || op == "===" || op == "!==" ||
                                op == "==?" || op == "!=?" || op == "<" || op == "<=" ||
                                op == ">" || op == ">=";
        const bool logical = op == "&&" || op == "||";
        const bool signed_operands = lhs->is_signed && rhs->is_signed;
        const size_t width = (op == "<<" || op == ">>" || op == "<<<" || op == ">>>")
                                 ? lhs->bit_width
                                 : std::max(lhs->bit_width, rhs->bit_width);
        if (!comparison && !logical && (width == 0 || width > options.maximum_width)) {
            error("InvalidConstantWidth", "expression result width is invalid", expression.location);
            return std::nullopt;
        }
        auto left = resized(*lhs, width, signed_operands);
        auto right = resized(*rhs, width, signed_operands);

        if (logical) {
            const auto a = logical_truth(left);
            const auto b = logical_truth(right);
            if (op == "&&") {
                if (a == VerilogLogicBit::Zero || b == VerilogLogicBit::Zero)
                    return make_value(1, false, VerilogLogicBit::Zero);
                if (a == VerilogLogicBit::X || b == VerilogLogicBit::X)
                    return make_value(1, false, VerilogLogicBit::X);
                return make_value(1, false, VerilogLogicBit::One);
            }
            if (a == VerilogLogicBit::One || b == VerilogLogicBit::One)
                return make_value(1, false, VerilogLogicBit::One);
            if (a == VerilogLogicBit::X || b == VerilogLogicBit::X)
                return make_value(1, false, VerilogLogicBit::X);
            return make_value(1, false, VerilogLogicBit::Zero);
        }

        if (op == "==" || op == "!=" || op == "===" || op == "!==" || op == "==?" ||
            op == "!=?") {
            bool equal = true;
            bool uncertain = false;
            for (size_t i = 0; i < width; ++i) {
                if ((op == "===" || op == "!==") && left.bits[i] != right.bits[i]) {
                    equal = false;
                    break;
                }
                if ((op == "==?" || op == "!=?") &&
                    (right.bits[i] == VerilogLogicBit::X || right.bits[i] == VerilogLogicBit::Z))
                    continue;
                if (!known(left.bits[i]) || !known(right.bits[i])) {
                    uncertain = true;
                    continue;
                }
                if (left.bits[i] != right.bits[i])
                    equal = false;
            }
            if ((op == "==" || op == "!=") && uncertain)
                return make_value(1, false, VerilogLogicBit::X);
            const bool invert = op == "!=" || op == "!==" || op == "!=?";
            return make_value(1, false, equal != invert ? VerilogLogicBit::One
                                                        : VerilogLogicBit::Zero);
        }
        if (op == "<" || op == "<=" || op == ">" || op == ">=") {
            if (has_unknown(left) || has_unknown(right))
                return make_value(1, false, VerilogLogicBit::X);
            const int comparison_result = compare(left.bits, right.bits, signed_operands);
            const bool value = op == "<"    ? comparison_result < 0
                               : op == "<=" ? comparison_result <= 0
                               : op == ">"  ? comparison_result > 0
                                            : comparison_result >= 0;
            return make_value(1, false, value ? VerilogLogicBit::One : VerilogLogicBit::Zero);
        }
        if (op == "&" || op == "|" || op == "^" || op == "~^" || op == "^~") {
            auto result = make_value(width, signed_operands);
            for (size_t i = 0; i < width; ++i) {
                const auto a = normalized(left.bits[i]);
                const auto b = normalized(right.bits[i]);
                if ((op == "&" && (a == VerilogLogicBit::Zero || b == VerilogLogicBit::Zero)) ||
                    (op == "|" && (a == VerilogLogicBit::One || b == VerilogLogicBit::One))) {
                    result.bits[i] = op == "&" ? VerilogLogicBit::Zero : VerilogLogicBit::One;
                }
                else if (!known(a) || !known(b)) {
                    result.bits[i] = VerilogLogicBit::X;
                }
                else {
                    const bool av = a == VerilogLogicBit::One;
                    const bool bv = b == VerilogLogicBit::One;
                    const bool bit = op == "&" ? av && bv
                                     : op == "|" ? av || bv
                                                 : av != bv;
                    const bool inverted = (op == "~^" || op == "^~") ? !bit : bit;
                    result.bits[i] = inverted ? VerilogLogicBit::One : VerilogLogicBit::Zero;
                }
            }
            return result;
        }
        if (has_unknown(left) || has_unknown(right))
            return make_value(width, signed_operands, VerilogLogicBit::X);
        if (op == "+" || op == "-") {
            auto result = make_value(width, signed_operands);
            unsigned carry = op == "-" ? 1U : 0U;
            for (size_t i = 0; i < width; ++i) {
                unsigned a = left.bits[i] == VerilogLogicBit::One ? 1U : 0U;
                unsigned b = right.bits[i] == VerilogLogicBit::One ? 1U : 0U;
                if (op == "-")
                    b ^= 1U;
                const unsigned sum = a + b + carry;
                result.bits[i] = (sum & 1U) ? VerilogLogicBit::One : VerilogLogicBit::Zero;
                carry = sum >> 1;
            }
            return result;
        }
        if (op == "*") {
            const bool negative =
                signed_operands && ((left.bits.back() == VerilogLogicBit::One) !=
                                    (right.bits.back() == VerilogLogicBit::One));
            if (signed_operands && left.bits.back() == VerilogLogicBit::One)
                twos_complement(left.bits);
            if (signed_operands && right.bits.back() == VerilogLogicBit::One)
                twos_complement(right.bits);
            auto result = multiply_values(left, right);
            result.bit_width = width;
            result.bits.resize(width, VerilogLogicBit::Zero);
            result.is_signed = signed_operands;
            if (negative)
                twos_complement(result.bits);
            return result;
        }
        if (op == "/" || op == "%")
            return divide(left, right, signed_operands, op == "%", expression.location);
        if (op == "<<" || op == "<<<" || op == ">>" || op == ">>>")
            return shift(left, *rhs, op, expression.location);
        if (op == "**")
            return power(left, right, expression.location);
        error("UnsupportedConstantOperator", "unsupported binary constant operator '" + op + "'",
              expression.location);
        return std::nullopt;
    }

    static int compare(const Bits& a, const Bits& b, bool signed_values) {
        if (signed_values && a.back() != b.back())
            return a.back() == VerilogLogicBit::One ? -1 : 1;
        for (size_t i = a.size(); i-- > 0;) {
            if (a[i] != b[i])
                return a[i] == VerilogLogicBit::One ? 1 : -1;
        }
        return 0;
    }

    static Bits magnitude(const Bits& input) {
        Bits result = input;
        if (!result.empty() && result.back() == VerilogLogicBit::One)
            twos_complement(result);
        return result;
    }

    std::optional<VerilogConstantValue> divide(const VerilogConstantValue& lhs,
                                                const VerilogConstantValue& rhs,
                                                bool signed_values, bool modulo,
                                                const VerilogSyntaxLocation& location) {
        if (std::all_of(rhs.bits.begin(), rhs.bits.end(),
                        [](auto bit) { return bit == VerilogLogicBit::Zero; })) {
            error("DivisionByZero", "division or modulo by zero in constant expression", location);
            return std::nullopt;
        }
        const bool lhs_negative = signed_values && lhs.bits.back() == VerilogLogicBit::One;
        const bool rhs_negative = signed_values && rhs.bits.back() == VerilogLogicBit::One;
        Bits dividend = lhs_negative ? magnitude(lhs.bits) : lhs.bits;
        Bits divisor = rhs_negative ? magnitude(rhs.bits) : rhs.bits;
        Bits quotient(lhs.bit_width, VerilogLogicBit::Zero);
        Bits remainder(lhs.bit_width, VerilogLogicBit::Zero);
        for (size_t i = lhs.bit_width; i-- > 0;) {
            for (size_t j = remainder.size(); j-- > 1;)
                remainder[j] = remainder[j - 1];
            remainder[0] = dividend[i];
            if (compare_unsigned(remainder, divisor) >= 0) {
                subtract_unsigned(remainder, divisor);
                quotient[i] = VerilogLogicBit::One;
            }
        }
        Bits bits = modulo ? remainder : quotient;
        if ((!modulo && lhs_negative != rhs_negative) || (modulo && lhs_negative))
            twos_complement(bits);
        auto result = make_value(lhs.bit_width, signed_values);
        result.bits = std::move(bits);
        return result;
    }

    static int compare_unsigned(const Bits& a, const Bits& b) {
        for (size_t i = a.size(); i-- > 0;) {
            const auto right = i < b.size() ? b[i] : VerilogLogicBit::Zero;
            if (a[i] != right)
                return a[i] == VerilogLogicBit::One ? 1 : -1;
        }
        for (size_t i = a.size(); i < b.size(); ++i)
            if (b[i] == VerilogLogicBit::One)
                return -1;
        return 0;
    }

    static void subtract_unsigned(Bits& a, const Bits& b) {
        int borrow = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            const int lhs = a[i] == VerilogLogicBit::One ? 1 : 0;
            const int rhs = i < b.size() && b[i] == VerilogLogicBit::One ? 1 : 0;
            int value = lhs - rhs - borrow;
            borrow = value < 0;
            if (borrow)
                value += 2;
            a[i] = value ? VerilogLogicBit::One : VerilogLogicBit::Zero;
        }
    }

    std::optional<VerilogConstantValue> shift(const VerilogConstantValue& lhs,
                                               const VerilogConstantValue& rhs,
                                               std::string_view op,
                                               const VerilogSyntaxLocation& location) {
        const auto amount = shift_amount(rhs);
        if (!amount) {
            error("InvalidShift", "shift amount must be a known nonnegative integer", location);
            return std::nullopt;
        }
        auto result = make_value(lhs.bit_width, lhs.is_signed);
        const size_t shift_by = *amount;
        if (shift_by >= lhs.bit_width) {
            const bool arithmetic = op == ">>>" && lhs.is_signed && !lhs.bits.empty() &&
                                    lhs.bits.back() == VerilogLogicBit::One;
            if (op == ">>" || op == ">>>")
                std::fill(result.bits.begin(), result.bits.end(),
                          arithmetic ? VerilogLogicBit::One : VerilogLogicBit::Zero);
            return result;
        }
        const bool right_shift = op == ">>" || op == ">>>";
        const bool arithmetic = op == ">>>" && lhs.is_signed;
        for (size_t i = 0; i < result.bit_width; ++i) {
            if (right_shift) {
                const size_t source = i + shift_by;
                result.bits[i] = source < lhs.bit_width
                                     ? lhs.bits[source]
                                     : arithmetic && !lhs.bits.empty() ? lhs.bits.back()
                                                                        : VerilogLogicBit::Zero;
            }
            else {
                result.bits[i] = i >= shift_by && i - shift_by < lhs.bit_width
                                     ? lhs.bits[i - shift_by]
                                     : VerilogLogicBit::Zero;
            }
        }
        return result;
    }

    std::optional<VerilogConstantValue> power(const VerilogConstantValue& base,
                                               const VerilogConstantValue& exponent,
                                               const VerilogSyntaxLocation& location) {
        const auto amount = exponent.to_uint64();
        if (!amount || *amount > options.maximum_width) {
            error("InvalidExponent", "constant exponent is unknown or exceeds supported limit",
                  location);
            return std::nullopt;
        }
        auto result = make_value(base.bit_width, base.is_signed);
        result.bits[0] = VerilogLogicBit::One;
        auto factor = base;
        uint64_t n = *amount;
        while (n) {
            if (n & 1U)
                result = multiply_values(result, factor);
            n >>= 1;
            if (n)
                factor = multiply_values(factor, factor);
        }
        return result;
    }

    static VerilogConstantValue multiply_values(const VerilogConstantValue& lhs,
                                                 const VerilogConstantValue& rhs) {
        VerilogConstantValue result;
        result.bit_width = std::max(lhs.bit_width, rhs.bit_width);
        result.is_signed = lhs.is_signed && rhs.is_signed;
        result.bits.assign(result.bit_width, VerilogLogicBit::Zero);
        for (size_t i = 0; i < lhs.bit_width; ++i) {
            if (lhs.bits[i] != VerilogLogicBit::One)
                continue;
            unsigned carry = 0;
            for (size_t j = 0; j + i < result.bit_width; ++j) {
                const unsigned sum = (j < rhs.bit_width && rhs.bits[j] == VerilogLogicBit::One) +
                                     (result.bits[i + j] == VerilogLogicBit::One) + carry;
                result.bits[i + j] = (sum & 1U) ? VerilogLogicBit::One
                                                : VerilogLogicBit::Zero;
                carry = sum >> 1;
            }
        }
        return result;
    }

    static VerilogLogicBit logical_truth(const VerilogConstantValue& value) {
        bool unknown = false;
        for (auto bit : value.bits) {
            bit = normalized(bit);
            if (bit == VerilogLogicBit::One)
                return VerilogLogicBit::One;
            unknown |= bit == VerilogLogicBit::X;
        }
        return unknown ? VerilogLogicBit::X : VerilogLogicBit::Zero;
    }

    static std::optional<size_t> shift_amount(const VerilogConstantValue& value) {
        if (!value.is_fully_known() ||
            (value.is_signed && !value.bits.empty() &&
             value.bits.back() == VerilogLogicBit::One))
            return std::nullopt;
        for (size_t i = 64; i < value.bits.size(); ++i)
            if (value.bits[i] == VerilogLogicBit::One)
                return std::nullopt;
        uint64_t result = 0;
        for (size_t i = 0; i < std::min<size_t>(64, value.bits.size()); ++i)
            if (value.bits[i] == VerilogLogicBit::One)
                result |= uint64_t{1} << i;
        if (result > std::numeric_limits<size_t>::max())
            return std::nullopt;
        return static_cast<size_t>(result);
    }

    std::optional<VerilogConstantValue> eval_conditional(
        const VerilogExpression& expression) {
        if (!expression.first || !expression.second || !expression.third) {
            error("InvalidConstantExpression", "conditional expression is incomplete",
                  expression.location);
            return std::nullopt;
        }
        auto condition = eval(*expression.first);
        if (!condition)
            return std::nullopt;
        auto when_true = eval(*expression.second);
        auto when_false = eval(*expression.third);
        if (!when_true || !when_false)
            return std::nullopt;
        const size_t width = std::max(when_true->bit_width, when_false->bit_width);
        const bool sign = when_true->is_signed && when_false->is_signed;
        auto left = resized(*when_true, width, sign);
        auto right = resized(*when_false, width, sign);
        const auto condition_truth = logical_truth(*condition);
        if (condition_truth == VerilogLogicBit::One)
            return left;
        if (condition_truth == VerilogLogicBit::Zero)
            return right;
        auto result = make_value(width, sign);
        for (size_t i = 0; i < width; ++i)
            result.bits[i] = left.bits[i] == right.bits[i] ? left.bits[i] : VerilogLogicBit::X;
        return result;
    }

    std::optional<VerilogConstantValue> eval_concatenation(
        const VerilogExpression& expression) {
        if (expression.operands.empty()) {
            error("InvalidConcatenation", "constant concatenation cannot be empty",
                  expression.location);
            return std::nullopt;
        }
        std::vector<VerilogConstantValue> values;
        size_t width = 0;
        for (const auto& operand : expression.operands) {
            if (!operand)
                continue;
            auto value = eval(*operand);
            if (!value)
                return std::nullopt;
            if (value->bit_width > options.maximum_width - width) {
                error("InvalidConstantWidth", "concatenation exceeds maximum supported width",
                      expression.location);
                return std::nullopt;
            }
            width += value->bit_width;
            values.push_back(std::move(*value));
        }
        auto result = make_value(width, false);
        size_t destination = 0;
        for (auto value = values.rbegin(); value != values.rend(); ++value)
            for (auto bit : value->bits)
                result.bits[destination++] = bit;
        return result;
    }

    std::optional<VerilogConstantValue> eval_replication(
        const VerilogExpression& expression) {
        if (!expression.first || expression.operands.empty()) {
            error("InvalidReplication", "constant replication requires a count and expression",
                  expression.location);
            return std::nullopt;
        }
        auto count_value = eval(*expression.first);
        if (!count_value)
            return std::nullopt;
        const auto count = count_value->to_uint64();
        if (!count || *count == 0 || *count > options.maximum_replication_count) {
            error("InvalidReplicationCount", "replication count must be a positive known integer",
                  expression.first->location);
            return std::nullopt;
        }
        VerilogExpression concatenation;
        concatenation.kind = VerilogExpressionKind::Concatenation;
        concatenation.location = expression.location;
        for (const auto& operand : expression.operands)
            concatenation.operands.push_back(clone_expression(operand.get()));
        auto value = eval_concatenation(concatenation);
        if (!value)
            return std::nullopt;
        if (value->bit_width > options.maximum_width / *count) {
            error("InvalidConstantWidth", "replication exceeds maximum supported width",
                  expression.location);
            return std::nullopt;
        }
        auto result = make_value(value->bit_width * static_cast<size_t>(*count), false);
        for (size_t repeat = 0; repeat < *count; ++repeat)
            std::copy(value->bits.begin(), value->bits.end(),
                      result.bits.begin() + static_cast<std::ptrdiff_t>(repeat * value->bit_width));
        return result;
    }

    static std::unique_ptr<VerilogExpression> clone_expression(const VerilogExpression* source) {
        if (!source)
            return nullptr;
        auto result = std::make_unique<VerilogExpression>();
        result->kind = source->kind;
        result->select_kind = source->select_kind;
        result->location = source->location;
        result->text = source->text;
        result->member_separator = source->member_separator;
        result->first = clone_expression(source->first.get());
        result->second = clone_expression(source->second.get());
        result->third = clone_expression(source->third.get());
        for (const auto& operand : source->operands)
            result->operands.push_back(clone_expression(operand.get()));
        return result;
    }

    std::optional<VerilogConstantValue> eval_select(const VerilogExpression& expression) {
        if (!expression.first || !expression.second) {
            error("InvalidConstantSelect", "constant select is incomplete", expression.location);
            return std::nullopt;
        }
        if (expression.select_kind == VerilogSelectKind::IndexedUp ||
            expression.select_kind == VerilogSelectKind::IndexedDown) {
            error("UnsupportedIndexedPartSelectExpression",
                  "indexed part-select reads are not supported by the constant evaluator",
                  expression.location);
            return std::nullopt;
        }
        auto value = eval(*expression.first);
        auto left_index = eval(*expression.second);
        auto right_index = expression.third ? eval(*expression.third) : std::nullopt;
        if (!value || !left_index || (expression.third && !right_index))
            return std::nullopt;
        const auto left = left_index->to_int64();
        const auto right = right_index ? right_index->to_int64() : left;
        if (!left || !right || *left < 0 || *right < 0 ||
            static_cast<uint64_t>(std::max(*left, *right)) >= value->bit_width) {
            error("InvalidConstantSelect", "constant select index is outside the value width",
                  expression.location);
            return std::nullopt;
        }
        const size_t width = static_cast<size_t>(std::llabs(*left - *right) + 1);
        auto result = make_value(width, false);
        const size_t low = static_cast<size_t>(std::min(*left, *right));
        std::copy_n(value->bits.begin() + static_cast<std::ptrdiff_t>(low), width,
                    result.bits.begin());
        return result;
    }

    std::optional<VerilogConstantValue> eval_call(const VerilogExpression& expression) {
        const auto* callee = expression.first.get();
        if (callee && callee->kind == VerilogExpressionKind::Identifier &&
            callee->text == "$clog2" && expression.operands.size() == 1) {
            auto argument = eval(*expression.operands.front());
            if (!argument || has_unknown(*argument))
                return std::nullopt;
            if (argument->is_signed && argument->to_int64() &&
                *argument->to_int64() < 0) {
                error("InvalidConstantFunctionArgument",
                      "$clog2 requires a nonnegative argument", expression.location);
                return std::nullopt;
            }
            const auto value = argument->to_uint64();
            if (!value) {
                error("InvalidConstantFunctionArgument", "$clog2 requires a nonnegative argument",
                      expression.location);
                return std::nullopt;
            }
            uint64_t n = *value > 0 ? *value - 1 : 0;
            size_t bits = 0;
            while (n) {
                ++bits;
                n >>= 1;
            }
            auto result = make_value(kDefaultWidth, true);
            size_t value_bits = bits;
            size_t bit = 0;
            while (value_bits) {
                if (value_bits & 1U)
                    result.bits[bit] = VerilogLogicBit::One;
                value_bits >>= 1;
                ++bit;
            }
            return result;
        }
        if (callee && callee->kind == VerilogExpressionKind::Identifier &&
            (callee->text == "$signed" || callee->text == "$unsigned") &&
            expression.operands.size() == 1) {
            auto value = eval(*expression.operands.front());
            if (value)
                value->is_signed = callee->text == "$signed";
            return value;
        }
        error("UnsupportedConstantExpression", "function call is not a supported constant function",
              expression.location);
        return std::nullopt;
    }
};

} // namespace

bool VerilogConstantValue::is_fully_known() const noexcept {
    return std::all_of(bits.begin(), bits.end(), [](auto bit) {
        return bit == VerilogLogicBit::Zero || bit == VerilogLogicBit::One;
    });
}

std::optional<uint64_t> VerilogConstantValue::to_uint64() const noexcept {
    if (!is_fully_known() || bit_width > 64 || bits.size() != bit_width)
        return std::nullopt;
    uint64_t value = 0;
    for (size_t i = 0; i < bits.size(); ++i)
        if (bits[i] == VerilogLogicBit::One)
            value |= uint64_t{1} << i;
    return value;
}

std::optional<int64_t> VerilogConstantValue::to_int64() const noexcept {
    if (!is_fully_known() || bit_width > 64 || bits.size() != bit_width)
        return std::nullopt;
    uint64_t value = 0;
    for (size_t i = 0; i < bits.size(); ++i)
        if (bits[i] == VerilogLogicBit::One)
            value |= uint64_t{1} << i;
    if (!is_signed && value > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
        return std::nullopt;
    if (is_signed && bit_width != 0 && bit_width < 64 &&
        bits.back() == VerilogLogicBit::One)
        value |= (~uint64_t{0}) << bit_width;
    return std::bit_cast<int64_t>(value);
}

std::string VerilogConstantValue::to_binary_string() const {
    std::string result;
    result.reserve(bits.size());
    for (auto bit = bits.rbegin(); bit != bits.rend(); ++bit) {
        switch (*bit) {
            case VerilogLogicBit::Zero:
                result.push_back('0');
                break;
            case VerilogLogicBit::One:
                result.push_back('1');
                break;
            case VerilogLogicBit::X:
                result.push_back('x');
                break;
            case VerilogLogicBit::Z:
                result.push_back('z');
                break;
        }
    }
    return result;
}

VerilogConstantEvaluationResult evaluate_verilog_constant(
    const VerilogExpression& expression, const VerilogConstantEnvironment& environment,
    VerilogConstantResolver resolver, VerilogConstantEvaluationOptions options) {
    Evaluator evaluator{environment, std::move(resolver), options, {}};
    auto value = evaluator.eval(expression);
    if (value)
        value->location = expression.location;
    return {std::move(value), std::move(evaluator.diagnostics)};
}

} // namespace labneura::eda
