#include "verilog_byte_classifier.h"

#include <algorithm>
#include <bit>
#include <stdexcept>

#if defined(LABNEURA_VERILOG_LEXER_HAS_NEON)
#    include <arm_neon.h>
#endif

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#    include <intrin.h>
#endif

namespace labneura::eda::detail::verilog_lexer {
#if defined(LABNEURA_VERILOG_LEXER_HAS_AVX2)
ClassificationBlock classify_avx2(const unsigned char* input);
#endif

namespace {

uint16_t scalar_flags(unsigned char ch) noexcept {
    uint16_t result = 0;
    const bool lower = ch >= 'a' && ch <= 'z';
    const bool upper = ch >= 'A' && ch <= 'Z';
    const bool identifier_start = lower || upper || ch == '_';
    if (ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r' || ch == '\f' || ch == '\v')
        result |= mask(ByteClass::Whitespace);
    if (ch == '\n')
        result |= mask(ByteClass::Newline);
    if (ch == '\r')
        result |= mask(ByteClass::CarriageReturn);
    if (ch >= '0' && ch <= '9')
        result |= mask(ByteClass::DecimalDigit);
    if (identifier_start)
        result |= mask(ByteClass::IdentifierStart);
    if (identifier_start || (ch >= '0' && ch <= '9') || ch == '$')
        result |= mask(ByteClass::IdentifierContinue);
    if (ch == '"')
        result |= mask(ByteClass::DoubleQuote);
    if (ch == '\\')
        result |= mask(ByteClass::Backslash);
    if (ch == '/')
        result |= mask(ByteClass::Slash);
    if (ch == '*')
        result |= mask(ByteClass::Asterisk);
    if (ch == '(' || ch == ')' || ch == '[' || ch == ']' || ch == '{' || ch == '}' ||
        ch == ',' || ch == ';' || ch == ':' || ch == '.' || ch == '@' || ch == '#' ||
        ch == '\'')
        result |= mask(ByteClass::Delimiter);
    if (ch == '+' || ch == '-' || ch == '*' || ch == '/' || ch == '%' || ch == '&' ||
        ch == '|' || ch == '^' || ch == '~' || ch == '!' || ch == '<' || ch == '>' ||
        ch == '=' || ch == '?')
        result |= mask(ByteClass::Operator);
    return result;
}

ClassificationBlock classify_scalar(const unsigned char* input, size_t size) {
    ClassificationBlock result;
    result.size = size;
    if (size == 0)
        return result;

    result.all_low = 0xff;
    result.all_high = 0xff;
    for (size_t i = 0; i < size; ++i) {
        const uint16_t flags = scalar_flags(input[i]);
        const uint8_t low = static_cast<uint8_t>(flags);
        const uint8_t high = static_cast<uint8_t>(flags >> 8);
        result.low_flags[i] = low;
        result.high_flags[i] = high;
        result.any_low |= low;
        result.any_high |= high;
        result.all_low &= low;
        result.all_high &= high;
    }
    return result;
}

#if defined(LABNEURA_VERILOG_LEXER_HAS_NEON)
uint8x16_t equal_to(uint8x16_t bytes, unsigned char value) {
    return vceqq_u8(bytes, vdupq_n_u8(value));
}

uint8x16_t identifier_start_mask(uint8x16_t bytes) {
    const uint8x16_t lower =
        vandq_u8(vcgeq_u8(bytes, vdupq_n_u8('a')), vcleq_u8(bytes, vdupq_n_u8('z')));
    const uint8x16_t upper =
        vandq_u8(vcgeq_u8(bytes, vdupq_n_u8('A')), vcleq_u8(bytes, vdupq_n_u8('Z')));
    return vorrq_u8(vorrq_u8(lower, upper), equal_to(bytes, '_'));
}

uint8x16_t combine_equalities(uint8x16_t bytes, std::string_view values) {
    uint8x16_t result = vdupq_n_u8(0);
    for (unsigned char value : values)
        result = vorrq_u8(result, equal_to(bytes, value));
    return result;
}

uint8_t reduce_or(uint8x16_t values) {
    values = vorrq_u8(values, vextq_u8(values, values, 8));
    values = vorrq_u8(values, vextq_u8(values, values, 4));
    values = vorrq_u8(values, vextq_u8(values, values, 2));
    values = vorrq_u8(values, vextq_u8(values, values, 1));
    return vgetq_lane_u8(values, 0);
}

uint8_t reduce_and(uint8x16_t values) {
    values = vandq_u8(values, vextq_u8(values, values, 8));
    values = vandq_u8(values, vextq_u8(values, values, 4));
    values = vandq_u8(values, vextq_u8(values, values, 2));
    values = vandq_u8(values, vextq_u8(values, values, 1));
    return vgetq_lane_u8(values, 0);
}

uint8x16_t add_flag(uint8x16_t current, uint8x16_t predicate, uint8_t flag) {
    return vorrq_u8(current, vandq_u8(predicate, vdupq_n_u8(flag)));
}

ClassificationBlock classify_neon_16(const unsigned char* input) {
    const uint8x16_t bytes = vld1q_u8(input);
    const uint8x16_t zero = vdupq_n_u8(0);
    const uint8x16_t digit =
        vandq_u8(vcgeq_u8(bytes, vdupq_n_u8('0')), vcleq_u8(bytes, vdupq_n_u8('9')));
    const uint8x16_t start = identifier_start_mask(bytes);
    const uint8x16_t whitespace = combine_equalities(bytes, " \t\n\r\f\v");
    const uint8x16_t delimiter = combine_equalities(bytes, "()[]{}.,;:@#'");
    const uint8x16_t op =
        combine_equalities(bytes, "+-*/%&|^~!<>=?");

    uint8x16_t low = zero;
    low = add_flag(low, whitespace, 1u << 0);
    low = add_flag(low, equal_to(bytes, '\n'), 1u << 1);
    low = add_flag(low, equal_to(bytes, '\r'), 1u << 2);
    low = add_flag(low, digit, 1u << 3);
    low = add_flag(low, start, 1u << 4);
    low = add_flag(low, vorrq_u8(vorrq_u8(start, digit), equal_to(bytes, '$')), 1u << 5);
    low = add_flag(low, equal_to(bytes, '"'), 1u << 6);
    low = add_flag(low, equal_to(bytes, '\\'), 1u << 7);

    uint8x16_t high = zero;
    high = add_flag(high, equal_to(bytes, '/'), 1u << 0);
    high = add_flag(high, equal_to(bytes, '*'), 1u << 1);
    high = add_flag(high, delimiter, 1u << 2);
    high = add_flag(high, op, 1u << 3);

    ClassificationBlock result;
    result.size = 16;
    vst1q_u8(result.low_flags.data(), low);
    vst1q_u8(result.high_flags.data(), high);
    result.any_low = reduce_or(low);
    result.any_high = reduce_or(high);
    result.all_low = reduce_and(low);
    result.all_high = reduce_and(high);
    return result;
}
#endif

#if defined(LABNEURA_VERILOG_LEXER_HAS_AVX2)
bool detect_avx2_runtime_support() noexcept {
#    if defined(__clang__) || defined(__GNUC__)
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2");
#    elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    int registers[4];
    __cpuid(registers, 1);
    constexpr int avx = 1 << 28;
    constexpr int osxsave = 1 << 27;
    if ((registers[2] & (avx | osxsave)) != (avx | osxsave) || (_xgetbv(0) & 0x6) != 0x6)
        return false;
    __cpuidex(registers, 7, 0);
    return (registers[1] & (1 << 5)) != 0;
#    else
    return false;
#    endif
}
#endif

#if defined(LABNEURA_VERILOG_LEXER_HAS_NEON)
ClassificationBlock merge_blocks(const ClassificationBlock& first,
                                 const ClassificationBlock& second) {
    ClassificationBlock result;
    result.size = first.size + second.size;
    for (size_t i = 0; i < first.size; ++i) {
        result.low_flags[i] = first.low_flags[i];
        result.high_flags[i] = first.high_flags[i];
    }
    for (size_t i = 0; i < second.size; ++i) {
        result.low_flags[first.size + i] = second.low_flags[i];
        result.high_flags[first.size + i] = second.high_flags[i];
    }
    result.any_low = first.any_low | second.any_low;
    result.any_high = first.any_high | second.any_high;
    if (first.size == 0) {
        result.all_low = second.all_low;
        result.all_high = second.all_high;
    }
    else if (second.size == 0) {
        result.all_low = first.all_low;
        result.all_high = first.all_high;
    }
    else {
        result.all_low = first.all_low & second.all_low;
        result.all_high = first.all_high & second.all_high;
    }
    return result;
}
#endif

void validate_range(std::string_view text, size_t start, size_t end) {
    if (start > end || end > text.size())
        throw std::out_of_range("Verilog classification range is outside the source buffer");
}

void validate_backend(VerilogLexerBackend backend) {
    if (backend == VerilogLexerBackend::Auto)
        throw std::invalid_argument("classification requires a concrete backend");
    if (backend == VerilogLexerBackend::Neon && !neon_available())
        throw std::invalid_argument("NEON Verilog byte classification is not available");
    if (backend == VerilogLexerBackend::Avx2 && !avx2_available())
        throw std::invalid_argument("AVX2 Verilog byte classification is not available");
    if (backend != VerilogLexerBackend::Scalar && backend != VerilogLexerBackend::Neon &&
        backend != VerilogLexerBackend::Avx2)
        throw std::invalid_argument("unknown Verilog byte-classification backend");
}

ClassificationBlock classify_block_unchecked(const unsigned char* input, size_t size,
                                             VerilogLexerBackend backend) {
#if defined(LABNEURA_VERILOG_LEXER_HAS_AVX2)
    if (backend == VerilogLexerBackend::Avx2 && size == classification_block_width)
        return classify_avx2(input);
#endif
#if defined(LABNEURA_VERILOG_LEXER_HAS_NEON)
    if (backend == VerilogLexerBackend::Neon && size == classification_block_width) {
        return merge_blocks(classify_neon_16(input),
                            classify_neon_16(input + classification_block_width / 2));
    }
#endif
    return classify_scalar(input, size);
}

} // namespace

uint16_t ClassificationBlock::flags_at(size_t index) const noexcept {
    if (index >= size)
        return 0;
    return static_cast<uint16_t>(low_flags[index]) |
           (static_cast<uint16_t>(high_flags[index]) << 8);
}

bool ClassificationBlock::has_any(ByteClassMask classes) const noexcept {
    const auto low = static_cast<uint8_t>(classes);
    const auto high = static_cast<uint8_t>(classes >> 8);
    return (any_low & low) != 0 || (any_high & high) != 0;
}

bool ClassificationBlock::every_has(ByteClass byte_class) const noexcept {
    const auto classes = mask(byte_class);
    const auto low = static_cast<uint8_t>(classes);
    const auto high = static_cast<uint8_t>(classes >> 8);
    return size != 0 && (all_low & low) == low && (all_high & high) == high;
}

bool neon_available() noexcept {
#if defined(LABNEURA_VERILOG_LEXER_HAS_NEON)
    return true;
#else
    return false;
#endif
}

bool avx2_available() noexcept {
#if defined(LABNEURA_VERILOG_LEXER_HAS_AVX2)
    static const bool supported = detect_avx2_runtime_support();
    return supported;
#else
    return false;
#endif
}

ClassificationBlock classify_block(const unsigned char* input, size_t size,
                                   VerilogLexerBackend backend) {
    if (size > classification_block_width)
        throw std::invalid_argument("Verilog classification block exceeds 32 bytes");
    if (size != 0 && input == nullptr)
        throw std::invalid_argument("Verilog classification input must not be null");
    validate_backend(backend);
    return classify_block_unchecked(input, size, backend);
}

ClassificationScanner::ClassificationScanner(std::string_view text,
                                             VerilogLexerBackend backend) :
    text_(text), backend_(backend) {
    validate_backend(backend);
}

const ClassificationBlock& ClassificationScanner::block_at(size_t source_offset) {
    const size_t block_start =
        source_offset - source_offset % classification_block_width;
    if (!has_cached_block_ || cached_block_start_ != block_start) {
        const size_t block_size =
            std::min(classification_block_width, text_.size() - block_start);
        const auto* bytes =
            reinterpret_cast<const unsigned char*>(text_.data() + block_start);
        cached_block_ = classify_block_unchecked(bytes, block_size, backend_);
        cached_block_start_ = block_start;
        has_cached_block_ = true;
    }
    return cached_block_;
}

size_t ClassificationScanner::find_first(size_t start, size_t end,
                                         ByteClassMask classes) {
    validate_range(text_, start, end);
    while (start < end) {
        const auto& block = block_at(start);
        const size_t local_start = start - cached_block_start_;
        const size_t local_end = std::min(block.size, end - cached_block_start_);
        if (block.has_any(classes)) {
            for (size_t i = local_start; i < local_end; ++i) {
                if ((block.flags_at(i) & classes) != 0)
                    return cached_block_start_ + i;
            }
        }
        start = cached_block_start_ + block.size;
    }
    return end;
}

size_t ClassificationScanner::find_first_not(size_t start, size_t end,
                                             ByteClass byte_class) {
    validate_range(text_, start, end);
    const ByteClassMask classes = mask(byte_class);
    if (classes == 0 || !std::has_single_bit(classes))
        throw std::invalid_argument("find_first_not requires one byte class");
    while (start < end) {
        const auto& block = block_at(start);
        const size_t local_start = start - cached_block_start_;
        const size_t local_end = std::min(block.size, end - cached_block_start_);
        if (!block.every_has(byte_class)) {
            for (size_t i = local_start; i < local_end; ++i) {
                if ((block.flags_at(i) & classes) == 0)
                    return cached_block_start_ + i;
            }
        }
        start = cached_block_start_ + block.size;
    }
    return end;
}

size_t find_first(std::string_view text, size_t start, size_t end,
                  ByteClassMask classes, VerilogLexerBackend backend) {
    return ClassificationScanner(text, backend).find_first(start, end, classes);
}

size_t find_first_not(std::string_view text, size_t start, size_t end,
                      ByteClass byte_class, VerilogLexerBackend backend) {
    return ClassificationScanner(text, backend).find_first_not(start, end, byte_class);
}

} // namespace labneura::eda::detail::verilog_lexer
