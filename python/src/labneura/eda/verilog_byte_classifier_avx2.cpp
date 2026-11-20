#include "verilog_byte_classifier.h"

#include <array>
#include <cstring>
#include <immintrin.h>

namespace labneura::eda::detail::verilog_lexer {
namespace {

__m256i equal_to(__m256i bytes, unsigned char value) {
    return _mm256_cmpeq_epi8(bytes, _mm256_set1_epi8(static_cast<char>(value)));
}

__m256i combine_equalities(__m256i bytes, std::string_view values) {
    __m256i result = _mm256_setzero_si256();
    for (unsigned char value : values)
        result = _mm256_or_si256(result, equal_to(bytes, value));
    return result;
}

uint32_t lane_mask(__m256i value) {
    return static_cast<uint32_t>(_mm256_movemask_epi8(value));
}

} // namespace

ClassificationBlock classify_avx2(const unsigned char* input) {
    __m256i bytes;
    std::memcpy(&bytes, input, sizeof(bytes));
    const __m256i digit =
        _mm256_and_si256(_mm256_cmpgt_epi8(bytes, _mm256_set1_epi8('0' - 1)),
                         _mm256_cmpgt_epi8(_mm256_set1_epi8('9' + 1), bytes));
    const __m256i lower =
        _mm256_and_si256(_mm256_cmpgt_epi8(bytes, _mm256_set1_epi8('a' - 1)),
                         _mm256_cmpgt_epi8(_mm256_set1_epi8('z' + 1), bytes));
    const __m256i upper =
        _mm256_and_si256(_mm256_cmpgt_epi8(bytes, _mm256_set1_epi8('A' - 1)),
                         _mm256_cmpgt_epi8(_mm256_set1_epi8('Z' + 1), bytes));
    const __m256i identifier_start =
        _mm256_or_si256(_mm256_or_si256(lower, upper), equal_to(bytes, '_'));
    const __m256i whitespace = combine_equalities(bytes, " \t\n\r\f\v");
    const __m256i delimiter = combine_equalities(bytes, "()[]{}.,;:@#'");
    const __m256i op = combine_equalities(bytes, "+-*/%&|^~!<>=?");

    constexpr std::array<uint16_t, 12> flags = {
        mask(ByteClass::Whitespace), mask(ByteClass::Newline),
        mask(ByteClass::CarriageReturn), mask(ByteClass::DecimalDigit),
        mask(ByteClass::IdentifierStart), mask(ByteClass::IdentifierContinue),
        mask(ByteClass::DoubleQuote), mask(ByteClass::Backslash),
        mask(ByteClass::Slash), mask(ByteClass::Asterisk),
        mask(ByteClass::Delimiter), mask(ByteClass::Operator)
    };
    const std::array<uint32_t, 12> masks = {
        lane_mask(whitespace),
        lane_mask(equal_to(bytes, '\n')),
        lane_mask(equal_to(bytes, '\r')),
        lane_mask(digit),
        lane_mask(identifier_start),
        lane_mask(_mm256_or_si256(_mm256_or_si256(identifier_start, digit),
                                  equal_to(bytes, '$'))),
        lane_mask(equal_to(bytes, '"')),
        lane_mask(equal_to(bytes, '\\')),
        lane_mask(equal_to(bytes, '/')),
        lane_mask(equal_to(bytes, '*')),
        lane_mask(delimiter),
        lane_mask(op)
    };

    ClassificationBlock result;
    result.size = classification_block_width;
    for (size_t i = 0; i < classification_block_width; ++i) {
        uint16_t byte_flags = 0;
        const uint32_t lane = uint32_t{1} << i;
        for (size_t class_index = 0; class_index < masks.size(); ++class_index) {
            if ((masks[class_index] & lane) != 0)
                byte_flags |= flags[class_index];
        }
        result.low_flags[i] = static_cast<uint8_t>(byte_flags);
        result.high_flags[i] = static_cast<uint8_t>(byte_flags >> 8);
        result.any_low |= result.low_flags[i];
        result.any_high |= result.high_flags[i];
        if (i == 0) {
            result.all_low = result.low_flags[i];
            result.all_high = result.high_flags[i];
        }
        else {
            result.all_low &= result.low_flags[i];
            result.all_high &= result.high_flags[i];
        }
    }

    return result;
}

} // namespace labneura::eda::detail::verilog_lexer
