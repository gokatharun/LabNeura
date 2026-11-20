#include "verilog_byte_classifier.h"

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

__m256i add_flag(__m256i current, __m256i predicate, unsigned char flag) {
    return _mm256_or_si256(
        current,
        _mm256_and_si256(predicate, _mm256_set1_epi8(static_cast<char>(flag))));
}

uint8_t reduce_or(__m256i values) {
    __m128i lanes = _mm_or_si128(_mm256_castsi256_si128(values),
                                 _mm256_extracti128_si256(values, 1));
    lanes = _mm_or_si128(lanes, _mm_srli_si128(lanes, 8));
    lanes = _mm_or_si128(lanes, _mm_srli_si128(lanes, 4));
    lanes = _mm_or_si128(lanes, _mm_srli_si128(lanes, 2));
    lanes = _mm_or_si128(lanes, _mm_srli_si128(lanes, 1));
    return static_cast<uint8_t>(_mm_cvtsi128_si32(lanes));
}

uint8_t reduce_and(__m256i values) {
    __m128i lanes = _mm_and_si128(_mm256_castsi256_si128(values),
                                  _mm256_extracti128_si256(values, 1));
    lanes = _mm_and_si128(lanes, _mm_srli_si128(lanes, 8));
    lanes = _mm_and_si128(lanes, _mm_srli_si128(lanes, 4));
    lanes = _mm_and_si128(lanes, _mm_srli_si128(lanes, 2));
    lanes = _mm_and_si128(lanes, _mm_srli_si128(lanes, 1));
    return static_cast<uint8_t>(_mm_cvtsi128_si32(lanes));
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
    const __m256i zero = _mm256_setzero_si256();

    __m256i low = zero;
    low = add_flag(low, whitespace, 1u << 0);
    low = add_flag(low, equal_to(bytes, '\n'), 1u << 1);
    low = add_flag(low, equal_to(bytes, '\r'), 1u << 2);
    low = add_flag(low, digit, 1u << 3);
    low = add_flag(low, identifier_start, 1u << 4);
    low = add_flag(low,
                   _mm256_or_si256(_mm256_or_si256(identifier_start, digit),
                                   equal_to(bytes, '$')),
                   1u << 5);
    low = add_flag(low, equal_to(bytes, '"'), 1u << 6);
    low = add_flag(low, equal_to(bytes, '\\'), 1u << 7);

    __m256i high = zero;
    high = add_flag(high, equal_to(bytes, '/'), 1u << 0);
    high = add_flag(high, equal_to(bytes, '*'), 1u << 1);
    high = add_flag(high, delimiter, 1u << 2);
    high = add_flag(high, op, 1u << 3);

    ClassificationBlock result;
    result.size = classification_block_width;
    std::memcpy(result.low_flags.data(), &low, sizeof(low));
    std::memcpy(result.high_flags.data(), &high, sizeof(high));
    result.any_low = reduce_or(low);
    result.any_high = reduce_or(high);
    result.all_low = reduce_and(low);
    result.all_high = reduce_and(high);

    return result;
}

uint32_t matching_lanes_avx2(const ClassificationBlock& block,
                             ByteClassMask classes) {
    __m256i low;
    __m256i high;
    std::memcpy(&low, block.low_flags.data(), sizeof(low));
    std::memcpy(&high, block.high_flags.data(), sizeof(high));
    const __m256i zero = _mm256_setzero_si256();
    const __m256i low_selected = _mm256_and_si256(
        low, _mm256_set1_epi8(static_cast<char>(classes)));
    const __m256i high_selected = _mm256_and_si256(
        high, _mm256_set1_epi8(static_cast<char>(classes >> 8)));
    const __m256i selected = _mm256_or_si256(low_selected, high_selected);
    return ~static_cast<uint32_t>(
        _mm256_movemask_epi8(_mm256_cmpeq_epi8(selected, zero)));
}

} // namespace labneura::eda::detail::verilog_lexer
