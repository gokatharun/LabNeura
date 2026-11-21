#pragma once

#include "labneura/eda/verilog_lexer.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace labneura::eda::detail::verilog_lexer {

inline constexpr size_t classification_block_width = 32;

enum class ByteClass : uint16_t {
    Whitespace = 1u << 0,
    Newline = 1u << 1,
    CarriageReturn = 1u << 2,
    DecimalDigit = 1u << 3,
    IdentifierStart = 1u << 4,
    IdentifierContinue = 1u << 5,
    DoubleQuote = 1u << 6,
    Backslash = 1u << 7,
    Slash = 1u << 8,
    Asterisk = 1u << 9,
    Delimiter = 1u << 10,
    Operator = 1u << 11
};

using ByteClassMask = uint16_t;

constexpr ByteClassMask mask(ByteClass byte_class) noexcept {
    return static_cast<ByteClassMask>(byte_class);
}

constexpr ByteClassMask operator|(ByteClass left, ByteClass right) noexcept {
    return mask(left) | mask(right);
}

constexpr ByteClassMask operator|(ByteClassMask left, ByteClass right) noexcept {
    return left | mask(right);
}

struct ClassificationBlock {
    size_t size = 0;
    std::array<uint8_t, classification_block_width> low_flags{};
    std::array<uint8_t, classification_block_width> high_flags{};
    uint8_t any_low = 0;
    uint8_t any_high = 0;
    uint8_t all_low = 0;
    uint8_t all_high = 0;

    uint16_t flags_at(size_t index) const noexcept;
    bool has_any(ByteClassMask classes) const noexcept;
    bool every_has(ByteClass byte_class) const noexcept;
};

class ClassificationScanner {
public:
    ClassificationScanner(std::string_view text, VerilogLexerBackend backend);

    size_t find_first(size_t start, size_t end, ByteClassMask classes);
    size_t find_first_not(size_t start, size_t end, ByteClass byte_class);

private:
    const ClassificationBlock& block_at(size_t source_offset);

    std::string_view text_;
    VerilogLexerBackend backend_;
    size_t cached_block_start_ = 0;
    bool has_cached_block_ = false;
    ClassificationBlock cached_block_{};
};

bool neon_available() noexcept;
bool avx2_available() noexcept;
ClassificationBlock classify_block(const unsigned char* input, size_t size,
                                   VerilogLexerBackend backend);
size_t find_first(std::string_view text, size_t start, size_t end,
                  ByteClassMask classes, VerilogLexerBackend backend);
size_t find_first_not(std::string_view text, size_t start, size_t end,
                      ByteClass byte_class, VerilogLexerBackend backend);

} // namespace labneura::eda::detail::verilog_lexer
