#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace labneura::eda {

enum class VerilogLanguageVersion : uint8_t {
    Verilog1995,
    Verilog2001,
    Verilog2005,
    SystemVerilog2005,
    SystemVerilog2009,
    SystemVerilog2012,
    SystemVerilog2017,
    SystemVerilog2023
};

enum class VerilogTokenKind : uint8_t {
    Identifier,
    EscapedIdentifier,
    SystemIdentifier,
    Keyword,
    CompilerDirective,
    Backtick,
    DecimalNumber,
    BasedNumber,
    UnbasedNumber,
    RealNumber,
    StringLiteral,
    LineComment,
    BlockComment,
    Operator,
    Punctuation,
    Invalid,
    EndOfFile
};

enum class VerilogLexErrorKind : uint8_t {
    InvalidCharacter,
    InvalidEscapedIdentifier,
    InvalidNumericLiteral,
    UnterminatedString,
    NewlineInString,
    UnterminatedBlockComment
};

struct VerilogSourceLocation {
    size_t line = 1;
    size_t column = 1;
};

class VerilogSourceBuffer {
public:
    explicit VerilogSourceBuffer(std::string text, std::string name = {});

    VerilogSourceBuffer(const VerilogSourceBuffer&) = delete;
    VerilogSourceBuffer& operator=(const VerilogSourceBuffer&) = delete;
    VerilogSourceBuffer(VerilogSourceBuffer&&) = delete;
    VerilogSourceBuffer& operator=(VerilogSourceBuffer&&) = delete;

    uint32_t id() const noexcept { return id_; }
    std::string_view name() const noexcept { return name_; }
    std::string_view text() const noexcept { return text_; }
    VerilogSourceLocation location(size_t byte_offset) const;
    // Optimized for monotonically increasing offsets. line_index_hint is
    // updated in place and safely reset if the caller moves backwards.
    VerilogSourceLocation location(size_t byte_offset,
                                   size_t& line_index_hint) const;

private:
    uint32_t id_;
    std::string name_;
    std::string text_;
    std::vector<size_t> line_starts_;
};

struct VerilogToken {
    VerilogTokenKind kind = VerilogTokenKind::Invalid;
    uint32_t source_id = 0;
    // Offsets and columns are byte-based; lines and columns are one-based.
    size_t byte_offset = 0;
    size_t length = 0;
    size_t line = 1;
    size_t column = 1;
};

struct VerilogLexError {
    VerilogLexErrorKind kind = VerilogLexErrorKind::InvalidCharacter;
    uint32_t source_id = 0;
    size_t byte_offset = 0;
    size_t length = 0;
    size_t line = 1;
    size_t column = 1;
};

enum class VerilogLexerBackend : uint8_t { Auto, Scalar, Neon, Avx2 };

struct VerilogLexerOptions {
    VerilogLanguageVersion language_version = VerilogLanguageVersion::SystemVerilog2017;
    VerilogLexerBackend backend = VerilogLexerBackend::Auto;
};

struct VerilogLexResult {
    std::shared_ptr<const VerilogSourceBuffer> source;
    std::vector<VerilogToken> tokens;
    std::vector<VerilogLexError> errors;

    std::string_view spelling(const VerilogToken& token) const;
};

enum class VerilogLexerSelection : uint8_t { Slang, Scalar };

// Reports the requested lexer; parse_verilog remains Slang-backed in this milestone.
VerilogLexerSelection configured_verilog_lexer();
// Returns the effective classifier backend selected by LABNEURA_VERILOG_LEXER_BACKEND.
VerilogLexerBackend configured_verilog_lexer_backend();

VerilogLexResult lex_verilog(
    std::shared_ptr<const VerilogSourceBuffer> source,
    VerilogLexerOptions options = {});
VerilogLexResult lex_verilog(std::string source, VerilogLexerOptions options = {},
                             std::string source_name = {});

} // namespace labneura::eda
