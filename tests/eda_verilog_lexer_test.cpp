#include "labneura/eda/verilog_lexer.h"

#if defined(LABNEURA_ENABLE_SLANG_LEXER_ORACLE)
#    include <slang/diagnostics/Diagnostics.h>
#    include <slang/parsing/Lexer.h>
#    include <slang/parsing/LexerFacts.h>
#    include <slang/text/SourceManager.h>
#    include <slang/util/BumpAllocator.h>
#endif

#include "labneura/eda/verilog_byte_classifier.h"

#include <cstdlib>
#include <array>
#include <fstream>
#include <iostream>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#ifndef LABNEURA_LEXER_CORPUS_PATH
#    error "LABNEURA_LEXER_CORPUS_PATH must be configured by CMake"
#endif

namespace {

using labneura::eda::VerilogLanguageVersion;
using labneura::eda::VerilogLexErrorKind;
using labneura::eda::VerilogLexerOptions;
using labneura::eda::VerilogLexerBackend;
using labneura::eda::VerilogLexerSelection;
using labneura::eda::VerilogTokenKind;
namespace byte_classifier = labneura::eda::detail::verilog_lexer;

void require(bool condition, std::string_view message) {
    if (!condition)
        throw std::runtime_error(std::string(message));
}

void set_lexer_environment(const char* value) {
#if defined(_WIN32)
    const int status = _putenv_s("LABNEURA_VERILOG_LEXER", value ? value : "");
#else
    const int status = value ? setenv("LABNEURA_VERILOG_LEXER", value, 1)
                             : unsetenv("LABNEURA_VERILOG_LEXER");
#endif
    if (status != 0)
        throw std::runtime_error("could not update LABNEURA_VERILOG_LEXER for the test");
}

void set_backend_environment(const char* value) {
#if defined(_WIN32)
    const int status = _putenv_s("LABNEURA_VERILOG_LEXER_BACKEND", value ? value : "");
#else
    const int status = value ? setenv("LABNEURA_VERILOG_LEXER_BACKEND", value, 1)
                             : unsetenv("LABNEURA_VERILOG_LEXER_BACKEND");
#endif
    if (status != 0)
        throw std::runtime_error(
            "could not update LABNEURA_VERILOG_LEXER_BACKEND for the test");
}

void test_empty_and_source_identity() {
    auto empty = labneura::eda::lex_verilog(std::string{});
    require(empty.tokens.size() == 1, "empty input must emit EOF");
    require(empty.tokens.front().kind == VerilogTokenKind::EndOfFile, "missing empty EOF");
    require(empty.tokens.front().line == 1 && empty.tokens.front().column == 1,
            "empty EOF location must be 1:1");
    require(empty.errors.empty(), "empty input must not produce errors");

    auto first = std::make_shared<const labneura::eda::VerilogSourceBuffer>("wire a;");
    auto second = std::make_shared<const labneura::eda::VerilogSourceBuffer>("wire a;");
    auto one = labneura::eda::lex_verilog(first);
    auto two = labneura::eda::lex_verilog(first);
    auto three = labneura::eda::lex_verilog(second);
    require(one.tokens[0].source_id == two.tokens[0].source_id,
            "repeated lexing must preserve source identity");
    require(one.tokens[0].source_id != three.tokens[0].source_id,
            "independent buffers must have independent source identities");
    bool rejected_foreign_token = false;
    try {
        (void)three.spelling(one.tokens.front());
    }
    catch (const std::invalid_argument&) {
        rejected_foreign_token = true;
    }
    require(rejected_foreign_token, "spelling lookup must reject a foreign token");
}

void test_identifiers_keywords_and_directives() {
    auto result = labneura::eda::lex_verilog(
        "module foo_2; \\escaped.name  $display `define ` ;");
    const std::vector kinds = {
        VerilogTokenKind::Keyword, VerilogTokenKind::Identifier, VerilogTokenKind::Punctuation,
        VerilogTokenKind::EscapedIdentifier, VerilogTokenKind::SystemIdentifier,
        VerilogTokenKind::CompilerDirective, VerilogTokenKind::Backtick,
        VerilogTokenKind::Punctuation, VerilogTokenKind::EndOfFile
    };
    const std::vector<std::string_view> spellings = {
        "module", "foo_2", ";", "\\escaped.name", "$display", "`define", "`", ";", ""
    };
    require(result.errors.empty(), "valid identifier and directive source must lex cleanly");
    require(result.tokens.size() == kinds.size(), "identifier token count mismatch");
    for (size_t i = 0; i < kinds.size(); ++i) {
        require(result.tokens[i].kind == kinds[i], "identifier category mismatch");
        require(result.spelling(result.tokens[i]) == spellings[i],
                "identifier or directive spelling mismatch");
    }

    auto macro_controls = labneura::eda::lex_verilog("`` `\" `\"\"\" `\\`\"");
    require(macro_controls.errors.empty(), "macro control tokens must lex cleanly");
    require(macro_controls.tokens.size() == 5, "macro control token count mismatch");
    for (size_t i = 0; i < 4; ++i)
        require(macro_controls.tokens[i].kind == VerilogTokenKind::Backtick,
                "macro quote and paste tokens must retain backtick category");

    auto legacy = labneura::eda::lex_verilog(
        "logic wire", {VerilogLanguageVersion::Verilog2005});
    require(legacy.tokens[0].kind == VerilogTokenKind::Identifier,
            "SystemVerilog keyword must be an identifier in Verilog mode");
    require(legacy.tokens[1].kind == VerilogTokenKind::Keyword,
            "Verilog keyword must remain a keyword in Verilog mode");
    auto old_verilog = labneura::eda::lex_verilog(
        "automatic", {VerilogLanguageVersion::Verilog1995});
    auto newer_verilog = labneura::eda::lex_verilog(
        "automatic", {VerilogLanguageVersion::Verilog2001});
    require(old_verilog.tokens[0].kind == VerilogTokenKind::Identifier &&
                newer_verilog.tokens[0].kind == VerilogTokenKind::Keyword,
            "Verilog keyword classification must follow introduction versions");
    auto modern = labneura::eda::lex_verilog(
        "logic", {VerilogLanguageVersion::SystemVerilog2017});
    require(modern.tokens[0].kind == VerilogTokenKind::Keyword,
            "SystemVerilog keyword must be classified for the selected version");
    auto earlier_keywords = labneura::eda::lex_verilog(
        "let", {VerilogLanguageVersion::SystemVerilog2005});
    auto later_keywords = labneura::eda::lex_verilog(
        "let nettype", {VerilogLanguageVersion::SystemVerilog2012});
    require(earlier_keywords.tokens[0].kind == VerilogTokenKind::Identifier &&
                later_keywords.tokens[0].kind == VerilogTokenKind::Keyword &&
                later_keywords.tokens[1].kind == VerilogTokenKind::Keyword,
            "keyword classification must follow introduction versions");

    auto invalid_escape = labneura::eda::lex_verilog("\\ wire");
    require(invalid_escape.errors.size() == 1 &&
                invalid_escape.errors[0].kind == VerilogLexErrorKind::InvalidEscapedIdentifier,
            "empty escaped identifier must report a lexical error");
}

void test_numbers() {
    auto result = labneura::eda::lex_verilog(
        "0 123 8'hFF 8'sb10xz 4'd? '0 '1 'x 'z 'hF 1.25 2. 3e+2 1e-3");
    const std::vector kinds = {
        VerilogTokenKind::DecimalNumber, VerilogTokenKind::DecimalNumber,
        VerilogTokenKind::BasedNumber, VerilogTokenKind::BasedNumber,
        VerilogTokenKind::BasedNumber, VerilogTokenKind::UnbasedNumber,
        VerilogTokenKind::UnbasedNumber, VerilogTokenKind::UnbasedNumber,
        VerilogTokenKind::UnbasedNumber, VerilogTokenKind::BasedNumber,
        VerilogTokenKind::RealNumber, VerilogTokenKind::RealNumber,
        VerilogTokenKind::RealNumber, VerilogTokenKind::RealNumber,
        VerilogTokenKind::EndOfFile
    };
    require(result.errors.empty(), "valid numeric literal variants must lex cleanly");
    require(result.tokens.size() == kinds.size(), "numeric token count mismatch");
    for (size_t i = 0; i < kinds.size(); ++i)
        require(result.tokens[i].kind == kinds[i], "numeric token kind mismatch");

    auto malformed = labneura::eda::lex_verilog("8'hG 3'b2 1e+ 12abc");
    require(malformed.errors.size() == 4, "malformed numeric literals must each report an error");
    for (size_t i = 0; i < 4; ++i)
        require(malformed.tokens[i].kind == VerilogTokenKind::Invalid,
                "malformed numeric token must be marked invalid");
}

void test_strings_comments_and_errors() {
    auto valid = labneura::eda::lex_verilog(
        "\"escaped \\\" quote\" // line\n/* block */ \"line\\\ncontinued\"");
    const std::vector kinds = {
        VerilogTokenKind::StringLiteral, VerilogTokenKind::LineComment,
        VerilogTokenKind::BlockComment, VerilogTokenKind::StringLiteral,
        VerilogTokenKind::EndOfFile
    };
    require(valid.errors.empty(), "valid strings and comments must lex cleanly");
    require(valid.tokens.size() == kinds.size(), "string/comment token count mismatch");
    for (size_t i = 0; i < kinds.size(); ++i)
        require(valid.tokens[i].kind == kinds[i], "string/comment token kind mismatch");

    auto malformed = labneura::eda::lex_verilog("\"newline\n /* never closed");
    require(malformed.errors.size() == 2, "malformed string and comment need two diagnostics");
    require(malformed.errors[0].kind == VerilogLexErrorKind::NewlineInString,
            "newline in string diagnostic mismatch");
    require(malformed.errors[1].kind == VerilogLexErrorKind::UnterminatedBlockComment,
            "unterminated block comment diagnostic mismatch");
    require(malformed.errors[1].line == 2, "block-comment diagnostic location mismatch");
    auto unterminated_string = labneura::eda::lex_verilog("\"unterminated");
    require(unterminated_string.errors.size() == 1 &&
                unterminated_string.errors[0].kind == VerilogLexErrorKind::UnterminatedString,
            "unterminated string must report its lexical error");
    auto invalid_system_name = labneura::eda::lex_verilog("$");
    require(invalid_system_name.errors.size() == 1 &&
                invalid_system_name.errors[0].kind == VerilogLexErrorKind::InvalidCharacter,
            "bare system identifier marker must report an invalid character");
}

void test_operators_and_punctuation() {
    constexpr std::string_view source =
        "<<< >>> <<<= >>>= === !== ==? !=? &&& ->> <<= >>= **= |-> |=> #-# #=# "
        "+/- +%- <-> ++ -- ** && || <= >= == != "
        "~& ~| ~^ ^~ << >> +: -: := :/ :: .* -> => *> ## += -= *= /= %= &= |= ^= = + - * / % "
        "& | ^ ~ ! < > ? ( ) [ ] { } , ; : . @ # '{ @@";
    auto result = labneura::eda::lex_verilog(std::string(source));
    require(result.errors.empty(), "operator and punctuation source must lex cleanly");
    require(result.tokens.back().kind == VerilogTokenKind::EndOfFile, "operator EOF missing");
    for (size_t i = 0; i + 1 < result.tokens.size(); ++i) {
        const auto token = result.tokens[i];
        const auto text = result.spelling(token);
        const bool punctuation = text == "'{" || text == "@@" ||
            (text.size() == 1 &&
             std::string_view("()[]{}.,;:@#").find(text.front()) != std::string_view::npos);
        require(token.kind == (punctuation ? VerilogTokenKind::Punctuation
                                           : VerilogTokenKind::Operator),
                "longest-match operator or punctuation category mismatch");
    }
    require(result.spelling(result.tokens[0]) == "<<<", "shift operator longest match failed");
    require(result.spelling(result.tokens[6]) == "==?", "wildcard equality longest match failed");
}

void test_locations_and_boundaries() {
    auto result = labneura::eda::lex_verilog("wire a;\r\n  logic b;\n");
    require(result.tokens[0].line == 1 && result.tokens[0].column == 1,
            "first token location mismatch");
    require(result.tokens[3].line == 2 && result.tokens[3].column == 3,
            "CRLF line/column location mismatch");
    require(result.tokens.back().line == 3 && result.tokens.back().column == 1,
            "EOF after newline location mismatch");
    bool rejected_out_of_range_location = false;
    try {
        (void)result.source->location(result.source->text().size() + 1);
    }
    catch (const std::out_of_range&) {
        rejected_out_of_range_location = true;
    }
    require(rejected_out_of_range_location, "source location must reject an out-of-range offset");

    size_t line_hint = 0;
    for (size_t offset = 0; offset <= result.source->text().size(); ++offset) {
        const auto expected = result.source->location(offset);
        const auto hinted = result.source->location(offset, line_hint);
        require(expected.line == hinted.line && expected.column == hinted.column,
                "monotonic hinted source location differs from binary-search lookup");
    }
    for (size_t offset = result.source->text().size(); offset != 0; --offset) {
        const auto expected = result.source->location(offset - 1);
        const auto hinted = result.source->location(offset - 1, line_hint);
        require(expected.line == hinted.line && expected.column == hinted.column,
                "backward hinted source location did not safely reset");
    }

    constexpr std::string_view complete = "module m; logic [3:0] a = 4'hf; endmodule";
    for (size_t size = 0; size <= complete.size(); ++size) {
        auto prefix = labneura::eda::lex_verilog(std::string(complete.substr(0, size)));
        require(prefix.tokens.back().kind == VerilogTokenKind::EndOfFile,
                "every prefix must terminate with EOF");
        require(prefix.tokens.back().byte_offset == size, "prefix EOF offset mismatch");
    }

    std::string large;
    large.reserve(2 * 1024 * 1024);
    while (large.size() < 2 * 1024 * 1024)
        large += "wire signal_123; // realistic repeated declaration\n";
    auto large_result = labneura::eda::lex_verilog(std::move(large));
    require(large_result.errors.empty(), "large source must lex without errors");
    require(large_result.tokens.size() > 150000, "large source token count unexpectedly small");
}

void test_environment_selection() {
    const char* existing = std::getenv("LABNEURA_VERILOG_LEXER");
    const std::optional<std::string> old_value =
        existing ? std::optional<std::string>(existing) : std::nullopt;
    auto restore = [&] {
        if (old_value)
            set_lexer_environment(old_value->c_str());
        else
            set_lexer_environment(nullptr);
    };

    try {
        set_lexer_environment(nullptr);
        require(labneura::eda::configured_verilog_lexer() == VerilogLexerSelection::Slang,
                "Slang must remain the default selection");
        set_lexer_environment("scalar");
        require(labneura::eda::configured_verilog_lexer() == VerilogLexerSelection::Scalar,
                "scalar environment selection failed");
        set_lexer_environment("unknown");
        bool rejected = false;
        try {
            (void)labneura::eda::configured_verilog_lexer();
        }
        catch (const std::invalid_argument&) {
            rejected = true;
        }
        require(rejected, "unknown environment selection must be rejected");
    }
    catch (...) {
        restore();
        throw;
    }
    restore();
}

std::string read_corpus() {
    std::ifstream input(LABNEURA_LEXER_CORPUS_PATH, std::ios::binary);
    if (!input)
        throw std::runtime_error("could not read the SystemVerilog lexer corpus");
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

uint16_t expected_byte_classes(unsigned char ch) {
    using byte_classifier::ByteClass;
    uint16_t flags = 0;
    const bool start = (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || ch == '_';
    if (ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r' || ch == '\f' || ch == '\v')
        flags |= byte_classifier::mask(ByteClass::Whitespace);
    if (ch == '\n')
        flags |= byte_classifier::mask(ByteClass::Newline);
    if (ch == '\r')
        flags |= byte_classifier::mask(ByteClass::CarriageReturn);
    if (ch >= '0' && ch <= '9')
        flags |= byte_classifier::mask(ByteClass::DecimalDigit);
    if (start)
        flags |= byte_classifier::mask(ByteClass::IdentifierStart);
    if (start || (ch >= '0' && ch <= '9') || ch == '$')
        flags |= byte_classifier::mask(ByteClass::IdentifierContinue);
    if (ch == '"')
        flags |= byte_classifier::mask(ByteClass::DoubleQuote);
    if (ch == '\\')
        flags |= byte_classifier::mask(ByteClass::Backslash);
    if (ch == '/')
        flags |= byte_classifier::mask(ByteClass::Slash);
    if (ch == '*')
        flags |= byte_classifier::mask(ByteClass::Asterisk);
    if (std::string_view("()[]{}.,;:@#'").find(static_cast<char>(ch)) !=
        std::string_view::npos)
        flags |= byte_classifier::mask(ByteClass::Delimiter);
    if (std::string_view("+-*/%&|^~!<>=?").find(static_cast<char>(ch)) !=
        std::string_view::npos)
        flags |= byte_classifier::mask(ByteClass::Operator);
    return flags;
}

void compare_classification_blocks(const unsigned char* input, size_t size,
                                   VerilogLexerBackend backend) {
    const auto reference =
        byte_classifier::classify_block(input, size, VerilogLexerBackend::Scalar);
    const auto actual = byte_classifier::classify_block(input, size, backend);
    require(reference.size == actual.size, "classification block length differs by backend");
    require(reference.any_low == actual.any_low && reference.any_high == actual.any_high &&
                reference.all_low == actual.all_low && reference.all_high == actual.all_high,
            "classification aggregate differs by backend");
    for (size_t i = 0; i < byte_classifier::classification_block_width; ++i) {
        require(reference.flags_at(i) == actual.flags_at(i),
                "byte classification differs by backend");
        if (i < size)
            require(actual.flags_at(i) == expected_byte_classes(input[i]),
                    "scalar byte classification does not match the ASCII reference");
    }
}

void compare_classification_backends(const unsigned char* input, size_t size,
                                     VerilogLexerBackend left_backend,
                                     VerilogLexerBackend right_backend) {
    const auto left = byte_classifier::classify_block(input, size, left_backend);
    const auto right = byte_classifier::classify_block(input, size, right_backend);
    require(left.size == right.size && left.any_low == right.any_low &&
                left.any_high == right.any_high && left.all_low == right.all_low &&
                left.all_high == right.all_high,
            "optimized classifier aggregates differ");
    for (size_t i = 0; i < byte_classifier::classification_block_width; ++i)
        require(left.flags_at(i) == right.flags_at(i),
                "optimized byte classifiers differ");
}

void test_byte_classifiers() {
    std::array<unsigned char, 256> every_byte{};
    for (size_t i = 0; i < every_byte.size(); ++i)
        every_byte[i] = static_cast<unsigned char>(i);

    for (size_t offset = 0; offset < every_byte.size();
         offset += byte_classifier::classification_block_width) {
        const auto size = std::min(byte_classifier::classification_block_width,
                                   every_byte.size() - offset);
        compare_classification_blocks(every_byte.data() + offset, size,
                                      VerilogLexerBackend::Scalar);
        if (byte_classifier::neon_available())
            compare_classification_blocks(every_byte.data() + offset, size,
                                          VerilogLexerBackend::Neon);
        if (byte_classifier::avx2_available())
            compare_classification_blocks(every_byte.data() + offset, size,
                                          VerilogLexerBackend::Avx2);
        if (byte_classifier::neon_available() && byte_classifier::avx2_available())
            compare_classification_backends(every_byte.data() + offset, size,
                                            VerilogLexerBackend::Neon,
                                            VerilogLexerBackend::Avx2);
    }

    std::array<unsigned char, 16 * 5 + 32> unaligned{};
    for (size_t i = 0; i < unaligned.size(); ++i)
        unaligned[i] = every_byte[(i * 73 + 19) % every_byte.size()];
    for (size_t address_offset = 0; address_offset < 16; ++address_offset) {
        for (size_t size = 0; size <= 16 * 4 + 15; ++size) {
            const auto* input = unaligned.data() + address_offset;
            size_t consumed = 0;
            do {
                const size_t block_size = std::min(
                    byte_classifier::classification_block_width, size - consumed);
                compare_classification_blocks(input + consumed, block_size,
                                              VerilogLexerBackend::Scalar);
                if (byte_classifier::neon_available())
                    compare_classification_blocks(input + consumed, block_size,
                                                  VerilogLexerBackend::Neon);
                if (byte_classifier::avx2_available())
                    compare_classification_blocks(input + consumed, block_size,
                                                  VerilogLexerBackend::Avx2);
                if (byte_classifier::neon_available() && byte_classifier::avx2_available())
                    compare_classification_backends(input + consumed, block_size,
                                                    VerilogLexerBackend::Neon,
                                                    VerilogLexerBackend::Avx2);
                consumed += block_size;
            } while (consumed < size);
        }
    }
    compare_classification_blocks(nullptr, 0, VerilogLexerBackend::Scalar);
    if (byte_classifier::neon_available())
        compare_classification_blocks(nullptr, 0, VerilogLexerBackend::Neon);
    if (byte_classifier::avx2_available())
        compare_classification_blocks(nullptr, 0, VerilogLexerBackend::Avx2);
    if (byte_classifier::neon_available() && byte_classifier::avx2_available())
        compare_classification_backends(nullptr, 0, VerilogLexerBackend::Neon,
                                        VerilogLexerBackend::Avx2);

    std::array<unsigned char, 4096 + 32> randomized{};
    uint32_t state = 0x4c455834u;
    for (auto& byte : randomized) {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        byte = static_cast<unsigned char>(state);
    }
    constexpr std::array<size_t, 12> boundary_lengths = {
        0, 1, 15, 16, 31, 32, 33, 63, 64, 65, 257, 4096
    };
    for (size_t address_offset = 0; address_offset < 32; ++address_offset) {
        for (const size_t length : boundary_lengths) {
            size_t consumed = 0;
            do {
                const size_t block_size = std::min(
                    byte_classifier::classification_block_width, length - consumed);
                const auto* input = randomized.data() + address_offset + consumed;
                if (byte_classifier::neon_available())
                    compare_classification_blocks(input, block_size,
                                                  VerilogLexerBackend::Neon);
                if (byte_classifier::avx2_available())
                    compare_classification_blocks(input, block_size,
                                                  VerilogLexerBackend::Avx2);
                consumed += block_size;
            } while (consumed < length);
        }
    }

    if (!byte_classifier::avx2_available())
        std::cout << "AVX2 classification parity explicitly skipped: unavailable at runtime\n";

    require(byte_classifier::find_first({}, 0, 0,
                                        byte_classifier::mask(byte_classifier::ByteClass::Slash),
                                        VerilogLexerBackend::Scalar) == 0,
            "empty classifier search must return its end position");
}

void test_classification_scanner() {
    using byte_classifier::ByteClass;

    std::string storage(1, 'x');
    for (size_t i = 0; i < 160; ++i) {
        storage.push_back(static_cast<char>((i * 73 + 19) & 0xff));
    }
    const std::string_view text(storage.data() + 1, storage.size() - 1);
    const std::array<byte_classifier::ByteClassMask, 7> candidate_sets = {
        byte_classifier::mask(ByteClass::Whitespace),
        ByteClass::Newline | ByteClass::CarriageReturn,
        byte_classifier::mask(ByteClass::IdentifierContinue),
        ByteClass::DoubleQuote | ByteClass::Backslash,
        ByteClass::Slash | ByteClass::Asterisk,
        byte_classifier::mask(ByteClass::Delimiter),
        byte_classifier::mask(ByteClass::Operator)
    };
    const std::array<ByteClass, 3> excluded_classes = {
        ByteClass::Whitespace, ByteClass::IdentifierContinue, ByteClass::Newline
    };
    std::vector<VerilogLexerBackend> backends = {VerilogLexerBackend::Scalar};
    if (byte_classifier::neon_available())
        backends.push_back(VerilogLexerBackend::Neon);
    if (byte_classifier::avx2_available())
        backends.push_back(VerilogLexerBackend::Avx2);

    const std::string boundary_text(32, 'x');
    const std::string boundary_source = boundary_text + " ";
    for (const auto backend : backends) {
        byte_classifier::ClassificationScanner boundary_scanner(boundary_source, backend);
        require(boundary_scanner.find_first(
                    32, 33, byte_classifier::mask(ByteClass::Whitespace)) == 32,
                "first classification lookup at a block boundary failed");
        require(boundary_scanner.find_first_not(
                    32, 33, ByteClass::IdentifierContinue) == 32,
                "first exclusion lookup at a block boundary failed");

        byte_classifier::ClassificationScanner scanner(text, backend);
        for (size_t start = 0; start <= text.size(); ++start) {
            const size_t max_length = std::min<size_t>(40, text.size() - start);
            for (size_t length = 0; length <= max_length; ++length) {
                const size_t end = start + length;
                for (const auto classes : candidate_sets) {
                    size_t expected = end;
                    for (size_t i = start; i < end; ++i) {
                        if ((expected_byte_classes(static_cast<unsigned char>(text[i])) &
                             classes) != 0) {
                            expected = i;
                            break;
                        }
                    }
                    require(scanner.find_first(start, end, classes) == expected,
                            "cached classification candidate search differs from scalar");
                }
                for (const auto byte_class : excluded_classes) {
                    size_t expected = end;
                    for (size_t i = start; i < end; ++i) {
                        if ((expected_byte_classes(static_cast<unsigned char>(text[i])) &
                             byte_classifier::mask(byte_class)) == 0) {
                            expected = i;
                            break;
                        }
                    }
                    require(scanner.find_first_not(start, end, byte_class) == expected,
                            "cached classification exclusion search differs from scalar");
                }
            }
        }
    }
}

void require_same_lex_result(const labneura::eda::VerilogLexResult& expected,
                             const labneura::eda::VerilogLexResult& actual) {
    require(expected.source->id() == actual.source->id(),
            "lexer backends changed the source identifier");
    require(expected.tokens.size() == actual.tokens.size(),
            "lexer backends emitted different token counts");
    require(expected.errors.size() == actual.errors.size(),
            "lexer backends emitted different diagnostic counts");
    for (size_t i = 0; i < expected.tokens.size(); ++i) {
        const auto& left = expected.tokens[i];
        const auto& right = actual.tokens[i];
        require(left.kind == right.kind && left.source_id == right.source_id &&
                    left.byte_offset == right.byte_offset && left.length == right.length &&
                    left.line == right.line && left.column == right.column,
                "lexer backend token metadata differs");
        require(expected.spelling(left) == actual.spelling(right),
                "lexer backend token spelling differs");
    }
    for (size_t i = 0; i < expected.errors.size(); ++i) {
        const auto& left = expected.errors[i];
        const auto& right = actual.errors[i];
        require(left.kind == right.kind && left.source_id == right.source_id &&
                    left.byte_offset == right.byte_offset && left.length == right.length &&
                    left.line == right.line && left.column == right.column,
                "lexer backend diagnostic differs");
    }
}

void test_lexer_backend_equivalence() {
    std::vector<std::string> sources = {
        read_corpus(),
        "module m;\r\n  logic [7:0] value = 8'hxZ;\n",
        "\"escaped \\\" quote\" \"line\\\r\ncontinued\" // comment\r\n/* block **/",
        "\\escaped.identifier $system_name `define `include `` `\" `\"\"\"",
        "\"unterminated\n/* never closed",
        std::string("wire ") + std::string(80, 'a') + ";\n/* " + std::string(80, 'x') +
            " */\n",
        std::string(30, ' ') + "//" + std::string(32, 'c') + "\r\n" +
            "\"" + std::string(70, 's') + "\\\"tail\" " +
            "\\" + std::string(70, 'e') + " \n" +
            "/*" + std::string(90, '*') + "x/" + std::string(60, 'z') + "*/\n",
        std::string("wire \0 high", 11) + static_cast<char>(0xff) + ";\n"
    };
    std::string large;
    large.reserve(2 * 1024 * 1024);
    while (large.size() < 2 * 1024 * 1024)
        large += "wire signal_123; // realistic repeated declaration\n";
    sources.push_back(std::move(large));

    constexpr std::array<std::string_view, 20> random_fragments = {
        "wire value_17;\n", "logic [31:0] packed_name;\r\n",
        "// comment with /* nested-looking */ text\n", "/* block // marker */\n",
        "assign y = a && b || !c;\n", "8'hfF 16'sd42 1.25e-3 ",
        "\"string with \\\" quote\" ", "\\escaped.identifier ",
        "`define TOKEN value\n", "$display ", "<<< >>> <<<= >>>= ",
        "module endmodule always_comb logic ", "let nettype automatic ",
        "( ) [ ] { } , ; : . @ # ' ", "\r\n\t\f\v",
        "identifier$tail another_123 ", "'0 '1 'x 'z ",
        "casez casex unique priority ", "/* stars ****/ ", "// final comment\r\n"
    };
    std::string randomized_source;
    uint32_t random_state = 0x4c455834u;
    for (size_t i = 0; i < 1024; ++i) {
        random_state ^= random_state << 13;
        random_state ^= random_state >> 17;
        random_state ^= random_state << 5;
        randomized_source += random_fragments[random_state % random_fragments.size()];
    }
    sources.push_back(std::move(randomized_source));

    for (const auto& text : sources) {
        auto source = std::make_shared<const labneura::eda::VerilogSourceBuffer>(text);
        auto scalar = labneura::eda::lex_verilog(
            source, {VerilogLanguageVersion::SystemVerilog2017, VerilogLexerBackend::Scalar});
        std::optional<labneura::eda::VerilogLexResult> neon_result;
        std::optional<labneura::eda::VerilogLexResult> avx2_result;
        for (const auto backend : {VerilogLexerBackend::Neon, VerilogLexerBackend::Avx2}) {
            const bool available = backend == VerilogLexerBackend::Neon
                                       ? byte_classifier::neon_available()
                                       : byte_classifier::avx2_available();
            if (!available)
                continue;
            auto accelerated = labneura::eda::lex_verilog(
                source, {VerilogLanguageVersion::SystemVerilog2017, backend});
            require_same_lex_result(scalar, accelerated);
            if (backend == VerilogLexerBackend::Neon)
                neon_result = std::move(accelerated);
            else
                avx2_result = std::move(accelerated);
        }
        if (neon_result && avx2_result)
            require_same_lex_result(*neon_result, *avx2_result);
    }
    if (!byte_classifier::avx2_available())
        std::cout << "AVX2 full-lexer parity explicitly skipped: unavailable at runtime\n";
}

void test_backend_environment_selection() {
    const char* existing = std::getenv("LABNEURA_VERILOG_LEXER_BACKEND");
    const std::optional<std::string> old_value =
        existing ? std::optional<std::string>(existing) : std::nullopt;
    auto restore = [&] {
        if (old_value)
            set_backend_environment(old_value->c_str());
        else
            set_backend_environment(nullptr);
    };

    try {
        set_backend_environment("scalar");
        require(labneura::eda::configured_verilog_lexer_backend() ==
                    VerilogLexerBackend::Scalar,
                "scalar backend environment selection failed");
        set_backend_environment("auto");
        require(labneura::eda::configured_verilog_lexer_backend() ==
                    (byte_classifier::avx2_available()
                         ? VerilogLexerBackend::Avx2
                         : byte_classifier::neon_available() ? VerilogLexerBackend::Neon
                                                              : VerilogLexerBackend::Scalar),
                "automatic backend selection did not choose the available implementation");
        if (byte_classifier::neon_available()) {
            set_backend_environment("neon");
            require(labneura::eda::configured_verilog_lexer_backend() ==
                        VerilogLexerBackend::Neon,
                    "NEON backend environment selection failed");
        }
        else {
            set_backend_environment("neon");
            bool rejected = false;
            try {
                (void)labneura::eda::configured_verilog_lexer_backend();
            }
            catch (const std::invalid_argument&) {
                rejected = true;
            }
            require(rejected, "unavailable NEON backend must report an error");
        }
        if (byte_classifier::avx2_available()) {
            set_backend_environment("avx2");
            require(labneura::eda::configured_verilog_lexer_backend() ==
                        VerilogLexerBackend::Avx2,
                    "AVX2 backend environment selection failed");
        }
        else {
            set_backend_environment("avx2");
            bool rejected = false;
            try {
                (void)labneura::eda::configured_verilog_lexer_backend();
            }
            catch (const std::invalid_argument&) {
                rejected = true;
            }
            require(rejected, "unavailable AVX2 backend must report an error");
        }
        set_backend_environment("unknown");
        bool rejected = false;
        try {
            (void)labneura::eda::configured_verilog_lexer_backend();
        }
        catch (const std::invalid_argument&) {
            rejected = true;
        }
        require(rejected, "unknown byte-classification backend must be rejected");
    }
    catch (...) {
        restore();
        throw;
    }
    restore();
}

#if defined(LABNEURA_ENABLE_SLANG_LEXER_ORACLE)
void test_differential_against_slang() {
    const std::string source = read_corpus();
    auto scalar = labneura::eda::lex_verilog(
        source, {VerilogLanguageVersion::SystemVerilog2017,
                 VerilogLexerBackend::Scalar});

    slang::SourceManager source_manager;
    auto slang_buffer = source_manager.assignText("lexer_corpus.sv", source);
    slang::BumpAllocator allocator;
    slang::Diagnostics diagnostics;
    slang::parsing::Lexer slang_lexer(slang_buffer, allocator, diagnostics, source_manager);

    std::vector<slang::parsing::Token> slang_tokens;
    while (true) {
        auto token = slang_lexer.lex(slang::parsing::KeywordVersion::v1800_2017);
        if (token.kind == slang::parsing::TokenKind::EndOfFile)
            break;
        slang_tokens.push_back(token);
    }

    size_t slang_index = 0;
    size_t scalar_token_count = 0;
    for (const auto& scalar_token : scalar.tokens) {
        if (scalar_token.kind == VerilogTokenKind::LineComment ||
            scalar_token.kind == VerilogTokenKind::BlockComment) {
            continue;
        }
        if (scalar_token.kind == VerilogTokenKind::EndOfFile)
            break;
        require(slang_index < slang_tokens.size(), "scalar lexer ended after Slang");
        require(slang_tokens[slang_index].location().offset() == scalar_token.byte_offset,
                "Slang and scalar token offsets differ");

        const auto spelling = scalar.spelling(scalar_token);
        size_t combined_size = 0;
        std::string combined_spelling;
        const size_t first_slang_index = slang_index;
        while (combined_size < spelling.size()) {
            require(slang_index < slang_tokens.size(), "Slang token group ended early");
            const auto& slang_token = slang_tokens[slang_index];
            require(slang_token.location().offset() == scalar_token.byte_offset + combined_size,
                    "Slang token group is not contiguous");
            combined_spelling.append(slang_token.rawText());
            combined_size += slang_token.rawText().size();
            ++slang_index;
        }
        require(combined_spelling == spelling, "Slang and scalar token spellings differ");
        const auto& first_slang_token = slang_tokens[first_slang_index];
        require(source_manager.getLineNumber(first_slang_token.location()) == scalar_token.line &&
                    source_manager.getColumnNumber(first_slang_token.location()) ==
                        scalar_token.column,
                "Slang and scalar line/column locations differ");
        if (slang_index == first_slang_index + 1) {
            require((scalar_token.kind == VerilogTokenKind::Keyword) ==
                        slang::parsing::LexerFacts::isKeyword(first_slang_token.kind),
                    "Slang and scalar keyword classifications differ");
        }
        ++scalar_token_count;
    }
    require(slang_index == slang_tokens.size(),
            "scalar lexer has trailing tokens after Slang");
    require(diagnostics.empty(), "Slang reported diagnostics for the differential corpus");
    require(scalar_token_count > 40, "differential corpus is not representative enough");
    require(scalar.errors.empty(), "scalar lexer reported diagnostics for the corpus");
}

void test_keyword_tables_against_slang() {
    struct VersionPair {
        slang::parsing::KeywordVersion slang;
        VerilogLanguageVersion scalar;
    };
    constexpr VersionPair versions[] = {
        {slang::parsing::KeywordVersion::v1364_1995, VerilogLanguageVersion::Verilog1995},
        {slang::parsing::KeywordVersion::v1364_2001, VerilogLanguageVersion::Verilog2001},
        {slang::parsing::KeywordVersion::v1364_2005, VerilogLanguageVersion::Verilog2005},
        {slang::parsing::KeywordVersion::v1800_2005, VerilogLanguageVersion::SystemVerilog2005},
        {slang::parsing::KeywordVersion::v1800_2009, VerilogLanguageVersion::SystemVerilog2009},
        {slang::parsing::KeywordVersion::v1800_2012, VerilogLanguageVersion::SystemVerilog2012},
        {slang::parsing::KeywordVersion::v1800_2017, VerilogLanguageVersion::SystemVerilog2017},
        {slang::parsing::KeywordVersion::v1800_2023, VerilogLanguageVersion::SystemVerilog2023}
    };
    const auto* all_keywords =
        slang::parsing::LexerFacts::getKeywordTable(slang::parsing::KeywordVersion::v1800_2023);
    require(all_keywords != nullptr, "Slang keyword table is unavailable");

    for (const auto& [word, unused_kind] : *all_keywords) {
        (void)unused_kind;
        for (const auto& version : versions) {
            const auto* table = slang::parsing::LexerFacts::getKeywordTable(version.slang);
            require(table != nullptr, "Slang versioned keyword table is unavailable");
            const bool expected_keyword = table->contains(word);
            auto result = labneura::eda::lex_verilog(
                std::string(word),
                {version.scalar, VerilogLexerBackend::Scalar});
            if ((result.tokens[0].kind == VerilogTokenKind::Keyword) != expected_keyword) {
                throw std::runtime_error(
                    "scalar keyword table differs from Slang for '" + std::string(word) +
                    "' at language version " +
                    std::to_string(static_cast<unsigned>(version.scalar)));
            }
        }
    }
}
#endif

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && (std::string_view(argv[1]) == "SCALAR" ||
                          std::string_view(argv[1]) == "NEON" ||
                          std::string_view(argv[1]) == "AVX2")) {
            const std::string_view mode(argv[1]);
            const auto backend = mode == "SCALAR" ? VerilogLexerBackend::Scalar :
                                 mode == "NEON" ? VerilogLexerBackend::Neon :
                                                  VerilogLexerBackend::Avx2;
            const bool available = backend == VerilogLexerBackend::Scalar ||
                                   (backend == VerilogLexerBackend::Neon &&
                                    byte_classifier::neon_available()) ||
                                   (backend == VerilogLexerBackend::Avx2 &&
                                    byte_classifier::avx2_available());
            if (!available) {
                std::cout << mode
                          << " lexer tests skipped: runtime backend unavailable\n";
                return 77;
            }
            test_byte_classifiers();
            test_classification_scanner();
            test_lexer_backend_equivalence();
            const auto source = std::make_shared<const labneura::eda::VerilogSourceBuffer>(
                read_corpus(), "lexer_corpus.sv");
            const auto scalar = labneura::eda::lex_verilog(
                source, {VerilogLanguageVersion::SystemVerilog2017,
                         VerilogLexerBackend::Scalar});
            const auto selected = labneura::eda::lex_verilog(
                source, {VerilogLanguageVersion::SystemVerilog2017, backend});
            require_same_lex_result(scalar, selected);
            std::cout << mode
                      << " byte-classifier and full-lexer parity tests passed\n";
            return 0;
        }
        if (argc != 1)
            throw std::invalid_argument("unknown Verilog lexer test mode");

        test_empty_and_source_identity();
        test_identifiers_keywords_and_directives();
        test_numbers();
        test_strings_comments_and_errors();
        test_operators_and_punctuation();
        test_locations_and_boundaries();
        test_environment_selection();
        test_byte_classifiers();
        test_classification_scanner();
        test_lexer_backend_equivalence();
        test_backend_environment_selection();
#if defined(LABNEURA_ENABLE_SLANG_LEXER_ORACLE)
        test_differential_against_slang();
        test_keyword_tables_against_slang();
#endif
        std::cout << "Verilog lexer tests passed\n";
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << "Verilog lexer test failure: " << error.what() << '\n';
        return 1;
    }
}
