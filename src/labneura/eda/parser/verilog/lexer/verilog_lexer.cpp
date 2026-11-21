#include "labneura/eda/verilog_lexer.h"
#include "verilog_byte_classifier.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <utility>

namespace labneura::eda {
namespace {

std::atomic<uint64_t> next_source_id{1};

uint32_t allocate_source_id() {
    const uint64_t id = next_source_id.fetch_add(1, std::memory_order_relaxed);
    if (id > std::numeric_limits<uint32_t>::max())
        throw std::overflow_error("Verilog source identifier space exhausted");
    return static_cast<uint32_t>(id);
}

constexpr std::string_view verilog_1995_keywords =
    "always and assign begin buf bufif0 bufif1 case casex casez cmos deassign default "
    "defparam disable edge else end endcase endfunction endmodule endprimitive endspecify "
    "endtable endtask event for force forever fork function highz0 highz1 if ifnone initial "
    "inout input integer join large macromodule medium module nand negedge nmos nor not "
    "notif0 notif1 or output parameter pmos posedge primitive pull0 pull1 pulldown pullup "
    "rcmos real realtime reg release repeat rnmos rpmos rtran rtranif0 rtranif1 scalared "
    "small specify specparam strong0 strong1 supply0 supply1 table task time tran tranif0 "
    "tranif1 tri tri0 tri1 triand trior trireg vectored wait wand weak0 weak1 while wire wor "
    "xor xnor";

constexpr std::string_view verilog_2001_keywords =
    "automatic endgenerate generate genvar localparam noshowcancelled pulsestyle_ondetect "
    "pulsestyle_onevent showcancelled signed unsigned cell config design endconfig incdir "
    "include instance liblist library use";

constexpr std::string_view verilog_2005_keywords = "uwire";

constexpr std::string_view systemverilog_2005_keywords =
    "alias always_comb always_ff always_latch assert assume before bind bins binsof bit break "
    "byte chandle class clocking const constraint context continue cover covergroup coverpoint "
    "cross dist do endclass endclocking endgroup endinterface endpackage endprogram "
    "endproperty endsequence enum expect export extends extern final first_match foreach "
    "forkjoin iff ignore_bins illegal_bins import inside int interface intersect join_any "
    "join_none local logic longint matches modport new null package packed priority program "
    "property protected pure rand randc randcase randsequence ref return sequence shortint "
    "shortreal solve static string struct super tagged this throughout timeprecision timeunit "
    "type typedef union unique var virtual void wait_order wildcard with within";

constexpr std::string_view systemverilog_2009_keywords =
    "accept_on checker endchecker eventually global implies let nexttime reject_on restrict "
    "s_always s_eventually s_nexttime s_until s_until_with strong sync_accept_on "
    "sync_reject_on unique0 until until_with untyped weak";

constexpr std::string_view systemverilog_2012_keywords = "implements interconnect nettype soft";

constexpr size_t keyword_hash(std::string_view word) {
    size_t hash = sizeof(size_t) == 8 ? size_t{1469598103934665603ULL} : size_t{2166136261U};
    constexpr size_t prime = sizeof(size_t) == 8 ? size_t{1099511628211ULL} : size_t{16777619U};
    for (const unsigned char ch : word) {
        hash ^= ch;
        hash *= prime;
    }
    return hash;
}

template <size_t Capacity>
class KeywordLookup {
    static_assert((Capacity & (Capacity - 1)) == 0,
                  "keyword lookup capacity must be a power of two");

public:
    consteval explicit KeywordLookup(std::string_view words) {
        size_t start = 0;
        while (start < words.size()) {
            const size_t end = words.find(' ', start);
            const auto word = words.substr(start, end == std::string_view::npos
                                                      ? words.size() - start
                                                      : end - start);
            size_t slot = keyword_hash(word) & (Capacity - 1);
            while (!slots_[slot].empty())
                slot = (slot + 1) & (Capacity - 1);
            slots_[slot] = word;
            if (end == std::string_view::npos)
                break;
            start = end + 1;
        }
    }

    constexpr bool contains(std::string_view word) const {
        size_t slot = keyword_hash(word) & (Capacity - 1);
        while (!slots_[slot].empty()) {
            if (slots_[slot] == word)
                return true;
            slot = (slot + 1) & (Capacity - 1);
        }
        return false;
    }

private:
    std::array<std::string_view, Capacity> slots_{};
};

constexpr KeywordLookup<256> verilog_1995_keyword_lookup{verilog_1995_keywords};
constexpr KeywordLookup<64> verilog_2001_keyword_lookup{verilog_2001_keywords};
constexpr KeywordLookup<4> verilog_2005_keyword_lookup{verilog_2005_keywords};
constexpr KeywordLookup<256> systemverilog_2005_keyword_lookup{systemverilog_2005_keywords};
constexpr KeywordLookup<64> systemverilog_2009_keyword_lookup{systemverilog_2009_keywords};
constexpr KeywordLookup<16> systemverilog_2012_keyword_lookup{systemverilog_2012_keywords};

bool is_identifier_start(unsigned char ch) {
    return (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || ch == '_';
}

bool is_identifier_continue(unsigned char ch) {
    return is_identifier_start(ch) || (ch >= '0' && ch <= '9') || ch == '$';
}

bool is_space(unsigned char ch) {
    return ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r' || ch == '\f' || ch == '\v';
}

char lower_ascii(char ch) {
    if (ch >= 'A' && ch <= 'Z')
        return static_cast<char>(ch - 'A' + 'a');
    return ch;
}

bool valid_separated_digits(std::string_view digits, bool allow_unknown, unsigned base) {
    if (digits.empty() || digits.front() == '_' || digits.back() == '_')
        return false;

    bool previous_underscore = false;
    for (unsigned char ch : digits) {
        if (ch == '_') {
            if (previous_underscore)
                return false;
            previous_underscore = true;
            continue;
        }
        previous_underscore = false;

        if (ch >= '0' && ch <= '9') {
            if (ch - '0' < base)
                continue;
        }
        else if (base == 16 && ((ch >= 'a' && ch <= 'f') || (ch >= 'A' && ch <= 'F'))) {
            continue;
        }
        if (allow_unknown && (ch == 'x' || ch == 'X' || ch == 'z' || ch == 'Z' || ch == '?'))
            continue;
        return false;
    }
    return true;
}

bool is_numeric_sequence_char(unsigned char ch) {
    return is_identifier_continue(ch) || ch == '?';
}

namespace byte_classifier = detail::verilog_lexer;

VerilogLexerBackend configured_backend(VerilogLexerBackend requested) {
    if (requested != VerilogLexerBackend::Auto &&
        requested != VerilogLexerBackend::Scalar &&
        requested != VerilogLexerBackend::Neon &&
        requested != VerilogLexerBackend::Avx2) {
        throw std::invalid_argument("unknown Verilog lexer backend");
    }

    if (requested == VerilogLexerBackend::Auto) {
        const char* value = std::getenv("LABNEURA_VERILOG_LEXER_BACKEND");
        const std::string_view setting = value && value[0] != '\0' ? value : "auto";
        if (setting == "scalar")
            requested = VerilogLexerBackend::Scalar;
        else if (setting == "neon")
            requested = VerilogLexerBackend::Neon;
        else if (setting == "avx2")
            requested = VerilogLexerBackend::Avx2;
        else if (setting != "auto")
            throw std::invalid_argument(
                "LABNEURA_VERILOG_LEXER_BACKEND must be 'auto', 'scalar', 'neon', or 'avx2'");
    }

    if (requested == VerilogLexerBackend::Auto) {
        if (byte_classifier::avx2_available())
            requested = VerilogLexerBackend::Avx2;
        else if (byte_classifier::neon_available())
            requested = VerilogLexerBackend::Neon;
        else
            requested = VerilogLexerBackend::Scalar;
    }
    if (requested == VerilogLexerBackend::Neon && !byte_classifier::neon_available())
        throw std::invalid_argument(
            "NEON Verilog lexer backend is unavailable on this build and architecture");
    if (requested == VerilogLexerBackend::Avx2 && !byte_classifier::avx2_available())
        throw std::invalid_argument(
            "AVX2 Verilog lexer backend is unavailable or unsupported by this CPU");
    return requested;
}

class Scanner {
public:
    Scanner(std::shared_ptr<const VerilogSourceBuffer> source, VerilogLexerOptions options,
            VerilogLexerBackend backend) :
        source_(std::move(source)), text_(source_->text()), options_(options),
        classifier_(text_, backend) {}

    VerilogLexResult run() {
        while (offset_ < text_.size()) {
            const auto ch = static_cast<unsigned char>(text_[offset_]);
            if (is_space(ch)) {
                offset_ = scan_whitespace(offset_);
                continue;
            }
            if (ch == '/' && offset_ + 1 < text_.size()) {
                if (text_[offset_ + 1] == '/') {
                    scan_line_comment();
                    continue;
                }
                if (text_[offset_ + 1] == '*') {
                    scan_block_comment();
                    continue;
                }
            }
            if (ch == '"') {
                scan_string();
                continue;
            }
            if (ch == '\\') {
                scan_escaped_identifier();
                continue;
            }
            if (ch == '`') {
                scan_backtick();
                continue;
            }
            if (ch == '$') {
                scan_system_identifier();
                continue;
            }
            if ((ch >= '0' && ch <= '9') ||
                (ch == '\'' && is_based_literal_start(offset_ + 1))) {
                scan_number();
                continue;
            }
            if (is_identifier_start(ch)) {
                scan_identifier();
                continue;
            }
            if (scan_operator_or_punctuation())
                continue;

            const size_t start = offset_++;
            emit(VerilogTokenKind::Invalid, start, offset_);
            add_error(VerilogLexErrorKind::InvalidCharacter, start, 1);
        }

        emit(VerilogTokenKind::EndOfFile, text_.size(), text_.size());
        return {std::move(source_), std::move(tokens_), std::move(errors_)};
    }

private:
    bool is_based_literal_start(size_t offset) const {
        if (offset >= text_.size())
            return false;
        const char ch = text_[offset];
        return ch == '0' || ch == '1' || ch == 'x' || ch == 'X' || ch == 'z' || ch == 'Z' ||
               ch == 's' || ch == 'S' || ch == 'b' || ch == 'B' || ch == 'o' || ch == 'O' ||
               ch == 'd' || ch == 'D' || ch == 'h' || ch == 'H';
    }

    bool is_keyword(std::string_view word) const {
        if (verilog_1995_keyword_lookup.contains(word))
            return true;

        switch (options_.language_version) {
            case VerilogLanguageVersion::Verilog1995:
                return false;
            case VerilogLanguageVersion::Verilog2001:
                return verilog_2001_keyword_lookup.contains(word);
            case VerilogLanguageVersion::Verilog2005:
                return verilog_2001_keyword_lookup.contains(word) ||
                       verilog_2005_keyword_lookup.contains(word);
            case VerilogLanguageVersion::SystemVerilog2005:
                return verilog_2001_keyword_lookup.contains(word) ||
                       verilog_2005_keyword_lookup.contains(word) ||
                       systemverilog_2005_keyword_lookup.contains(word);
            case VerilogLanguageVersion::SystemVerilog2009:
                return verilog_2001_keyword_lookup.contains(word) ||
                       verilog_2005_keyword_lookup.contains(word) ||
                       systemverilog_2005_keyword_lookup.contains(word) ||
                       systemverilog_2009_keyword_lookup.contains(word);
            case VerilogLanguageVersion::SystemVerilog2012:
                return verilog_2001_keyword_lookup.contains(word) ||
                       verilog_2005_keyword_lookup.contains(word) ||
                       systemverilog_2005_keyword_lookup.contains(word) ||
                       systemverilog_2009_keyword_lookup.contains(word) ||
                       systemverilog_2012_keyword_lookup.contains(word);
            case VerilogLanguageVersion::SystemVerilog2017:
                return verilog_2001_keyword_lookup.contains(word) ||
                       verilog_2005_keyword_lookup.contains(word) ||
                       systemverilog_2005_keyword_lookup.contains(word) ||
                       systemverilog_2009_keyword_lookup.contains(word) ||
                       systemverilog_2012_keyword_lookup.contains(word);
            case VerilogLanguageVersion::SystemVerilog2023:
                return verilog_2001_keyword_lookup.contains(word) ||
                       verilog_2005_keyword_lookup.contains(word) ||
                       systemverilog_2005_keyword_lookup.contains(word) ||
                       systemverilog_2009_keyword_lookup.contains(word) ||
                       systemverilog_2012_keyword_lookup.contains(word);
        }
        return false;
    }

    void emit(VerilogTokenKind kind, size_t start, size_t end) {
        const auto location = source_->location(start, location_line_hint_);
        tokens_.push_back({kind, source_->id(), start, end - start, location.line, location.column});
    }

    void add_error(VerilogLexErrorKind kind, size_t start, size_t length) {
        const auto location = source_->location(start, location_line_hint_);
        errors_.push_back({kind, source_->id(), start, length, location.line, location.column});
    }

    size_t scan_whitespace(size_t offset) {
        constexpr size_t scalar_prefix = 8;
        size_t remaining = scalar_prefix;
        while (offset < text_.size() && remaining != 0 &&
               is_space(static_cast<unsigned char>(text_[offset]))) {
            ++offset;
            --remaining;
        }
        if (offset == text_.size() ||
            !is_space(static_cast<unsigned char>(text_[offset])))
            return offset;
        return classifier_.find_first_not(
            offset, text_.size(), byte_classifier::ByteClass::Whitespace);
    }

    bool scan_decimal_sequence() {
        bool saw_digit = false;
        bool previous_underscore = false;
        bool valid = true;
        while (offset_ < text_.size()) {
            const unsigned char ch = static_cast<unsigned char>(text_[offset_]);
            if (ch >= '0' && ch <= '9') {
                saw_digit = true;
                previous_underscore = false;
            }
            else if (ch == '_') {
                if (!saw_digit || previous_underscore)
                    valid = false;
                previous_underscore = true;
            }
            else {
                break;
            }
            ++offset_;
        }
        return valid && saw_digit && !previous_underscore;
    }

    void scan_identifier() {
        const size_t start = offset_++;
        offset_ = classifier_.find_first_not(
            offset_, text_.size(), byte_classifier::ByteClass::IdentifierContinue);
        const auto spelling = text_.substr(start, offset_ - start);
        emit(is_keyword(spelling) ? VerilogTokenKind::Keyword : VerilogTokenKind::Identifier,
             start, offset_);
    }

    void scan_escaped_identifier() {
        const size_t start = offset_++;
        offset_ = classifier_.find_first(
            offset_, text_.size(), byte_classifier::mask(byte_classifier::ByteClass::Whitespace));
        if (offset_ == start + 1) {
            emit(VerilogTokenKind::Invalid, start, offset_);
            add_error(VerilogLexErrorKind::InvalidEscapedIdentifier, start, 1);
            return;
        }
        emit(VerilogTokenKind::EscapedIdentifier, start, offset_);
    }

    void scan_system_identifier() {
        const size_t start = offset_++;
        if (offset_ == text_.size() ||
            !is_identifier_start(static_cast<unsigned char>(text_[offset_]))) {
            emit(VerilogTokenKind::Invalid, start, offset_);
            add_error(VerilogLexErrorKind::InvalidCharacter, start, 1);
            return;
        }
        offset_ = classifier_.find_first_not(
            offset_, text_.size(), byte_classifier::ByteClass::IdentifierContinue);
        emit(VerilogTokenKind::SystemIdentifier, start, offset_);
    }

    void scan_backtick() {
        const size_t start = offset_++;
        size_t special_length = 0;
        if (text_.substr(start, 4) == "`\"\"\"" || text_.substr(start, 4) == "`\\`\"")
            special_length = 4;
        else if (text_.substr(start, 2) == "``" || text_.substr(start, 2) == "`\"")
            special_length = 2;
        if (special_length != 0) {
            offset_ = start + special_length;
            emit(VerilogTokenKind::Backtick, start, offset_);
            return;
        }
        if (offset_ < text_.size() && text_[offset_] == '\\') {
            offset_ = classifier_.find_first(
                offset_, text_.size(),
                byte_classifier::mask(byte_classifier::ByteClass::Whitespace));
            emit(VerilogTokenKind::CompilerDirective, start, offset_);
            return;
        }
        if (offset_ < text_.size() &&
            is_identifier_start(static_cast<unsigned char>(text_[offset_]))) {
            offset_ = classifier_.find_first_not(
                offset_, text_.size(), byte_classifier::ByteClass::IdentifierContinue);
            emit(VerilogTokenKind::CompilerDirective, start, offset_);
            return;
        }
        emit(VerilogTokenKind::Backtick, start, offset_);
    }

    void scan_line_comment() {
        const size_t start = offset_;
        offset_ += 2;
        offset_ = classifier_.find_first(
            offset_, text_.size(),
            byte_classifier::ByteClass::Newline | byte_classifier::ByteClass::CarriageReturn);
        emit(VerilogTokenKind::LineComment, start, offset_);
    }

    void scan_block_comment() {
        const size_t start = offset_;
        offset_ += 2;
        while (offset_ < text_.size()) {
            offset_ = classifier_.find_first(
                offset_, text_.size(), byte_classifier::mask(byte_classifier::ByteClass::Asterisk));
            if (offset_ == text_.size())
                break;
            if (offset_ + 1 < text_.size() && text_[offset_ + 1] == '/')
                break;
            ++offset_;
        }
        if (offset_ == text_.size()) {
            emit(VerilogTokenKind::BlockComment, start, offset_);
            add_error(VerilogLexErrorKind::UnterminatedBlockComment, start, offset_ - start);
            return;
        }
        offset_ += 2;
        emit(VerilogTokenKind::BlockComment, start, offset_);
    }

    void scan_string() {
        const size_t start = offset_++;
        while (offset_ < text_.size()) {
            offset_ = classifier_.find_first(
                offset_, text_.size(),
                byte_classifier::ByteClass::DoubleQuote |
                    byte_classifier::ByteClass::Backslash |
                    byte_classifier::ByteClass::Newline |
                    byte_classifier::ByteClass::CarriageReturn);
            if (offset_ == text_.size())
                break;
            const char ch = text_[offset_];
            if (ch == '"') {
                ++offset_;
                emit(VerilogTokenKind::StringLiteral, start, offset_);
                return;
            }
            if (ch == '\n' || ch == '\r') {
                emit(VerilogTokenKind::Invalid, start, offset_);
                add_error(VerilogLexErrorKind::NewlineInString, start, offset_ - start);
                return;
            }
            if (ch == '\\') {
                ++offset_;
                if (offset_ < text_.size() && text_[offset_] == '\r') {
                    ++offset_;
                    if (offset_ < text_.size() && text_[offset_] == '\n')
                        ++offset_;
                }
                else if (offset_ < text_.size()) {
                    ++offset_;
                }
                continue;
            }
            ++offset_;
        }
        emit(VerilogTokenKind::Invalid, start, offset_);
        add_error(VerilogLexErrorKind::UnterminatedString, start, offset_ - start);
    }

    void scan_number() {
        const size_t start = offset_;
        if (text_[offset_] == '\'') {
            scan_apostrophe_number();
            return;
        }

        const bool valid_integer = scan_decimal_sequence();

        if (offset_ < text_.size() && text_[offset_] == '\'') {
            ++offset_;
            const bool is_signed = offset_ < text_.size() &&
                                   (text_[offset_] == 's' || text_[offset_] == 'S');
            if (is_signed)
                ++offset_;
            if (offset_ == text_.size()) {
                emit(VerilogTokenKind::Invalid, start, offset_);
                add_error(VerilogLexErrorKind::InvalidNumericLiteral, start, offset_ - start);
                return;
            }

            const char base_char = lower_ascii(text_[offset_++]);
            const unsigned base = base_char == 'b' ? 2 : base_char == 'o' ? 8 :
                                  base_char == 'd' ? 10 : base_char == 'h' ? 16 : 0;
            if (!base) {
                consume_numeric_tail();
                emit(VerilogTokenKind::Invalid, start, offset_);
                add_error(VerilogLexErrorKind::InvalidNumericLiteral, start, offset_ - start);
                return;
            }

            const size_t digits_start = offset_;
            while (offset_ < text_.size() &&
                   is_numeric_sequence_char(static_cast<unsigned char>(text_[offset_]))) {
                ++offset_;
            }
            const auto digits = text_.substr(digits_start, offset_ - digits_start);
            const bool valid_size = valid_integer;
            const bool valid_value = valid_separated_digits(digits, true, base);
            if (!valid_size || !valid_value) {
                emit(VerilogTokenKind::Invalid, start, offset_);
                add_error(VerilogLexErrorKind::InvalidNumericLiteral, start, offset_ - start);
                return;
            }
            emit(VerilogTokenKind::BasedNumber, start, offset_);
            return;
        }

        bool is_real = false;
        bool valid = valid_integer;
        if (offset_ < text_.size() && text_[offset_] == '.' &&
            (offset_ + 1 == text_.size() || text_[offset_ + 1] != '.')) {
            is_real = true;
            ++offset_;
            const size_t fraction_start = offset_;
            const bool valid_fraction = scan_decimal_sequence();
            if (offset_ != fraction_start)
                valid = valid && valid_fraction;
        }

        if (offset_ < text_.size() && (text_[offset_] == 'e' || text_[offset_] == 'E')) {
            is_real = true;
            ++offset_;
            if (offset_ < text_.size() && (text_[offset_] == '+' || text_[offset_] == '-'))
                ++offset_;
            const bool valid_exponent = scan_decimal_sequence();
            valid = valid && valid_exponent;
            if (offset_ + 1 < text_.size() && text_[offset_] == '.' &&
                text_[offset_ + 1] >= '0' && text_[offset_ + 1] <= '9') {
                ++offset_;
                while (offset_ < text_.size() &&
                       ((text_[offset_] >= '0' && text_[offset_] <= '9') ||
                        text_[offset_] == '_')) {
                    ++offset_;
                }
                valid = false;
            }
        }

        if (offset_ < text_.size() &&
            is_identifier_start(static_cast<unsigned char>(text_[offset_]))) {
            consume_numeric_tail();
            valid = false;
        }
        if (!valid) {
            emit(VerilogTokenKind::Invalid, start, offset_);
            add_error(VerilogLexErrorKind::InvalidNumericLiteral, start, offset_ - start);
            return;
        }
        emit(is_real ? VerilogTokenKind::RealNumber : VerilogTokenKind::DecimalNumber,
             start, offset_);
    }

    void scan_apostrophe_number() {
        const size_t start = offset_++;
        bool is_signed = false;
        if (offset_ < text_.size() && (text_[offset_] == 's' || text_[offset_] == 'S')) {
            is_signed = true;
            ++offset_;
        }
        if (offset_ == text_.size()) {
            emit(VerilogTokenKind::Invalid, start, offset_);
            add_error(VerilogLexErrorKind::InvalidNumericLiteral, start, offset_ - start);
            return;
        }

        const char next = lower_ascii(text_[offset_]);
        if (!is_signed && (next == '0' || next == '1' || next == 'x' || next == 'z')) {
            ++offset_;
            emit(VerilogTokenKind::UnbasedNumber, start, offset_);
            return;
        }

        const unsigned base = next == 'b' ? 2 : next == 'o' ? 8 :
                              next == 'd' ? 10 : next == 'h' ? 16 : 0;
        if (!base) {
            --offset_;
            emit(VerilogTokenKind::Punctuation, start, offset_);
            return;
        }
        ++offset_;
        const size_t digits_start = offset_;
        while (offset_ < text_.size() &&
               is_numeric_sequence_char(static_cast<unsigned char>(text_[offset_]))) {
            ++offset_;
        }
        const auto digits = text_.substr(digits_start, offset_ - digits_start);
        if (!valid_separated_digits(digits, true, base)) {
            emit(VerilogTokenKind::Invalid, start, offset_);
            add_error(VerilogLexErrorKind::InvalidNumericLiteral, start, offset_ - start);
            return;
        }
        (void)is_signed;
        emit(VerilogTokenKind::BasedNumber, start, offset_);
    }

    void consume_numeric_tail() {
        while (offset_ < text_.size() &&
               is_numeric_sequence_char(static_cast<unsigned char>(text_[offset_]))) {
            ++offset_;
        }
    }

    bool scan_operator_or_punctuation() {
        switch (text_[offset_]) {
            case '(':
            case ')':
            case '[':
            case ']':
            case '{':
            case '}':
            case ',':
            case ';': {
                const size_t start = offset_++;
                emit(VerilogTokenKind::Punctuation, start, offset_);
                return true;
            }
            default:
                break;
        }

        const auto match_operator = [this](std::initializer_list<std::string_view> candidates) {
            for (const auto op : candidates) {
                if (text_.substr(offset_, op.size()) != op)
                    continue;
                const size_t start = offset_;
                offset_ += op.size();
                emit(VerilogTokenKind::Operator, start, offset_);
                return true;
            }
            return false;
        };

        switch (text_[offset_]) {
            case '<':
                if (match_operator({"<<<=", "<->", "<<<", "<<=", "<=", "<<", "<"}))
                    return true;
                break;
            case '>':
                if (match_operator({">>>=", ">>>", ">>=", ">=", ">>", ">"}))
                    return true;
                break;
            case '|':
                if (match_operator({"|->", "|=>", "||", "|=", "|"}))
                    return true;
                break;
            case '#':
                if (match_operator({"#-#", "#=#", "##"}))
                    return true;
                break;
            case '+':
                if (match_operator({"+/-", "+%-", "++", "+:", "+=", "+"}))
                    return true;
                break;
            case '-':
                if (match_operator({"->>", "--", "-:", "->", "-=", "-"}))
                    return true;
                break;
            case '=':
                if (match_operator({"===", "==?", "==", "=>", "="}))
                    return true;
                break;
            case '!':
                if (match_operator({"!==", "!=?", "!=", "!"}))
                    return true;
                break;
            case '&':
                if (match_operator({"&&&", "&&", "&=", "&"}))
                    return true;
                break;
            case '*':
                if (match_operator({"**=", "**", "*>", "*=", "*"}))
                    return true;
                break;
            case '~':
                if (match_operator({"~&", "~|", "~^", "~"}))
                    return true;
                break;
            case '^':
                if (match_operator({"^~", "^=", "^"}))
                    return true;
                break;
            case ':':
                if (match_operator({":=", ":/", "::"}))
                    return true;
                break;
            case '.':
                if (match_operator({".*"}))
                    return true;
                break;
            case '/':
                if (match_operator({"/=", "/"}))
                    return true;
                break;
            case '%':
                if (match_operator({"%=", "%"}))
                    return true;
                break;
            case '?':
                if (match_operator({"?"}))
                    return true;
                break;
            default:
                break;
        }

        constexpr std::string_view multi_punctuation[] = {"'{", "@@"};
        for (const auto punctuation : multi_punctuation) {
            if (text_.substr(offset_, punctuation.size()) == punctuation) {
                const size_t start = offset_;
                offset_ += punctuation.size();
                emit(VerilogTokenKind::Punctuation, start, offset_);
                return true;
            }
        }

        constexpr std::string_view punctuation = "()[]{}.,;:@#'";
        if (punctuation.find(text_[offset_]) != std::string_view::npos) {
            const size_t start = offset_++;
            emit(VerilogTokenKind::Punctuation, start, offset_);
            return true;
        }
        return false;
    }

    std::shared_ptr<const VerilogSourceBuffer> source_;
    const std::string_view text_;
    VerilogLexerOptions options_;
    byte_classifier::ClassificationScanner classifier_;
    size_t offset_ = 0;
    size_t location_line_hint_ = 0;
    std::vector<VerilogToken> tokens_;
    std::vector<VerilogLexError> errors_;
};

} // namespace

VerilogSourceBuffer::VerilogSourceBuffer(std::string text, std::string name) :
    id_(allocate_source_id()), name_(std::move(name)), text_(std::move(text)), line_starts_{0} {
    for (size_t i = 0; i < text_.size(); ++i) {
        if (text_[i] == '\r') {
            if (i + 1 < text_.size() && text_[i + 1] == '\n')
                ++i;
            line_starts_.push_back(i + 1);
        }
        else if (text_[i] == '\n') {
            line_starts_.push_back(i + 1);
        }
    }
}

VerilogSourceLocation VerilogSourceBuffer::location(size_t byte_offset) const {
    if (byte_offset > text_.size())
        throw std::out_of_range("Verilog source offset is past the end of the buffer");
    const auto it = std::upper_bound(line_starts_.begin(), line_starts_.end(), byte_offset);
    const size_t line_index = static_cast<size_t>(it - line_starts_.begin() - 1);
    return {line_index + 1, byte_offset - line_starts_[line_index] + 1};
}

VerilogSourceLocation VerilogSourceBuffer::location(
    size_t byte_offset, size_t& line_index_hint) const {
    if (byte_offset > text_.size())
        throw std::out_of_range("Verilog source offset is past the end of the buffer");
    if (line_index_hint >= line_starts_.size() ||
        line_starts_[line_index_hint] > byte_offset) {
        const auto it = std::upper_bound(line_starts_.begin(), line_starts_.end(),
                                         byte_offset);
        line_index_hint = static_cast<size_t>(it - line_starts_.begin() - 1);
    } else {
        while (line_index_hint + 1 < line_starts_.size() &&
               line_starts_[line_index_hint + 1] <= byte_offset)
            ++line_index_hint;
    }
    return {line_index_hint + 1,
            byte_offset - line_starts_[line_index_hint] + 1};
}

std::string_view VerilogLexResult::spelling(const VerilogToken& token) const {
    if (!source || token.source_id != source->id() ||
        token.byte_offset > source->text().size() ||
        token.length > source->text().size() - token.byte_offset) {
        throw std::invalid_argument("Verilog token does not refer to this source buffer");
    }
    return source->text().substr(token.byte_offset, token.length);
}

VerilogLexerSelection configured_verilog_lexer() {
    const char* value = std::getenv("LABNEURA_VERILOG_LEXER");
    if (!value || value[0] == '\0' || std::string_view(value) == "slang")
        return VerilogLexerSelection::Slang;
    if (std::string_view(value) == "scalar")
        return VerilogLexerSelection::Scalar;
    throw std::invalid_argument("LABNEURA_VERILOG_LEXER must be 'slang' or 'scalar'");
}

VerilogLexerBackend configured_verilog_lexer_backend() {
    return configured_backend(VerilogLexerBackend::Auto);
}

VerilogLexResult lex_verilog(std::shared_ptr<const VerilogSourceBuffer> source,
                             VerilogLexerOptions options) {
    if (!source)
        throw std::invalid_argument("Verilog source buffer must not be null");
    return Scanner(std::move(source), options, configured_backend(options.backend)).run();
}

VerilogLexResult lex_verilog(std::string source, VerilogLexerOptions options,
                             std::string source_name) {
    auto buffer = std::make_shared<const VerilogSourceBuffer>(
        std::move(source), std::move(source_name));
    return lex_verilog(std::move(buffer), options);
}

} // namespace labneura::eda
