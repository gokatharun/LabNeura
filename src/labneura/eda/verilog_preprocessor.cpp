#include "labneura/eda/verilog_preprocessor.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iterator>
#include <limits>
#include <set>
#include <stdexcept>
#include <system_error>
#include <utility>

namespace labneura::eda {
namespace {

bool identifier_start(unsigned char ch) {
    return (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || ch == '_';
}

bool identifier_continue(unsigned char ch) {
    return identifier_start(ch) || (ch >= '0' && ch <= '9') || ch == '$';
}

bool whitespace(unsigned char ch) {
    return ch == ' ' || ch == '\t' || ch == '\r' || ch == '\n' || ch == '\f' || ch == '\v';
}

std::string_view trim(std::string_view value) {
    while (!value.empty() && whitespace(static_cast<unsigned char>(value.front())))
        value.remove_prefix(1);
    while (!value.empty() && whitespace(static_cast<unsigned char>(value.back())))
        value.remove_suffix(1);
    return value;
}

bool same_frames(const std::vector<VerilogMacroExpansionFrame>& left,
                const std::vector<VerilogMacroExpansionFrame>& right) {
    if (left.size() != right.size())
        return false;
    for (size_t i = 0; i < left.size(); ++i) {
        const auto& a = left[i];
        const auto& b = right[i];
        if (a.macro_name != b.macro_name || a.invocation_source_id != b.invocation_source_id ||
            a.invocation_location.line != b.invocation_location.line ||
            a.invocation_location.column != b.invocation_location.column ||
            a.definition_source_id != b.definition_source_id ||
            a.definition_location.line != b.definition_location.line ||
            a.definition_location.column != b.definition_location.column)
            return false;
    }
    return true;
}

struct Macro {
    struct ParameterOccurrence {
        size_t start = 0;
        size_t end = 0;
        size_t parameter_index = 0;
    };

    VerilogMacroInfo info;
    size_t replacement_offset = 0;
    std::vector<ParameterOccurrence> parameter_occurrences;
};

struct ConditionalFrame {
    bool parent_active = false;
    bool active = false;
    bool branch_taken = false;
    bool saw_else = false;
    std::shared_ptr<const VerilogSourceBuffer> source;
    size_t offset = 0;
};

struct MacroArguments {
    bool valid = false;
    size_t end = 0;
    std::vector<std::pair<size_t, size_t>> ranges;
};

class Preprocessor {
public:
    explicit Preprocessor(VerilogPreprocessorOptions options) : options_(std::move(options)) {
        if (options_.max_macro_expansion_depth == 0 || options_.max_include_depth == 0)
            throw std::invalid_argument("preprocessor depth limits must be greater than zero");
    }

    VerilogPreprocessorResult run(std::shared_ptr<const VerilogSourceBuffer> source,
                                  std::filesystem::path source_path = {}) {
        if (!source)
            throw std::invalid_argument("Verilog preprocessor source must not be null");
        add_source(source);
        for (const auto& [name, replacement] : options_.predefined_macros) {
            if (name.empty() || !identifier_start(static_cast<unsigned char>(name.front())) ||
                !std::all_of(name.begin() + 1, name.end(), [](unsigned char ch) {
                    return identifier_continue(ch);
                })) {
                throw std::invalid_argument("invalid predefined Verilog macro name: " + name);
            }
            auto predefined = std::make_shared<const VerilogSourceBuffer>(
                replacement, "<predefined:" + name + ">");
            add_source(predefined);
            Macro macro;
            macro.info.name = name;
            macro.info.replacement = replacement;
            macro.info.definition_source = std::move(predefined);
            macros_.insert_or_assign(name, std::move(macro));
        }

        if (source_path.empty() && !source->name().empty() && source->name().front() != '<')
            source_path = std::filesystem::path(source->name());
        process_file(std::move(source), std::move(source_path));
        for (auto it = conditionals_.rbegin(); it != conditionals_.rend(); ++it)
            diagnose("MissingEndif", "`endif missing before end of preprocessing", it->source,
                     it->offset);

        result_.macros.clear();
        for (const auto& [name, macro] : macros_)
            result_.macros.emplace(name, macro.info);
        return std::move(result_);
    }

private:
    bool active() const {
        return conditionals_.empty() || conditionals_.back().active;
    }

    void add_source(const std::shared_ptr<const VerilogSourceBuffer>& source) {
        if (std::none_of(result_.sources.begin(), result_.sources.end(),
                         [&](const auto& known) { return known->id() == source->id(); }))
            result_.sources.push_back(source);
    }

    void diagnose(std::string code, std::string message,
                  const std::shared_ptr<const VerilogSourceBuffer>& source, size_t offset,
                  VerilogPreprocessorSeverity severity = VerilogPreprocessorSeverity::Error,
                  std::vector<VerilogMacroExpansionFrame> expansion = {}) {
        VerilogPreprocessorDiagnostic diagnostic;
        diagnostic.severity = severity;
        diagnostic.code = std::move(code);
        diagnostic.message = std::move(message);
        diagnostic.source = source;
        diagnostic.byte_offset = std::min(offset, source->text().size());
        diagnostic.location = source->location(diagnostic.byte_offset);
        diagnostic.include_stack = include_stack_;
        diagnostic.expansion_stack = std::move(expansion);
        result_.diagnostics.push_back(std::move(diagnostic));
    }

    void append(std::string_view text, const std::shared_ptr<const VerilogSourceBuffer>& source,
                size_t source_offset, VerilogPreprocessedOriginKind kind,
                const std::vector<VerilogMacroExpansionFrame>& expansion) {
        if (text.empty())
            return;
        if (text.size() > std::numeric_limits<size_t>::max() - result_.text.size())
            throw std::length_error("preprocessed Verilog output is too large");

        const size_t output_offset = result_.text.size();
        result_.text.append(text);
        if (!result_.source_map.empty()) {
            auto& previous = result_.source_map.back();
            const bool contiguous_source =
                kind == VerilogPreprocessedOriginKind::Source &&
                previous.source_offset <= std::numeric_limits<size_t>::max() - previous.length &&
                previous.source_offset + previous.length == source_offset;
            const bool same_macro_origin =
                kind == VerilogPreprocessedOriginKind::MacroExpansion &&
                previous.source_offset == source_offset;
            if (previous.output_offset + previous.length == output_offset &&
                previous.source == source && previous.kind == kind &&
                same_frames(previous.expansion_stack, expansion) &&
                (contiguous_source || same_macro_origin)) {
                previous.length += text.size();
                return;
            }
        }
        result_.source_map.push_back(
            {output_offset, text.size(), source, source_offset, kind, expansion});
    }

    void append_source(std::string_view text,
                       const std::shared_ptr<const VerilogSourceBuffer>& source,
                       size_t offset) {
        append(text, source, offset, VerilogPreprocessedOriginKind::Source, {});
    }

    size_t directive_start(std::string_view line, bool in_block_comment) const {
        size_t i = 0;
        while (i < line.size()) {
            if (in_block_comment) {
                const size_t close = line.find("*/", i);
                if (close == std::string_view::npos)
                    return line.size();
                i = close + 2;
                in_block_comment = false;
                continue;
            }
            while (i < line.size() && whitespace(static_cast<unsigned char>(line[i])))
                ++i;
            if (line.substr(i, 2) == "/*") {
                i += 2;
                in_block_comment = true;
                continue;
            }
            break;
        }
        return i < line.size() && line[i] == '`' ? i : line.size();
    }

    void update_comment_state(std::string_view text, bool& in_block_comment) const {
        size_t i = 0;
        while (i < text.size()) {
            if (in_block_comment) {
                const size_t close = text.find("*/", i);
                if (close == std::string_view::npos)
                    return;
                i = close + 2;
                in_block_comment = false;
                continue;
            }
            if (text.substr(i, 2) == "//")
                return;
            if (text.substr(i, 2) == "/*") {
                in_block_comment = true;
                i += 2;
                continue;
            }
            if (text[i] == '"') {
                ++i;
                while (i < text.size()) {
                    if (text[i] == '\\') {
                        i += std::min<size_t>(2, text.size() - i);
                        continue;
                    }
                    if (text[i++] == '"')
                        break;
                }
                continue;
            }
            if (text[i] == '\\') {
                ++i;
                while (i < text.size() && !whitespace(static_cast<unsigned char>(text[i])))
                    ++i;
                continue;
            }
            ++i;
        }
    }

    std::pair<std::string_view, size_t> directive_parts(std::string_view line,
                                                        size_t start) const {
        size_t i = start + 1;
        const size_t name_start = i;
        while (i < line.size() && identifier_continue(static_cast<unsigned char>(line[i])))
            ++i;
        return {line.substr(name_start, i - name_start), i};
    }

    void append_newline(const std::shared_ptr<const VerilogSourceBuffer>& source,
                        std::string_view text, size_t line_end, size_t newline_size) {
        if (newline_size != 0)
            append_source(text.substr(line_end, newline_size), source, line_end);
    }

    size_t find_stringification_end(std::string_view text, size_t start) const {
        bool in_string = false;
        bool in_block_comment = false;
        for (size_t i = start; i < text.size(); ++i) {
            if (in_block_comment) {
                if (text.substr(i, 2) == "*/") {
                    in_block_comment = false;
                    ++i;
                }
                continue;
            }
            if (in_string) {
                if (text[i] == '\\') {
                    if (i + 1 < text.size())
                        ++i;
                }
                else if (text[i] == '"') {
                    in_string = false;
                }
                continue;
            }
            if (text.substr(i, 2) == "//")
                return std::string_view::npos;
            if (text.substr(i, 2) == "/*") {
                in_block_comment = true;
                ++i;
                continue;
            }
            if (text[i] == '"') {
                in_string = true;
                continue;
            }
            if (text.substr(i, 2) == "`\"")
                return i;
        }
        return std::string_view::npos;
    }

    void process_file(std::shared_ptr<const VerilogSourceBuffer> source,
                      std::filesystem::path path) {
        if (include_stack_.size() >= options_.max_include_depth) {
            diagnose("IncludeDepthExceeded", "maximum include depth exceeded", source, 0);
            return;
        }
        if (!path.empty()) {
            std::error_code ec;
            path = std::filesystem::weakly_canonical(path, ec);
            if (ec)
                path = std::filesystem::absolute(path, ec).lexically_normal();
            const std::string key = path.string();
            if (std::find(active_paths_.begin(), active_paths_.end(), key) != active_paths_.end()) {
                diagnose("IncludeCycle", "include cycle detected for '" + key + "'", source, 0);
                return;
            }
            active_paths_.push_back(key);
            include_stack_.push_back(key);
        }
        else {
            include_stack_.push_back(std::string(source->name()));
        }

        const std::string_view text = source->text();
        size_t offset = 0;
        bool in_block_comment = false;
        while (offset < text.size()) {
            size_t content_end = offset;
            while (content_end < text.size() && text[content_end] != '\r' &&
                   text[content_end] != '\n')
                ++content_end;
            size_t line_end = content_end;
            if (line_end < text.size() && text[line_end] == '\r')
                ++line_end;
            if (line_end < text.size() && text[line_end] == '\n')
                ++line_end;
            size_t logical_content_end = content_end;
            size_t logical_line_end = line_end;
            std::string continued_line(text.substr(offset, content_end - offset));
            std::string_view line = continued_line;
            const size_t start = directive_start(line, in_block_comment);
            bool is_define = false;
            if (start < line.size()) {
                const auto [name, unused_args_start] = directive_parts(line, start);
                (void)unused_args_start;
                is_define = name == "define";
            }
            if (is_define) {
                while (logical_content_end > offset &&
                       text[logical_content_end - 1] == '\\' &&
                       logical_line_end > logical_content_end) {
                    continued_line.pop_back();
                    size_t next_content_end = logical_line_end;
                    while (next_content_end < text.size() &&
                           text[next_content_end] != '\r' && text[next_content_end] != '\n')
                        ++next_content_end;
                    size_t next_line_end = next_content_end;
                    if (next_line_end < text.size() && text[next_line_end] == '\r')
                        ++next_line_end;
                    if (next_line_end < text.size() && text[next_line_end] == '\n')
                        ++next_line_end;
                    continued_line.append(
                        text.substr(logical_line_end, next_content_end - logical_line_end));
                    logical_content_end = next_content_end;
                    logical_line_end = next_line_end;
                    line = continued_line;
                }
            }
            bool consumed_directive = false;
            if (start < line.size()) {
                const auto [name, args_start] = directive_parts(line, start);
                if (!name.empty() && is_directive_name(name)) {
                    const auto prefix = line.substr(0, start);
                    if (active() &&
                        (in_block_comment ||
                         prefix.find("/*") != std::string_view::npos))
                        append_source(line.substr(0, start), source, offset);
                    update_comment_state(prefix, in_block_comment);
                    consumed_directive = process_directive(
                        name, line.substr(args_start), source, offset + start, path);
                }
            }
            if (!consumed_directive) {
                if (active()) {
                    expand_text(line, source, offset, false, {}, 0, in_block_comment);
                }
                else {
                    update_comment_state(line, in_block_comment);
                }
            }
            append_newline(source, text, logical_content_end,
                           logical_line_end - logical_content_end);
            offset = logical_line_end;
        }

        include_stack_.pop_back();
        if (!path.empty())
            active_paths_.pop_back();
    }

    bool process_directive(std::string_view name, std::string_view raw_args,
                           const std::shared_ptr<const VerilogSourceBuffer>& source,
                           size_t directive_offset, const std::filesystem::path& current_path) {
        const std::string_view args = trim(raw_args);
        const size_t name_offset = directive_offset + 1;
        const size_t args_offset = name_offset + name.size();
        if (name == "ifdef" || name == "ifndef") {
            const std::string macro_name = read_identifier(args);
            if (macro_name.empty())
                diagnose("MalformedConditional", "conditional directive requires a macro name",
                         source, name_offset);
            const bool parent_active = active();
            const bool defined = parent_active && macros_.contains(macro_name);
            const bool branch_active =
                parent_active && (name == "ifdef" ? defined : !defined);
            conditionals_.push_back(
                {parent_active, branch_active, branch_active, false, source, directive_offset});
            return true;
        }
        if (name == "elsif") {
            if (conditionals_.empty()) {
                diagnose("UnmatchedElsif", "unmatched `elsif", source, directive_offset);
                return true;
            }
            auto& frame = conditionals_.back();
            if (frame.saw_else) {
                diagnose("ElsifAfterElse", "`elsif cannot follow `else", source, directive_offset);
                frame.active = false;
                return true;
            }
            const std::string macro_name = read_identifier(args);
            if (macro_name.empty())
                diagnose("MalformedConditional", "`elsif requires a macro name", source,
                         name_offset);
            frame.active = frame.parent_active && !frame.branch_taken &&
                           macros_.contains(macro_name);
            frame.branch_taken = frame.branch_taken || frame.active;
            return true;
        }
        if (name == "else") {
            if (conditionals_.empty()) {
                diagnose("UnmatchedElse", "unmatched `else", source, directive_offset);
                return true;
            }
            auto& frame = conditionals_.back();
            if (frame.saw_else)
                diagnose("DuplicateElse", "duplicate `else", source, directive_offset);
            frame.saw_else = true;
            frame.active = frame.parent_active && !frame.branch_taken;
            frame.branch_taken = true;
            return true;
        }
        if (name == "endif") {
            if (conditionals_.empty())
                diagnose("UnmatchedEndif", "unmatched `endif", source, directive_offset);
            else
                conditionals_.pop_back();
            return true;
        }

        if (!active())
            return true;

        if (name == "define") {
            define_macro(raw_args, source, args_offset);
            return true;
        }
        if (name == "undef") {
            const std::string macro_name = read_identifier(args);
            if (macro_name.empty())
                diagnose("MalformedUndef", "`undef requires a macro name", source, name_offset);
            else
                macros_.erase(macro_name);
            return true;
        }
        if (name == "include") {
            include_file(args, source, directive_offset, current_path);
            return true;
        }
        if (name == "timescale" || name == "default_nettype") {
            record_directive(name, args, source, directive_offset);
            return true;
        }
        if (name == "celldefine" || name == "endcelldefine" ||
            name == "unconnected_drive" || name == "nounconnected_drive" ||
            name == "resetall" || name == "begin_keywords" || name == "end_keywords" ||
            name == "line") {
            record_directive(name, args, source, directive_offset,
                             VerilogPreprocessorDirectiveKind::Deferred);
            return true;
        }
        if (name == "pragma")
            return true;
        diagnose("UnknownDirective", "unknown compiler directive `" + std::string(name),
                 source, directive_offset, VerilogPreprocessorSeverity::Warning);
        record_directive(name, args, source, directive_offset,
                         VerilogPreprocessorDirectiveKind::Deferred);
        return true;
    }

    std::string read_identifier(std::string_view text) const {
        text = trim(text);
        size_t length = 0;
        if (text.empty() || !identifier_start(static_cast<unsigned char>(text.front())))
            return {};
        while (length < text.size() &&
               identifier_continue(static_cast<unsigned char>(text[length])))
            ++length;
        return std::string(text.substr(0, length));
    }

    void collect_parameter_occurrences(Macro& macro) const {
        const auto text = std::string_view(macro.info.replacement);
        auto collect_tokens = [&](size_t start, size_t end) {
            const auto lexed = lex_verilog(
                std::string(text.substr(start, end - start)),
                {options_.language_version, VerilogLexerBackend::Scalar});
            for (const auto& token : lexed.tokens) {
                if (token.kind != VerilogTokenKind::Identifier &&
                    token.kind != VerilogTokenKind::Keyword)
                    continue;
                const size_t token_start = start + token.byte_offset;
                const size_t token_end = token_start + token.length;
                const auto parameter = std::find(
                    macro.info.parameters.begin(), macro.info.parameters.end(),
                    text.substr(token_start, token_end - token_start));
                if (parameter != macro.info.parameters.end()) {
                    macro.parameter_occurrences.push_back(
                        {token_start, token_end,
                         static_cast<size_t>(parameter - macro.info.parameters.begin())});
                }
            }
        };

        size_t segment_start = 0;
        bool in_string = false;
        bool in_block_comment = false;
        for (size_t i = 0; i < text.size();) {
            if (in_block_comment) {
                if (text.substr(i, 2) == "*/") {
                    in_block_comment = false;
                    i += 2;
                }
                else {
                    ++i;
                }
                continue;
            }
            if (in_string) {
                if (text[i] == '\\') {
                    i += std::min<size_t>(2, text.size() - i);
                }
                else {
                    if (text[i] == '"')
                        in_string = false;
                    ++i;
                }
                continue;
            }
            if (text.substr(i, 2) == "//") {
                const size_t newline = text.find_first_of("\r\n", i);
                i = newline == std::string_view::npos ? text.size() : newline;
                continue;
            }
            if (text.substr(i, 2) == "/*") {
                in_block_comment = true;
                i += 2;
                continue;
            }
            if (text.substr(i, 2) == "`\"") {
                collect_tokens(segment_start, i);
                const size_t body_start = i + 2;
                const size_t close = find_stringification_end(text, body_start);
                if (close == std::string_view::npos) {
                    collect_tokens(body_start, text.size());
                    return;
                }
                collect_tokens(body_start, close);
                i = close + 2;
                segment_start = i;
                continue;
            }
            if (text[i] == '"') {
                in_string = true;
                ++i;
                continue;
            }
            if (text[i] == '\\') {
                ++i;
                while (i < text.size() && !whitespace(static_cast<unsigned char>(text[i])))
                    ++i;
                continue;
            }
            ++i;
        }
        collect_tokens(segment_start, text.size());
    }

    void define_macro(std::string_view args,
                      const std::shared_ptr<const VerilogSourceBuffer>& source,
                      size_t args_offset) {
        size_t i = 0;
        while (i < args.size() && whitespace(static_cast<unsigned char>(args[i])))
            ++i;
        const size_t name_start = i;
        if (i == args.size() || !identifier_start(static_cast<unsigned char>(args[i]))) {
            diagnose("MalformedDefine", "`define requires a macro name", source, args_offset);
            return;
        }
        while (i < args.size() && identifier_continue(static_cast<unsigned char>(args[i])))
            ++i;

        Macro macro;
        macro.info.name = std::string(args.substr(name_start, i - name_start));
        macro.info.definition_source = source;
        macro.info.definition_offset = args_offset + name_start;
        if (i < args.size() && args[i] == '(') {
            macro.info.function_like = true;
            ++i;
            bool expect_parameter = true;
            while (true) {
                while (i < args.size() && whitespace(static_cast<unsigned char>(args[i])))
                    ++i;
                if (i < args.size() && args[i] == ')') {
                    if (!macro.info.parameters.empty() && expect_parameter) {
                        diagnose("MalformedDefine", "trailing comma in macro parameter list",
                                 source, args_offset + i);
                        return;
                    }
                    ++i;
                    break;
                }
                const size_t parameter_start = i;
                if (i == args.size() ||
                    !identifier_start(static_cast<unsigned char>(args[i]))) {
                    diagnose("MalformedDefine", "invalid function-like macro parameter list",
                             source, args_offset + i);
                    return;
                }
                while (i < args.size() &&
                       identifier_continue(static_cast<unsigned char>(args[i])))
                    ++i;
                macro.info.parameters.emplace_back(args.substr(parameter_start, i - parameter_start));
                expect_parameter = false;
                while (i < args.size() && whitespace(static_cast<unsigned char>(args[i])))
                    ++i;
                if (i < args.size() && args[i] == ',') {
                    ++i;
                    expect_parameter = true;
                    continue;
                }
                if (i < args.size() && args[i] == ')') {
                    ++i;
                    break;
                }
                diagnose("MalformedDefine", "unterminated function-like macro parameter list",
                         source, args_offset + i);
                return;
            }
        }
        while (i < args.size() && whitespace(static_cast<unsigned char>(args[i])))
            ++i;
        const size_t replacement_start = i;
        macro.info.replacement = std::string(args.substr(replacement_start));
        macro.replacement_offset = args_offset + replacement_start;
        if (macro.info.function_like)
            collect_parameter_occurrences(macro);

        const auto previous = macros_.find(macro.info.name);
        if (previous != macros_.end() &&
            (previous->second.info.function_like != macro.info.function_like ||
             previous->second.info.parameters != macro.info.parameters ||
             previous->second.info.replacement != macro.info.replacement)) {
            diagnose("MacroRedefined", "macro '" + macro.info.name + "' redefined",
                     source, args_offset + name_start, VerilogPreprocessorSeverity::Warning);
        }
        macros_.insert_or_assign(macro.info.name, std::move(macro));
    }

    void record_directive(std::string_view name, std::string_view value,
                          const std::shared_ptr<const VerilogSourceBuffer>& source,
                          size_t offset,
                          VerilogPreprocessorDirectiveKind kind =
                              VerilogPreprocessorDirectiveKind::Timescale) {
        if (kind == VerilogPreprocessorDirectiveKind::Timescale &&
            name == "default_nettype")
            kind = VerilogPreprocessorDirectiveKind::DefaultNettype;
        result_.directives.push_back(
            {kind, std::string(name), std::string(value), source, offset,
             source->location(offset)});
    }

    void include_file(std::string_view argument,
                      const std::shared_ptr<const VerilogSourceBuffer>& source,
                      size_t directive_offset, const std::filesystem::path& current_path) {
        argument = trim(argument);
        if (argument.size() < 2 || argument.front() != '"' || argument.back() != '"') {
            diagnose("MalformedInclude", "`include requires a quoted path", source,
                     directive_offset);
            return;
        }
        const std::filesystem::path requested(std::string(argument.substr(1, argument.size() - 2)));
        std::vector<std::filesystem::path> candidates;
        if (requested.is_absolute())
            candidates.push_back(requested);
        else {
            if (!current_path.empty())
                candidates.push_back(current_path.parent_path() / requested);
            for (const auto& include_path : options_.include_paths)
                candidates.push_back(include_path / requested);
        }

        std::filesystem::path found;
        std::error_code ec;
        for (const auto& candidate : candidates) {
            if (std::filesystem::is_regular_file(candidate, ec) && !ec) {
                found = std::filesystem::weakly_canonical(candidate, ec);
                if (ec)
                    found = candidate.lexically_normal();
                break;
            }
            ec.clear();
        }
        if (found.empty()) {
            diagnose("IncludeNotFound", "included file not found: " + requested.string(),
                     source, directive_offset);
            return;
        }
        const std::string key = found.string();
        auto cached = file_cache_.find(key);
        std::shared_ptr<const VerilogSourceBuffer> include_source;
        if (cached != file_cache_.end()) {
            include_source = cached->second;
        }
        else {
            std::ifstream input(found, std::ios::binary);
            if (!input) {
                diagnose("IncludeReadFailed", "could not read included file: " + key,
                         source, directive_offset);
                return;
            }
            std::string contents{std::istreambuf_iterator<char>(input),
                                 std::istreambuf_iterator<char>()};
            if (input.bad()) {
                diagnose("IncludeReadFailed", "failed while reading included file: " + key,
                         source, directive_offset);
                return;
            }
            include_source =
                std::make_shared<const VerilogSourceBuffer>(std::move(contents), key);
            file_cache_.emplace(key, include_source);
            add_source(include_source);
        }
        process_file(std::move(include_source), found);
    }

    MacroArguments parse_arguments(std::string_view text, size_t open) const {
        MacroArguments result;
        if (open >= text.size() || text[open] != '(')
            return result;
        size_t argument_start = open + 1;
        size_t paren_depth = 1;
        size_t bracket_depth = 0;
        size_t brace_depth = 0;
        bool in_string = false;
        bool in_block_comment = false;
        bool saw_non_whitespace = false;
        bool saw_separator = false;
        for (size_t i = open + 1; i < text.size(); ++i) {
            const char ch = text[i];
            if (in_block_comment) {
                if (text.substr(i, 2) == "*/") {
                    in_block_comment = false;
                    ++i;
                }
                continue;
            }
            if (in_string) {
                if (ch == '\\') {
                    if (i + 1 < text.size())
                        ++i;
                }
                else if (ch == '"') {
                    in_string = false;
                }
                continue;
            }
            if (text.substr(i, 2) == "//")
                break;
            if (text.substr(i, 2) == "/*") {
                in_block_comment = true;
                ++i;
                continue;
            }
            if (ch == '"') {
                in_string = true;
                saw_non_whitespace = true;
            }
            else if (ch == '(') {
                ++paren_depth;
                saw_non_whitespace = true;
            }
            else if (ch == ')') {
                if (--paren_depth == 0) {
                    const auto tail = trim(text.substr(argument_start, i - argument_start));
                    if (saw_separator || saw_non_whitespace || !tail.empty())
                        result.ranges.emplace_back(argument_start, i);
                    result.valid = true;
                    result.end = i + 1;
                    return result;
                }
            }
            else if (ch == '[') {
                ++bracket_depth;
                saw_non_whitespace = true;
            }
            else if (ch == ']' && bracket_depth != 0) {
                --bracket_depth;
            }
            else if (ch == '{') {
                ++brace_depth;
                saw_non_whitespace = true;
            }
            else if (ch == '}' && brace_depth != 0) {
                --brace_depth;
            }
            else if (ch == ',' && paren_depth == 1 && bracket_depth == 0 &&
                     brace_depth == 0) {
                result.ranges.emplace_back(argument_start, i);
                argument_start = i + 1;
                saw_non_whitespace = false;
                saw_separator = true;
            }
            else if (!whitespace(static_cast<unsigned char>(ch))) {
                saw_non_whitespace = true;
            }
        }
        return result;
    }

    bool paste_operator_at(std::string_view text, size_t offset) const {
        return text.substr(offset, 2) == "``";
    }

    std::string paste_tokens(
        std::string substituted,
        const std::shared_ptr<const VerilogSourceBuffer>& source,
        size_t invocation_offset,
        const std::vector<VerilogMacroExpansionFrame>& expansion_stack) {
        std::string output;
        size_t i = 0;
        bool in_string = false;
        bool in_block_comment = false;
        bool in_macro_string = false;
        while (i < substituted.size()) {
            if (in_block_comment) {
                if (substituted.substr(i, 2) == "*/") {
                    output.append("*/");
                    i += 2;
                    in_block_comment = false;
                }
                else {
                    output.push_back(substituted[i++]);
                }
                continue;
            }
            if (in_string) {
                output.push_back(substituted[i]);
                if (substituted[i] == '\\' && i + 1 < substituted.size()) {
                    output.push_back(substituted[i + 1]);
                    i += 2;
                }
                else {
                    if (substituted[i++] == '"')
                        in_string = false;
                }
                continue;
            }
            if (in_macro_string && substituted.substr(i, 2) == "`\"") {
                output.append("`\"");
                i += 2;
                in_macro_string = false;
                continue;
            }
            if (!in_macro_string && substituted.substr(i, 2) == "//") {
                output.append(substituted.substr(i));
                break;
            }
            if (!in_macro_string && substituted.substr(i, 2) == "/*") {
                output.append("/*");
                i += 2;
                in_block_comment = true;
                continue;
            }
            if (!in_macro_string && substituted.substr(i, 2) == "`\"") {
                output.append("`\"");
                i += 2;
                in_macro_string = true;
                continue;
            }
            if (!in_macro_string && substituted[i] == '"') {
                output.push_back(substituted[i++]);
                in_string = true;
                continue;
            }
            if (!in_macro_string && substituted[i] == '\\') {
                do {
                    output.push_back(substituted[i++]);
                } while (i < substituted.size() &&
                         !whitespace(static_cast<unsigned char>(substituted[i])));
                continue;
            }
            if (in_macro_string || !paste_operator_at(substituted, i)) {
                output.push_back(substituted[i++]);
                continue;
            }

            const auto left_lexed = lex_verilog(
                output, {options_.language_version, VerilogLexerBackend::Scalar});
            const VerilogToken* left = nullptr;
            for (const auto& token : left_lexed.tokens) {
                if (token.kind != VerilogTokenKind::EndOfFile &&
                    token.kind != VerilogTokenKind::LineComment &&
                    token.kind != VerilogTokenKind::BlockComment)
                    left = &token;
            }
            size_t right_start = i + 2;
            while (right_start < substituted.size() &&
                   whitespace(static_cast<unsigned char>(substituted[right_start])))
                ++right_start;
            const auto right_lexed = lex_verilog(
                std::string(substituted.substr(right_start)),
                {options_.language_version, VerilogLexerBackend::Scalar});
            const VerilogToken* right = nullptr;
            for (const auto& token : right_lexed.tokens) {
                if (token.kind != VerilogTokenKind::EndOfFile &&
                    token.kind != VerilogTokenKind::LineComment &&
                    token.kind != VerilogTokenKind::BlockComment) {
                    right = &token;
                    break;
                }
            }
            if (!left || !right) {
                diagnose("MalformedTokenPaste", "token-paste operator requires two tokens",
                         source, invocation_offset, VerilogPreprocessorSeverity::Error,
                         expansion_stack);
                output.append("``");
                i += 2;
                continue;
            }

            const std::string left_text = std::string(left_lexed.spelling(*left));
            const std::string right_text = std::string(right_lexed.spelling(*right));
            const std::string pasted = left_text + right_text;
            const auto pasted_lexed = lex_verilog(
                pasted, {options_.language_version, VerilogLexerBackend::Scalar});
            size_t significant_tokens = 0;
            const VerilogToken* pasted_token = nullptr;
            for (const auto& token : pasted_lexed.tokens) {
                if (token.kind == VerilogTokenKind::EndOfFile)
                    continue;
                ++significant_tokens;
                pasted_token = &token;
            }
            if (!pasted_lexed.errors.empty() || significant_tokens != 1 || !pasted_token ||
                pasted_token->byte_offset != 0 || pasted_token->length != pasted.size()) {
                diagnose("InvalidTokenPaste",
                         "token concatenation does not form one valid Verilog token: '" +
                             pasted + "'",
                         source, invocation_offset, VerilogPreprocessorSeverity::Error,
                         expansion_stack);
            }

            const size_t left_start = left->byte_offset;
            output.resize(left_start);
            output.append(pasted);
            i = right_start + right->byte_offset + right->length;
        }
        return output;
    }

    std::string substitute_parameters(
        std::string_view replacement, const Macro& macro,
        const std::vector<std::string>& arguments,
        const std::shared_ptr<const VerilogSourceBuffer>& source, size_t invocation_offset,
        const std::vector<VerilogMacroExpansionFrame>& expansion_stack) {
        std::string result;
        size_t copied_until = 0;
        for (const auto& occurrence : macro.parameter_occurrences) {
            result.append(replacement.substr(copied_until, occurrence.start - copied_until));
            result.append(arguments[occurrence.parameter_index]);
            copied_until = occurrence.end;
        }
        result.append(replacement.substr(copied_until));
        return paste_tokens(std::move(result), source, invocation_offset, expansion_stack);
    }

    bool is_directive_name(std::string_view name) const {
        static constexpr std::string_view names[] = {
            "define", "undef", "ifdef", "ifndef", "elsif", "else", "endif", "include",
            "timescale", "default_nettype", "celldefine", "endcelldefine",
            "unconnected_drive", "nounconnected_drive", "resetall", "begin_keywords",
            "end_keywords", "line", "pragma"
        };
        return std::find(std::begin(names), std::end(names), name) != std::end(names);
    }

    void expand_text(std::string_view text,
                     const std::shared_ptr<const VerilogSourceBuffer>& source,
                     size_t source_offset, bool synthetic,
                     std::vector<VerilogMacroExpansionFrame> expansion_stack, size_t depth,
                     bool& in_block_comment, bool stringify_output = false) {
        size_t i = 0;
        size_t copy_start = 0;
        auto emit = [&](std::string_view value,
                        const std::shared_ptr<const VerilogSourceBuffer>& origin_source,
                        size_t origin_offset, VerilogPreprocessedOriginKind kind,
                        const std::vector<VerilogMacroExpansionFrame>& stack) {
            if (!stringify_output) {
                append(value, origin_source, origin_offset, kind, stack);
                return;
            }
            std::string escaped;
            escaped.reserve(value.size());
            for (const char ch : value) {
                if (ch == '\\' || ch == '"')
                    escaped.push_back('\\');
                escaped.push_back(ch);
            }
            append(escaped, origin_source, origin_offset, kind, stack);
        };
        auto flush = [&](size_t end) {
            if (end > copy_start) {
                const size_t mapped_offset = synthetic ? source_offset : source_offset + copy_start;
                emit(text.substr(copy_start, end - copy_start), source, mapped_offset,
                     synthetic ? VerilogPreprocessedOriginKind::MacroExpansion
                               : VerilogPreprocessedOriginKind::Source,
                     expansion_stack);
            }
        };

        while (i < text.size()) {
            if (in_block_comment) {
                const size_t close = text.find("*/", i);
                if (close == std::string_view::npos) {
                    i = text.size();
                    break;
                }
                i = close + 2;
                in_block_comment = false;
                continue;
            }
            if (text.substr(i, 2) == "//") {
                i = text.size();
                break;
            }
            if (text.substr(i, 2) == "/*") {
                in_block_comment = true;
                i += 2;
                continue;
            }
            if (text[i] == '"') {
                ++i;
                while (i < text.size()) {
                    if (text[i] == '\\' && i + 1 < text.size()) {
                        i += 2;
                        continue;
                    }
                    if (text[i++] == '"')
                        break;
                }
                continue;
            }
            if (text[i] == '\\') {
                ++i;
                while (i < text.size() && !whitespace(static_cast<unsigned char>(text[i])))
                    ++i;
                continue;
            }
            if (synthetic && text.substr(i, 2) == "`\"") {
                const size_t close = find_stringification_end(text, i + 2);
                flush(i);
                const size_t mapped_offset = source_offset;
                if (close == std::string_view::npos) {
                    diagnose("MalformedMacroStringification",
                             "macro stringification is missing its closing `\" delimiter",
                             source, mapped_offset, VerilogPreprocessorSeverity::Error,
                             expansion_stack);
                    emit(text.substr(i, 2), source, mapped_offset,
                         VerilogPreprocessedOriginKind::MacroExpansion, expansion_stack);
                    i += 2;
                    copy_start = i;
                    continue;
                }
                emit("\"", source, mapped_offset,
                     VerilogPreprocessedOriginKind::MacroExpansion, expansion_stack);
                bool string_comment_state = false;
                expand_text(text.substr(i + 2, close - i - 2), source, mapped_offset, true,
                            expansion_stack, depth, string_comment_state, true);
                emit("\"", source, mapped_offset,
                     VerilogPreprocessedOriginKind::MacroExpansion, expansion_stack);
                i = close + 2;
                copy_start = i;
                continue;
            }
            if (text[i] != '`' || i + 1 >= text.size()) {
                ++i;
                continue;
            }

            if (text[i + 1] == '`') {
                flush(i);
                const size_t mapped_offset = synthetic ? source_offset : source_offset + i;
                emit(text.substr(i, 2), source, mapped_offset,
                     synthetic ? VerilogPreprocessedOriginKind::MacroExpansion
                               : VerilogPreprocessedOriginKind::Source,
                     expansion_stack);
                i += 2;
                copy_start = i;
                continue;
            }
            if (text[i + 1] == '"') {
                ++i;
                continue;
            }

            size_t name_end = i + 1;
            if (!identifier_start(static_cast<unsigned char>(text[name_end]))) {
                ++i;
                continue;
            }
            ++name_end;
            while (name_end < text.size() &&
                   identifier_continue(static_cast<unsigned char>(text[name_end])))
                ++name_end;
            const std::string name(text.substr(i + 1, name_end - i - 1));
            const auto macro_it = macros_.find(name);
            if (macro_it == macros_.end()) {
                if (is_directive_name(name)) {
                    i = name_end;
                    continue;
                }
                flush(i);
                const size_t mapped_offset = synthetic ? source_offset : source_offset + i;
                diagnose("UndefinedMacro", "undefined macro '" + name + "'", source,
                         mapped_offset, VerilogPreprocessorSeverity::Error, expansion_stack);
                append(text.substr(i, name_end - i), source, mapped_offset,
                       synthetic ? VerilogPreprocessedOriginKind::MacroExpansion
                                 : VerilogPreprocessedOriginKind::Source,
                       expansion_stack);
                i = name_end;
                copy_start = i;
                continue;
            }

            const Macro& macro = macro_it->second;
            size_t invocation_end = name_end;
            std::vector<std::string> arguments;
            if (macro.info.function_like) {
                size_t open = name_end;
                while (open < text.size() &&
                       whitespace(static_cast<unsigned char>(text[open])))
                    ++open;
                const MacroArguments parsed = parse_arguments(text, open);
                if (!parsed.valid) {
                    flush(i);
                    const size_t mapped_offset = synthetic ? source_offset : source_offset + i;
                    diagnose("MacroArgumentsExpected",
                             "function-like macro '" + name + "' requires arguments", source,
                             mapped_offset, VerilogPreprocessorSeverity::Error, expansion_stack);
                    i = name_end;
                    copy_start = i;
                    continue;
                }
                invocation_end = parsed.end;
                for (const auto& [start, end] : parsed.ranges)
                    arguments.emplace_back(trim(text.substr(start, end - start)));
                if (arguments.empty() && !macro.info.parameters.empty())
                    arguments.emplace_back();
                if (arguments.size() != macro.info.parameters.size()) {
                    flush(i);
                    const size_t mapped_offset = synthetic ? source_offset : source_offset + i;
                    diagnose("MacroArgumentCount",
                             "macro '" + name + "' expects " +
                                 std::to_string(macro.info.parameters.size()) + " arguments, got " +
                                 std::to_string(arguments.size()),
                             source, mapped_offset, VerilogPreprocessorSeverity::Error,
                             expansion_stack);
                    append(text.substr(i, invocation_end - i), source, mapped_offset,
                           synthetic ? VerilogPreprocessedOriginKind::MacroExpansion
                                     : VerilogPreprocessedOriginKind::Source,
                           expansion_stack);
                    i = invocation_end;
                    copy_start = i;
                    continue;
                }
            }
            if (depth >= options_.max_macro_expansion_depth ||
                std::any_of(expansion_stack.begin(), expansion_stack.end(),
                            [&](const auto& frame) { return frame.macro_name == name; })) {
                flush(i);
                const size_t mapped_offset = synthetic ? source_offset : source_offset + i;
                diagnose(depth >= options_.max_macro_expansion_depth
                             ? "MacroExpansionDepthExceeded"
                             : "RecursiveMacroExpansion",
                         depth >= options_.max_macro_expansion_depth
                             ? "maximum macro expansion depth exceeded"
                             : "recursive macro expansion detected for '" + name + "'",
                         source, mapped_offset, VerilogPreprocessorSeverity::Error,
                         expansion_stack);
                append(text.substr(i, invocation_end - i), source, mapped_offset,
                       synthetic ? VerilogPreprocessedOriginKind::MacroExpansion
                                 : VerilogPreprocessedOriginKind::Source,
                       expansion_stack);
                i = invocation_end;
                copy_start = i;
                continue;
            }

            flush(i);
            const size_t invocation_offset = synthetic ? source_offset : source_offset + i;
            const auto definition_source = macro.info.definition_source;
            const auto frame = VerilogMacroExpansionFrame{
                name, source->id(), source->location(invocation_offset),
                definition_source->id(),
                definition_source->location(macro.info.definition_offset)};
            auto nested_stack = expansion_stack;
            nested_stack.push_back(frame);
            std::string substituted = substitute_parameters(
                macro.info.replacement, macro, arguments, source, invocation_offset, nested_stack);
            bool replacement_block_comment = in_block_comment;
            expand_text(substituted, source, invocation_offset, true, std::move(nested_stack),
                        depth + 1, replacement_block_comment, stringify_output);
            in_block_comment = replacement_block_comment;
            i = invocation_end;
            copy_start = i;
        }
        flush(text.size());
    }

    VerilogPreprocessorOptions options_;
    VerilogPreprocessorResult result_;
    std::map<std::string, Macro, std::less<>> macros_;
    std::vector<ConditionalFrame> conditionals_;
    std::vector<std::string> include_stack_;
    std::vector<std::string> active_paths_;
    std::map<std::string, std::shared_ptr<const VerilogSourceBuffer>, std::less<>> file_cache_;
};

std::shared_ptr<const VerilogSourceBuffer> read_source_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input)
        throw std::runtime_error("could not read Verilog source file: " + path.string());
    std::string contents{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    if (input.bad())
        throw std::runtime_error("failed while reading Verilog source file: " + path.string());
    return std::make_shared<const VerilogSourceBuffer>(std::move(contents), path.string());
}

} // namespace

std::optional<VerilogPreprocessedOrigin> VerilogPreprocessorResult::origin_at(
    size_t output_offset) const {
    if (output_offset >= text.size())
        return std::nullopt;
    const auto it = std::upper_bound(
        source_map.begin(), source_map.end(), output_offset,
        [](size_t offset, const VerilogSourceMapSegment& segment) {
            return offset < segment.output_offset;
        });
    if (it == source_map.begin())
        return std::nullopt;
    const auto& segment = *std::prev(it);
    if (output_offset >= segment.output_offset + segment.length)
        return std::nullopt;
    const size_t source_offset =
        segment.kind == VerilogPreprocessedOriginKind::Source
            ? segment.source_offset + output_offset - segment.output_offset
            : segment.source_offset;
    return VerilogPreprocessedOrigin{
        segment.source, source_offset, segment.source->location(source_offset), segment.kind,
        segment.expansion_stack};
}

bool VerilogPreprocessorResult::has_errors() const noexcept {
    return std::any_of(diagnostics.begin(), diagnostics.end(), [](const auto& diagnostic) {
        return diagnostic.severity == VerilogPreprocessorSeverity::Error;
    });
}

VerilogPreprocessorResult preprocess_verilog(
    std::shared_ptr<const VerilogSourceBuffer> source, VerilogPreprocessorOptions options) {
    return Preprocessor(std::move(options)).run(std::move(source));
}

VerilogPreprocessorResult preprocess_verilog(std::string source,
                                             VerilogPreprocessorOptions options,
                                             std::string source_name) {
    return preprocess_verilog(
        std::make_shared<const VerilogSourceBuffer>(std::move(source), std::move(source_name)),
        std::move(options));
}

VerilogPreprocessorResult preprocess_verilog_file(const std::filesystem::path& path,
                                                  VerilogPreprocessorOptions options) {
    std::error_code ec;
    const auto absolute = std::filesystem::weakly_canonical(path, ec);
    const auto resolved = ec ? path : absolute;
    return Preprocessor(std::move(options)).run(read_source_file(resolved), resolved);
}

} // namespace labneura::eda
