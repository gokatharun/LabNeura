#include "labneura/eda/verilog_preprocessor.h"

#include <slang/diagnostics/Diagnostics.h>
#include <slang/parsing/Preprocessor.h>
#include <slang/parsing/Token.h>
#include <slang/text/SourceManager.h>
#include <slang/util/BumpAllocator.h>

#include <algorithm>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace labneura::eda;

void require(bool condition, std::string_view message) {
    if (!condition)
        throw std::runtime_error(std::string(message));
}

class TemporaryDirectory {
public:
    TemporaryDirectory() {
        static std::atomic<uint64_t> next_id{0};
        path_ = std::filesystem::temp_directory_path() /
                ("labneura-preprocessor-" + std::to_string(next_id.fetch_add(1)));
        std::filesystem::create_directories(path_);
    }

    ~TemporaryDirectory() {
        std::error_code ec;
        std::filesystem::remove_all(path_, ec);
    }

    const std::filesystem::path& path() const noexcept { return path_; }

private:
    std::filesystem::path path_;
};

void write_file(const std::filesystem::path& path, std::string_view contents) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary);
    if (!output)
        throw std::runtime_error("could not create test source file");
    output.write(contents.data(), static_cast<std::streamsize>(contents.size()));
    if (!output)
        throw std::runtime_error("could not write test source file");
}

bool has_diagnostic(const VerilogPreprocessorResult& result, std::string_view code) {
    return std::any_of(result.diagnostics.begin(), result.diagnostics.end(),
                       [&](const auto& diagnostic) { return diagnostic.code == code; });
}

void require_complete_source_map(const VerilogPreprocessorResult& result) {
    size_t output_offset = 0;
    for (const auto& segment : result.source_map) {
        require(segment.length != 0 && segment.output_offset == output_offset &&
                    segment.source && segment.source_offset <= segment.source->text().size(),
                "source map segments must be ordered, contiguous, and owned");
        if (segment.kind == VerilogPreprocessedOriginKind::Source)
            require(segment.length <= segment.source->text().size() - segment.source_offset,
                    "ordinary source mapping exceeds its input buffer");
        output_offset += segment.length;
    }
    require(output_offset == result.text.size(),
            "source map must cover every emitted byte exactly once");
}

std::vector<std::string> active_tokens(std::string_view source) {
    slang::SourceManager source_manager;
    slang::BumpAllocator allocator;
    slang::Diagnostics diagnostics;
    slang::parsing::Preprocessor preprocessor(source_manager, allocator, diagnostics);
    preprocessor.pushSource(source, "differential.sv");
    std::vector<std::string> tokens;
    while (true) {
        const auto token = preprocessor.next();
        if (token.kind == slang::parsing::TokenKind::EndOfFile)
            break;
        tokens.emplace_back(token.rawText());
    }
    return tokens;
}

void test_macro_expansion_and_lexical_boundaries() {
    const std::string source =
        "`define WIDTH 32\n"
        "`define WIDTHX 64\n"
        "`define ADD(a,b) ((a) + (b))\n"
        "`define TWICE(x) `ADD(x, x)\n"
        "logic [`WIDTH-1:0] data;\n"
        "logic [`WIDTHX-1:0] other;\n"
        "assign result = `TWICE(value + (offset, 1));\n"
        "string s = \"`WIDTH // not a comment\"; // `ADD(x,y)\n"
        "/* `WIDTH and `TWICE(a,b) */ \\escaped`WIDTH \n";
    const auto result = preprocess_verilog(source);
    require(!result.has_errors(), "valid macro source must preprocess without errors");
    require(result.text.find("logic [32-1:0] data;") != std::string::npos,
            "object-like macro did not expand");
    require(result.text.find("logic [64-1:0] other;") != std::string::npos,
            "macro names must be matched as complete identifiers");
    require(result.text.find("((value + (offset, 1)) + (value + (offset, 1)))") !=
                std::string::npos,
            "nested function-like macro arguments did not expand");
    require(result.text.find("\"`WIDTH // not a comment\"") != std::string::npos,
            "macro-like text inside strings must remain unchanged");
    require(result.text.find("// `ADD(x,y)") != std::string::npos &&
                result.text.find("/* `WIDTH and `TWICE(a,b) */") != std::string::npos,
            "macro-like text inside comments must remain unchanged");
    require(result.text.find("\\escaped`WIDTH") != std::string::npos,
            "macro-like text inside escaped identifiers must remain unchanged");
    require(result.macros.size() == 4 && result.macros.contains("ADD"),
            "final macro configuration was not returned");
    require_complete_source_map(result);
    const auto undefined = preprocess_verilog(
        "`define TEMP 1\n`undef TEMP\n`ifdef TEMP\nwrong\n`else\nright\n`endif\n");
    require(!undefined.has_errors() && !undefined.macros.contains("TEMP") &&
                undefined.text.find("right\n") != std::string::npos &&
                undefined.text.find("wrong\n") == std::string::npos,
            "`undef` must remove a macro from subsequent conditional evaluation");
}

void test_macro_arguments_diagnostics_and_origins() {
    const auto source = std::make_shared<const VerilogSourceBuffer>(
        "`define F(a,b) a + b\n"
        "`define EMPTY(x) [x]\n"
        "`define PAIR(a,b) {a,b}\n"
        "`define LOOP `LOOP\n"
        "wire x = `F( , (a, b));\n"
        "wire y = `EMPTY();\n"
        "wire p = `PAIR(,);\n"
        "`define F(a,b) a - b\n"
        "wire z = `LOOP;\n"
        "wire q = `MISSING;\n",
        "macro-origin.sv");
    const auto result = preprocess_verilog(source);
    require(result.has_errors(), "recursive and undefined macros must be errors");
    require(has_diagnostic(result, "MacroRedefined"), "macro redefinition warning is missing");
    require(has_diagnostic(result, "RecursiveMacroExpansion"),
            "recursive expansion diagnostic is missing");
    require(has_diagnostic(result, "UndefinedMacro"), "undefined macro diagnostic is missing");
    require(result.text.find(" + (a, b)") != std::string::npos,
            "empty function macro argument was not accepted");
    require(result.text.find("[]") != std::string::npos,
            "empty single macro argument was not accepted");
    require(result.text.find("{,}") != std::string::npos,
            "multiple empty macro arguments were not accepted");

    const size_t expansion_offset = result.text.find(" + (a, b)");
    const auto origin = result.origin_at(expansion_offset);
    require(origin.has_value() && origin->kind == VerilogPreprocessedOriginKind::MacroExpansion,
            "expanded output is missing macro-origin mapping");
    require(origin->source->id() == source->id() && origin->location.line == 5 &&
                origin->expansion_stack.size() == 1 &&
                origin->expansion_stack.front().macro_name == "F" &&
                origin->expansion_stack.front().definition_location.line == 1,
            "macro expansion mapping does not point to the invocation and definition");
    require(result.origin_at(result.text.size()).has_value() == false,
            "origin lookup past the output must return empty");
    require_complete_source_map(result);
}

void test_macro_token_pasting() {
    const std::string source =
        "`define OBJECT_PASTE pre``fixed\n"
        "`define CAT(a,b) a``b\n"
        "`define HEX(a,b) a``b\n"
        "`define JOIN3(a,b,c) a``b``c\n"
        "`define LITERAL(a) pre``a``_suffix\n"
        "`define BIN(b) 8'b1010 + b\n"
        "wire `OBJECT_PASTE;\n"
        "wire `CAT(signal,7);\n"
        "wire [`HEX(8,'hFF)-1:0] value;\n"
        "wire `JOIN3(pre,fix,_name);\n"
        "wire `LITERAL(tag);\n"
        "wire `BIN(signal);\n";
    const auto result = preprocess_verilog(source);
    require(!result.has_errors(), "valid macro token pasting must preprocess cleanly");
    require(result.text.find("wire signal7;") != std::string::npos &&
                result.text.find("wire prefixed;") != std::string::npos &&
                result.text.find("wire [8'hFF-1:0] value;") != std::string::npos &&
                result.text.find("wire prefix_name;") != std::string::npos &&
                result.text.find("wire pretag_suffix;") != std::string::npos &&
                result.text.find("wire 8'b1010 + signal;") != std::string::npos,
            "identifier, numeric, or chained token pasting produced incorrect text");
    require_complete_source_map(result);
    require(active_tokens(result.text) == active_tokens(source),
            "token-paste token stream differs from Slang preprocessing");

    const auto invalid = preprocess_verilog(
        "`define CAT(a,b) a``b\nwire `CAT(+,*);\n");
    require(has_diagnostic(invalid, "InvalidTokenPaste"),
            "invalid pasted tokens must have a structured diagnostic");
    const auto origin = invalid.origin_at(invalid.text.find("+*"));
    require(origin && origin->kind == VerilogPreprocessedOriginKind::MacroExpansion &&
                origin->expansion_stack.front().macro_name == "CAT",
            "invalid token-paste output must retain macro expansion mapping");
    const auto malformed = preprocess_verilog(
        "`define BAD_STRING(x) `\"x\nwire `BAD_STRING(value);\n");
    require(has_diagnostic(malformed, "MalformedMacroStringification"),
            "unterminated macro stringification must be diagnosed");
}

void test_macro_stringification() {
    const std::string source =
        "`define WORD hello\n"
        "`define STR(x) `\"x`\"\n"
        "string first = `STR(world);\n"
        "string escaped = `STR(a \"quoted\" \\path );\n"
        "string expanded = `STR(`WORD);\n"
        "string ordinary = \"keep `WORD and \\\"quotes\\\"\";\n";
    const auto native = preprocess_verilog(source);
    require(!native.has_errors(), "valid macro stringification must preprocess cleanly");
    require(native.text.find("string first = \"world\";") != std::string::npos &&
                native.text.find("string escaped = \"a \\\"quoted\\\" \\\\path\";") !=
                    std::string::npos &&
                native.text.find("string expanded = \"hello\";") != std::string::npos,
            "macro stringification did not substitute or escape its argument");
    require(native.text.find("string ordinary = \"keep `WORD and \\\"quotes\\\"\";") !=
                std::string::npos,
            "ordinary strings must not be changed by macro stringification");
    const auto string_origin = native.origin_at(native.text.find("\"world\""));
    require(string_origin &&
                string_origin->kind == VerilogPreprocessedOriginKind::MacroExpansion &&
                !string_origin->expansion_stack.empty() &&
                string_origin->expansion_stack.back().macro_name == "STR",
            "stringified output must retain macro expansion source mapping");
    require_complete_source_map(native);

    const std::string differential_source =
        "`define WORD hello\n"
        "`define STR(x) `\"x`\"\n"
        "string first = `STR(world);\n"
        "string expanded = `STR(`WORD);\n"
        "string quoted = `STR(a \"quoted\");\n";
    const auto differential_native = preprocess_verilog(differential_source);
    const auto native_tokens = active_tokens(differential_native.text);
    const auto slang_tokens = active_tokens(differential_source);
    require(native_tokens == slang_tokens,
            "macro stringification token stream differs from Slang preprocessing");
}

void test_macro_definition_line_continuations() {
    const std::string lf =
        "`define LF_VALUE 8 + \\\n"
        "1\n"
        "wire [(`LF_VALUE)-1:0] lf_value;\n";
    const auto lf_result = preprocess_verilog(lf);
    require(!lf_result.has_errors() &&
                lf_result.text.find("wire [(8 + 1)-1:0] lf_value;") != std::string::npos,
            "LF macro continuation did not preserve the joined replacement text");
    const auto lf_definition = lf_result.macros.find("LF_VALUE");
    require(lf_definition != lf_result.macros.end() &&
                lf_definition->second.definition_source
                        ->location(lf_definition->second.definition_offset)
                        .line == 1,
            "continued macro definition location must point to its original line");

    const std::string crlf =
        "`define CRLF_VALUE 4 + \\\r\n"
        "2\r\n"
        "wire [(`CRLF_VALUE)-1:0] crlf_value;\r\n";
    const auto crlf_result = preprocess_verilog(crlf);
    require(!crlf_result.has_errors() &&
                crlf_result.text.find("wire [(4 + 2)-1:0] crlf_value;") != std::string::npos,
            "CRLF macro continuation did not preserve the joined replacement text");
    require(crlf_result.text.find("2\r\nwire") == std::string::npos,
            "continued physical lines must not leak into output as separate text");

    require(active_tokens(lf_result.text) == active_tokens(lf),
            "LF continued macro token stream differs from Slang preprocessing");
    require(active_tokens(crlf_result.text) == active_tokens(crlf),
            "CRLF continued macro token stream differs from Slang preprocessing");
}

void test_conditionals_and_malformed_directives() {
    const std::string source =
        "`define A\n"
        "`ifdef A\n"
        "one\n"
        "`ifndef B\n"
        "two\n"
        "`else\n"
        "wrong\n"
        "`endif\n"
        "`elsif B\n"
        "wrong2\n"
        "`else\n"
        "wrong3\n"
        "`endif\n"
        "`ifndef A\n"
        "wrong4\n"
        "`elsif B\n"
        "wrong5\n"
        "`else\n"
        "three\n"
        "`endif\n";
    const auto result = preprocess_verilog(source);
    require(!result.has_errors(), "valid nested conditional source must be accepted");
    require(result.text.find("one\n") != std::string::npos &&
                result.text.find("two\n") != std::string::npos &&
                result.text.find("three\n") != std::string::npos,
            "active conditional branches were not emitted");
    for (const auto* inactive : {"wrong\n", "wrong2\n", "wrong3\n", "wrong4\n", "wrong5\n"})
        require(result.text.find(inactive) == std::string::npos,
                "inactive conditional branch was emitted");

    const auto malformed = preprocess_verilog(
        "`else\n`elsif A\n`ifdef A\nbody\n");
    require(has_diagnostic(malformed, "UnmatchedElse") &&
                has_diagnostic(malformed, "UnmatchedElsif") &&
                has_diagnostic(malformed, "MissingEndif"),
            "unmatched and unterminated conditional diagnostics are required");
    const auto missing_name = preprocess_verilog("`ifdef\nx\n`endif\n");
    require(has_diagnostic(missing_name, "MalformedConditional"),
            "malformed conditional directive diagnostic is missing");
    const auto duplicate_else = preprocess_verilog(
        "`ifdef A\nfirst\n`else\nsecond\n`else\nthird\n`endif\n");
    require(has_diagnostic(duplicate_else, "DuplicateElse"),
            "duplicate else directives must be diagnosed");
    const auto elsif_after_else = preprocess_verilog(
        "`ifdef A\nfirst\n`else\nsecond\n`elsif B\nthird\n`endif\n");
    require(has_diagnostic(elsif_after_else, "ElsifAfterElse"),
            "elsif after else must be diagnosed");
    const auto inactive_side_effects = preprocess_verilog(
        "`ifdef NOT_DEFINED\n"
        "`define HIDDEN 1\n"
        "`include \"should-not-be-opened.vh\"\n"
        "`endif\n"
        "`ifdef HIDDEN\nwrong\n`else\nright\n`endif\n");
    require(!inactive_side_effects.has_errors() &&
                inactive_side_effects.text.find("right\n") != std::string::npos &&
                inactive_side_effects.text.find("wrong\n") == std::string::npos &&
                !inactive_side_effects.macros.contains("HIDDEN"),
            "inactive branches must not define macros or load included files");
}

void test_includes_and_file_source_mapping() {
    TemporaryDirectory temp;
    const auto root = temp.path() / "rtl" / "top.sv";
    const auto relative_include = temp.path() / "rtl" / "headers" / "defs.vh";
    const auto nested_include = temp.path() / "shared" / "nested.vh";
    write_file(nested_include, "`define DEPTH 12\nwire nested;\n");
    write_file(relative_include, "`include \"../../shared/nested.vh\"\nwire [`DEPTH-1:0] bus;\n");
    write_file(root, "`include \"headers/defs.vh\"\n"
                     "`include \"headers/defs.vh\"\n"
                     "wire top;\n");

    auto result = preprocess_verilog_file(root);
    require(!result.has_errors(), "relative and nested includes must preprocess");
    require(result.text.find("wire nested;") != std::string::npos &&
                result.text.find("wire [12-1:0] bus;") != std::string::npos &&
                result.text.find("wire top;") != std::string::npos,
            "nested include contents or macro state are missing");
    require(result.sources.size() == 3,
            "included source buffers must be cached and identities retained");
    require_complete_source_map(result);
    const size_t nested_offset = result.text.find("wire nested;");
    const auto nested_origin = result.origin_at(nested_offset);
    std::error_code canonical_error;
    const auto canonical_nested = std::filesystem::weakly_canonical(nested_include, canonical_error);
    require(nested_origin && nested_origin->source->name() == canonical_nested.string() &&
                nested_origin->location.line == 2,
            "included text must map to the included source file");
    const size_t top_offset = result.text.find("wire top;");
    const auto top_origin = result.origin_at(top_offset);
    const auto canonical_root = std::filesystem::weakly_canonical(root, canonical_error);
    require(top_origin && top_origin->source->name() == canonical_root.string() &&
                top_origin->location.line == 3,
            "parent text mapping was lost across included files");

    VerilogPreprocessorOptions include_search;
    include_search.include_paths.push_back(temp.path() / "shared");
    const auto searched = preprocess_verilog(
        "`include \"nested.vh\"\n", include_search, (temp.path() / "memory.sv").string());
    require(!searched.has_errors() && searched.text.find("wire nested;") != std::string::npos,
            "configured include search path was not used");

    const auto cycle_a = temp.path() / "cycle_a.vh";
    const auto cycle_b = temp.path() / "cycle_b.vh";
    write_file(cycle_a, "`include \"cycle_b.vh\"\n");
    write_file(cycle_b, "`include \"cycle_a.vh\"\n");
    const auto cycle = preprocess_verilog_file(cycle_a);
    require(has_diagnostic(cycle, "IncludeCycle"), "include cycle must be diagnosed");
    const auto missing = preprocess_verilog(
        "`include \"does-not-exist.vh\"\n", {}, (temp.path() / "missing.sv").string());
    require(has_diagnostic(missing, "IncludeNotFound"),
            "missing include must produce an explicit diagnostic");
    const auto missing_diagnostic = std::find_if(
        missing.diagnostics.begin(), missing.diagnostics.end(),
        [](const auto& diagnostic) { return diagnostic.code == "IncludeNotFound"; });
    require(missing_diagnostic != missing.diagnostics.end() &&
                missing_diagnostic->location.line == 1 &&
                !missing_diagnostic->include_stack.empty(),
            "include diagnostics must retain source and include context");
}

void test_directive_metadata_empty_and_line_endings() {
    const auto result = preprocess_verilog(
        "`timescale 1ns/1ps\r\n"
        "`default_nettype none\n"
        "`begin_keywords \"1800-2012\"\n"
        "wire a;\r\n"
        "wire b;");
    require(result.directives.size() == 3, "compiler directive metadata count differs");
    require(result.directives[0].kind == VerilogPreprocessorDirectiveKind::Timescale &&
                result.directives[0].value == "1ns/1ps",
            "timescale metadata was not preserved");
    require(result.directives[1].kind == VerilogPreprocessorDirectiveKind::DefaultNettype &&
                result.directives[1].value == "none",
            "default_nettype metadata was not preserved");
    require(result.directives[2].kind == VerilogPreprocessorDirectiveKind::Deferred &&
                result.directives[2].name == "begin_keywords",
            "deferred directive metadata was not retained");
    require(result.text.find("wire a;\r\nwire b;") != std::string::npos,
            "ordinary text or CRLF line endings were not preserved");
    const auto empty = preprocess_verilog(std::string{});
    require(empty.text.empty() && empty.source_map.empty() && !empty.has_errors(),
            "empty source must produce an empty successful result");
    const auto no_newline = preprocess_verilog("wire x;");
    require(no_newline.text == "wire x;" && no_newline.origin_at(0),
            "input without trailing newline was not preserved and mapped");

    const auto comment_prefix = preprocess_verilog(
        "/* comment starts\n"
        "*/ `define AFTER_COMMENT 1\n"
        "wire [`AFTER_COMMENT-1:0] value;\n");
    require(!comment_prefix.has_errors() &&
                comment_prefix.text.find("*/ \n") != std::string::npos &&
                comment_prefix.text.find("wire [1-1:0] value;") != std::string::npos,
            "block comments before directives must preserve comment state and text");
}

void test_large_input_and_expansion_limit() {
    std::string large;
    large.reserve(1024 * 1024);
    for (size_t i = 0; i < 30000; ++i)
        large += "wire signal_" + std::to_string(i) + ";\n";
    const auto large_result = preprocess_verilog(large);
    require(!large_result.has_errors() && large_result.text == large,
            "large ordinary input must be preserved");
    require(large_result.source_map.size() == 1,
            "contiguous ordinary source should use a compact mapping segment");

    VerilogPreprocessorOptions limited;
    limited.max_macro_expansion_depth = 2;
    const auto nested = preprocess_verilog(
        "`define A `B\n`define B `C\n`define C 1\n`A\n", limited);
    require(has_diagnostic(nested, "MacroExpansionDepthExceeded"),
            "macro nesting beyond configured depth must be diagnosed");
}

void test_slang_preprocessor_differential() {
    const std::string source =
        "`define WIDTH 8\n"
        "`define ADD(a,b) ((a)+(b))\n"
        "`ifdef WIDTH\n"
        "logic [`WIDTH-1:0] value = `ADD(2, 3);\n"
        "`else\n"
        "logic wrong;\n"
        "`endif\n";
    const auto native = preprocess_verilog(source);
    require(!native.has_errors(), "native differential source must preprocess cleanly");
    const auto native_tokens = active_tokens(native.text);
    const auto slang_tokens = active_tokens(source);
    require(native_tokens == slang_tokens,
            "native output token stream differs from Slang preprocessing");
}

} // namespace

int main() {
    try {
        test_macro_expansion_and_lexical_boundaries();
        test_macro_arguments_diagnostics_and_origins();
        test_macro_token_pasting();
        test_macro_stringification();
        test_macro_definition_line_continuations();
        test_conditionals_and_malformed_directives();
        test_includes_and_file_source_mapping();
        test_directive_metadata_empty_and_line_endings();
        test_large_input_and_expansion_limit();
        test_slang_preprocessor_differential();
    }
    catch (const std::exception& error) {
        std::cerr << "EDA Verilog preprocessor test failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
