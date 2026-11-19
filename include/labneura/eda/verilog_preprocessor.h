#pragma once

#include "labneura/eda/verilog_lexer.h"

#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace labneura::eda {

enum class VerilogPreprocessorSeverity : uint8_t { Note, Warning, Error };

enum class VerilogPreprocessedOriginKind : uint8_t { Source, MacroExpansion };

struct VerilogMacroExpansionFrame {
    std::string macro_name;
    uint32_t invocation_source_id = 0;
    VerilogSourceLocation invocation_location;
    uint32_t definition_source_id = 0;
    VerilogSourceLocation definition_location;
};

struct VerilogPreprocessedOrigin {
    std::shared_ptr<const VerilogSourceBuffer> source;
    size_t byte_offset = 0;
    VerilogSourceLocation location;
    VerilogPreprocessedOriginKind kind = VerilogPreprocessedOriginKind::Source;
    std::vector<VerilogMacroExpansionFrame> expansion_stack;
};

struct VerilogSourceMapSegment {
    size_t output_offset = 0;
    size_t length = 0;
    std::shared_ptr<const VerilogSourceBuffer> source;
    size_t source_offset = 0;
    VerilogPreprocessedOriginKind kind = VerilogPreprocessedOriginKind::Source;
    std::vector<VerilogMacroExpansionFrame> expansion_stack;
};

struct VerilogPreprocessorDiagnostic {
    VerilogPreprocessorSeverity severity = VerilogPreprocessorSeverity::Error;
    std::string code;
    std::string message;
    std::shared_ptr<const VerilogSourceBuffer> source;
    size_t byte_offset = 0;
    VerilogSourceLocation location;
    std::vector<std::string> include_stack;
    std::vector<VerilogMacroExpansionFrame> expansion_stack;
};

enum class VerilogPreprocessorDirectiveKind : uint8_t {
    Timescale,
    DefaultNettype,
    Deferred
};

struct VerilogPreprocessorDirective {
    VerilogPreprocessorDirectiveKind kind = VerilogPreprocessorDirectiveKind::Deferred;
    std::string name;
    std::string value;
    std::shared_ptr<const VerilogSourceBuffer> source;
    size_t byte_offset = 0;
    VerilogSourceLocation location;
};

struct VerilogMacroInfo {
    std::string name;
    bool function_like = false;
    std::vector<std::string> parameters;
    std::string replacement;
    std::shared_ptr<const VerilogSourceBuffer> definition_source;
    size_t definition_offset = 0;
};

struct VerilogPreprocessorOptions {
    std::vector<std::filesystem::path> include_paths;
    std::map<std::string, std::string, std::less<>> predefined_macros;
    VerilogLanguageVersion language_version = VerilogLanguageVersion::SystemVerilog2017;
    size_t max_macro_expansion_depth = 128;
    size_t max_include_depth = 128;
};

struct VerilogPreprocessorResult {
    std::string text;
    std::vector<VerilogSourceMapSegment> source_map;
    std::vector<VerilogPreprocessorDiagnostic> diagnostics;
    std::vector<VerilogPreprocessorDirective> directives;
    std::map<std::string, VerilogMacroInfo, std::less<>> macros;
    std::vector<std::shared_ptr<const VerilogSourceBuffer>> sources;

    std::optional<VerilogPreprocessedOrigin> origin_at(size_t output_offset) const;
    bool has_errors() const noexcept;
};

VerilogPreprocessorResult preprocess_verilog(
    std::shared_ptr<const VerilogSourceBuffer> source,
    VerilogPreprocessorOptions options = {});
VerilogPreprocessorResult preprocess_verilog(
    std::string source, VerilogPreprocessorOptions options = {},
    std::string source_name = {});
VerilogPreprocessorResult preprocess_verilog_file(
    const std::filesystem::path& path, VerilogPreprocessorOptions options = {});

} // namespace labneura::eda
