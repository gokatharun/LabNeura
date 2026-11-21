#pragma once

#include "labneura/eda/verilog_elaboration.h"
#include "labneura/eda/verilog_process_analysis.h"
#include "labneura/eda/verilog_syntax.h"

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace labneura::eda::detail {

struct NativeVerilogSourceUnit {
    std::string text;
    std::string name;
};

// Owns native frontend products in dependency order. Syntax owns the AST and
// retains source text through shared VerilogSourceBuffer references in source
// locations. Elaboration points into that syntax and owns the semantic model;
// type and eligibility results point into the syntax and semantic model. Keep
// these members in this order so destruction runs eligibility, types,
// elaboration, then syntax. The non-copyable context must outlive all borrowed
// syntax and semantic identities exposed by its results.
struct NativeVerilogProjectContext {
    NativeVerilogProjectContext() = default;
    NativeVerilogProjectContext(const NativeVerilogProjectContext&) = delete;
    NativeVerilogProjectContext& operator=(const NativeVerilogProjectContext&) = delete;

    VerilogSyntaxParseResult parsed;
    std::optional<VerilogElaborationResult> elaboration;
    std::optional<VerilogTypeAnalysisResult> types;
    std::optional<VerilogSynthesisEligibilityResult> eligibility;
    std::vector<std::string> source_files;

    // True means syntax, elaboration, and type analysis completed without
    // errors and eligibility analysis ran; it does not mean synthesis-eligible.
    bool ready() const noexcept;
};

std::shared_ptr<NativeVerilogProjectContext> build_native_verilog_project_context(
    std::vector<NativeVerilogSourceUnit> sources,
    VerilogPreprocessorOptions preprocessor_options = {},
    VerilogSyntaxParserOptions parser_options = {},
    VerilogElaborationOptions elaboration_options = {});

std::shared_ptr<NativeVerilogProjectContext>
build_native_verilog_project_context_from_files(
    const std::vector<std::filesystem::path>& paths,
    VerilogPreprocessorOptions preprocessor_options = {},
    VerilogSyntaxParserOptions parser_options = {},
    VerilogElaborationOptions elaboration_options = {});

} // namespace labneura::eda::detail
