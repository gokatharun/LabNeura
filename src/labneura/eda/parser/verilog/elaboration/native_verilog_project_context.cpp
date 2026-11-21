#include "native_verilog_project_context.h"

#include "labneura/eda/verilog_type.h"

#include <algorithm>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <utility>

namespace labneura::eda::detail {
namespace {

void append_syntax(VerilogSyntaxParseResult& destination,
                   VerilogSyntaxParseResult source) {
    if (!destination.syntax.location.source && source.syntax.location.source)
        destination.syntax.location = source.syntax.location;
    destination.diagnostics.insert(
        destination.diagnostics.end(),
        std::make_move_iterator(source.diagnostics.begin()),
        std::make_move_iterator(source.diagnostics.end()));
    destination.syntax.modules.insert(
        destination.syntax.modules.end(),
        std::make_move_iterator(source.syntax.modules.begin()),
        std::make_move_iterator(source.syntax.modules.end()));
    destination.syntax.interfaces.insert(
        destination.syntax.interfaces.end(),
        std::make_move_iterator(source.syntax.interfaces.begin()),
        std::make_move_iterator(source.syntax.interfaces.end()));
    destination.syntax.packages.insert(
        destination.syntax.packages.end(),
        std::make_move_iterator(source.syntax.packages.begin()),
        std::make_move_iterator(source.syntax.packages.end()));
}

} // namespace

bool NativeVerilogProjectContext::ready() const noexcept {
    return !parsed.has_errors() && elaboration.has_value() &&
           !elaboration->has_errors() && types.has_value() &&
           !types->has_errors() && eligibility.has_value();
}

std::shared_ptr<NativeVerilogProjectContext> build_native_verilog_project_context(
    std::vector<NativeVerilogSourceUnit> sources,
    VerilogPreprocessorOptions preprocessor_options,
    VerilogSyntaxParserOptions parser_options,
    VerilogElaborationOptions elaboration_options) {
    auto context = std::make_shared<NativeVerilogProjectContext>();
    context->source_files.reserve(sources.size());
    for (auto& source : sources) {
        context->source_files.push_back(source.name);
        append_syntax(
            context->parsed,
            parse_verilog_syntax(std::move(source.text), preprocessor_options,
                                 parser_options, source.name));
    }
    if (context->parsed.has_errors())
        return context;

    context->elaboration.emplace(elaborate_verilog_design(
        context->parsed.syntax, std::move(elaboration_options)));
    if (context->elaboration->has_errors())
        return context;

    context->types.emplace(analyze_verilog_expression_types(
        context->parsed.syntax, context->elaboration->semantic));
    if (context->types->has_errors())
        return context;

    context->eligibility.emplace(analyze_verilog_synthesis_eligibility(
        context->parsed.syntax, context->elaboration->semantic,
        *context->types));
    return context;
}

std::shared_ptr<NativeVerilogProjectContext>
build_native_verilog_project_context_from_files(
    const std::vector<std::filesystem::path>& paths,
    VerilogPreprocessorOptions preprocessor_options,
    VerilogSyntaxParserOptions parser_options,
    VerilogElaborationOptions elaboration_options) {
    std::vector<NativeVerilogSourceUnit> sources;
    sources.reserve(paths.size());
    for (const auto& path : paths) {
        std::ifstream input(path, std::ios::binary);
        if (!input)
            throw std::runtime_error("could not open Verilog file: " + path.string());
        sources.push_back({std::string(std::istreambuf_iterator<char>(input), {}),
                           path.string()});
        const auto directory = path.has_parent_path() ? path.parent_path()
                                                       : std::filesystem::path(".");
        if (std::find(preprocessor_options.include_paths.begin(),
                      preprocessor_options.include_paths.end(), directory) ==
            preprocessor_options.include_paths.end())
            preprocessor_options.include_paths.push_back(directory);
    }
    return build_native_verilog_project_context(
        std::move(sources), std::move(preprocessor_options), parser_options,
        std::move(elaboration_options));
}

} // namespace labneura::eda::detail
