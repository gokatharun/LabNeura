#include "frontend_profile_common.h"

#include "labneura/eda/verilog_elaboration.h"
#include "labneura/eda/verilog_lexer.h"
#include "labneura/eda/verilog_native_lowering.h"
#include "labneura/eda/verilog_process_analysis.h"
#include "labneura/eda/verilog_semantic.h"
#include "labneura/eda/verilog_syntax.h"
#include "labneura/eda/verilog_type.h"

#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>

#ifndef LABNEURA_FRONTEND_PROFILE_BUILD_TYPE
#define LABNEURA_FRONTEND_PROFILE_BUILD_TYPE "unknown"
#endif

namespace {
using namespace labneura::eda;
using namespace labneura::eda::benchmark;

constexpr std::size_t kWarmups = 2;
constexpr std::size_t kRepetitions = 9;

struct NativePipeline {
    VerilogSyntaxParseResult parsed;
    VerilogElaborationResult elaborated;
    VerilogTypeAnalysisResult types;
    VerilogSynthesisEligibilityResult eligibility;
    VerilogNativeLoweringResult lowered;
};

NativePipeline run_pipeline(const std::string& source, const std::string& top,
                            VerilogLexerBackend backend, bool lower) {
    NativePipeline result;
    result.parsed = parse_verilog_syntax(
        source, {}, {.language_version = VerilogLanguageVersion::SystemVerilog2017,
                     .lexer_backend = backend},
        "<frontend-profile>");
    if (result.parsed.has_errors())
        throw std::runtime_error("native syntax parsing failed");
    result.elaborated = elaborate_verilog_design(
        result.parsed.syntax, VerilogElaborationOptions{.top_module = top});
    if (result.elaborated.has_errors())
        throw std::runtime_error("native elaboration failed");
    result.types = analyze_verilog_expression_types(result.parsed.syntax,
                                                     result.elaborated.semantic);
    if (result.types.has_errors())
        throw std::runtime_error("native type analysis failed");
    result.eligibility = analyze_verilog_synthesis_eligibility(
        result.parsed.syntax, result.elaborated.semantic, result.types);
    if (lower) {
        result.lowered = lower_verilog_native(
            result.parsed.syntax, result.elaborated, result.types,
            result.eligibility, top);
        if (!result.lowered.succeeded()) {
            std::string message = "native lowering failed";
            if (!result.lowered.diagnostics.empty())
                message += ": " + result.lowered.diagnostics.front().code + ": " +
                           result.lowered.diagnostics.front().message;
            throw std::runtime_error(message);
        }
    }
    return result;
}

const char* backend_name(VerilogLexerBackend backend) {
    switch (backend) {
        case VerilogLexerBackend::Scalar: return "scalar";
        case VerilogLexerBackend::Neon: return "neon";
        case VerilogLexerBackend::Avx2: return "avx2";
        default: return "auto";
    }
}

VerilogLexerBackend parse_backend(std::string_view value) {
    if (value == "scalar") return VerilogLexerBackend::Scalar;
    if (value == "neon") return VerilogLexerBackend::Neon;
    if (value == "avx2") return VerilogLexerBackend::Avx2;
    if (value == "auto") return VerilogLexerBackend::Auto;
    throw std::runtime_error("backend must be auto, scalar, neon, or avx2");
}

std::size_t process_count(const VerilogCompilationUnit& syntax) {
    std::size_t count = 0;
    for (const auto& module : syntax.modules)
        count += module.assignments.size() + module.procedural_blocks.size();
    return count;
}

std::size_t signal_count(const VerilogCompilationUnit& syntax) {
    std::size_t count = 0;
    for (const auto& module : syntax.modules)
        count += module.ports.size() + module.declarations.size();
    return count;
}
} // namespace

int main(int argc, char** argv) {
    if (argc < 3 || argc > 5) {
        std::cerr << "usage: labneura_native_frontend_profile SOURCE TOP [BACKEND] [WORKLOAD]\n";
        return 2;
    }
    try {
        std::ifstream input(argv[1], std::ios::binary);
        if (!input)
            throw std::runtime_error("could not read source");
        const std::string source{std::istreambuf_iterator<char>(input),
                                 std::istreambuf_iterator<char>()};
        const std::string top = argv[2];
        const auto backend = parse_backend(argc >= 4 ? argv[3] : "auto");
        const std::string workload = argc >= 5 ? argv[4] : argv[1];
        const auto hash = fnv1a64(source);

        const auto baseline = run_pipeline(source, top, backend, true);
        const auto& syntax = baseline.parsed.syntax;
        const auto tokens = lex_verilog(
            std::make_shared<const VerilogSourceBuffer>(source, argv[1]),
            {VerilogLanguageVersion::SystemVerilog2017, backend});
        if (!tokens.errors.empty())
            throw std::runtime_error("native lexer failed during metadata collection");
        const auto modules = syntax.modules.size();
        const auto processes = process_count(syntax);
        const auto signals = signal_count(syntax);
        const auto nets = baseline.lowered.module->get_nets().size();
        const auto instances = baseline.lowered.module->get_instances().size();
        const auto& index =
            baseline.elaborated.semantic.reference_index_statistics;

        const auto syntax_timing = measure(kWarmups, kRepetitions, [&] {
            auto parsed = parse_verilog_syntax(
                source, {}, {.language_version = VerilogLanguageVersion::SystemVerilog2017,
                             .lexer_backend = backend},
                "<frontend-profile>");
            if (parsed.has_errors()) throw std::runtime_error("syntax failed");
        });
        emit_result("syntax", "native", backend_name(backend), workload, hash,
                    source.size(), tokens.tokens.size() - 1, modules, processes,
                    signals, std::nullopt, std::nullopt, std::nullopt,
                    kWarmups, kRepetitions,
                    syntax_timing, "accepted", LABNEURA_FRONTEND_PROFILE_BUILD_TYPE);

        const auto semantic_timing = measure(kWarmups, kRepetitions, [&] {
            (void)run_pipeline(source, top, backend, false);
        });
        emit_result("semantic", "native", backend_name(backend), workload, hash,
                    source.size(), tokens.tokens.size() - 1, modules, processes,
                    signals, std::nullopt, std::nullopt, std::nullopt,
                    kWarmups, kRepetitions,
                    semantic_timing, "accepted", LABNEURA_FRONTEND_PROFILE_BUILD_TYPE,
                    "cpu", index.reference_count, 1,
                    index.construction_nanoseconds, index.estimated_storage_bytes,
                    index.duplicate_location_count);

        const auto complete_timing = measure(kWarmups, kRepetitions, [&] {
            (void)run_pipeline(source, top, backend, true);
        });
        emit_result("source_to_netlist", "native", backend_name(backend), workload,
                    hash, source.size(), tokens.tokens.size() - 1, modules,
                    processes, signals, baseline.lowered.module->get_ports().size(),
                    nets, instances, kWarmups, kRepetitions,
                    complete_timing, "lowered", LABNEURA_FRONTEND_PROFILE_BUILD_TYPE,
                    "cpu", index.reference_count, 1,
                    index.construction_nanoseconds, index.estimated_storage_bytes,
                    index.duplicate_location_count);
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << "native frontend profile failed: " << error.what() << '\n';
        return 1;
    }
}
