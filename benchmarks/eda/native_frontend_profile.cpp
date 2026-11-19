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
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

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

struct LifecycleTiming {
    TimingSummary construction;
    TimingSummary destruction;
};

TimingSummary summarize(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    return {samples.front(), percentile(samples, 0.50),
            percentile(samples, 0.95),
            percentile(samples, 0.75) - percentile(samples, 0.25)};
}

template <typename Construct, typename Destroy>
LifecycleTiming measure_lifecycle(std::size_t warmups, std::size_t repetitions,
                                  Construct&& construct, Destroy&& destroy) {
    std::vector<double> construction;
    std::vector<double> destruction;
    construction.reserve(repetitions);
    destruction.reserve(repetitions);
    for (std::size_t iteration = 0; iteration < warmups + repetitions; ++iteration) {
        const auto start = std::chrono::steady_clock::now();
        construct();
        const auto constructed = std::chrono::steady_clock::now();
        destroy();
        const auto destroyed = std::chrono::steady_clock::now();
        if (iteration >= warmups) {
            construction.push_back(
                std::chrono::duration<double>(constructed - start).count());
            destruction.push_back(
                std::chrono::duration<double>(destroyed - constructed).count());
        }
    }
    return {summarize(std::move(construction)),
            summarize(std::move(destruction))};
}

void emit_phase_result(std::string_view phase, std::string_view workload,
                       std::string_view hash, std::size_t source_bytes,
                       std::size_t repetitions, const LifecycleTiming& timing,
                       std::size_t expression_type_count = 0,
                       std::size_t net_count = 0,
                       std::size_t instance_count = 0) {
    std::cout << std::setprecision(12)
              << "{\"schema_version\":1,\"phase\":\"" << json_escape(phase)
              << "\",\"workload\":\"" << json_escape(workload)
              << "\",\"content_hash_fnv1a64\":\"" << hash
              << "\",\"source_bytes\":" << source_bytes
              << ",\"warmups\":" << kWarmups
              << ",\"repetitions\":" << repetitions
              << ",\"construction_median_seconds\":"
              << timing.construction.median_seconds
              << ",\"construction_p95_seconds\":"
              << timing.construction.p95_seconds
              << ",\"construction_iqr_seconds\":"
              << timing.construction.iqr_seconds
              << ",\"destruction_median_seconds\":"
              << timing.destruction.median_seconds
              << ",\"destruction_p95_seconds\":"
              << timing.destruction.p95_seconds
              << ",\"destruction_iqr_seconds\":"
              << timing.destruction.iqr_seconds
              << ",\"expression_type_count\":" << expression_type_count
              << ",\"net_count\":" << net_count
              << ",\"instance_count\":" << instance_count << "}\n";
}

void profile_phase(const std::string& phase, const std::string& source,
                   const std::string& top, VerilogLexerBackend backend,
                   const std::string& workload, const std::string& hash,
                   std::size_t repetitions) {
    if (phase == "parser") {
        std::optional<VerilogSyntaxParseResult> result;
        const auto timing = measure_lifecycle(kWarmups, repetitions, [&] {
            result.emplace(parse_verilog_syntax(
                source, {}, {.language_version = VerilogLanguageVersion::SystemVerilog2017,
                             .lexer_backend = backend},
                "<frontend-phase-profile>"));
            if (result->has_errors()) throw std::runtime_error("syntax failed");
        }, [&] { result.reset(); });
        emit_phase_result(phase, workload, hash, source.size(), repetitions, timing);
        return;
    }

    auto baseline = run_pipeline(source, top, backend, false);
    if (phase == "type") {
        std::optional<VerilogTypeAnalysisResult> result;
        const auto timing = measure_lifecycle(kWarmups, repetitions, [&] {
            result.emplace(analyze_verilog_expression_types(
                baseline.parsed.syntax, baseline.elaborated.semantic));
            if (result->has_errors()) throw std::runtime_error("type analysis failed");
        }, [&] { result.reset(); });
        const auto count = analyze_verilog_expression_types(
            baseline.parsed.syntax, baseline.elaborated.semantic).expression_types.size();
        emit_phase_result(phase, workload, hash, source.size(), repetitions,
                          timing, count);
        return;
    }
    if (phase == "netlist") {
        std::optional<VerilogNativeLoweringResult> result;
        const auto timing = measure_lifecycle(kWarmups, repetitions, [&] {
            result.emplace(lower_verilog_native(
                baseline.parsed.syntax, baseline.elaborated, baseline.types,
                baseline.eligibility, top));
            if (!result->succeeded()) {
                std::string message = "lowering failed";
                if (!result->diagnostics.empty())
                    message += ": " + result->diagnostics.front().code + ": " +
                               result->diagnostics.front().message;
                throw std::runtime_error(message);
            }
        }, [&] { result.reset(); });
        auto check = lower_verilog_native(
            baseline.parsed.syntax, baseline.elaborated, baseline.types,
            baseline.eligibility, top);
        emit_phase_result(phase, workload, hash, source.size(), repetitions,
                          timing, 0, check.module->get_nets().size(),
                          check.module->get_instances().size());
        return;
    }
    throw std::runtime_error("phase must be parser, type, or netlist");
}
} // namespace

int main(int argc, char** argv) {
    if (argc < 3 || argc > 7) {
        std::cerr << "usage: labneura_native_frontend_profile SOURCE TOP [BACKEND] "
                     "[WORKLOAD] [PHASE] [REPETITIONS]\n";
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
        if (argc >= 6) {
            const auto repetitions = argc >= 7
                ? static_cast<std::size_t>(std::stoull(argv[6]))
                : kRepetitions;
            if (repetitions == 0)
                throw std::runtime_error("repetitions must be positive");
            profile_phase(argv[5], source, top, backend, workload, hash,
                          repetitions);
            return 0;
        }

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
