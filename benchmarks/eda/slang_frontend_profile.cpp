#include "labneura/eda/netlist.h"

#include "slang/ast/Compilation.h"
#include "slang/ast/symbols/CompilationUnitSymbols.h"
#include "slang/ast/symbols/InstanceSymbols.h"
#include "slang/syntax/SyntaxTree.h"
#include "slang/text/SourceManager.h"
#include "slang/util/TimeTrace.h"
#include "slang/util/VersionInfo.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

volatile std::size_t result_sink = 0;

struct Options {
    std::filesystem::path input;
    std::filesystem::path trace;
    std::size_t warmups = 2;
    std::size_t repetitions = 9;
};

struct PassResult {
    std::size_t checksum = 0;
    std::size_t modules = 0;
    std::size_t processes = 0;
    std::size_t signals = 0;
    std::size_t ports = 0;
    std::size_t nets = 0;
    std::size_t instances = 0;
};

struct Measurement {
    double minimum_seconds = 0.0;
    double median_seconds = 0.0;
    double p95_seconds = 0.0;
    double iqr_seconds = 0.0;
    PassResult result;
};

double percentile(const std::vector<double>& sorted, double fraction) {
    const auto rank = static_cast<std::size_t>(
        std::ceil(fraction * static_cast<double>(sorted.size())));
    return sorted[std::min(std::max<std::size_t>(rank, 1) - 1,
                           sorted.size() - 1)];
}

template <typename Function>
Measurement measure(const Options& options, Function&& function) {
    std::vector<double> samples;
    samples.reserve(options.repetitions);
    Measurement measurement;
    for (std::size_t iteration = 0;
         iteration < options.warmups + options.repetitions; ++iteration) {
        const auto start = Clock::now();
        auto pass = function();
        result_sink = result_sink ^ pass.checksum;
        const auto stop = Clock::now();
        if (iteration >= options.warmups) {
            samples.push_back(std::chrono::duration<double>(stop - start).count());
            measurement.result = pass;
        }
    }
    std::sort(samples.begin(), samples.end());
    measurement.minimum_seconds = samples.front();
    measurement.median_seconds = percentile(samples, 0.50);
    measurement.p95_seconds = percentile(samples, 0.95);
    measurement.iqr_seconds =
        percentile(samples, 0.75) - percentile(samples, 0.25);
    return measurement;
}

std::uint64_t content_hash(std::string_view source) {
    std::uint64_t hash = 14695981039346656037ULL;
    for (const auto character : source) {
        hash ^= static_cast<unsigned char>(character);
        hash *= 1099511628211ULL;
    }
    return hash;
}

std::string hex_hash(std::uint64_t value) {
    static constexpr char digits[] = "0123456789abcdef";
    std::string result(16, '0');
    for (std::size_t index = 0; index < result.size(); ++index) {
        result[result.size() - index - 1] = digits[value & 0xf];
        value >>= 4;
    }
    return result;
}

std::string json_escape(std::string_view value) {
    std::string result;
    result.reserve(value.size() + 2);
    for (const char character : value) {
        switch (character) {
            case '\\': result += "\\\\"; break;
            case '"': result += "\\\""; break;
            case '\n': result += "\\n"; break;
            case '\r': result += "\\r"; break;
            case '\t': result += "\\t"; break;
            default: result += character; break;
        }
    }
    return result;
}

std::string platform_name() {
#if defined(__APPLE__)
    return "macos";
#elif defined(__linux__)
    return "linux";
#else
    return "unknown";
#endif
}

std::string architecture_name() {
#if defined(__aarch64__) || defined(__arm64__)
    return "arm64";
#elif defined(__x86_64__) || defined(_M_X64)
    return "x86_64";
#else
    return "unknown";
#endif
}

std::string compiler_name() {
#if defined(__clang__)
    return "clang-" + std::to_string(__clang_major__) + "." +
           std::to_string(__clang_minor__);
#elif defined(__GNUC__)
    return "gcc-" + std::to_string(__GNUC__) + "." +
           std::to_string(__GNUC_MINOR__);
#else
    return "unknown";
#endif
}

std::optional<std::string> environment(std::string_view name) {
    const std::string owned(name);
    if (const auto* value = std::getenv(owned.c_str()))
        return std::string(value);
    return std::nullopt;
}

void count_semantic_members(const slang::ast::InstanceSymbol& top,
                            PassResult& result) {
    using slang::ast::SymbolKind;
    ++result.modules;
    for (const auto& member : top.body.members()) {
        switch (member.kind) {
            case SymbolKind::Port:
                ++result.ports;
                ++result.signals;
                break;
            case SymbolKind::Net:
            case SymbolKind::Variable:
                ++result.signals;
                break;
            case SymbolKind::ProceduralBlock:
            case SymbolKind::ContinuousAssign:
                ++result.processes;
                break;
            default:
                break;
        }
    }
}

PassResult run_syntax(std::string_view source) {
    slang::SourceManager source_manager;
    auto tree = slang::syntax::SyntaxTree::fromText(
        source, source_manager, "profile.sv", "profile.sv");
    std::size_t errors = 0;
    for (const auto& diagnostic : tree->diagnostics())
        errors += diagnostic.isError() ? 1 : 0;
    if (errors != 0)
        throw std::runtime_error("Slang syntax parsing produced errors");
    PassResult result;
    result.checksum = source.size() ^ tree->diagnostics().size();
    return result;
}

PassResult run_semantic(std::string_view source) {
    slang::SourceManager source_manager;
    auto tree = slang::syntax::SyntaxTree::fromText(
        source, source_manager, "profile.sv", "profile.sv");
    slang::ast::Compilation compilation;
    compilation.addSyntaxTree(tree);
    const auto& root = compilation.getRoot();
    const auto& diagnostics = compilation.getAllDiagnostics();
    for (const auto& diagnostic : diagnostics)
        if (diagnostic.isError())
            throw std::runtime_error("Slang semantic analysis produced errors");

    PassResult result;
    for (const auto* top : root.topInstances) {
        count_semantic_members(*top, result);
        // Walking members forces symbol/type materialization that can otherwise
        // remain lazy after compilation object construction.
        for (const auto& member : top->body.members())
            result.checksum ^= static_cast<std::size_t>(member.kind) +
                               member.name.size() * 1315423911u;
    }
    result.checksum ^= diagnostics.size();
    return result;
}

PassResult run_complete(const std::filesystem::path& input) {
    auto module = labneura::eda::parse_verilog(input.string());
    PassResult result;
    result.modules = 1 + module.get_child_modules().size();
    result.ports = module.get_port_count();
    result.nets = module.get_nets().size();
    result.instances = module.get_instance_count();
    result.checksum = module.get_name().size() ^ (result.ports << 8) ^
                      (result.nets << 16) ^ (result.instances << 24);
    return result;
}

void print_nullable_string(std::string_view name,
                           const std::optional<std::string>& value) {
    std::cout << ",\"" << name << "\":";
    if (value)
        std::cout << '"' << json_escape(*value) << '"';
    else
        std::cout << "null";
}

void print_result(std::string_view category, const Options& options,
                  std::string_view workload, std::string_view hash,
                  std::size_t source_size, const Measurement& measurement,
                  bool includes_file_io) {
    std::cout << "{\"schema_version\":1,\"frontend\":\"slang\""
              << ",\"category\":\"" << category << "\""
              << ",\"backend\":\"scalar\""
              << ",\"execution_backend\":\"cpu\""
              << ",\"workload\":\"" << json_escape(workload) << "\""
              << ",\"content_hash_fnv1a64\":\"" << hash << "\""
              << ",\"source_bytes\":" << source_size
              << ",\"token_count\":null"
              << ",\"warmups\":" << options.warmups
              << ",\"repetitions\":" << options.repetitions
              << ",\"minimum_seconds\":" << measurement.minimum_seconds
              << ",\"median_seconds\":" << measurement.median_seconds
              << ",\"p95_seconds\":" << measurement.p95_seconds
              << ",\"iqr_seconds\":" << measurement.iqr_seconds
              << ",\"cpu_seconds\":null,\"peak_rss_bytes\":null"
              << ",\"allocation_count\":null,\"allocation_bytes\":null"
              << ",\"module_count\":" << measurement.result.modules
              << ",\"process_count\":" << measurement.result.processes
              << ",\"signal_count\":" << measurement.result.signals
              << ",\"generated_port_count\":";
    if (category == "source_to_netlist")
        std::cout << measurement.result.ports;
    else
        std::cout << "null";
    std::cout
              << ",\"generated_net_count\":";
    if (category == "source_to_netlist")
        std::cout << measurement.result.nets;
    else
        std::cout << "null";
    std::cout << ",\"generated_instance_count\":";
    if (category == "source_to_netlist")
        std::cout << measurement.result.instances;
    else
        std::cout << "null";
    std::cout << ",\"correctness\":\"accepted\""
              << ",\"includes_file_io\":"
              << (includes_file_io ? "true" : "false")
              << ",\"os\":\"" << platform_name() << "\""
              << ",\"architecture\":\"" << architecture_name() << "\""
              << ",\"cpu_model\":null"
              << ",\"compiler\":\"" << compiler_name() << "\""
              << ",\"compiler_flags\":null"
              << ",\"build_configuration\":\"Release\""
              << ",\"slang_version\":\""
              << json_escape(slang::VersionInfo::getVersionString()) << "\"";
    print_nullable_string("git_commit", environment("LABNEURA_BENCH_GIT_SHA"));
    print_nullable_string("git_dirty", environment("LABNEURA_BENCH_GIT_DIRTY"));
    std::cout << "}\n";
}

Options parse_options(int argc, char** argv) {
    if (argc < 2)
        throw std::runtime_error(
            "usage: labneura_slang_frontend_profile <source.sv> "
            "[--warmups N] [--repetitions N] [--trace output.json]");
    Options result;
    result.input = argv[1];
    for (int index = 2; index < argc; ++index) {
        const std::string_view option(argv[index]);
        if ((option == "--warmups" || option == "--repetitions" ||
             option == "--trace") && index + 1 >= argc)
            throw std::runtime_error("missing value after " + std::string(option));
        if (option == "--warmups")
            result.warmups = std::stoull(argv[++index]);
        else if (option == "--repetitions")
            result.repetitions = std::stoull(argv[++index]);
        else if (option == "--trace")
            result.trace = argv[++index];
        else
            throw std::runtime_error("unknown option: " + std::string(option));
    }
    if (result.repetitions == 0)
        throw std::runtime_error("repetitions must be positive");
    return result;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse_options(argc, argv);
        std::ifstream input(options.input, std::ios::binary);
        if (!input)
            throw std::runtime_error("could not read input source");
        const std::string source{std::istreambuf_iterator<char>(input),
                                 std::istreambuf_iterator<char>()};
        const auto hash = hex_hash(content_hash(source));
        const auto workload = options.input.filename().string();

        // Correctness is established before timing; every measured pass still
        // rejects diagnostics so profiling cannot silently change the result.
        (void)run_syntax(source);
        const auto semantic_baseline = run_semantic(source);
        const auto complete_baseline = run_complete(options.input);
        if (semantic_baseline.modules != complete_baseline.modules)
            throw std::runtime_error(
                "semantic and complete Slang paths disagree on module count");

        print_result("syntax", options, workload, hash, source.size(),
                     measure(options, [&] { return run_syntax(source); }), false);
        print_result("semantic", options, workload, hash, source.size(),
                     measure(options, [&] { return run_semantic(source); }), false);
        // parse_verilog is intentionally measured through its existing public
        // path. It performs source loading internally, so this category reports
        // includes_file_io=true instead of pretending to be in-memory.
        print_result("source_to_netlist", options, workload, hash, source.size(),
                     measure(options, [&] { return run_complete(options.input); }), true);

        if (!options.trace.empty()) {
            slang::TimeTrace::initialize();
            (void)run_semantic(source);
            std::ofstream trace(options.trace, std::ios::binary);
            if (!trace)
                throw std::runtime_error("could not create Slang time trace");
            slang::TimeTrace::write(trace);
            slang::TimeTrace::destroy();
        }
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << "slang frontend profile failed: " << error.what() << '\n';
        if (slang::TimeTrace::isEnabled())
            slang::TimeTrace::destroy();
        return 1;
    }
}
