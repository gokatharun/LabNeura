#include "labneura/eda/verilog_preprocessor.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

struct Sample {
    double median = 0;
    double iqr = 0;
    size_t output_bytes = 0;
    size_t source_count = 0;
    size_t directive_count = 0;
    size_t macro_count = 0;
};

struct PassResult {
    size_t output_bytes = 0;
    size_t source_count = 0;
    size_t directive_count = 0;
    size_t macro_count = 0;
    bool has_errors = false;
};

class TemporaryDirectory {
public:
    TemporaryDirectory() {
        const auto ticks = std::chrono::steady_clock::now().time_since_epoch().count();
        path_ = std::filesystem::temp_directory_path() /
                ("labneura-preprocessor-benchmark-" + std::to_string(ticks));
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

double percentile(const std::vector<double>& sorted, double fraction) {
    const size_t index = static_cast<size_t>(std::ceil(fraction * sorted.size())) - 1;
    return sorted[std::min(index, sorted.size() - 1)];
}

template <typename Run>
Sample measure(Run&& run) {
    constexpr size_t warmups = 2;
    constexpr size_t repetitions = 9;
    std::vector<double> durations;
    durations.reserve(repetitions);
    Sample result;
    for (size_t i = 0; i < warmups + repetitions; ++i) {
        const auto start = std::chrono::steady_clock::now();
        const PassResult output = run();
        const auto stop = std::chrono::steady_clock::now();
        if (output.has_errors)
            throw std::runtime_error("preprocessor benchmark workload has diagnostics");
        if (i >= warmups) {
            durations.push_back(std::chrono::duration<double>(stop - start).count());
            result.output_bytes = output.output_bytes;
            result.source_count = output.source_count;
            result.directive_count = output.directive_count;
            result.macro_count = output.macro_count;
        }
    }
    std::sort(durations.begin(), durations.end());
    result.median = percentile(durations, 0.5);
    result.iqr = percentile(durations, 0.75) - percentile(durations, 0.25);
    return result;
}

template <typename Run>
void report(std::string_view name, size_t input_bytes, Run&& run) {
    const Sample sample = measure(std::forward<Run>(run));
    std::cout << "case=" << name << " input_bytes=" << input_bytes
              << " output_bytes=" << sample.output_bytes << " median_seconds=" << sample.median
              << " iqr_seconds=" << sample.iqr
              << " median_MB_per_second=" << input_bytes / sample.median / 1'000'000.0
              << " macros=" << sample.macro_count << " directives=" << sample.directive_count
              << " loaded_sources=" << sample.source_count << " warmups=2 repetitions=9\n";
}

PassResult summarize(const labneura::eda::VerilogPreprocessorResult& result) {
    return {result.text.size(), result.sources.size(), result.directives.size(),
            result.macros.size(), result.has_errors()};
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: labneura_preprocessor_benchmark <verilog-source> [more sources...]\n";
        return 2;
    }

    std::vector<std::filesystem::path> paths;
    size_t total_bytes = 0;
    labneura::eda::VerilogPreprocessorOptions options;
    for (int i = 1; i < argc; ++i) {
        const std::filesystem::path path(argv[i]);
        std::ifstream file(path, std::ios::binary);
        if (!file) {
            std::cerr << "could not read input source: " << path << '\n';
            return 2;
        }
        std::string contents{std::istreambuf_iterator<char>(file),
                             std::istreambuf_iterator<char>()};
        total_bytes += contents.size();
        paths.push_back(path);
        options.include_paths.push_back(path.parent_path());
    }

    report("representative_corpus", total_bytes, [&] {
        PassResult pass;
        for (const auto& path : paths) {
            auto result = labneura::eda::preprocess_verilog_file(path, options);
            const auto one = summarize(result);
            pass.output_bytes += one.output_bytes;
            pass.source_count += one.source_count;
            pass.directive_count += one.directive_count;
            pass.macro_count += one.macro_count;
            pass.has_errors = pass.has_errors || one.has_errors;
        }
        return pass;
    });

    std::string macro_source =
        "`define M7_WIDTH 32\n"
        "`define M7_ID(x) x\n"
        "`define M7_ADD(a,b) ((a)+(b))\n";
    for (size_t i = 0; i < 10'000; ++i) {
        macro_source += "wire [`M7_WIDTH-1:0] output_" + std::to_string(i) + " = `M7_ADD(" +
                        std::to_string(i) + ", `M7_ID(1));\n";
    }
    const size_t macro_bytes = macro_source.size();
    const auto macro_buffer = std::make_shared<const labneura::eda::VerilogSourceBuffer>(
        std::move(macro_source), "<macro-benchmark>");
    report("macro_expansion", macro_bytes, [&] {
        return summarize(labneura::eda::preprocess_verilog(macro_buffer));
    });

    TemporaryDirectory include_directory;
    const auto header_path = include_directory.path() / "shared.vh";
    const auto root_path = include_directory.path() / "include_workload.sv";
    constexpr std::string_view header_contents =
        "`define M7_INCLUDE_WIDTH 16\n"
        "wire [`M7_INCLUDE_WIDTH-1:0] included_signal;\n";
    constexpr size_t include_repetitions = 1'000;
    {
        std::ofstream header(header_path, std::ios::binary);
        header << header_contents;
        std::ofstream root(root_path, std::ios::binary);
        for (size_t i = 0; i < include_repetitions; ++i)
            root << "`include \"shared.vh\"\n";
    }
    std::ifstream root_file(root_path, std::ios::binary);
    std::string include_source{std::istreambuf_iterator<char>(root_file),
                               std::istreambuf_iterator<char>()};
    const size_t include_bytes =
        include_source.size() + include_repetitions * header_contents.size();
    labneura::eda::VerilogPreprocessorOptions include_options;
    include_options.include_paths.push_back(include_directory.path());
    report("include_workload", include_bytes, [&] {
        return summarize(
            labneura::eda::preprocess_verilog_file(root_path, include_options));
    });

    std::cout << "architecture="
#if defined(__aarch64__) || defined(__arm64__)
              << "arm64"
#elif defined(__x86_64__) || defined(_M_X64)
              << "x86_64"
#else
              << "unknown"
#endif
              << " hardware_threads=" << std::thread::hardware_concurrency()
#if defined(LABNEURA_LEXER_BUILD_TYPE)
              << " build_type=" << LABNEURA_LEXER_BUILD_TYPE
#endif
#if defined(__clang__)
              << " compiler=clang " << __clang_version__
#elif defined(__GNUC__)
              << " compiler=gcc " << __VERSION__
#endif
              << '\n';
    return 0;
}
