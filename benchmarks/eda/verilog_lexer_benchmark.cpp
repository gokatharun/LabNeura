#include "labneura/eda/verilog_lexer.h"
#include "labneura/eda/verilog_byte_classifier.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#if defined(__APPLE__)
#    include <sys/sysctl.h>
#endif

namespace {

namespace byte_classifier = labneura::eda::detail::verilog_lexer;

struct Input {
    std::string name;
    std::shared_ptr<const labneura::eda::VerilogSourceBuffer> source;
    size_t tokens = 0;
};

struct Measurements {
    double median_seconds = 0;
    double iqr_seconds = 0;
    double p95_seconds = 0;
    uint64_t checksum = 0;
    size_t tokens = 0;
    size_t retained_bytes = 0;
};

double percentile(const std::vector<double>& sorted, double fraction) {
    const size_t index = static_cast<size_t>(std::ceil(fraction * sorted.size())) - 1;
    return sorted[std::min(index, sorted.size() - 1)];
}

template <typename RunPass>
Measurements measure(RunPass&& run_pass) {
    constexpr size_t warmups = 2;
    constexpr size_t repetitions = 9;
    Measurements result;
    std::vector<double> durations;
    durations.reserve(repetitions);
    for (size_t iteration = 0; iteration < warmups + repetitions; ++iteration) {
        const auto start = std::chrono::steady_clock::now();
        const Measurements pass = run_pass();
        const auto stop = std::chrono::steady_clock::now();
        if (iteration >= warmups) {
            durations.push_back(std::chrono::duration<double>(stop - start).count());
            result.checksum = pass.checksum;
            result.tokens = pass.tokens;
            result.retained_bytes = pass.retained_bytes;
        }
    }
    std::sort(durations.begin(), durations.end());
    result.median_seconds = percentile(durations, 0.50);
    result.iqr_seconds = percentile(durations, 0.75) - percentile(durations, 0.25);
    result.p95_seconds = percentile(durations, 0.95);
    return result;
}

const char* backend_name(labneura::eda::VerilogLexerBackend backend) {
    switch (backend) {
        case labneura::eda::VerilogLexerBackend::Neon:
            return "neon";
        case labneura::eda::VerilogLexerBackend::Avx2:
            return "avx2";
        default:
            return "scalar";
    }
}

Measurements measure_classification(
    const std::vector<Input>& inputs, labneura::eda::VerilogLexerBackend backend) {
    return measure([&] {
        Measurements pass;
        for (const auto& input : inputs) {
            const auto text = input.source->text();
            size_t offset = 0;
            while (offset < text.size()) {
                const size_t size =
                    std::min(byte_classifier::classification_block_width, text.size() - offset);
                const auto* bytes =
                    reinterpret_cast<const unsigned char*>(text.data() + offset);
                const auto block = byte_classifier::classify_block(bytes, size, backend);
                pass.checksum = (pass.checksum * 1315423911u) ^
                                (static_cast<uint64_t>(block.any_low) << 8) ^
                                block.any_high ^ block.all_low ^ block.all_high;
                offset += size;
            }
        }
        return pass;
    });
}

Measurements measure_lexer(const std::vector<Input>& inputs,
                           labneura::eda::VerilogLexerBackend backend) {
    return measure([&] {
        Measurements pass;
        const labneura::eda::VerilogLexerOptions options{
            labneura::eda::VerilogLanguageVersion::SystemVerilog2017, backend};
        for (const auto& input : inputs) {
            auto result = labneura::eda::lex_verilog(input.source, options);
            pass.tokens += result.tokens.size() - 1;
            pass.retained_bytes +=
                result.tokens.capacity() * sizeof(labneura::eda::VerilogToken) +
                result.errors.capacity() * sizeof(labneura::eda::VerilogLexError);
            if (!result.errors.empty()) {
                std::cerr << "lexical errors in benchmark source: " << input.name << '\n';
                std::exit(1);
            }
        }
        return pass;
    });
}

void print_measurements(std::string_view category,
                        labneura::eda::VerilogLexerBackend backend, size_t input_bytes,
                        size_t corpus_tokens, const Measurements& results,
                        std::string_view input_name = {}) {
    std::cout << "case=" << category << " backend=" << backend_name(backend)
              << " input_bytes=" << input_bytes << " corpus_tokens=" << corpus_tokens
              << (input_name.empty() ? "" : " input=") << input_name
              << " median_seconds=" << results.median_seconds
              << " iqr_seconds=" << results.iqr_seconds
              << " p95_seconds=" << results.p95_seconds
              << " median_MB_per_second="
              << input_bytes / results.median_seconds / 1'000'000.0;
    if (category == "lexer_end_to_end") {
        std::cout << " tokens_per_pass=" << results.tokens
                  << " median_tokens_per_second="
                  << results.tokens / results.median_seconds
                  << " retained_token_and_error_capacity_bytes=" << results.retained_bytes;
    }
    else {
        std::cout << " classification_only=true checksum=" << results.checksum;
    }
    std::cout << '\n';
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: labneura_lexer_benchmark <verilog-source> [more sources...]\n";
        return 2;
    }

    std::vector<Input> inputs;
    size_t input_bytes = 0;
    for (int i = 1; i < argc; ++i) {
        std::ifstream file(argv[i], std::ios::binary);
        if (!file) {
            std::cerr << "could not read input source: " << argv[i] << '\n';
            return 2;
        }
        std::string source{std::istreambuf_iterator<char>(file),
                           std::istreambuf_iterator<char>()};
        input_bytes += source.size();
        inputs.push_back({argv[i], std::make_shared<const labneura::eda::VerilogSourceBuffer>(
                                       std::move(source), argv[i])});
    }

    size_t corpus_tokens = 0;
    for (auto& input : inputs) {
        const auto baseline = labneura::eda::lex_verilog(
            input.source, {labneura::eda::VerilogLanguageVersion::SystemVerilog2017,
                           labneura::eda::VerilogLexerBackend::Scalar});
        if (!baseline.errors.empty()) {
            std::cerr << "lexical errors in benchmark source: " << input.name << '\n';
            return 1;
        }
        input.tokens = baseline.tokens.size() - 1;
        corpus_tokens += input.tokens;
    }

    std::cout << "sources=" << inputs.size() << " input_bytes=" << input_bytes
              << " corpus_tokens=" << corpus_tokens
              << " warmups=2 repetitions=9 hardware_threads="
              << std::thread::hardware_concurrency() << '\n';
    const std::vector backends = {labneura::eda::VerilogLexerBackend::Scalar,
                                  labneura::eda::VerilogLexerBackend::Neon,
                                  labneura::eda::VerilogLexerBackend::Avx2};
    for (const auto backend : backends) {
        if ((backend == labneura::eda::VerilogLexerBackend::Neon &&
             !byte_classifier::neon_available()) ||
            (backend == labneura::eda::VerilogLexerBackend::Avx2 &&
             !byte_classifier::avx2_available())) {
            std::cout << "backend=" << backend_name(backend)
                      << " status=unavailable_at_runtime\n";
            continue;
        }
        print_measurements("classification_only", backend, input_bytes, corpus_tokens,
                           measure_classification(inputs, backend));
        print_measurements("lexer_end_to_end", backend, input_bytes, corpus_tokens,
                           measure_lexer(inputs, backend));
        for (const auto& input : inputs) {
            const std::vector<Input> single_input = {input};
            const size_t bytes = input.source->text().size();
            print_measurements("classification_only", backend, bytes, input.tokens,
                               measure_classification(single_input, backend), input.name);
            print_measurements("lexer_end_to_end", backend, bytes, input.tokens,
                               measure_lexer(single_input, backend), input.name);
        }
    }

    std::string cpu_model = "unknown";
#if defined(__APPLE__)
    char cpu_model_buffer[256] = {};
    size_t cpu_model_size = sizeof(cpu_model_buffer);
    if (sysctlbyname("machdep.cpu.brand_string", cpu_model_buffer, &cpu_model_size, nullptr, 0) ==
        0) {
        cpu_model = cpu_model_buffer;
    }
#endif

#if defined(__aarch64__) || defined(__arm64__)
    constexpr const char* architecture = "arm64";
#elif defined(__x86_64__) || defined(_M_X64)
    constexpr const char* architecture = "x86_64";
#else
    constexpr const char* architecture = "unknown";
#endif

    std::cout << "architecture=" << architecture << " cpu_model=" << cpu_model << '\n'
#if defined(__APPLE__)
              << "os=macOS\n"
#elif defined(__linux__)
              << "os=Linux\n"
#elif defined(_WIN32)
              << "os=Windows\n"
#endif
#ifdef LABNEURA_LEXER_BUILD_TYPE
              << "build_type=" << LABNEURA_LEXER_BUILD_TYPE << '\n'
#endif
#if defined(__clang__)
              << "compiler=clang " << __clang_version__ << '\n'
#elif defined(__GNUC__)
              << "compiler=gcc " << __VERSION__ << '\n'
#elif defined(_MSC_VER)
              << "compiler=msvc " << _MSC_VER << '\n'
#endif
        ;

    return 0;
}
