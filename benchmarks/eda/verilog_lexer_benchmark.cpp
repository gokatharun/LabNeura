#include "labneura/eda/verilog_lexer.h"
#include "labneura/eda/verilog_byte_classifier.h"

#if defined(LABNEURA_ENABLE_SLANG_LEXER_BENCHMARK)
#    include <slang/diagnostics/Diagnostics.h>
#    include <slang/parsing/Lexer.h>
#    include <slang/text/SourceManager.h>
#    include <slang/util/BumpAllocator.h>
#    include <slang/util/VersionInfo.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <stdexcept>
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

#if defined(LABNEURA_ENABLE_SLANG_LEXER_BENCHMARK)
uint64_t consume_spelling(uint64_t checksum, std::string_view spelling) {
    for (const unsigned char ch : spelling) {
        checksum ^= ch;
        checksum *= 1099511628211ULL;
    }
    return checksum;
}
#endif

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
#if defined(LABNEURA_ENABLE_SLANG_LEXER_BENCHMARK)
            for (const auto& token : result.tokens) {
                pass.checksum ^= static_cast<uint64_t>(token.kind) << 56;
                pass.checksum ^= token.byte_offset ^ (token.length << 17);
                pass.checksum ^= (token.line << 7) ^ token.column;
                pass.checksum = consume_spelling(pass.checksum,
                                                 result.spelling(token));
            }
#endif
        }
        return pass;
    });
}

#if defined(LABNEURA_ENABLE_SLANG_LEXER_BENCHMARK)
struct SlangInput {
    slang::SourceManager source_manager;
    slang::SourceBuffer buffer;

    explicit SlangInput(const Input& input) :
        buffer(source_manager.assignText(input.name,
                                         std::string(input.source->text()))) {}
};

Measurements measure_slang_lexer(const std::vector<Input>& inputs) {
    std::vector<std::unique_ptr<SlangInput>> slang_inputs;
    slang_inputs.reserve(inputs.size());
    for (const auto& input : inputs)
        slang_inputs.push_back(std::make_unique<SlangInput>(input));

    return measure([&] {
        Measurements pass;
        for (const auto& input : slang_inputs) {
            slang::BumpAllocator allocator;
            slang::Diagnostics diagnostics;
            slang::parsing::Lexer lexer(input->buffer, allocator, diagnostics,
                                        input->source_manager);
            while (true) {
                const auto token = lexer.lex(
                    slang::parsing::KeywordVersion::v1800_2017);
                pass.checksum ^= static_cast<uint64_t>(token.kind) << 56;
                pass.checksum ^= token.location().offset();
                pass.checksum ^=
                    static_cast<uint64_t>(input->source_manager.getLineNumber(
                        token.location())) << 7;
                pass.checksum ^=
                    input->source_manager.getColumnNumber(token.location());
                pass.checksum = consume_spelling(pass.checksum, token.rawText());
                if (token.kind == slang::parsing::TokenKind::EndOfFile)
                    break;
                ++pass.tokens;
            }
            if (!diagnostics.empty()) {
                std::cerr << "Slang lexical diagnostics in benchmark source\n";
                std::exit(1);
            }
        }
        return pass;
    });
}

void verify_slang_parity(const Input& input) {
    const auto native = labneura::eda::lex_verilog(
        input.source,
        {labneura::eda::VerilogLanguageVersion::SystemVerilog2017,
         labneura::eda::VerilogLexerBackend::Scalar});
    if (!native.errors.empty())
        throw std::runtime_error("native lexer diagnostics prevent comparison");

    SlangInput slang_input(input);
    slang::BumpAllocator allocator;
    slang::Diagnostics diagnostics;
    slang::parsing::Lexer lexer(slang_input.buffer, allocator, diagnostics,
                                slang_input.source_manager);
    std::vector<slang::parsing::Token> slang_tokens;
    while (true) {
        auto token = lexer.lex(slang::parsing::KeywordVersion::v1800_2017);
        if (token.kind == slang::parsing::TokenKind::EndOfFile)
            break;
        slang_tokens.push_back(token);
    }
    if (!diagnostics.empty())
        throw std::runtime_error("Slang diagnostics prevent comparison");

    size_t slang_index = 0;
    for (const auto& token : native.tokens) {
        if (token.kind == labneura::eda::VerilogTokenKind::LineComment ||
            token.kind == labneura::eda::VerilogTokenKind::BlockComment)
            continue;
        if (token.kind == labneura::eda::VerilogTokenKind::EndOfFile)
            break;
        if (slang_index >= slang_tokens.size())
            throw std::runtime_error("native lexer has a token after Slang EOF");
        const auto spelling = native.spelling(token);
        size_t combined_size = 0;
        std::string combined_spelling;
        const size_t first = slang_index;
        while (combined_size < spelling.size()) {
            if (slang_index >= slang_tokens.size())
                throw std::runtime_error("Slang token group ended early");
            const auto& slang_token = slang_tokens[slang_index];
            if (slang_token.location().offset() != token.byte_offset + combined_size)
                throw std::runtime_error("native and Slang offsets differ");
            combined_spelling.append(slang_token.rawText());
            combined_size += slang_token.rawText().size();
            ++slang_index;
        }
        if (combined_spelling != spelling)
            throw std::runtime_error("native and Slang spellings differ");
        const auto& first_token = slang_tokens[first];
        if (slang_input.source_manager.getLineNumber(first_token.location()) != token.line ||
            slang_input.source_manager.getColumnNumber(first_token.location()) != token.column)
            throw std::runtime_error("native and Slang source positions differ");
    }
    if (slang_index != slang_tokens.size())
        throw std::runtime_error("Slang has trailing tokens after native EOF");
}

void print_slang_measurements(size_t input_bytes, size_t corpus_tokens,
                              const Measurements& results,
                              std::string_view input_name = {}) {
    std::cout << "case=lexer_end_to_end backend=slang"
              << " input_bytes=" << input_bytes
              << " corpus_tokens=" << corpus_tokens
              << (input_name.empty() ? "" : " input=") << input_name
              << " median_seconds=" << results.median_seconds
              << " iqr_seconds=" << results.iqr_seconds
              << " p95_seconds=" << results.p95_seconds
              << " median_MB_per_second="
              << input_bytes / results.median_seconds / 1'000'000.0
              << " tokens_per_pass=" << results.tokens
              << " median_tokens_per_second="
              << results.tokens / results.median_seconds
              << " retained_token_and_error_capacity_bytes=unavailable"
              << " slang_version=" << slang::VersionInfo::getVersionString()
              << " checksum=" << results.checksum << '\n';
}
#endif

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

#if defined(LABNEURA_ENABLE_SLANG_LEXER_BENCHMARK)
    for (const auto& input : inputs)
        verify_slang_parity(input);
    print_slang_measurements(input_bytes, corpus_tokens,
                             measure_slang_lexer(inputs));
    for (const auto& input : inputs) {
        const std::vector<Input> single_input = {input};
        print_slang_measurements(input.source->text().size(), input.tokens,
                                 measure_slang_lexer(single_input), input.name);
    }
#elif defined(__linux__)
    std::ifstream cpu_info("/proc/cpuinfo");
    for (std::string line; std::getline(cpu_info, line);) {
        const auto separator = line.find(':');
        if (separator == std::string::npos)
            continue;
        const auto key = line.substr(0, separator);
        if (key.find("model name") == std::string::npos &&
            key.find("Hardware") == std::string::npos)
            continue;
        const auto first = line.find_first_not_of(" \t", separator + 1);
        cpu_model = first == std::string::npos ? "unknown" : line.substr(first);
        break;
    }
#elif defined(_WIN32)
    if (const char* identifier = std::getenv("PROCESSOR_IDENTIFIER"))
        cpu_model = identifier;
#endif

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
