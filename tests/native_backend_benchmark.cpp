#include "labneura/backends/common/cpu_features.h"
#include "labneura/backends/cpu/generic.h"
#include "labneura/backends/common/base.h"

#if defined(LABNEURA_TEST_AVX2)
#include "labneura/backends/cpu/avx2.h"
#endif

#if defined(LABNEURA_TEST_NEON)
#include "labneura/backends/cpu/neon.h"
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using BackendFactory =
    std::function<std::unique_ptr<labneura::TensorBackend>(
        std::size_t, labneura::QuantizationMode)>;

struct BackendSpec {
    std::string name;
    BackendFactory create;
};

struct Timing {
    double median_ns;
    double p95_ns;
};

constexpr std::array<std::size_t, 7> kSizes{
    1, 8, 32, 64, 1024, 65536, 1048576};
constexpr int kWarmupIterations = 5;
constexpr int kSamples = 31;

const char* operation_name(labneura::OperationType operation) {
    switch (operation) {
        case labneura::OperationType::ADD: return "ADD";
        case labneura::OperationType::SUB: return "SUB";
        case labneura::OperationType::MUL: return "MUL";
        case labneura::OperationType::DIV: return "DIV";
    }
    throw std::runtime_error("Unknown operation");
}

int8_t saturate_int8(int value) {
    return static_cast<int8_t>(std::clamp(value, -128, 127));
}

int8_t expected_int8(int lhs, int rhs, labneura::OperationType operation) {
    switch (operation) {
        case labneura::OperationType::ADD:
            return saturate_int8(lhs + rhs);
        case labneura::OperationType::SUB:
            return saturate_int8(lhs - rhs);
        case labneura::OperationType::MUL:
            return saturate_int8(lhs * rhs);
        case labneura::OperationType::DIV:
            return rhs == 0 ? 0 : saturate_int8(lhs / rhs);
    }
    throw std::runtime_error("Unknown operation");
}

void apply(labneura::TensorBackend& lhs,
           const labneura::TensorBackend& rhs,
           labneura::OperationType operation) {
    lhs.operation(rhs, operation);
}

template <typename T>
Timing measure(labneura::TensorBackend& lhs,
               const labneura::TensorBackend& rhs,
               const std::vector<T>& initial_values,
               labneura::OperationType operation,
               T* lhs_data) {
    for (int i = 0; i < kWarmupIterations; ++i) {
        std::copy(initial_values.begin(), initial_values.end(), lhs_data);
        apply(lhs, rhs, operation);
    }

    std::vector<double> samples;
    samples.reserve(kSamples);
    for (int i = 0; i < kSamples; ++i) {
        std::copy(initial_values.begin(), initial_values.end(), lhs_data);
        const auto start = std::chrono::steady_clock::now();
        apply(lhs, rhs, operation);
        const auto stop = std::chrono::steady_clock::now();
        samples.push_back(
            std::chrono::duration<double, std::nano>(stop - start).count());
    }
    std::sort(samples.begin(), samples.end());
    return {samples[samples.size() / 2],
            samples[(samples.size() - 1) * 95 / 100]};
}

std::vector<float> make_fp32_values(std::size_t size, bool divisor) {
    std::vector<float> values(size);
    for (std::size_t i = 0; i < size; ++i) {
        if (divisor) {
            values[i] = static_cast<float>((i % 13) + 1) * 0.25f;
        } else {
            values[i] = static_cast<float>(static_cast<int>(i % 101) - 50) *
                        0.125f;
        }
    }
    return values;
}

std::vector<int8_t> make_int8_values(std::size_t size, bool divisor) {
    std::vector<int8_t> values(size);
    for (std::size_t i = 0; i < size; ++i) {
        int value = static_cast<int>((i * 73 + 19) % 256) - 128;
        if (divisor) {
            value = static_cast<int>((i * 31 + 1) % 255) - 127;
            if (value == 0) value = 1;
        }
        values[i] = static_cast<int8_t>(value);
    }
    return values;
}

void benchmark_fp32(const BackendSpec& backend,
                    std::size_t size,
                    labneura::OperationType operation,
                    const std::string& architecture,
                    std::ostream& output) {
    auto lhs = backend.create(size, labneura::QuantizationMode::FP32);
    auto rhs = backend.create(size, labneura::QuantizationMode::FP32);
    auto lhs_values = make_fp32_values(size, false);
    const auto rhs_values =
        make_fp32_values(size, operation == labneura::OperationType::DIV);
    std::copy(rhs_values.begin(), rhs_values.end(), rhs->data_fp32());

    std::vector<float> expected(size);
    for (std::size_t i = 0; i < size; ++i) {
        switch (operation) {
            case labneura::OperationType::ADD:
                expected[i] = lhs_values[i] + rhs_values[i];
                break;
            case labneura::OperationType::SUB:
                expected[i] = lhs_values[i] - rhs_values[i];
                break;
            case labneura::OperationType::MUL:
                expected[i] = lhs_values[i] * rhs_values[i];
                break;
            case labneura::OperationType::DIV:
                expected[i] = lhs_values[i] / rhs_values[i];
                break;
        }
    }
    std::copy(lhs_values.begin(), lhs_values.end(), lhs->data_fp32());
    apply(*lhs, *rhs, operation);
    for (std::size_t i = 0; i < size; ++i) {
        const float actual = lhs->data_fp32()[i];
        const float tolerance = 1e-5f * std::max(1.0f, std::abs(expected[i]));
        if (!std::isfinite(actual) || std::abs(actual - expected[i]) > tolerance) {
            throw std::runtime_error("FP32 correctness check failed for " +
                                     backend.name + " " +
                                     operation_name(operation));
        }
    }

    const Timing timing = measure(*lhs, *rhs, lhs_values, operation,
                                  lhs->data_fp32());
    output << architecture << ',' << backend.name << ",FP32,"
           << operation_name(operation) << ',' << size << ','
           << timing.median_ns << ',' << timing.p95_ns << '\n';
}

void benchmark_int8(const BackendSpec& backend,
                    std::size_t size,
                    labneura::OperationType operation,
                    const std::string& architecture,
                    std::ostream& output) {
    auto lhs = backend.create(size, labneura::QuantizationMode::INT8);
    auto rhs = backend.create(size, labneura::QuantizationMode::INT8);
    auto lhs_values = make_int8_values(size, false);
    const auto rhs_values =
        make_int8_values(size, operation == labneura::OperationType::DIV);
    std::copy(rhs_values.begin(), rhs_values.end(), rhs->data_int8());

    std::vector<int8_t> expected(size);
    for (std::size_t i = 0; i < size; ++i) {
        expected[i] = expected_int8(lhs_values[i], rhs_values[i], operation);
    }
    std::copy(lhs_values.begin(), lhs_values.end(), lhs->data_int8());
    apply(*lhs, *rhs, operation);
    if (!std::equal(expected.begin(), expected.end(), lhs->data_int8())) {
        throw std::runtime_error("INT8 correctness check failed for " +
                                 backend.name + " " +
                                 operation_name(operation));
    }

    const Timing timing = measure(*lhs, *rhs, lhs_values, operation,
                                  lhs->data_int8());
    output << architecture << ',' << backend.name << ",INT8,"
           << operation_name(operation) << ',' << size << ','
           << timing.median_ns << ',' << timing.p95_ns << '\n';
}

std::string architecture_name() {
#if defined(__x86_64__) || defined(_M_X64)
    return "x86_64";
#elif defined(__aarch64__) || defined(_M_ARM64)
    return "aarch64";
#else
    return "unknown";
#endif
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: native_backend_benchmark <output.csv>\n";
        return 2;
    }

    try {
        std::vector<BackendSpec> backends;
        backends.push_back({"GENERIC", [](std::size_t size,
                                          labneura::QuantizationMode mode) {
                                return std::make_unique<labneura::GenericBackend>(
                                    size, mode);
                            }});

#if defined(LABNEURA_TEST_AVX2)
        if (labneura::cpu_supports_avx2()) {
            backends.push_back({"AVX2", [](std::size_t size,
                                           labneura::QuantizationMode mode) {
                                    return std::make_unique<labneura::AVX2Backend>(
                                        size, mode);
                                }});
        }
#endif

#if defined(LABNEURA_TEST_NEON)
        if (labneura::cpu_supports_neon()) {
            backends.push_back({"NEON", [](std::size_t size,
                                           labneura::QuantizationMode mode) {
                                    return std::make_unique<labneura::NEONBackend>(
                                        size, mode);
                                }});
        }
#endif

        if (backends.size() < 2) {
            throw std::runtime_error(
                "No supported SIMD backend is available for benchmarking");
        }

        std::ofstream output(argv[1]);
        if (!output) {
            throw std::runtime_error("Could not open benchmark output file");
        }
        output << "architecture,backend,dtype,operation,size,median_ns,p95_ns\n";
        output << std::fixed << std::setprecision(2);
        for (const auto& backend : backends) {
            for (const auto size : kSizes) {
                for (const auto operation : {
                         labneura::OperationType::ADD,
                         labneura::OperationType::SUB,
                         labneura::OperationType::MUL,
                         labneura::OperationType::DIV}) {
                    benchmark_fp32(backend, size, operation, architecture_name(),
                                   output);
                    benchmark_int8(backend, size, operation, architecture_name(),
                                   output);
                }
            }
        }
        output.close();
        if (!output) {
            throw std::runtime_error("Failed writing benchmark results");
        }
        std::cout << "Wrote " << argv[1] << " for " << architecture_name()
                  << " (" << backends.size() << " backends)\n";
    } catch (const std::exception& error) {
        std::cerr << "Benchmark failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
