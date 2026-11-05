#include "labneura/backends/common/backend_factory.h"
#include "labneura/backends/cpu/generic.h"
#if defined(LABNEURA_TEST_AVX2)
#include "labneura/backends/cpu/avx2.h"
#endif
#if defined(LABNEURA_TEST_NEON)
#include "labneura/backends/cpu/neon.h"
#endif

#include <array>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

class BackendOverride {
public:
    explicit BackendOverride(const char* value) {
        if (const char* current = std::getenv("LABNEURA_BACKEND")) {
            previous_ = current;
        }
        if (value) {
            setenv("LABNEURA_BACKEND", value, 1);
        } else {
            unsetenv("LABNEURA_BACKEND");
        }
    }

    ~BackendOverride() {
        if (previous_) {
            setenv("LABNEURA_BACKEND", previous_->c_str(), 1);
        } else {
            unsetenv("LABNEURA_BACKEND");
        }
    }

private:
    std::optional<std::string> previous_;
};

void expect_unsupported(const std::function<void()>& construct,
                        const std::string& dtype) {
    try {
        construct();
    } catch (const std::invalid_argument& error) {
        const std::string expected = "Unsupported tensor dtype: " + dtype;
        if (error.what() != expected) {
            throw std::runtime_error(
                "Expected error '" + expected + "', got '" + error.what() + "'");
        }
        return;
    } catch (const std::exception& error) {
        throw std::runtime_error(
            "Expected std::invalid_argument for " + dtype + ", got: " +
            error.what());
    }
    throw std::runtime_error("Construction accepted unsupported dtype " + dtype);
}

void check_tensor_constructors(labneura::QuantizationMode mode,
                               const std::string& dtype) {
    const std::array<const char*, 4> backends = {
        "GENERIC", "AVX2", "NEON", nullptr};

    for (const char* backend : backends) {
        BackendOverride override(backend);
        expect_unsupported([&] {
            labneura::Tensor tensor(std::vector<float>{1.0f, 2.0f}, mode);
        }, dtype);
        expect_unsupported([&] {
            labneura::Tensor tensor(std::vector<float>{}, mode);
        }, dtype);
        expect_unsupported([&] {
            labneura::Tensor tensor(std::vector<int>{1, 2}, mode);
        }, dtype);
        expect_unsupported([&] {
            labneura::Tensor tensor(std::vector<int>{}, mode);
        }, dtype);
        expect_unsupported([&] {
            labneura::Tensor tensor(
                std::vector<std::vector<float>>{{1.0f, 2.0f}}, mode);
        }, dtype);
        expect_unsupported([&] {
            labneura::Tensor tensor(1.0f, mode);
        }, dtype);
        expect_unsupported([&] {
            labneura::Tensor tensor(1, mode);
        }, dtype);
        expect_unsupported([&] {
            labneura::create_best_backend(2, mode);
        }, dtype);
    }
}

void check_direct_backend_constructors(labneura::QuantizationMode mode,
                                       const std::string& dtype) {
    expect_unsupported([&] {
        labneura::GenericBackend backend(2, mode);
    }, dtype);
#if defined(LABNEURA_TEST_AVX2)
    expect_unsupported([&] {
        labneura::AVX2Backend backend(2, mode);
    }, dtype);
#endif
#if defined(LABNEURA_TEST_NEON)
    expect_unsupported([&] {
        labneura::NEONBackend backend(2, mode);
    }, dtype);
#endif
}

}  // namespace

int main() {
    try {
        check_tensor_constructors(labneura::QuantizationMode::FP64, "FP64");
        check_tensor_constructors(labneura::QuantizationMode::INT32, "INT32");
        check_direct_backend_constructors(
            labneura::QuantizationMode::FP64, "FP64");
        check_direct_backend_constructors(
            labneura::QuantizationMode::INT32, "INT32");
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
