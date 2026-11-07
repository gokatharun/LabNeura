#include "labneura/backends/common/backend_factory.h"
#include "labneura/backends/common/cpu_features.h"
#include "labneura/tensor.h"
#include "test_environment.h"

#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

class SkipTest : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::vector<float> make_values(std::size_t size, float offset) {
    std::vector<float> values(size);
    for (std::size_t i = 0; i < size; ++i) {
        values[i] = offset + static_cast<float>((i * 13) % 97) * 0.25f;
    }
    return values;
}

void check_values(const labneura::Tensor& actual,
                  const std::vector<float>& expected,
                  const std::string& context) {
    require(actual.size() == expected.size(), context + ": size mismatch");
    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (actual.data_fp32()[i] != expected[i]) {
            throw std::runtime_error(context + ": mismatch at index " +
                                     std::to_string(i));
        }
    }
}

void check_size(std::size_t size, const std::string& backend) {
    const auto a_values = make_values(size, -3.0f);
    const auto b_values = make_values(size, 5.0f);
    std::vector<float> expected(size);
    for (std::size_t i = 0; i < size; ++i) {
        expected[i] = a_values[i] + b_values[i];
    }

    const labneura::Tensor a(a_values);
    const labneura::Tensor b(b_values);
    labneura::Tensor out(size == 0 ? std::vector<float>{}
                                   : std::vector<float>(size, -999.0f));
    float* const output_storage = out.data_fp32();

    a.add_into(b, out);
    check_values(out, expected, backend + " output size=" + std::to_string(size));
    require(out.data_fp32() == output_storage,
            backend + ": output storage changed after add_into");
    check_values(a, a_values, backend + ": add_into modified a");
    check_values(b, b_values, backend + ": add_into modified b");

    for (std::size_t i = 0; i < size; ++i) out.data_fp32()[i] = -123.0f;
    a.add_into(b, out);
    check_values(out, expected,
                 backend + " repeated output size=" + std::to_string(size));
    require(out.data_fp32() == output_storage,
            backend + ": repeated call replaced output storage");
}

void check_aliasing(const std::string& backend) {
    const std::vector<float> a_values = {1.0f, -2.0f, 3.5f, 4.0f, -5.25f};
    const std::vector<float> b_values = {2.0f, 3.0f, -1.5f, 0.25f, 5.0f};
    std::vector<float> expected(a_values.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        expected[i] = a_values[i] + b_values[i];
    }

    const labneura::Tensor b(b_values);
    labneura::Tensor out_alias_a(a_values);
    out_alias_a.add_into(b, out_alias_a);
    check_values(out_alias_a, expected, backend + ": out aliases a");

    const labneura::Tensor a(a_values);
    labneura::Tensor out_alias_b(b_values);
    a.add_into(out_alias_b, out_alias_b);
    check_values(out_alias_b, expected, backend + ": out aliases b");

    labneura::Tensor self_alias(a_values);
    std::vector<float> doubled(a_values.size());
    for (std::size_t i = 0; i < doubled.size(); ++i) {
        doubled[i] = a_values[i] + a_values[i];
    }
    self_alias.add_into(self_alias, self_alias);
    check_values(self_alias, doubled, backend + ": all arguments alias");
}

template <typename Exception, typename Function>
void require_throws(Function&& function, const std::string& context) {
    try {
        function();
    } catch (const Exception&) {
        return;
    }
    throw std::runtime_error(context + ": expected exception");
}

void check_validation(const std::string& backend) {
    const labneura::Tensor a(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f});
    const labneura::Tensor same_shape(std::vector<float>{5.0f, 6.0f, 7.0f, 8.0f});
    labneura::Tensor output(std::vector<float>(4, 0.0f));
    const labneura::Tensor matrix(
        std::vector<std::vector<float>>{{1.0f, 2.0f}, {3.0f, 4.0f}});
    const labneura::Tensor wrong_size(std::vector<float>{1.0f, 2.0f, 3.0f});
    const labneura::Tensor wrong_dtype(
        std::vector<int>{1, 2, 3, 4}, labneura::QuantizationMode::INT8);
    labneura::Tensor wrong_dtype_output(
        std::vector<int>{0, 0, 0, 0}, labneura::QuantizationMode::INT8);

    require_throws<std::invalid_argument>(
        [&] { a.add_into(matrix, output); },
        backend + ": equal numel but different shape");
    require_throws<std::invalid_argument>(
        [&] { a.add_into(wrong_size, output); },
        backend + ": input size mismatch");
    require_throws<std::invalid_argument>(
        [&] { a.add_into(wrong_dtype, output); },
        backend + ": non-FP32 input");
    require_throws<std::invalid_argument>(
        [&] { a.add_into(same_shape, wrong_dtype_output); },
        backend + ": non-FP32 output");
}

void check_backend_mismatch(const std::string& backend) {
    std::string other_backend = "GENERIC";
    if (backend == "GENERIC") {
        if (labneura::cpu_supports_neon()) {
            other_backend = "NEON";
        } else if (labneura::cpu_supports_avx2()) {
            other_backend = "AVX2";
        } else {
            return;
        }
    }

    std::unique_ptr<labneura::Tensor> other;
    {
        labneura::test::ScopedEnvironmentVariable override(
            "LABNEURA_BACKEND", other_backend);
        other = std::make_unique<labneura::Tensor>(
            std::vector<float>{5.0f, 6.0f});
    }

    const labneura::Tensor a(std::vector<float>{1.0f, 2.0f});
    labneura::Tensor output(std::vector<float>{0.0f, 0.0f});
    require_throws<std::invalid_argument>(
        [&] { a.add_into(*other, output); },
        backend + ": mismatched backend");
}

void run_backend(const std::string& backend) {
    labneura::test::ScopedEnvironmentVariable override(
        "LABNEURA_BACKEND", backend);
    require(labneura::detect_backend() == backend,
            "Requested backend " + backend + " was not selected");

    for (std::size_t size : std::vector<std::size_t>{
             0, 1, 7, 9, 31, 32, 33, 65, 16383, 16384, 16385, 1000000}) {
        check_size(size, backend);
    }
    check_aliasing(backend);
    check_validation(backend);
    check_backend_mismatch(backend);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::invalid_argument("Expected GENERIC, AVX2 or NEON");
        const std::string backend = argv[1];
        if (backend == "GENERIC") {
            run_backend(backend);
        } else if (backend == "AVX2") {
#if !defined(LABNEURA_TEST_AVX2)
            throw SkipTest("AVX2 backend was not compiled for this target");
#else
            if (!labneura::cpu_supports_avx2()) {
                throw SkipTest("CPU/OS does not support AVX2 state");
            }
            run_backend(backend);
#endif
        } else if (backend == "NEON") {
#if !defined(LABNEURA_TEST_NEON)
            throw SkipTest("NEON backend was not compiled for this target");
#else
            if (!labneura::cpu_supports_neon()) {
                throw SkipTest("CPU does not support NEON");
            }
            run_backend(backend);
#endif
        } else {
            throw std::invalid_argument("Unknown backend: " + backend);
        }
        std::cout << backend << ": add_into tests passed\n";
        return 0;
    } catch (const SkipTest& skip) {
        std::cout << "SKIP: " << skip.what() << '\n';
        return 77;
    } catch (const std::exception& error) {
        std::cerr << "add_into test failed: " << error.what() << '\n';
        return 1;
    }
}
