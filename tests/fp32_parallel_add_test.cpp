#include "labneura/backends/common/backend_factory.h"
#include "labneura/runtime.h"
#include "labneura/tensor.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

class BackendEnvironment {
public:
    explicit BackendEnvironment(const char* backend) {
        const char* old_value = std::getenv("LABNEURA_BACKEND");
        if (old_value) {
            had_old_value_ = true;
            old_value_ = old_value;
        }
        if (setenv("LABNEURA_BACKEND", backend, 1) != 0) {
            throw std::runtime_error("failed to set backend environment");
        }
        if (labneura::detect_backend() != backend) {
            throw std::runtime_error(std::string("requested backend unavailable: ") +
                                     backend);
        }
    }

    ~BackendEnvironment() {
        if (had_old_value_) {
            setenv("LABNEURA_BACKEND", old_value_.c_str(), 1);
        } else {
            unsetenv("LABNEURA_BACKEND");
        }
    }

private:
    bool had_old_value_{false};
    std::string old_value_;
};

std::vector<float> make_values(std::size_t size, std::size_t salt) {
    std::vector<float> values(size);
    for (std::size_t i = 0; i < size; ++i) {
        const int numerator = static_cast<int>((i * 17 + salt * 13) % 2048) - 1024;
        values[i] = static_cast<float>(numerator) / 128.0f;
    }
    return values;
}

void check_equal(const labneura::Tensor& actual,
                 const std::vector<float>& expected,
                 const std::string& context) {
    require(actual.size() == expected.size(), context + ": result size mismatch");
    const float* values = actual.data_fp32();
    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (values[i] != expected[i]) {
            throw std::runtime_error(context + ": mismatch at index " +
                                     std::to_string(i));
        }
    }
}

void test_backend(const char* backend, std::size_t available) {
    BackendEnvironment selected(backend);
    const std::vector<std::size_t> sizes = {
        1, 7, 8, 9, 31, 32, 33,
        16383, 16384, 16385,
        32767, 32768, 32769,
        49151, 49152, 49153,
        65536, 100000, 262144,
        999999, 1000000, 1000001,
    };
    std::vector<std::size_t> thread_counts = {1, 2, 4, 6, available};
    thread_counts.erase(
        std::remove_if(thread_counts.begin(), thread_counts.end(),
                       [available](std::size_t count) {
                           return count > available;
                       }),
        thread_counts.end());
    std::sort(thread_counts.begin(), thread_counts.end());
    thread_counts.erase(std::unique(thread_counts.begin(), thread_counts.end()),
                        thread_counts.end());

    for (std::size_t thread_count : thread_counts) {
        labneura::set_execution_thread_count(thread_count);
        for (std::size_t size : sizes) {
            const auto left = make_values(size, 1);
            const auto right = make_values(size, 7);
            std::vector<float> expected(size);
            for (std::size_t i = 0; i < size; ++i) {
                expected[i] = left[i] + right[i];
            }

            const labneura::Tensor lhs(left);
            const labneura::Tensor rhs(right);
            const labneura::Tensor allocated = lhs.add(rhs);
            check_equal(allocated, expected,
                        std::string(backend) + " allocating " +
                            std::to_string(size) + " threads=" +
                            std::to_string(thread_count));
            check_equal(lhs, left, "allocating ADD modified left input");
            check_equal(rhs, right, "allocating ADD modified right input");

            labneura::Tensor in_place(left);
            in_place.add_inplace(rhs);
            check_equal(in_place, expected,
                        std::string(backend) + " in-place " +
                            std::to_string(size) + " threads=" +
                            std::to_string(thread_count));

            std::vector<float> doubled(size);
            for (std::size_t i = 0; i < size; ++i) doubled[i] = left[i] + left[i];
            labneura::Tensor alias(left);
            alias.add_inplace(alias);
            check_equal(alias, doubled,
                        std::string(backend) + " exact alias " +
                            std::to_string(size) + " threads=" +
                            std::to_string(thread_count));
        }
    }
}

}  // namespace

int main() {
    try {
        const std::size_t available = labneura::available_cpu_count();
        test_backend("GENERIC", available);
#if defined(LABNEURA_TEST_NEON)
        test_backend("NEON", available);
#endif
        labneura::set_execution_thread_count(1);
        std::cout << "FP32 parallel ADD tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FP32 parallel ADD test failed: " << error.what() << '\n';
        return 1;
    }
}
