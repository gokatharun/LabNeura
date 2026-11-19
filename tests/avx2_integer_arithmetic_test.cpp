#include "labneura/backends/common/backend_factory.h"
#include "labneura/backends/common/cpu_features.h"
#if defined(LABNEURA_TEST_AVX2)
#include "labneura/backends/cpu/avx2.h"
#endif
#include "labneura/tensor.h"
#include "test_environment.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using labneura::OperationType;
using labneura::QuantizationMode;
using labneura::Tensor;

class SkipTest : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

const char* operation_name(OperationType operation) {
    switch (operation) {
        case labneura::ADD: return "ADD";
        case labneura::SUB: return "SUB";
        case labneura::MUL: return "MUL";
        case labneura::DIV: return "DIV";
    }
    return "UNKNOWN";
}

Tensor apply(const Tensor& lhs, const Tensor& rhs, OperationType operation) {
    switch (operation) {
        case labneura::ADD: return lhs.add(rhs);
        case labneura::SUB: return lhs.sub(rhs);
        case labneura::MUL: return lhs.mul(rhs);
        case labneura::DIV: return lhs.div(rhs);
    }
    throw std::invalid_argument("Unknown integer arithmetic operation");
}

void apply_inplace(Tensor& lhs, const Tensor& rhs, OperationType operation) {
    switch (operation) {
        case labneura::ADD: lhs.add_inplace(rhs); return;
        case labneura::SUB: lhs.sub_inplace(rhs); return;
        case labneura::MUL: lhs.mul_inplace(rhs); return;
        case labneura::DIV: lhs.div_inplace(rhs); return;
    }
    throw std::invalid_argument("Unknown integer arithmetic operation");
}

std::vector<int> make_lhs(std::size_t size, int minimum, int maximum) {
    std::vector<int> values(size);
    for (std::size_t i = 0; i < size; ++i) {
        switch (i % 8) {
            case 0: values[i] = maximum; break;
            case 1: values[i] = minimum; break;
            case 2: values[i] = maximum; break;
            case 3: values[i] = minimum; break;
            case 4: values[i] = maximum - 1; break;
            case 5: values[i] = minimum; break;
            case 6: values[i] = 7; break;
            case 7: values[i] = -7; break;
        }
    }
    return values;
}

std::vector<int> make_rhs(std::size_t size) {
    constexpr std::array<int, 8> kDenominators = {
        1, -1, -1, 1, 2, 2, 0, 3};
    std::vector<int> values(size);
    for (std::size_t i = 0; i < size; ++i) {
        values[i] = kDenominators[i % kDenominators.size()];
    }
    return values;
}

std::vector<int> scalar_expected(const std::vector<int>& lhs,
                                 const std::vector<int>& rhs,
                                 OperationType operation,
                                 int minimum, int maximum) {
    require(lhs.size() == rhs.size(), "Reference operands have different sizes");
    std::vector<int> expected(lhs.size());
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        std::int64_t result = 0;
        switch (operation) {
            case labneura::ADD:
                result = static_cast<std::int64_t>(lhs[i]) + rhs[i];
                break;
            case labneura::SUB:
                result = static_cast<std::int64_t>(lhs[i]) - rhs[i];
                break;
            case labneura::MUL:
                result = static_cast<std::int64_t>(lhs[i]) * rhs[i];
                break;
            case labneura::DIV:
                result = rhs[i] == 0
                             ? 0
                             : static_cast<std::int64_t>(lhs[i]) / rhs[i];
                break;
        }
        expected[i] = static_cast<int>(
            std::max<std::int64_t>(minimum,
                                   std::min<std::int64_t>(maximum, result)));
    }
    return expected;
}

std::vector<int> read_values(const Tensor& tensor, QuantizationMode dtype) {
    std::vector<int> result(tensor.numel());
    if (dtype == QuantizationMode::INT8) {
        for (std::size_t i = 0; i < result.size(); ++i) {
            result[i] = tensor.data_int8()[i];
        }
    } else {
        for (std::size_t i = 0; i < result.size(); ++i) {
            result[i] = tensor.data_int16()[i];
        }
    }
    return result;
}

void require_values(const std::vector<int>& actual,
                    const std::vector<int>& expected,
                    const std::string& context) {
    require(actual.size() == expected.size(),
            context + ": result size mismatch");
    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (actual[i] != expected[i]) {
            throw std::runtime_error(
                context + ": mismatch at index " + std::to_string(i) +
                ", expected " + std::to_string(expected[i]) + ", got " +
                std::to_string(actual[i]));
        }
    }
}

struct OperationResults {
    std::vector<int> allocating;
    std::vector<int> inplace;
};

OperationResults run_case(const std::string& backend, QuantizationMode dtype,
                          const std::vector<int>& lhs_values,
                          const std::vector<int>& rhs_values,
                          OperationType operation, bool use_offset_views) {
    labneura::test::ScopedEnvironmentVariable backend_override(
        "LABNEURA_BACKEND", backend);
    require(labneura::detect_backend() == backend,
            backend + ": requested backend was not selected");

    const std::size_t size = lhs_values.size();
    require(rhs_values.size() == size,
            "Test operands have different sizes");

    Tensor lhs_parent;
    Tensor rhs_parent;
    Tensor lhs;
    Tensor rhs;
    if (use_offset_views) {
        std::vector<int> lhs_storage(size + 2, -17);
        std::vector<int> rhs_storage(size + 2, 23);
        std::copy(lhs_values.begin(), lhs_values.end(),
                  lhs_storage.begin() + 1);
        std::copy(rhs_values.begin(), rhs_values.end(),
                  rhs_storage.begin() + 1);
        lhs_parent = Tensor::from_flat(
            lhs_storage, {lhs_storage.size()}, dtype);
        rhs_parent = Tensor::from_flat(
            rhs_storage, {rhs_storage.size()}, dtype);
        lhs = lhs_parent.slice_first_axis(1, size + 1);
        rhs = rhs_parent.slice_first_axis(1, size + 1);
        require(lhs.storage_offset() == 1 && rhs.storage_offset() == 1,
                backend + ": offset views did not retain their logical offset");
    } else {
        lhs = Tensor::from_flat(lhs_values, {size}, dtype);
        rhs = Tensor::from_flat(rhs_values, {size}, dtype);
    }

    const Tensor allocated = apply(lhs, rhs, operation);
    require(allocated.quantization_mode() == dtype &&
                allocated.shape() == lhs.shape() &&
                allocated.numel() == size,
            backend + ": allocating operation changed dtype or shape");
    OperationResults results;
    results.allocating = read_values(allocated, dtype);

    Tensor inplace_parent;
    Tensor inplace_rhs_parent;
    Tensor inplace;
    Tensor inplace_rhs;
    std::vector<int> expected_lhs_storage;
    std::vector<int> expected_rhs_storage;
    if (use_offset_views) {
        expected_lhs_storage.assign(size + 2, -17);
        expected_rhs_storage.assign(size + 2, 23);
        std::copy(lhs_values.begin(), lhs_values.end(),
                  expected_lhs_storage.begin() + 1);
        std::copy(rhs_values.begin(), rhs_values.end(),
                  expected_rhs_storage.begin() + 1);
        inplace_parent = Tensor::from_flat(
            expected_lhs_storage, {size + 2}, dtype);
        inplace_rhs_parent = Tensor::from_flat(
            expected_rhs_storage, {size + 2}, dtype);
        inplace = inplace_parent.slice_first_axis(1, size + 1);
        inplace_rhs = inplace_rhs_parent.slice_first_axis(1, size + 1);
    } else {
        inplace = Tensor::from_flat(lhs_values, {size}, dtype);
        inplace_rhs = Tensor::from_flat(rhs_values, {size}, dtype);
    }
    apply_inplace(inplace, inplace_rhs, operation);
    results.inplace = read_values(inplace, dtype);

    if (use_offset_views) {
        const auto lhs_after = read_values(inplace_parent, dtype);
        const auto rhs_after = read_values(inplace_rhs_parent, dtype);
        require(lhs_after.front() == -17 && lhs_after.back() == -17,
                backend + ": in-place view operation crossed the lhs bounds");
        require(rhs_after == expected_rhs_storage,
                backend + ": in-place view operation modified the rhs");
        for (std::size_t i = 0; i < size; ++i) {
            expected_lhs_storage[i + 1] = results.inplace[i];
        }
        require(lhs_after == expected_lhs_storage,
                backend + ": in-place view did not update only its logical range");
    }
    return results;
}

void check_backend_case(QuantizationMode dtype, const std::string& dtype_name,
                        std::size_t size, bool use_offset_views,
                        int minimum, int maximum) {
    const auto lhs = make_lhs(size, minimum, maximum);
    const auto rhs = make_rhs(size);
    const std::string case_name =
        dtype_name + " size=" + std::to_string(size) +
        (use_offset_views ? " offset-view" : "");

    for (const OperationType operation :
         {labneura::ADD, labneura::SUB, labneura::MUL, labneura::DIV}) {
        const auto expected =
            scalar_expected(lhs, rhs, operation, minimum, maximum);
        const auto generic =
            run_case("GENERIC", dtype, lhs, rhs, operation, use_offset_views);
        const auto avx2 =
            run_case("AVX2", dtype, lhs, rhs, operation, use_offset_views);
        const std::string context =
            case_name + " " + operation_name(operation);

        require_values(generic.allocating, expected,
                       context + " Generic allocating reference");
        require_values(generic.inplace, expected,
                       context + " Generic in-place reference");
        require_values(avx2.allocating, expected,
                       context + " AVX2 allocating reference");
        require_values(avx2.inplace, expected,
                       context + " AVX2 in-place reference");
        require_values(avx2.allocating, generic.allocating,
                       context + " AVX2/Generic allocating parity");
        require_values(avx2.inplace, generic.inplace,
                       context + " AVX2/Generic in-place parity");
    }
}

void verify_avx2_selection() {
#if !defined(LABNEURA_TEST_AVX2)
    throw SkipTest("AVX2 backend was not compiled for this target");
#else
    labneura::test::ScopedEnvironmentVariable backend_override(
        "LABNEURA_BACKEND", "AVX2");
    try {
        require(labneura::detect_backend() == "AVX2",
                "LABNEURA_BACKEND=AVX2 did not select AVX2");
    } catch (const std::invalid_argument& error) {
        if (std::string(error.what()) ==
            "Requested backend AVX2 is not available in this build") {
            throw SkipTest(error.what());
        }
        throw;
    }

    auto backend = labneura::create_best_backend(
        1, QuantizationMode::INT8);
    require(dynamic_cast<labneura::AVX2Backend*>(backend.get()) != nullptr,
            "Factory did not construct the actual AVX2 backend");
#endif
}

void run_tests() {
    const std::array<std::size_t, 14> sizes = {
        0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65};
    const int int8_min = std::numeric_limits<std::int8_t>::min();
    const int int8_max = std::numeric_limits<std::int8_t>::max();
    const int int16_min = std::numeric_limits<std::int16_t>::min();
    const int int16_max = std::numeric_limits<std::int16_t>::max();

    for (const std::size_t size : sizes) {
        check_backend_case(QuantizationMode::INT8, "INT8", size, false,
                           int8_min, int8_max);
        check_backend_case(QuantizationMode::INT16, "INT16", size, false,
                           int16_min, int16_max);
    }

    // Both views start one element into their storage. That makes the AVX2
    // INT8 and INT16 logical starts unaligned while spanning vector work and a
    // scalar tail (67 INT8 elements, 35 INT16 elements).
    check_backend_case(QuantizationMode::INT8, "INT8", 67, true,
                       int8_min, int8_max);
    check_backend_case(QuantizationMode::INT16, "INT16", 35, true,
                       int16_min, int16_max);
}

}  // namespace

int main() {
    try {
        if (!labneura::cpu_supports_avx2()) {
            throw SkipTest("CPU/OS does not support AVX2 state");
        }
        verify_avx2_selection();
        run_tests();
        std::cout << "AVX2 INT8/INT16 arithmetic and Generic parity passed\n";
        return 0;
    } catch (const SkipTest& skip) {
        std::cout << "SKIP: " << skip.what() << '\n';
        return 77;
    } catch (const std::exception& error) {
        std::cerr << "AVX2 integer arithmetic test failed: " << error.what()
                  << '\n';
        return 1;
    }
}
