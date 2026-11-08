#include "labneura/backends/common/backend_factory.h"
#include "labneura/backends/common/cpu_features.h"
#include "labneura/backends/cpu/generic.h"
#include "labneura/tensor.h"
#include "test_environment.h"

#if defined(LABNEURA_TEST_AVX2)
#include "labneura/backends/cpu/avx2.h"
#endif

#if defined(LABNEURA_TEST_NEON)
#include "labneura/backends/cpu/neon.h"
#endif

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

class SkipTest : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
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

void check_logical_not_result(
    const labneura::Tensor& result,
    const std::vector<std::size_t>& expected_shape,
    const std::vector<std::uint8_t>& expected_values,
    const std::string& context) {
    require(result.quantization_mode() == labneura::QuantizationMode::BOOL,
            context + ": result dtype was not BOOL");
    require(result.shape() == expected_shape,
            context + ": result shape mismatch");
    require(result.numel() == expected_values.size(),
            context + ": result element count mismatch");
    require(result.is_contiguous(), context + ": result is not contiguous");
    const std::uint8_t* actual = result.data_bool();
    for (std::size_t i = 0; i < expected_values.size(); ++i) {
        require(actual[i] == expected_values[i] && actual[i] <= 1,
                context + ": value mismatch at index " +
                    std::to_string(i));
    }
}

void check_logical_and_result(
    const labneura::Tensor& result,
    const std::vector<std::size_t>& expected_shape,
    const std::vector<std::uint8_t>& expected_values,
    const std::string& context) {
    require(result.quantization_mode() == labneura::QuantizationMode::BOOL,
            context + ": result dtype was not BOOL");
    require(result.shape() == expected_shape,
            context + ": result shape mismatch");
    require(result.numel() == expected_values.size(),
            context + ": result element count mismatch");
    require(result.is_contiguous(), context + ": result is not contiguous");
    const std::uint8_t* actual = result.data_bool();
    for (std::size_t i = 0; i < expected_values.size(); ++i) {
        require(actual[i] == expected_values[i] && actual[i] <= 1,
                context + ": value mismatch at index " +
                    std::to_string(i));
    }
}

void check_generic_logical_not() {
    using labneura::QuantizationMode;
    using labneura::Tensor;

    labneura::test::ScopedEnvironmentVariable backend_override(
        "LABNEURA_BACKEND", "GENERIC");
    require(labneura::detect_backend() == "GENERIC",
            "Generic logical_not backend was not explicitly selected");

    const Tensor mixed(std::vector<std::uint8_t>{1, 0, 1, 0},
                       QuantizationMode::BOOL);
    const std::vector<std::uint8_t> mixed_before(mixed.data_bool(),
                                                  mixed.data_bool() + 4);
    Tensor mixed_result = mixed.logical_not();
    check_logical_not_result(mixed_result, {4}, {0, 1, 0, 1},
                             "Mixed BOOL logical_not");
    require(mixed_result.data_bool() != mixed.data_bool(),
            "logical_not output aliases its input");
    mixed_result.data_bool()[0] = 1;
    require(std::vector<std::uint8_t>(mixed.data_bool(),
                                      mixed.data_bool() + 4) == mixed_before,
            "Mutating logical_not output changed its input");

    check_logical_not_result(
        Tensor(std::vector<std::uint8_t>{0, 0, 0}, QuantizationMode::BOOL)
            .logical_not(),
        {3}, {1, 1, 1}, "All-false logical_not");
    check_logical_not_result(
        Tensor(std::vector<std::uint8_t>{1, 1, 1}, QuantizationMode::BOOL)
            .logical_not(),
        {3}, {0, 0, 0}, "All-true logical_not");
    check_logical_not_result(
        Tensor::from_flat(std::vector<std::uint8_t>{0}, {}, QuantizationMode::BOOL)
            .logical_not(),
        {}, {1}, "Scalar-false logical_not");
    check_logical_not_result(
        Tensor::from_flat(std::vector<std::uint8_t>{1}, {}, QuantizationMode::BOOL)
            .logical_not(),
        {}, {0}, "Scalar-true logical_not");

    const Tensor empty = Tensor::from_flat(
        std::vector<std::uint8_t>{}, {0, 3}, QuantizationMode::BOOL);
    check_logical_not_result(empty.logical_not(), {0, 3}, {},
                             "Empty logical_not");

    const Tensor parent = Tensor::from_flat(
        std::vector<std::uint8_t>{0, 1, 1, 0, 0, 0, 1, 1}, {4, 2},
        QuantizationMode::BOOL);
    const Tensor view = parent.slice_first_axis(1, 3);
    require(view.storage_offset() == 2,
            "logical_not test view did not have a nonzero storage offset");
    const std::vector<std::uint8_t> parent_before(
        parent.data_bool(), parent.data_bool() + parent.numel());
    const Tensor view_result = view.logical_not();
    check_logical_not_result(view_result, {2, 2}, {0, 1, 1, 1},
                             "Offset-view logical_not");
    require(view_result.data_bool() != view.data_bool(),
            "Offset-view logical_not output aliases its input");
    require(std::vector<std::uint8_t>(
                parent.data_bool(), parent.data_bool() + parent.numel()) ==
                parent_before,
            "Offset-view logical_not mutated its parent");

    const Tensor repeated = mixed.logical_not();
    check_logical_not_result(repeated, {4}, {0, 1, 0, 1},
                             "Repeated logical_not");
    require(repeated.data_bool() != mixed_result.data_bool(),
            "Repeated logical_not calls reused output storage");
    require(std::vector<std::uint8_t>(mixed.data_bool(),
                                      mixed.data_bool() + mixed.numel()) ==
                mixed_before,
            "Repeated logical_not mutated its input");

    const Tensor numeric(std::vector<float>{0.0f, 1.0f});
    require_throws<std::runtime_error>(
        [&] { (void)numeric.logical_not(); },
        "logical_not accepted a non-BOOL tensor");
}

void check_generic_logical_and() {
    using labneura::QuantizationMode;
    using labneura::Tensor;

    labneura::test::ScopedEnvironmentVariable backend_override(
        "LABNEURA_BACKEND", "GENERIC");
    require(labneura::detect_backend() == "GENERIC",
            "Generic logical_and backend was not explicitly selected");

    const Tensor lhs(std::vector<std::uint8_t>{0, 0, 1, 1},
                     QuantizationMode::BOOL);
    const Tensor rhs(std::vector<std::uint8_t>{0, 1, 0, 1},
                     QuantizationMode::BOOL);
    const Tensor same_shape = lhs.logical_and(rhs);
    check_logical_and_result(same_shape, {4}, {0, 0, 0, 1},
                             "Generic same-shape logical_and");
    require(same_shape.data_bool() != lhs.data_bool() &&
                same_shape.data_bool() != rhs.data_bool(),
            "Generic logical_and output aliases an input");

    const Tensor row = Tensor::from_flat(
        std::vector<std::uint8_t>{0, 1}, {2, 1}, QuantizationMode::BOOL);
    const Tensor column = Tensor::from_flat(
        std::vector<std::uint8_t>{1, 0, 1}, {1, 3}, QuantizationMode::BOOL);
    check_logical_and_result(row.logical_and(column), {2, 3},
                             {0, 0, 0, 1, 0, 1},
                             "Generic two-axis broadcast logical_and");

    const Tensor scalar = Tensor::from_flat(
        std::vector<std::uint8_t>{1}, {}, QuantizationMode::BOOL);
    check_logical_and_result(scalar.logical_and(rhs), {4}, {0, 1, 0, 1},
                             "Generic scalar broadcast logical_and");
    check_logical_and_result(
        Tensor::from_flat(std::vector<std::uint8_t>{0}, {},
                          QuantizationMode::BOOL)
            .logical_and(Tensor::from_flat(std::vector<std::uint8_t>{1}, {},
                                           QuantizationMode::BOOL)),
        {}, {0}, "Generic scalar logical_and");
    check_logical_and_result(scalar.logical_and(scalar), {}, {1},
                             "Generic true scalar logical_and");

    const Tensor empty = Tensor::from_flat(
        std::vector<std::uint8_t>{}, {0, 3}, QuantizationMode::BOOL);
    const Tensor broadcast_empty = Tensor::from_flat(
        std::vector<std::uint8_t>{1, 0, 1}, {1, 3}, QuantizationMode::BOOL);
    check_logical_and_result(empty.logical_and(broadcast_empty), {0, 3}, {},
                             "Generic empty broadcast logical_and");

    std::vector<std::uint8_t> parent_values(66);
    for (std::size_t i = 0; i < parent_values.size(); ++i) {
        parent_values[i] = static_cast<std::uint8_t>(i % 2);
    }
    const Tensor parent =
        Tensor::from_flat(parent_values, {2, 33}, QuantizationMode::BOOL);
    const Tensor view = parent.slice_first_axis(1, 2);
    require(view.storage_offset() == 33,
            "Generic logical_and view has no storage offset");
    const Tensor view_rhs(
        std::vector<std::uint8_t>(33, std::uint8_t{1}),
        QuantizationMode::BOOL);
    const Tensor view_result = view.logical_and(view_rhs);
    check_logical_and_result(view_result, {1, 33},
                             std::vector<std::uint8_t>(
                                 parent_values.begin() + 33,
                                 parent_values.end()),
                             "Generic offset-view logical_and");
    require(view_result.data_bool() != view.data_bool(),
            "Generic offset-view logical_and output aliases its input");

    const Tensor numeric(std::vector<float>{1.0f});
    require_throws<std::runtime_error>(
        [&] { (void)lhs.logical_and(numeric); },
        "logical_and accepted a non-BOOL tensor");
    const Tensor incompatible = Tensor::from_flat(
        std::vector<std::uint8_t>{0, 1}, {2}, QuantizationMode::BOOL);
    require_throws<std::runtime_error>(
        [&] { (void)lhs.logical_and(incompatible); },
        "logical_and accepted incompatible shapes");
}

#if defined(LABNEURA_TEST_NEON)
void check_neon_logical_not() {
    using labneura::QuantizationMode;
    using labneura::Tensor;

    labneura::test::ScopedEnvironmentVariable backend_override(
        "LABNEURA_BACKEND", "NEON");
    require(labneura::detect_backend() == "NEON",
            "NEON logical_not backend was not explicitly selected");
    const auto selected_backend =
        labneura::create_best_backend(1, QuantizationMode::BOOL);
    require(dynamic_cast<labneura::NEONBackend*>(selected_backend.get()) !=
                nullptr,
            "Explicit NEON selection did not create a NEON backend");

    for (std::size_t size = 0; size <= 33; ++size) {
        std::vector<std::uint8_t> values(size);
        std::vector<std::uint8_t> expected(size);
        for (std::size_t i = 0; i < size; ++i) {
            values[i] = static_cast<std::uint8_t>(i % 2);
            expected[i] = static_cast<std::uint8_t>(values[i] == 0);
        }
        const Tensor input(values, QuantizationMode::BOOL);
        const std::vector<std::uint8_t> before(input.data_bool(),
                                               input.data_bool() + size);
        const Tensor result = input.logical_not();
        check_logical_not_result(result, {size}, expected,
                                 "NEON logical_not size " +
                                     std::to_string(size));
        require(result.data_bool() != input.data_bool(),
                "NEON logical_not output aliases its input");
        require(std::vector<std::uint8_t>(input.data_bool(),
                                          input.data_bool() + size) == before,
                "NEON logical_not mutated its input");
    }

    check_logical_not_result(
        Tensor(std::vector<std::uint8_t>(32, 0), QuantizationMode::BOOL)
            .logical_not(),
        {32}, std::vector<std::uint8_t>(32, 1),
        "NEON all-false logical_not");
    check_logical_not_result(
        Tensor(std::vector<std::uint8_t>(32, 1), QuantizationMode::BOOL)
            .logical_not(),
        {32}, std::vector<std::uint8_t>(32, 0),
        "NEON all-true logical_not");

    for (const std::uint8_t value : {std::uint8_t{0}, std::uint8_t{1}}) {
        const Tensor scalar = Tensor::from_flat(
            std::vector<std::uint8_t>{value}, {}, QuantizationMode::BOOL);
        check_logical_not_result(
            scalar.logical_not(), {},
            {static_cast<std::uint8_t>(value == 0)},
            "NEON scalar logical_not");
    }

    const Tensor empty = Tensor::from_flat(
        std::vector<std::uint8_t>{}, {0, 3}, QuantizationMode::BOOL);
    check_logical_not_result(empty.logical_not(), {0, 3}, {},
                             "NEON empty logical_not");

    std::vector<std::uint8_t> parent_values(34);
    for (std::size_t i = 0; i < parent_values.size(); ++i) {
        parent_values[i] = static_cast<std::uint8_t>(i % 2);
    }
    const Tensor parent =
        Tensor::from_flat(parent_values, {2, 17}, QuantizationMode::BOOL);
    const Tensor view = parent.slice_first_axis(1, 2);
    require(view.storage_offset() == 17,
            "NEON logical_not test view has no storage offset");
    const Tensor view_result = view.logical_not();
    std::vector<std::uint8_t> view_expected(17);
    for (std::size_t i = 0; i < view_expected.size(); ++i) {
        view_expected[i] =
            static_cast<std::uint8_t>(parent_values[17 + i] == 0);
    }
    check_logical_not_result(view_result, {1, 17}, view_expected,
                             "NEON unaligned offset-view logical_not");
    require(view_result.data_bool() != view.data_bool(),
            "NEON offset-view logical_not output aliases its input");

    const Tensor repeated = view.logical_not();
    require(repeated.data_bool() != view_result.data_bool() &&
                std::vector<std::uint8_t>(repeated.data_bool(),
                                          repeated.data_bool() + repeated.numel()) ==
                    view_expected,
            "Repeated NEON logical_not calls reused storage or changed values");
    require(std::vector<std::uint8_t>(parent.data_bool(),
                                      parent.data_bool() + parent.numel()) ==
                parent_values,
            "NEON logical_not mutated its offset-view parent");
}

void check_neon_logical_and() {
    using labneura::QuantizationMode;
    using labneura::Tensor;

    labneura::test::ScopedEnvironmentVariable backend_override(
        "LABNEURA_BACKEND", "NEON");
    require(labneura::detect_backend() == "NEON",
            "NEON logical_and backend was not explicitly selected");
    auto backend = labneura::create_best_backend(1, QuantizationMode::BOOL);
    require(dynamic_cast<labneura::NEONBackend*>(backend.get()) != nullptr,
            "Explicit NEON logical_and selection did not create NEON");

    for (std::size_t size = 0; size <= 33; ++size) {
        std::vector<std::uint8_t> lhs_values(size);
        std::vector<std::uint8_t> rhs_values(size);
        std::vector<std::uint8_t> expected(size);
        for (std::size_t i = 0; i < size; ++i) {
            lhs_values[i] = static_cast<std::uint8_t>((i / 2) % 2);
            rhs_values[i] = static_cast<std::uint8_t>(i % 2);
            expected[i] =
                static_cast<std::uint8_t>(lhs_values[i] & rhs_values[i]);
        }
        const Tensor lhs(lhs_values, QuantizationMode::BOOL);
        const Tensor rhs(rhs_values, QuantizationMode::BOOL);
        const std::vector<std::uint8_t> lhs_before(
            lhs.data_bool(), lhs.data_bool() + size);
        const std::vector<std::uint8_t> rhs_before(
            rhs.data_bool(), rhs.data_bool() + size);
        const Tensor result = lhs.logical_and(rhs);
        require(labneura::detect_backend() == "NEON",
                "NEON logical_and dispatch changed backend");
        check_logical_and_result(result, {size}, expected,
                                 "NEON logical_and size " +
                                     std::to_string(size));
        require(result.data_bool() != lhs.data_bool() &&
                    result.data_bool() != rhs.data_bool(),
                "NEON logical_and output aliases an input");
        require(std::vector<std::uint8_t>(lhs.data_bool(),
                                          lhs.data_bool() + size) == lhs_before &&
                    std::vector<std::uint8_t>(rhs.data_bool(),
                                              rhs.data_bool() + size) ==
                        rhs_before,
                "NEON logical_and mutated an input");
    }

    const Tensor lhs = Tensor::from_flat(
        std::vector<std::uint8_t>{0, 1}, {2, 1}, QuantizationMode::BOOL);
    const Tensor rhs = Tensor::from_flat(
        std::vector<std::uint8_t>{1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1,
                                  0, 1},
        {1, 17}, QuantizationMode::BOOL);
    check_logical_and_result(
        lhs.logical_and(rhs), {2, 17},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1},
        "NEON broadcast logical_and");

    const Tensor scalar = Tensor::from_flat(
        std::vector<std::uint8_t>{1}, {}, QuantizationMode::BOOL);
    check_logical_and_result(scalar.logical_and(rhs), {1, 17},
                             {1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1,
                              0, 1},
                             "NEON scalar broadcast logical_and");
    check_logical_and_result(scalar.logical_and(scalar), {}, {1},
                             "NEON true scalar logical_and");
    check_logical_and_result(
        Tensor::from_flat(std::vector<std::uint8_t>{0}, {},
                          QuantizationMode::BOOL)
            .logical_and(scalar),
        {}, {0}, "NEON false scalar logical_and");

    std::vector<std::uint8_t> parent_values(34);
    for (std::size_t i = 0; i < parent_values.size(); ++i) {
        parent_values[i] = static_cast<std::uint8_t>(i % 2);
    }
    const Tensor parent =
        Tensor::from_flat(parent_values, {2, 17}, QuantizationMode::BOOL);
    const Tensor view = parent.slice_first_axis(1, 2);
    const Tensor ones(std::vector<std::uint8_t>(17, std::uint8_t{1}),
                      QuantizationMode::BOOL);
    const Tensor offset_result = view.logical_and(ones);
    check_logical_and_result(
        offset_result, {1, 17},
        std::vector<std::uint8_t>(parent_values.begin() + 17,
                                  parent_values.end()),
        "NEON offset-view logical_and");
    require(offset_result.data_bool() != view.data_bool(),
            "NEON offset-view logical_and output aliases its input");
}
#endif

#if defined(LABNEURA_TEST_AVX2)
void check_avx2_logical_not() {
    using labneura::QuantizationMode;
    using labneura::Tensor;

    labneura::test::ScopedEnvironmentVariable backend_override(
        "LABNEURA_BACKEND", "AVX2");
    require(labneura::detect_backend() == "AVX2",
            "AVX2 logical_not backend was not explicitly selected");
    const auto selected_backend =
        labneura::create_best_backend(1, QuantizationMode::BOOL);
    require(dynamic_cast<labneura::AVX2Backend*>(selected_backend.get()) !=
                nullptr,
            "Explicit AVX2 selection did not create an AVX2 backend");

    for (std::size_t size = 0; size <= 66; ++size) {
        std::vector<std::uint8_t> values(size);
        std::vector<std::uint8_t> expected(size);
        for (std::size_t i = 0; i < size; ++i) {
            values[i] = static_cast<std::uint8_t>(i % 2);
            expected[i] = static_cast<std::uint8_t>(values[i] == 0);
        }
        const Tensor input(values, QuantizationMode::BOOL);
        const std::vector<std::uint8_t> before(input.data_bool(),
                                               input.data_bool() + size);
        const Tensor result = input.logical_not();
        check_logical_not_result(result, {size}, expected,
                                 "AVX2 logical_not size " +
                                     std::to_string(size));
        require(result.data_bool() != input.data_bool(),
                "AVX2 logical_not output aliases its input");
        require(std::vector<std::uint8_t>(input.data_bool(),
                                          input.data_bool() + size) == before,
                "AVX2 logical_not mutated its input");
    }

    check_logical_not_result(
        Tensor(std::vector<std::uint8_t>(64, 0), QuantizationMode::BOOL)
            .logical_not(),
        {64}, std::vector<std::uint8_t>(64, 1),
        "AVX2 all-false logical_not");
    check_logical_not_result(
        Tensor(std::vector<std::uint8_t>(64, 1), QuantizationMode::BOOL)
            .logical_not(),
        {64}, std::vector<std::uint8_t>(64, 0),
        "AVX2 all-true logical_not");

    for (const std::uint8_t value : {std::uint8_t{0}, std::uint8_t{1}}) {
        const Tensor scalar = Tensor::from_flat(
            std::vector<std::uint8_t>{value}, {}, QuantizationMode::BOOL);
        check_logical_not_result(
            scalar.logical_not(), {},
            {static_cast<std::uint8_t>(value == 0)},
            "AVX2 scalar logical_not");
    }

    const Tensor empty = Tensor::from_flat(
        std::vector<std::uint8_t>{}, {0, 3}, QuantizationMode::BOOL);
    check_logical_not_result(empty.logical_not(), {0, 3}, {},
                             "AVX2 empty logical_not");

    std::vector<std::uint8_t> parent_values(66);
    for (std::size_t i = 0; i < parent_values.size(); ++i) {
        parent_values[i] = static_cast<std::uint8_t>(i % 2);
    }
    const Tensor parent =
        Tensor::from_flat(parent_values, {2, 33}, QuantizationMode::BOOL);
    const Tensor view = parent.slice_first_axis(1, 2);
    require(view.storage_offset() == 33,
            "AVX2 logical_not test view has no storage offset");
    const Tensor view_result = view.logical_not();
    std::vector<std::uint8_t> view_expected(33);
    for (std::size_t i = 0; i < view_expected.size(); ++i) {
        view_expected[i] =
            static_cast<std::uint8_t>(parent_values[33 + i] == 0);
    }
    check_logical_not_result(view_result, {1, 33}, view_expected,
                             "AVX2 unaligned offset-view logical_not");
    require(view_result.data_bool() != view.data_bool(),
            "AVX2 offset-view logical_not output aliases its input");

    const Tensor repeated = view.logical_not();
    require(repeated.data_bool() != view_result.data_bool() &&
                std::vector<std::uint8_t>(repeated.data_bool(),
                                          repeated.data_bool() + repeated.numel()) ==
                    view_expected,
            "Repeated AVX2 logical_not calls reused storage or changed values");
    require(std::vector<std::uint8_t>(parent.data_bool(),
                                      parent.data_bool() + parent.numel()) ==
                parent_values,
            "AVX2 logical_not mutated its offset-view parent");
}

void check_avx2_logical_and() {
    using labneura::QuantizationMode;
    using labneura::Tensor;

    labneura::test::ScopedEnvironmentVariable backend_override(
        "LABNEURA_BACKEND", "AVX2");
    require(labneura::detect_backend() == "AVX2",
            "AVX2 logical_and backend was not explicitly selected");
    auto backend = labneura::create_best_backend(1, QuantizationMode::BOOL);
    require(dynamic_cast<labneura::AVX2Backend*>(backend.get()) != nullptr,
            "Explicit AVX2 logical_and selection did not create AVX2");

    for (std::size_t size = 0; size <= 65; ++size) {
        std::vector<std::uint8_t> lhs_values(size);
        std::vector<std::uint8_t> rhs_values(size);
        std::vector<std::uint8_t> expected(size);
        for (std::size_t i = 0; i < size; ++i) {
            lhs_values[i] = static_cast<std::uint8_t>((i / 2) % 2);
            rhs_values[i] = static_cast<std::uint8_t>(i % 2);
            expected[i] =
                static_cast<std::uint8_t>(lhs_values[i] & rhs_values[i]);
        }
        const Tensor lhs(lhs_values, QuantizationMode::BOOL);
        const Tensor rhs(rhs_values, QuantizationMode::BOOL);
        const std::vector<std::uint8_t> lhs_before(
            lhs.data_bool(), lhs.data_bool() + size);
        const std::vector<std::uint8_t> rhs_before(
            rhs.data_bool(), rhs.data_bool() + size);
        const Tensor result = lhs.logical_and(rhs);
        require(labneura::detect_backend() == "AVX2",
                "AVX2 logical_and dispatch changed backend");
        check_logical_and_result(result, {size}, expected,
                                 "AVX2 logical_and size " +
                                     std::to_string(size));
        require(result.data_bool() != lhs.data_bool() &&
                    result.data_bool() != rhs.data_bool(),
                "AVX2 logical_and output aliases an input");
        require(std::vector<std::uint8_t>(lhs.data_bool(),
                                          lhs.data_bool() + size) == lhs_before &&
                    std::vector<std::uint8_t>(rhs.data_bool(),
                                              rhs.data_bool() + size) ==
                        rhs_before,
                "AVX2 logical_and mutated an input");
    }

    const Tensor lhs = Tensor::from_flat(
        std::vector<std::uint8_t>{0, 1}, {2, 1}, QuantizationMode::BOOL);
    const Tensor rhs = Tensor::from_flat(
        std::vector<std::uint8_t>{1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1,
                                  0, 1},
        {1, 17}, QuantizationMode::BOOL);
    check_logical_and_result(
        lhs.logical_and(rhs), {2, 17},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1},
        "AVX2 broadcast logical_and");

    const Tensor scalar = Tensor::from_flat(
        std::vector<std::uint8_t>{1}, {}, QuantizationMode::BOOL);
    check_logical_and_result(scalar.logical_and(rhs), {1, 17},
                             {1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1,
                              0, 1},
                             "AVX2 scalar broadcast logical_and");
    check_logical_and_result(scalar.logical_and(scalar), {}, {1},
                             "AVX2 true scalar logical_and");
    check_logical_and_result(
        Tensor::from_flat(std::vector<std::uint8_t>{0}, {},
                          QuantizationMode::BOOL)
            .logical_and(scalar),
        {}, {0}, "AVX2 false scalar logical_and");

    std::vector<std::uint8_t> parent_values(66);
    for (std::size_t i = 0; i < parent_values.size(); ++i) {
        parent_values[i] = static_cast<std::uint8_t>(i % 2);
    }
    const Tensor parent =
        Tensor::from_flat(parent_values, {2, 33}, QuantizationMode::BOOL);
    const Tensor view = parent.slice_first_axis(1, 2);
    const Tensor ones(std::vector<std::uint8_t>(33, std::uint8_t{1}),
                      QuantizationMode::BOOL);
    const Tensor offset_result = view.logical_and(ones);
    check_logical_and_result(
        offset_result, {1, 33},
        std::vector<std::uint8_t>(parent_values.begin() + 33,
                                  parent_values.end()),
        "AVX2 offset-view logical_and");
    require(offset_result.data_bool() != view.data_bool(),
            "AVX2 offset-view logical_and output aliases its input");
}
#endif

void check_bool_backend(const std::string& backend_name) {
    using labneura::QuantizationMode;
    using labneura::Tensor;

    labneura::test::ScopedEnvironmentVariable backend_override(
        "LABNEURA_BACKEND", backend_name);
    try {
        require(labneura::detect_backend() == backend_name,
                "Explicit backend selection did not match request");
    } catch (const std::invalid_argument& error) {
        if (backend_name != "GENERIC") {
            throw SkipTest(error.what());
        }
        throw;
    }

    constexpr std::size_t test_size = 17;
    auto backend =
        labneura::create_best_backend(test_size, QuantizationMode::BOOL);
    require(backend->size() == test_size,
            backend_name + ": backend changed logical BOOL size");
    require(backend->quantization_mode() == QuantizationMode::BOOL,
            backend_name + ": backend changed BOOL dtype");
    require(backend->allocation_capacity() >= test_size,
            backend_name + ": allocation capacity is smaller than logical size");
    const auto* backend_data = backend->data_bool();
    require(reinterpret_cast<std::uintptr_t>(backend_data) %
                    backend->storage_alignment() ==
                0,
            backend_name + ": BOOL allocation does not meet backend alignment");

    std::vector<std::uint8_t> noncanonical{0, 1, 2, 0, 255, 1, 0, 7, 0,
                                           1, 2, 0, 0, 1, 0, 255, 1};
    Tensor tensor(noncanonical, QuantizationMode::BOOL);
    require(tensor.quantization_mode() == QuantizationMode::BOOL &&
                tensor.shape() == std::vector<std::size_t>({test_size}) &&
                tensor.numel() == test_size && tensor.size() == test_size,
            backend_name + ": BOOL construction metadata mismatch");
    const std::vector<std::uint8_t> expected{0, 1, 1, 0, 1, 1, 0, 1, 0,
                                             1, 1, 0, 0, 1, 0, 1, 1};
    const Tensor& const_tensor = tensor;
    const std::uint8_t* const_data = const_tensor.data_bool();
    for (std::size_t i = 0; i < test_size; ++i) {
        require(const_data[i] == expected[i],
                backend_name + ": BOOL input was not normalized at index " +
                    std::to_string(i));
    }

    std::uint8_t* mutable_data = tensor.data_bool();
    mutable_data[0] = 1;
    mutable_data[1] = 0;
    require(const_tensor.data_bool()[0] == 1 &&
                const_tensor.data_bool()[1] == 0,
            backend_name + ": mutable BOOL accessor did not update storage");

    Tensor copy(tensor);
    require(copy.quantization_mode() == QuantizationMode::BOOL &&
                copy.shape() == tensor.shape() &&
                copy.data_bool()[0] == 1 && copy.data_bool()[1] == 0,
            backend_name + ": BOOL clone did not preserve dtype/data");
    copy.data_bool()[0] = 0;
    require(tensor.data_bool()[0] == 1,
            backend_name + ": BOOL copy shares source storage");

    Tensor assigned(std::vector<std::uint8_t>{1}, QuantizationMode::BOOL);
    assigned = tensor;
    assigned.data_bool()[2] = 0;
    require(tensor.data_bool()[2] == 1,
            backend_name + ": BOOL copy assignment shares source storage");

    std::uint8_t* invalidated_data = tensor.data_bool();
    invalidated_data[0] = 255;
    require(tensor.data_bool()[0] == 1,
            backend_name + ": mutable access did not canonicalize prior writes");
    invalidated_data[0] = 255;
    require_throws<std::runtime_error>(
        [&] { (void)const_tensor.data_bool(); },
        backend_name + ": const BOOL access accepted a noncanonical raw write");
    require(tensor.data_bool()[0] == 1,
            backend_name + ": mutable BOOL access failed to normalize raw data");

    Tensor empty(std::vector<std::uint8_t>{}, QuantizationMode::BOOL);
    Tensor empty_nd = Tensor::from_flat(
        std::vector<std::uint8_t>{}, {0, 3}, QuantizationMode::BOOL);
    require(empty.quantization_mode() == QuantizationMode::BOOL &&
                empty.size() == 0 && empty.numel() == 0 &&
                empty.data_bool() != nullptr,
            backend_name + ": empty BOOL vector construction failed");
    require(empty_nd.shape() == std::vector<std::size_t>({0, 3}) &&
                empty_nd.size() == 0 && empty_nd.numel() == 0 &&
                empty_nd.data_bool() != nullptr,
            backend_name + ": zero-sized nD BOOL construction failed");

    Tensor non_aligned(std::vector<std::uint8_t>{1, 0, 1, 1, 0},
                       QuantizationMode::BOOL);
    require(non_aligned.size() == 5 && non_aligned.data_bool()[4] == 0,
            backend_name + ": non-vector-aligned BOOL length failed");

    Tensor view_lifetime = [&] {
        Tensor parent = Tensor::from_flat(
            std::vector<std::uint8_t>{0, 1, 1, 0, 1, 0}, {3, 2},
            QuantizationMode::BOOL);
        Tensor view = parent.slice_first_axis(1, 3);
        require(view.storage_offset() == 2 &&
                    view.data_bool()[0] == 1 &&
                    view.data_bool()[3] == 0,
                backend_name + ": BOOL view offset or data mismatch");
        return view;
    }();
    require(view_lifetime.quantization_mode() == QuantizationMode::BOOL &&
                view_lifetime.data_bool()[0] == 1 &&
                view_lifetime.data_bool()[3] == 0,
            backend_name + ": BOOL view did not retain storage lifetime");

    require_throws<std::runtime_error>(
        [&] { tensor.add_inplace(copy); },
        backend_name + ": BOOL arithmetic was not rejected");
    require_throws<std::runtime_error>(
        [&] { (void)tensor.add(copy); },
        backend_name + ": allocating BOOL arithmetic was not rejected");
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Expected GENERIC, NEON, or AVX2\n";
        return 1;
    }
    const std::string backend = argv[1];
    try {
        if (backend == "GENERIC") {
            check_bool_backend(backend);
            check_generic_logical_not();
            check_generic_logical_and();
        } else if (backend == "NEON") {
#if !defined(LABNEURA_TEST_NEON)
            throw SkipTest("NEON backend was not compiled for this target");
#else
            if (!labneura::cpu_supports_neon()) {
                throw SkipTest("CPU does not support NEON");
            }
            check_bool_backend(backend);
            check_neon_logical_not();
            check_neon_logical_and();
#endif
        } else if (backend == "AVX2") {
#if !defined(LABNEURA_TEST_AVX2)
            throw SkipTest("AVX2 backend was not compiled for this target");
#else
            if (!labneura::cpu_supports_avx2()) {
                throw SkipTest("CPU/OS does not support AVX2 state");
            }
            check_bool_backend(backend);
            check_avx2_logical_not();
            check_avx2_logical_and();
#endif
        } else {
            throw std::invalid_argument("Unknown backend: " + backend);
        }
        std::cout << backend << ": BOOL dtype tests passed\n";
        return 0;
    } catch (const SkipTest& skip) {
        std::cout << "SKIP: " << skip.what() << '\n';
        return 77;
    } catch (const std::exception& error) {
        std::cerr << "BOOL dtype test failed: " << error.what() << '\n';
        return 1;
    }
}
