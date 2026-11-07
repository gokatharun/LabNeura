#include "labneura/backends/common/backend_factory.h"
#include "labneura/backends/common/cpu_features.h"
#include "labneura/detail/broadcast.h"
#include "labneura/tensor.h"
#include "labneura/types.hpp"
#include "test_environment.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

class SkipTest : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
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

void check_values(const labneura::Tensor& tensor,
                  const std::vector<float>& expected,
                  const std::string& context) {
    require(tensor.size() == expected.size(), context + ": numel mismatch");
    for (std::size_t i = 0; i < expected.size(); ++i) {
        require(std::fabs(tensor.data_fp32()[i] - expected[i]) < 1e-5f,
                context + ": value mismatch at index " + std::to_string(i));
    }
}

void check_layout(const labneura::Tensor& tensor,
                  const std::vector<std::size_t>& expected_shape,
                  const std::vector<std::size_t>& expected_strides,
                  const std::string& context,
                  std::size_t expected_offset = 0) {
    require(tensor.shape() == expected_shape,
            context + ": shape mismatch");
    require(tensor.strides() == expected_strides,
            context + ": strides mismatch");
    require(tensor.storage_offset() == expected_offset,
            context + ": storage offset mismatch");
    require(tensor.is_contiguous(),
            context + ": contiguous tensor reported non-contiguous");
    require(tensor.numel() == tensor.size(),
            context + ": numel differs from logical element count");
}

void check_broadcast_planner() {
    using labneura::detail::make_broadcast_plan;

    const auto same_shape = make_broadcast_plan(
        {2, 3}, {3, 1}, {2, 3}, {3, 1});
    require(same_shape.shape == std::vector<std::size_t>({2, 3}) &&
                same_shape.lhs_strides ==
                    std::vector<std::size_t>({3, 1}) &&
                same_shape.rhs_strides ==
                    std::vector<std::size_t>({3, 1}),
            "Broadcast planner changed same-shape operand strides");

    const auto vector_to_matrix = make_broadcast_plan(
        {2, 3}, {3, 1}, {3}, {1});
    require(vector_to_matrix.shape == std::vector<std::size_t>({2, 3}) &&
                vector_to_matrix.lhs_strides ==
                    std::vector<std::size_t>({3, 1}) &&
                vector_to_matrix.rhs_strides ==
                    std::vector<std::size_t>({0, 1}),
            "Broadcast planner mishandled missing leading dimensions");

    const auto multiple_dimensions = make_broadcast_plan(
        {4, 1, 3}, {3, 3, 1}, {1, 5, 3}, {15, 3, 1});
    require(multiple_dimensions.shape ==
                std::vector<std::size_t>({4, 5, 3}) &&
                multiple_dimensions.lhs_strides ==
                    std::vector<std::size_t>({3, 0, 1}) &&
                multiple_dimensions.rhs_strides ==
                    std::vector<std::size_t>({0, 3, 1}),
            "Broadcast planner mishandled multiple expanded dimensions");

    const auto scalar = make_broadcast_plan({}, {}, {2, 3}, {3, 1});
    require(scalar.shape == std::vector<std::size_t>({2, 3}) &&
                scalar.lhs_strides ==
                    std::vector<std::size_t>({0, 0}) &&
                scalar.rhs_strides ==
                    std::vector<std::size_t>({3, 1}),
            "Broadcast planner mishandled a scalar operand");
    const auto two_scalars = make_broadcast_plan({}, {}, {}, {});
    require(two_scalars.shape.empty() &&
                two_scalars.lhs_strides.empty() &&
                two_scalars.rhs_strides.empty(),
            "Broadcasting two scalars should produce a scalar plan");

    const auto empty = make_broadcast_plan(
        {0, 3}, {3, 1}, {1, 3}, {3, 1});
    require(empty.shape == std::vector<std::size_t>({0, 3}) &&
                empty.lhs_strides == std::vector<std::size_t>({3, 1}) &&
                empty.rhs_strides == std::vector<std::size_t>({0, 1}),
            "Broadcast planner mishandled a zero dimension");
    const std::size_t max_dimension =
        std::numeric_limits<std::size_t>::max();
    const auto empty_large_dimensions = make_broadcast_plan(
        {0, max_dimension, max_dimension}, {max_dimension, 1, 1},
        {0, max_dimension, max_dimension}, {max_dimension, 1, 1});
    require(empty_large_dimensions.shape ==
                std::vector<std::size_t>({0, max_dimension, max_dimension}),
            "Zero-sized broadcast shape incorrectly overflowed");

    const auto non_contiguous = make_broadcast_plan(
        {2, 3}, {1, 2}, {3}, {2});
    require(non_contiguous.shape == std::vector<std::size_t>({2, 3}) &&
                non_contiguous.lhs_strides ==
                    std::vector<std::size_t>({1, 2}) &&
                non_contiguous.rhs_strides ==
                    std::vector<std::size_t>({0, 2}),
            "Broadcast planner changed valid non-contiguous strides");

    auto source = labneura::Tensor::from_flat(
        std::vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, {4, 3});
    const auto view = source.slice_first_axis(1, 3);
    const auto original_data = std::vector<float>(
        source.data_fp32(), source.data_fp32() + source.size());
    const float* const source_pointer = source.data_fp32();
    const float* const view_pointer = view.data_fp32();
    const std::size_t view_offset = view.storage_offset();
    const auto view_plan = make_broadcast_plan(
        view.shape(), view.strides(), {3}, {1});
    require(view_plan.shape == std::vector<std::size_t>({2, 3}) &&
                view_plan.lhs_strides == view.strides() &&
                view_plan.rhs_strides == std::vector<std::size_t>({0, 1}) &&
                source.data_fp32() == source_pointer &&
                view.data_fp32() == view_pointer &&
                view.storage_offset() == view_offset &&
                std::vector<float>(
                    source.data_fp32(), source.data_fp32() + source.size()) ==
                    original_data,
            "Broadcast planning mutated or replaced a view's storage");

    require_throws<std::invalid_argument>(
        [] { (void)make_broadcast_plan({2, 3}, {3}, {3}, {1}); },
        "Left shape/stride rank mismatch");
    require_throws<std::invalid_argument>(
        [] { (void)make_broadcast_plan({2, 3}, {3, 1}, {4, 3}, {3, 1}); },
        "Incompatible broadcast dimensions");
    require_throws<std::overflow_error>(
        [] {
            const std::size_t max =
                std::numeric_limits<std::size_t>::max();
            (void)make_broadcast_plan({max, 2}, {2, 1}, {1, 2}, {2, 1});
        },
        "Broadcast output element-count overflow");
}

void check_first_axis_views(const std::string& backend) {
    std::vector<float> values(24);
    for (std::size_t i = 0; i < values.size(); ++i) {
        values[i] = static_cast<float>(i);
    }
    auto source = labneura::Tensor::from_flat(values, {4, 2, 3});

    auto slice = source.slice_first_axis(1, 3);
    check_layout(slice, {2, 2, 3}, {6, 3, 1},
                 backend + ": first-axis 3-D slice", 6);
    require(slice.storage_offset() == 6 &&
                slice.data_fp32() == source.data_fp32() + 6,
            backend + ": slice did not expose its logical storage offset");
    for (std::size_t i = 0; i < slice.size(); ++i) {
        require(slice.data_fp32()[i] == values[i + 6],
                backend + ": 3-D slice returned incorrect row-major values");
    }

    const auto full = source.slice_first_axis(0, 4);
    const auto empty = source.slice_first_axis(2, 2);
    const auto first = source.slice_first_axis(0, 1);
    const auto last = source.slice_first_axis(3, 4);
    require(full.storage_offset() == 0 && full.size() == source.size() &&
                first.storage_offset() == 0 && first.size() == 6 &&
                last.storage_offset() == 18 && last.size() == 6,
            backend + ": full, first, or last range slice metadata mismatch");
    require(full.data_fp32()[23] == values[23] &&
                first.data_fp32()[5] == values[5] &&
                last.data_fp32()[0] == values[18],
            backend + ": full, first, or last range slice values mismatch");
    require(empty.shape() == std::vector<std::size_t>({0, 2, 3}) &&
                empty.numel() == 0 && empty.storage_offset() == 12 &&
                empty.is_contiguous() &&
                empty.data_fp32() == source.data_fp32() + 12,
            backend + ": empty slice metadata or pointer mismatch");

    const auto vector = labneura::Tensor::from_flat(
        std::vector<float>{0, 1, 2, 3, 4, 5}, {6});
    const auto vector_slice = vector.slice_first_axis(1, 4);
    require(vector_slice.shape() == std::vector<std::size_t>({3}) &&
                vector_slice.storage_offset() == 1 &&
                vector_slice.data_fp32()[0] == 1.0f &&
                vector_slice.data_fp32()[2] == 3.0f,
            backend + ": 1-D range slice mismatch");

    const auto matrix = labneura::Tensor::from_flat(
        std::vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, {3, 4});
    const auto matrix_slice = matrix.slice_first_axis(1, 3);
    require(matrix_slice.shape() == std::vector<std::size_t>({2, 4}) &&
                matrix_slice.storage_offset() == 4 &&
                matrix_slice.data_fp32()[0] == 4.0f &&
                matrix_slice.data_fp32()[7] == 11.0f,
            backend + ": 2-D range slice mismatch");

    auto selected = source.select_first_axis(2);
    check_layout(selected, {2, 3}, {3, 1},
                 backend + ": first-axis selection", 12);
    require(selected.storage_offset() == 12 &&
                selected.data_fp32() == source.data_fp32() + 12,
            backend + ": selection did not advance to the selected sub-tensor");
    const auto first_selection = source.select_first_axis(0);
    const auto last_selection = source.select_first_axis(3);
    require(first_selection.storage_offset() == 0 &&
                first_selection.data_fp32()[0] == values[0] &&
                last_selection.storage_offset() == 18 &&
                last_selection.data_fp32()[5] == values[23],
            backend + ": first/last valid selection mismatch");

    auto reshaped_view = slice.reshape_view({4, 3});
    require(reshaped_view.shape() == std::vector<std::size_t>({4, 3}) &&
                reshaped_view.storage_offset() == 6 &&
                reshaped_view.data_fp32() == source.data_fp32() + 6,
            backend + ": reshape_view did not preserve a slice storage offset");

    auto nested = source.slice_first_axis(1, 4).slice_first_axis(1, 2);
    require(nested.shape() == std::vector<std::size_t>({1, 2, 3}) &&
                nested.storage_offset() == 12 &&
                nested.data_fp32() == source.data_fp32() + 12,
            backend + ": nested slicing computed the wrong offset");

    auto rhs = labneura::Tensor::from_flat(
        std::vector<float>(6, 10.0f), {2, 3});
    auto allocated = selected.add(rhs);
    require(allocated.data_fp32() != selected.data_fp32() &&
                allocated.storage_offset() == 0,
            backend + ": allocating operation on a view reused its storage");
    for (std::size_t i = 0; i < allocated.size(); ++i) {
        require(allocated.data_fp32()[i] == values[i + 12] + 10.0f,
                backend + ": allocating operation on a view returned bad values");
    }
    selected.add_inplace(rhs);
    for (std::size_t i = 0; i < source.size(); ++i) {
        const float expected = i >= 12 && i < 18
                                   ? values[i] + 10.0f
                                   : values[i];
        require(source.data_fp32()[i] == expected,
                backend + ": in-place slice operation changed the wrong range");
    }

    auto copied = slice;
    require(copied.storage_offset() == 0 &&
                copied.data_fp32() != slice.data_fp32(),
            backend + ": copying a view must create independent offset-zero storage");
    auto slice_rhs = labneura::Tensor::from_flat(
        std::vector<float>(slice.size(), 1.0f), {2, 2, 3});
    copied.add_inplace(slice_rhs);
    require(slice.data_fp32()[0] == values[6] &&
                copied.data_fp32()[0] == values[6] + 1.0f,
            backend + ": copied view mutation was not independent");

    auto surviving_view = [] {
        auto owner = labneura::Tensor::from_flat(
            std::vector<float>{0, 1, 2, 3, 4, 5}, {3, 2});
        return owner.slice_first_axis(1, 3);
    }();
    require(surviving_view.storage_offset() == 2 &&
                surviving_view.data_fp32()[0] == 2.0f &&
                surviving_view.data_fp32()[3] == 5.0f,
            backend + ": slice did not retain storage after source destruction");
    {
        auto temporary_view = source.slice_first_axis(0, 1);
        temporary_view.add_inplace(labneura::Tensor::from_flat(
            std::vector<float>(6, 1.0f), {1, 2, 3}));
    }
    require(source.data_fp32()[0] == 1.0f,
            backend + ": destroying a view invalidated its source storage");

    const auto scalar_source =
        labneura::Tensor::from_flat(
            std::vector<int>{7, 8, 9}, {3},
            labneura::QuantizationMode::INT8);
    auto scalar = scalar_source.select_first_axis(2);
    require(scalar.shape().empty() && scalar.strides().empty() &&
                scalar.storage_offset() == 2 && scalar.numel() == 1 &&
                scalar.data_int8()[0] == 9,
            backend + ": selecting a 1-D element did not produce a scalar view");

    const auto zero_sized = labneura::Tensor::from_flat(
        std::vector<float>{}, {4, 0, 3});
    const auto zero_first_axis = labneura::Tensor::from_flat(
        std::vector<float>{}, {0, 2, 3});
    auto empty_rows = zero_sized.slice_first_axis(1, 3);
    auto empty_selection = zero_sized.select_first_axis(3);
    require(empty_rows.shape() == std::vector<std::size_t>({2, 0, 3}) &&
                empty_rows.numel() == 0 &&
                empty_selection.shape() == std::vector<std::size_t>({0, 3}) &&
                empty_selection.numel() == 0,
            backend + ": zero-sized first-axis views have incorrect shapes");

    const auto scalar_tensor =
        labneura::Tensor::from_flat(std::vector<float>{1.0f}, {});
    require_throws<std::invalid_argument>(
        [&] { (void)scalar_tensor.slice_first_axis(0, 0); },
        backend + ": slicing a scalar must fail");
    require_throws<std::invalid_argument>(
        [&] { (void)scalar_tensor.select_first_axis(0); },
        backend + ": selecting a scalar must fail");
    require_throws<std::invalid_argument>(
        [&] { (void)source.slice_first_axis(3, 2); },
        backend + ": reversed first-axis range must fail");
    require_throws<std::out_of_range>(
        [&] { (void)source.slice_first_axis(0, 5); },
        backend + ": out-of-range slice stop must fail");
    require_throws<std::out_of_range>(
        [&] { (void)source.select_first_axis(4); },
        backend + ": out-of-range first-axis index must fail");
    require_throws<std::out_of_range>(
        [&] { (void)zero_first_axis.select_first_axis(0); },
        backend + ": zero-sized first axis cannot be selected");

    auto overlap_source = labneura::Tensor::from_flat(
        std::vector<float>{1, 2, 3, 4, 5, 6}, {6});
    auto overlap_lhs = overlap_source.slice_first_axis(0, 4);
    auto overlap_rhs = overlap_source.slice_first_axis(1, 5);
    require_throws<std::invalid_argument>(
        [&] { overlap_lhs.add_inplace(overlap_rhs); },
        backend + ": partially overlapping in-place views must fail");
    auto exact_alias = overlap_source.slice_first_axis(0, 4);
    exact_alias.add_inplace(exact_alias);
    require(overlap_source.data_fp32()[0] == 2.0f &&
                overlap_source.data_fp32()[3] == 8.0f &&
                overlap_source.data_fp32()[4] == 5.0f,
            backend + ": exact in-place view aliasing is incorrect");
    auto add_into_output = overlap_source.slice_first_axis(1, 5);
    const auto add_into_input = overlap_source.slice_first_axis(0, 4);
    const auto add_into_rhs = labneura::Tensor::from_flat(
        std::vector<float>(4, 1.0f), {4});
    require_throws<std::invalid_argument>(
        [&] { add_into_input.add_into(add_into_rhs, add_into_output); },
        backend + ": partially overlapping add_into output must fail");

    const std::vector<float> odd_values(3 * 33, 1.0f);
    auto odd_source = labneura::Tensor::from_flat(odd_values, {3, 33});
    auto odd_view = odd_source.select_first_axis(2);
    require(odd_view.storage_offset() == 66 && odd_view.size() == 33 &&
                odd_view.data_fp32() == odd_source.data_fp32() + 66,
            backend + ": odd-sized selection range mismatch");
    auto odd_rhs = labneura::Tensor::from_flat(
        std::vector<float>(33, 2.0f), {33});
    odd_view.add_inplace(odd_rhs);
    for (std::size_t i = 0; i < odd_source.size(); ++i) {
        require(odd_source.data_fp32()[i] == (i >= 66 && i < 99 ? 3.0f : 1.0f),
                backend + ": odd view operation crossed its logical range");
    }

    auto int8_source = labneura::Tensor::from_flat(
        std::vector<int>(3 * 33, 1), {3, 33},
        labneura::QuantizationMode::INT8);
    auto int8_view = int8_source.select_first_axis(2);
    auto int8_rhs = labneura::Tensor::from_flat(
        std::vector<int>(33, 2), {33}, labneura::QuantizationMode::INT8);
    int8_view.add_inplace(int8_rhs);
    for (std::size_t i = 0; i < int8_source.size(); ++i) {
        require(int8_source.data_int8()[i] == (i >= 66 && i < 99 ? 3 : 1),
                backend + ": INT8 slice operation crossed its logical range");
    }

    if (backend != "GENERIC") {
        for (const auto dtype : {labneura::QuantizationMode::FP16,
                                 labneura::QuantizationMode::INT16}) {
            auto typed_source = labneura::Tensor::from_flat(
                std::vector<int>(3 * 33, 1), {3, 33}, dtype);
            auto typed_view = typed_source.select_first_axis(2);
            auto typed_rhs = labneura::Tensor::from_flat(
                std::vector<int>(33, 2), {33}, dtype);
            typed_view.add_inplace(typed_rhs);
            const int16_t* typed_data =
                dtype == labneura::QuantizationMode::FP16
                    ? typed_source.data_fp16()
                    : typed_source.data_int16();
            for (std::size_t i = 0; i < typed_source.size(); ++i) {
                const int16_t expected = dtype == labneura::QuantizationMode::FP16
                                             ? static_cast<int16_t>(
                                                   labneura::quantize_fp16(
                                                       i >= 66 && i < 99 ? 3.0f : 1.0f))
                                             : static_cast<int16_t>(
                                                   i >= 66 && i < 99 ? 3 : 1);
                require(typed_data[i] == expected,
                        backend + ": typed slice operation crossed its logical range");
            }
        }
    }
}

void run_backend(const std::string& backend) {
    labneura::test::ScopedEnvironmentVariable override(
        "LABNEURA_BACKEND", backend);
    require(labneura::detect_backend() == backend,
            "Requested backend " + backend + " was not selected");
    check_first_axis_views(backend);

    const std::vector<float> values = [] {
        std::vector<float> result(24);
        for (std::size_t i = 0; i < result.size(); ++i) {
            result[i] = static_cast<float>(i + 1);
        }
        return result;
    }();
    const std::vector<std::size_t> shape{2, 3, 4};
    auto a = labneura::Tensor::from_flat(values, shape);
    auto b = labneura::Tensor::from_flat(
        std::vector<float>(values.size(), 2.0f), shape);

    check_layout(a, {2, 3, 4}, {12, 4, 1}, backend + ": 3-D layout");
    require(a.shape() == shape && a.ndim() == 3 && a.numel() == 24,
            backend + ": nD shape metadata mismatch");
    for (std::size_t i = 0; i < values.size(); ++i) {
        require(a.data_fp32()[i] == values[i],
                backend + ": flat row-major data order changed");
    }

    const std::vector<float> six_values(values.begin(), values.begin() + 6);
    const auto vector = labneura::Tensor::from_flat(six_values, {6});
    check_layout(vector, {6}, {1}, backend + ": 1-D layout");
    const auto matrix = vector.reshape({2, 3});
    check_layout(matrix, {2, 3}, {3, 1}, backend + ": 2-D reshape layout");
    require(matrix.shape() == std::vector<std::size_t>({2, 3}) &&
                matrix.data_fp32()[0] == 1.0f &&
                matrix.data_fp32()[5] == 6.0f,
            backend + ": 1-D to 2-D reshape changed shape or order");
    const auto three_dimensional = matrix.reshape({1, 2, 3});
    check_layout(three_dimensional, {1, 2, 3}, {6, 3, 1},
                 backend + ": singleton layout");
    require(three_dimensional.shape() ==
                std::vector<std::size_t>({1, 2, 3}) &&
                three_dimensional.data_fp32()[3] == 4.0f,
            backend + ": 2-D to 3-D reshape changed shape or order");
    const auto flattened = a.flatten();
    check_layout(flattened, {24}, {1}, backend + ": flattened layout");
    require(flattened.shape() == std::vector<std::size_t>({24}) &&
                flattened.numel() == a.numel(),
            backend + ": nD flatten shape or size mismatch");
    for (std::size_t i = 0; i < values.size(); ++i) {
        require(flattened.data_fp32()[i] == values[i],
                backend + ": flatten changed row-major order");
    }

    auto reshaped_copy = a.reshape(shape);
    reshaped_copy.data_fp32()[0] = -100.0f;
    require(a.data_fp32()[0] == values[0],
            backend + ": reshape result shares storage with its source");

    auto first_view = a.reshape_view({4, 6});
    auto second_view = a.reshape_view({2, 2, 6});
    check_layout(first_view, {4, 6}, {6, 1},
                 backend + ": first reshape view layout");
    check_layout(second_view, {2, 2, 6}, {12, 6, 1},
                 backend + ": second reshape view layout");
    require(first_view.quantization_mode() == a.quantization_mode() &&
                first_view.data_fp32() == a.data_fp32() &&
                second_view.data_fp32() == a.data_fp32(),
            backend + ": reshape views did not preserve dtype or share storage");
    first_view.data_fp32()[3] = 300.0f;
    require(a.data_fp32()[3] == 300.0f &&
                second_view.data_fp32()[3] == 300.0f,
            backend + ": view mutation was not shared across aliases");
    a.data_fp32()[4] = 400.0f;
    require(first_view.data_fp32()[4] == 400.0f &&
                second_view.data_fp32()[4] == 400.0f,
            backend + ": source mutation was not visible through views");
    a.data_fp32()[3] = values[3];
    a.data_fp32()[4] = values[4];
    auto copied_view = first_view;
    copied_view.data_fp32()[0] = -500.0f;
    require(first_view.data_fp32()[0] == values[0] &&
                a.data_fp32()[0] == values[0],
            backend + ": ordinary copy of a view did not deep-copy storage");
    reshaped_copy = a.reshape(shape);
    reshaped_copy.data_fp32()[0] = -600.0f;
    require(a.data_fp32()[0] == values[0],
            backend + ": reshape stopped being a deep copy after view creation");

    auto surviving_view = [] {
        auto temporary = labneura::Tensor::from_flat(
            std::vector<float>{5.0f, 6.0f, 7.0f, 8.0f}, {2, 2});
        return temporary.reshape_view({4});
    }();
    surviving_view.data_fp32()[2] = 700.0f;
    require(surviving_view.data_fp32()[2] == 700.0f,
            backend + ": view became invalid after its source was destroyed");

    const auto odd_sized_source = labneura::Tensor::from_flat(
        std::vector<float>(13, 1.0f), {13});
    auto odd_sized_view = odd_sized_source.reshape_view({1, 13});
    const auto odd_sized_rhs = labneura::Tensor::from_flat(
        std::vector<float>(13, 2.0f), {1, 13});
    auto odd_sized_result = odd_sized_view.add(odd_sized_rhs);
    require(odd_sized_result.size() == 13 &&
                odd_sized_result.data_fp32() != odd_sized_source.data_fp32(),
            backend + ": allocating view operation reused source storage");
    for (std::size_t i = 0; i < 13; ++i) {
        require(odd_sized_result.data_fp32()[i] == 3.0f &&
                    odd_sized_source.data_fp32()[i] == 1.0f,
                backend + ": allocating view operation changed an input or value");
    }
    odd_sized_result.data_fp32()[0] = 99.0f;
    require(odd_sized_source.data_fp32()[0] == 1.0f,
            backend + ": allocating view output aliases source storage");

    const auto odd_sized_product = odd_sized_view.mul(odd_sized_rhs);
    require(odd_sized_product.size() == 13 &&
                odd_sized_product.data_fp32() != odd_sized_source.data_fp32(),
            backend + ": cloned view operation reused source storage");
    for (std::size_t i = 0; i < 13; ++i) {
        require(odd_sized_product.data_fp32()[i] == 2.0f &&
                    odd_sized_source.data_fp32()[i] == 1.0f,
                backend + ": cloned view operation changed an input or value");
    }

    odd_sized_view.add_inplace(odd_sized_rhs);
    require(odd_sized_source.data_fp32()[12] == 3.0f,
            backend + ": backend operation did not respect view logical size");

    const auto added = a.add(b);
    check_layout(added, shape, {12, 4, 1}, backend + ": operation output layout");
    const auto subtracted = a.sub(b);
    const auto multiplied = a.mul(b);
    const auto divided = a.div(b);
    std::vector<float> expected(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        expected[i] = values[i] + 2.0f;
    }
    check_values(added, expected, backend + ": allocating add");
    for (std::size_t i = 0; i < values.size(); ++i) {
        expected[i] = values[i] - 2.0f;
    }
    check_values(subtracted, expected, backend + ": allocating sub");
    for (std::size_t i = 0; i < values.size(); ++i) {
        expected[i] = values[i] * 2.0f;
    }
    check_values(multiplied, expected, backend + ": allocating mul");
    for (std::size_t i = 0; i < values.size(); ++i) {
        expected[i] = values[i] / 2.0f;
    }
    check_values(divided, expected, backend + ": allocating div");

    auto in_place = labneura::Tensor::from_flat(values, shape);
    in_place.add_inplace(b);
    for (std::size_t i = 0; i < values.size(); ++i) expected[i] = values[i] + 2.0f;
    check_values(in_place, expected, backend + ": in-place add");
    in_place = labneura::Tensor::from_flat(values, shape);
    in_place.sub_inplace(b);
    for (std::size_t i = 0; i < values.size(); ++i) expected[i] = values[i] - 2.0f;
    check_values(in_place, expected, backend + ": in-place sub");
    in_place = labneura::Tensor::from_flat(values, shape);
    in_place.mul_inplace(b);
    for (std::size_t i = 0; i < values.size(); ++i) expected[i] = values[i] * 2.0f;
    check_values(in_place, expected, backend + ": in-place mul");
    in_place = labneura::Tensor::from_flat(values, shape);
    in_place.div_inplace(b);
    for (std::size_t i = 0; i < values.size(); ++i) expected[i] = values[i] / 2.0f;
    check_values(in_place, expected, backend + ": in-place div");

    auto output = labneura::Tensor::from_flat(
        std::vector<float>(values.size(), -1.0f), shape);
    check_layout(output, shape, {12, 4, 1}, backend + ": add_into output layout");
    a.add_into(b, output);
    for (std::size_t i = 0; i < values.size(); ++i) expected[i] = values[i] + 2.0f;
    check_values(output, expected, backend + ": add_into");

    const auto four_dimensional =
        labneura::Tensor::from_flat(values, {2, 2, 3, 2});
    const auto four_dimensional_twos = labneura::Tensor::from_flat(
        std::vector<float>(values.size(), 2.0f), {2, 2, 3, 2});
    check_values(four_dimensional.add(four_dimensional_twos), expected,
                 backend + ": four-dimensional add");
    std::vector<float> tail_values(29);
    for (std::size_t i = 0; i < tail_values.size(); ++i) {
        tail_values[i] = static_cast<float>(i + 1);
    }
    const auto tail_input = labneura::Tensor::from_flat(tail_values, {1, 29});
    const auto tail_ones = labneura::Tensor::from_flat(
        std::vector<float>(tail_values.size(), 1.0f), {1, 29});
    for (float& value : tail_values) value += 1.0f;
    check_values(tail_input.add(tail_ones), tail_values,
                 backend + ": vector-tail add");

    auto copy = a;
    check_layout(copy, shape, {12, 4, 1}, backend + ": copied layout");
    copy.data_fp32()[0] = -100.0f;
    require(a.data_fp32()[0] == values[0],
            backend + ": copy did not own independent storage");
    labneura::Tensor assigned;
    assigned = a;
    check_layout(assigned, shape, {12, 4, 1}, backend + ": assigned layout");
    assigned.data_fp32()[1] = -200.0f;
    require(a.data_fp32()[1] == values[1],
            backend + ": copy assignment did not own independent storage");
    labneura::Tensor moved(std::move(assigned));
    check_layout(moved, shape, {12, 4, 1}, backend + ": moved layout");

    auto same_numel_different_shape =
        labneura::Tensor::from_flat(values, {4, 6});
    require_throws<std::runtime_error>(
        [&] { (void)a.add(same_numel_different_shape); },
        backend + ": equal-numel shape mismatch");
    auto in_place_mismatch = labneura::Tensor::from_flat(values, shape);
    require_throws<std::runtime_error>(
        [&] { in_place_mismatch.add_inplace(same_numel_different_shape); },
        backend + ": in-place equal-numel shape mismatch");
    require_throws<std::invalid_argument>(
        [&] { a.add_into(b, same_numel_different_shape); },
        backend + ": add_into equal-numel shape mismatch");

    const auto empty = labneura::Tensor::from_flat(
        std::vector<float>{}, {2, 0, 7});
    check_layout(empty, {2, 0, 7}, {0, 7, 1},
                 backend + ": zero-sized layout");
    require(empty.shape() == std::vector<std::size_t>({2, 0, 7}) &&
                empty.numel() == 0,
            backend + ": zero-sized dimension semantics mismatch");
    const auto scalar = labneura::Tensor::from_flat(
        std::vector<float>{42.0f}, std::vector<std::size_t>{});
    check_layout(scalar, {}, {}, backend + ": scalar layout");
    require(scalar.shape().empty() && scalar.ndim() == 0 &&
                scalar.numel() == 1 && scalar.data_fp32()[0] == 42.0f,
            backend + ": scalar shape semantics mismatch");
    const auto reshaped_scalar = scalar.reshape({});
    const auto one_element_scalar = scalar.reshape({1});
    const auto flattened_scalar = scalar.flatten();
    check_layout(reshaped_scalar, {}, {}, backend + ": reshaped scalar layout");
    check_layout(one_element_scalar, {1}, {1},
                 backend + ": one-element scalar layout");
    check_layout(flattened_scalar, {1}, {1},
                 backend + ": flattened scalar layout");
    require(reshaped_scalar.shape().empty() &&
                reshaped_scalar.data_fp32()[0] == 42.0f &&
                one_element_scalar.shape() ==
                    std::vector<std::size_t>({1}) &&
                one_element_scalar.data_fp32()[0] == 42.0f &&
                flattened_scalar.shape() == std::vector<std::size_t>({1}) &&
                flattened_scalar.data_fp32()[0] == 42.0f,
            backend + ": scalar reshape or flatten semantics mismatch");
    const auto reshaped_empty = empty.reshape({4, 0, 9});
    const auto flattened_empty = empty.flatten();
    check_layout(reshaped_empty, {4, 0, 9}, {0, 9, 1},
                 backend + ": reshaped empty layout");
    check_layout(flattened_empty, {0}, {1},
                 backend + ": flattened empty layout");
    require(reshaped_empty.shape() ==
                std::vector<std::size_t>({4, 0, 9}) &&
                reshaped_empty.numel() == 0 &&
                flattened_empty.shape() == std::vector<std::size_t>({0}) &&
                flattened_empty.numel() == 0,
            backend + ": zero-sized reshape or flatten semantics mismatch");

    const auto integer_tensor = labneura::Tensor::from_flat(
        std::vector<int>{7, -3, 9, 1}, {2, 2},
        labneura::QuantizationMode::INT8);
    check_layout(integer_tensor, {2, 2}, {2, 1},
                 backend + ": integer factory layout");
    auto reshaped_int8 = integer_tensor.reshape({4});
    require(integer_tensor.shape() == std::vector<std::size_t>({2, 2}) &&
                integer_tensor.data_int8()[1] == -3 &&
                reshaped_int8.quantization_mode() ==
                    labneura::QuantizationMode::INT8 &&
                reshaped_int8.data_int8()[1] == -3,
            backend + ": integer factory construction mismatch");
    reshaped_int8.data_int8()[0] = 100;
    require(integer_tensor.data_int8()[0] == 7,
            backend + ": integer reshape result shares source storage");
    auto int8_view = integer_tensor.reshape_view({4});
    require(int8_view.quantization_mode() ==
                labneura::QuantizationMode::INT8 &&
                int8_view.data_int8() == integer_tensor.data_int8(),
            backend + ": INT8 reshape view did not preserve shared storage");
    int8_view.data_int8()[1] = -30;
    require(integer_tensor.data_int8()[1] == -30,
            backend + ": INT8 view mutation was not shared");
    const auto int8_vector_source = labneura::Tensor::from_flat(
        std::vector<int>(35, 3), {35}, labneura::QuantizationMode::INT8);
    auto int8_vector_view = int8_vector_source.reshape_view({5, 7});
    const auto int8_vector_rhs = labneura::Tensor::from_flat(
        std::vector<int>(35, 2), {5, 7}, labneura::QuantizationMode::INT8);
    int8_vector_view.mul_inplace(int8_vector_rhs);
    for (std::size_t i = 0; i < 35; ++i) {
        require(int8_vector_source.data_int8()[i] == 6,
                backend + ": INT8 view operation missed vector or scalar tail");
    }

    if (backend != "GENERIC") {
        const auto half = labneura::Tensor::from_flat(
            std::vector<float>{1.0f, -2.0f, 3.5f, 4.0f}, {2, 2},
            labneura::QuantizationMode::FP16);
        auto reshaped_half = half.reshape({4});
        require(reshaped_half.quantization_mode() ==
                    labneura::QuantizationMode::FP16 &&
                    reshaped_half.data_fp16()[2] == half.data_fp16()[2],
                backend + ": FP16 reshape did not preserve dtype/data");
        reshaped_half.data_fp16()[0] = 0;
        require(half.data_fp16()[0] != 0,
                backend + ": FP16 reshape result shares source storage");
        auto half_view = half.reshape_view({4});
        require(half_view.quantization_mode() ==
                    labneura::QuantizationMode::FP16 &&
                    half_view.data_fp16() == half.data_fp16(),
                backend + ": FP16 view did not preserve dtype or storage");
        auto half_vector_source = labneura::Tensor::from_flat(
            std::vector<float>(13, 1.25f), {13},
            labneura::QuantizationMode::FP16);
        auto half_vector_view = half_vector_source.reshape_view({1, 13});
        const auto half_vector_rhs = labneura::Tensor::from_flat(
            std::vector<float>(13, 0.5f), {1, 13},
            labneura::QuantizationMode::FP16);
        half_vector_view.add_inplace(half_vector_rhs);
        const auto expected_half = static_cast<int16_t>(
            labneura::quantize_fp16(1.75f));
        for (std::size_t i = 0; i < 13; ++i) {
            require(half_vector_source.data_fp16()[i] == expected_half,
                    backend + ": FP16 view operation missed vector or scalar tail");
        }

        const auto int16 = labneura::Tensor::from_flat(
            std::vector<int>{-10, 20, -30, 40}, {2, 2},
            labneura::QuantizationMode::INT16);
        auto reshaped_int16 = int16.reshape({4});
        require(reshaped_int16.quantization_mode() ==
                    labneura::QuantizationMode::INT16 &&
                    reshaped_int16.data_int16()[2] == -30,
                backend + ": INT16 reshape did not preserve dtype/data");
        reshaped_int16.data_int16()[0] = 99;
        require(int16.data_int16()[0] == -10,
                backend + ": INT16 reshape result shares source storage");
        auto int16_view = int16.reshape_view({4});
        require(int16_view.quantization_mode() ==
                    labneura::QuantizationMode::INT16 &&
                    int16_view.data_int16() == int16.data_int16(),
                backend + ": INT16 view did not preserve dtype or storage");
        const auto int16_vector_source = labneura::Tensor::from_flat(
            std::vector<int>(35, 3), {35},
            labneura::QuantizationMode::INT16);
        auto int16_vector_view = int16_vector_source.reshape_view({5, 7});
        const auto int16_vector_rhs = labneura::Tensor::from_flat(
            std::vector<int>(35, 2), {5, 7},
            labneura::QuantizationMode::INT16);
        int16_vector_view.add_inplace(int16_vector_rhs);
        for (std::size_t i = 0; i < 35; ++i) {
            require(int16_vector_source.data_int16()[i] == 5,
                    backend + ": INT16 view operation missed vector or scalar tail");
        }
    }

    const auto scalar_view = scalar.reshape_view({1});
    require(scalar_view.shape() == std::vector<std::size_t>({1}) &&
                scalar_view.data_fp32() == scalar.data_fp32(),
            backend + ": scalar reshape view did not share storage");
    const auto empty_view = empty.reshape_view({0, 14});
    check_layout(empty_view, {0, 14}, {14, 1},
                 backend + ": zero-sized reshape view layout", 0);
}

void check_invalid_inputs() {
    using labneura::QuantizationMode;
    const auto max_size = std::numeric_limits<std::size_t>::max();
    require_throws<std::invalid_argument>(
        [] {
            (void)labneura::Tensor::from_flat(
                std::vector<float>{1.0f, 2.0f}, {3});
        },
        "shape/data length mismatch");
    const auto reshape_source = labneura::Tensor::from_flat(
        std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
    require_throws<std::invalid_argument>(
        [&] { (void)reshape_source.reshape({3, 2}); },
        "reshape element count mismatch");
    require_throws<std::invalid_argument>(
        [&] { (void)reshape_source.reshape_view({3, 2}); },
        "reshape view element count mismatch");
    require_throws<std::overflow_error>(
        [&] {
            (void)reshape_source.reshape(
                {std::numeric_limits<std::size_t>::max(), 2});
        },
        "reshape element count overflow");
    require_throws<std::overflow_error>(
        [&] {
            (void)reshape_source.reshape_view(
                {std::numeric_limits<std::size_t>::max(), 2});
        },
        "reshape view element count overflow");
    require_throws<std::invalid_argument>(
        [] {
            (void)labneura::Tensor::from_flat(
                std::vector<float>{}, std::vector<std::size_t>{});
        },
        "empty scalar data");

    require_throws<std::overflow_error>(
        [&] {
            (void)labneura::Tensor::from_flat(
                std::vector<float>{}, {0, max_size, max_size});
        },
        "row-major stride overflow");
    require_throws<std::overflow_error>(
        [&] {
            (void)labneura::Tensor::from_flat(
                std::vector<float>{}, {max_size, 2});
        },
        "element count overflow");
    require_throws<std::overflow_error>(
        [&] {
            (void)labneura::Tensor::from_flat(
                std::vector<float>{}, {max_size / sizeof(float) + 1});
        },
        "byte size overflow");
    require_throws<std::invalid_argument>(
        [] {
            (void)labneura::Tensor::from_flat(
                std::vector<float>{1.0f}, std::vector<std::size_t>{},
                QuantizationMode::FP64);
        },
        "unsupported dtype");

    const auto empty_with_large_dimensions =
        labneura::Tensor::from_flat(
            std::vector<float>{}, {max_size, 0, max_size});
    require(empty_with_large_dimensions.numel() == 0,
            "zero dimension should yield zero elements");
    require(empty_with_large_dimensions.strides() ==
                std::vector<std::size_t>({0, max_size, 1}) &&
                empty_with_large_dimensions.storage_offset() == 0 &&
                empty_with_large_dimensions.is_contiguous(),
            "large zero-sized tensor layout mismatch");

    const labneura::Tensor default_tensor;
    check_layout(default_tensor, {0}, {1}, "default constructor layout");
    const labneura::Tensor float_vector(std::vector<float>{1.0f, 2.0f});
    check_layout(float_vector, {2}, {1}, "float vector constructor layout");
    const labneura::Tensor int_vector(std::vector<int>{1, 2, 3});
    check_layout(int_vector, {3}, {1}, "int vector constructor layout");
    const labneura::Tensor rows(
        std::vector<std::vector<float>>{{1.0f, 2.0f}, {3.0f, 4.0f}});
    check_layout(rows, {2, 2}, {2, 1}, "nested rows constructor layout");
    const labneura::Tensor float_scalar(1.0f);
    check_layout(float_scalar, {1}, {1}, "float scalar constructor layout");
    const labneura::Tensor int_scalar(1);
    check_layout(int_scalar, {1}, {1}, "int scalar constructor layout");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::invalid_argument("Expected GENERIC, AVX2 or NEON");
        const std::string backend = argv[1];
        check_broadcast_planner();
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
        check_invalid_inputs();
        std::cout << backend << ": nD Tensor tests passed\n";
        return 0;
    } catch (const SkipTest& skip) {
        std::cout << "SKIP: " << skip.what() << '\n';
        return 77;
    } catch (const std::exception& error) {
        std::cerr << "nD Tensor test failed: " << error.what() << '\n';
        return 1;
    }
}
