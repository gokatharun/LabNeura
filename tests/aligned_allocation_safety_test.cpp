#include "labneura/utils/alignment.hpp"
#include "labneura/backends/common/tensor_storage.h"

#if defined(LABNEURA_TEST_AVX2)
#include "labneura/backends/common/cpu_features.h"
#include "labneura/backends/cpu/avx2.h"
#endif
#if defined(LABNEURA_TEST_NEON)
#include "labneura/backends/cpu/neon.h"
#endif
#include "labneura/backends/cpu/generic.h"

#include "labneura/types.hpp"

#include <array>
#include <cstddef>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

std::size_t storage_deletions = 0;

void counted_free(void* pointer) {
    ++storage_deletions;
    labneura::util::aligned_free(pointer);
}

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Exception, typename Function>
void require_throws(Function&& function, const std::string& message) {
    try {
        function();
    } catch (const Exception&) {
        return;
    } catch (const std::exception& error) {
        throw std::runtime_error(message + ": unexpected exception: " +
                                 error.what());
    }
    throw std::runtime_error(message + ": no exception was thrown");
}

void check_tensor_storage() {
    auto empty = labneura::TensorStorage::allocate(
        0, 0, labneura::QuantizationMode::FP32,
        labneura::util::DEFAULT_ALIGN);
    require(empty->logical_size == 0 &&
                empty->allocation_capacity == 0 &&
                empty->dtype == labneura::QuantizationMode::FP32 &&
                empty->data() != nullptr &&
                labneura::util::is_aligned(
                    empty->data(), labneura::util::DEFAULT_ALIGN),
            "Zero-sized storage metadata or alignment is incorrect");

    auto padded = labneura::TensorStorage::allocate(
        9, 16, labneura::QuantizationMode::FP32,
        labneura::util::AVX2_ALIGN);
    require(padded->logical_size == 9 &&
                padded->allocation_capacity == 16 &&
                padded->dtype == labneura::QuantizationMode::FP32 &&
                labneura::util::is_aligned(
                    padded->data(), labneura::util::AVX2_ALIGN),
            "Padded storage metadata or alignment is incorrect");

    void* allocation =
        labneura::util::allocate_aligned<std::uint8_t>(64, 64);
    require(allocation != nullptr, "Test storage allocation failed");
    storage_deletions = 0;
    auto owner = labneura::TensorStorage::adopt(
        allocation, 4, 8, labneura::QuantizationMode::FP32, 64,
        counted_free);
    auto retained_owner = owner;
    owner.reset();
    require(storage_deletions == 0,
            "Shared storage was released while an owner remained");
    retained_owner.reset();
    require(storage_deletions == 1,
            "Shared storage was not deallocated exactly once");

    labneura::GenericBackend generic(
        9, labneura::QuantizationMode::FP32);
    require(generic.size() == 9 && generic.allocation_capacity() == 9 &&
                generic.storage_alignment() == labneura::util::DEFAULT_ALIGN &&
                labneura::util::is_aligned(
                    generic.data_fp32(), generic.storage_alignment()),
            "Generic backend storage metadata is inconsistent");
}

void check_checked_round_up() {
    require(labneura::util::checked_round_up(0, 8) == 0,
            "Rounding zero should remain zero");
    require(labneura::util::checked_round_up(8, 8) == 8,
            "An aligned size should remain unchanged");
    require(labneura::util::checked_round_up(9, 8) == 16,
            "An unaligned size should round to the next multiple");
    require_throws<std::invalid_argument>(
        [] { labneura::util::checked_round_up(1, 0); },
        "A zero rounding multiple must be rejected");

    const std::size_t maximum = std::numeric_limits<std::size_t>::max();
    require_throws<std::bad_array_new_length>(
        [maximum] { labneura::util::checked_round_up(maximum, 8); },
        "Rounding overflow must be rejected");
}

void check_aligned_allocator() {
    float* empty = labneura::util::allocate_aligned<float>(0, 32);
    require(empty != nullptr, "Zero-sized allocation must return valid storage");
    require(labneura::util::is_aligned(empty, 32),
            "Zero-sized allocation must retain the requested alignment");
    labneura::util::aligned_free(empty);

    for (const std::size_t count : {1U, 7U, 8U, 9U, 31U, 32U, 33U}) {
        float* allocation = labneura::util::allocate_aligned<float>(count, 32);
        require(allocation != nullptr, "Normal allocation returned null");
        require(labneura::util::is_aligned(allocation, 32),
                "Allocation did not satisfy the requested alignment");
        labneura::util::aligned_free(allocation);
    }

    require_throws<std::bad_array_new_length>(
        [] {
            labneura::util::allocate_aligned<float>(
                std::numeric_limits<std::size_t>::max() / sizeof(float) + 1);
        },
        "Allocation byte-size overflow must be rejected");
    require_throws<std::invalid_argument>(
        [] { labneura::util::allocate_aligned<float>(1, 3); },
        "Invalid alignment must be rejected");

    require_throws<std::bad_alloc>(
        [] {
            labneura::util::allocate_aligned<std::uint8_t>(
                std::numeric_limits<std::size_t>::max());
        },
        "An unsatisfiable allocation must report allocation failure");
}

#if defined(LABNEURA_TEST_AVX2)
bool check_avx2_constructors() {
    if (!labneura::cpu_supports_avx2()) {
        return false;
    }

    const std::array<std::size_t, 13> sizes = {
        0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64};
    const std::array<labneura::QuantizationMode, 4> dtypes = {
        labneura::QuantizationMode::FP32,
        labneura::QuantizationMode::FP16,
        labneura::QuantizationMode::INT16,
        labneura::QuantizationMode::INT8};

    for (const auto dtype : dtypes) {
        for (const std::size_t size : sizes) {
            labneura::AVX2Backend backend(size, dtype);
            require(backend.size() == size,
                    "AVX2 backend changed the logical element count");
            require(backend.allocation_capacity() ==
                        labneura::util::checked_round_up(size, 8) &&
                        backend.storage_alignment() ==
                            labneura::util::AVX2_ALIGN,
                    "AVX2 storage capacity or alignment metadata is incorrect");

            const void* data = nullptr;
            switch (dtype) {
                case labneura::QuantizationMode::FP32:
                    data = backend.data_fp32();
                    break;
                case labneura::QuantizationMode::FP16:
                    data = backend.data_fp16();
                    break;
                case labneura::QuantizationMode::INT16:
                    data = backend.data_int16();
                    break;
                case labneura::QuantizationMode::INT8:
                    data = backend.data_int8();
                    break;
                default:
                    throw std::runtime_error("Unexpected AVX2 test dtype");
            }
            require(data != nullptr, "AVX2 storage must be non-null for size zero");
            require(labneura::util::is_aligned(
                        data, labneura::util::AVX2_ALIGN),
                    "AVX2 storage must satisfy the SIMD alignment");
        }
    }

    require_throws<std::bad_array_new_length>(
        [] {
            labneura::AVX2Backend backend(
                std::numeric_limits<std::size_t>::max(),
                labneura::QuantizationMode::FP32);
        },
        "AVX2 padded-size overflow must be rejected");
    return true;
}
#endif

#if defined(LABNEURA_TEST_NEON)
void check_neon_constructors() {
    const std::array<std::size_t, 9> sizes = {
        0, 1, 7, 8, 9, 15, 16, 17, 31};
    for (const std::size_t size : sizes) {
        labneura::NEONBackend backend(
            size, labneura::QuantizationMode::FP32);
        require(backend.size() == size &&
                    backend.allocation_capacity() ==
                        labneura::util::checked_round_up(size, 8) &&
                    backend.storage_alignment() ==
                        labneura::util::NEON_ALIGN &&
                    labneura::util::is_aligned(
                        backend.data_fp32(), labneura::util::NEON_ALIGN),
                "NEON storage capacity or alignment metadata is incorrect");
    }
}
#endif

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) {
            throw std::runtime_error("Expected UTILITY or AVX2 test mode");
        }
        const std::string mode(argv[1]);
        if (mode == "UTILITY") {
            check_tensor_storage();
            check_checked_round_up();
            check_aligned_allocator();
#if defined(LABNEURA_TEST_NEON)
            check_neon_constructors();
#endif
            return 0;
        }
        if (mode == "AVX2") {
#if defined(LABNEURA_TEST_AVX2)
            return check_avx2_constructors() ? 0 : 77;
#else
            return 77;
#endif
        }
        throw std::runtime_error("Unknown test mode: " + mode);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
