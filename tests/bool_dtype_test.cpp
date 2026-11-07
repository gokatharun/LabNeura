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
        } else if (backend == "NEON") {
#if !defined(LABNEURA_TEST_NEON)
            throw SkipTest("NEON backend was not compiled for this target");
#else
            if (!labneura::cpu_supports_neon()) {
                throw SkipTest("CPU does not support NEON");
            }
            check_bool_backend(backend);
#endif
        } else if (backend == "AVX2") {
#if !defined(LABNEURA_TEST_AVX2)
            throw SkipTest("AVX2 backend was not compiled for this target");
#else
            if (!labneura::cpu_supports_avx2()) {
                throw SkipTest("CPU/OS does not support AVX2 state");
            }
            check_bool_backend(backend);
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
