#include "labneura/backends/backend_factory.h"
#include "labneura/backends/cpu_features.h"
#include "labneura/backends/generic.h"
#include <iostream>
#if defined(__AVX2__)
#include "labneura/backends/avx2.h"
#endif

#if defined(__ARM_NEON)
#include "labneura/backends/neon.h"
#endif

namespace labneura {

// Return the preferred backend label based on runtime CPU features
std::string detect_backend() {
#if defined(__x86_64__) || defined(_M_X64)
    if (cpu_supports_avx2()) {
        return "AVX2";
    }
    return "GENERIC";
#elif defined(__aarch64__) || defined(__ARM_NEON)
    if (cpu_supports_neon()) {
        return "NEON";
    }
    return "GENERIC";
#else
    return "GENERIC";
#endif
}

std::unique_ptr<TensorBackend>
create_best_backend(std::size_t size, QuantizationMode mode) {

#if defined(__x86_64__) || defined(_M_X64)
    if (cpu_supports_avx2()) {
#if defined(__AVX2__)
        return std::make_unique<AVX2Backend>(size, mode);
#endif
    }
#endif

#if defined(__aarch64__) || defined(__ARM_NEON)
    if (cpu_supports_neon()) {
#if defined(__ARM_NEON)
        std::cout << "Using NEON backend" << std::endl;
        return std::make_unique<NEONBackend>(size, mode);
#endif
    }
#endif

    return std::make_unique<GenericBackend>(size, mode);
}

} // namespace labneura