#include "labneura/backends/common/backend_factory.h"
#include "labneura/backends/common/cpu_features.h"
#include "labneura/backends/cpu/generic.h"
#include <iostream>
#include <cstdlib>
#if defined(__AVX2__)
#include "labneura/backends/cpu/avx2.h"
#endif

#if defined(__ARM_NEON)
#include "labneura/backends/cpu/neon.h"
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

    // Optional override for testing via environment variable
    // LABNEURA_BACKEND can be set to "GENERIC", "NEON", or "AVX2"
    if (const char* override = std::getenv("LABNEURA_BACKEND")) {
        std::string value = override;
        if (value == "GENERIC") {
            return std::make_unique<GenericBackend>(size, mode);
        }
#if defined(__ARM_NEON)
        if (value == "NEON") {
            return std::make_unique<NEONBackend>(size, mode);
        }
#endif
#if defined(__AVX2__)
        if (value == "AVX2") {
            return std::make_unique<AVX2Backend>(size, mode);
        }
#endif
        // Fall through to auto-detection if value is unrecognized
    }


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
        return std::make_unique<NEONBackend>(size, mode);
#endif
    }
#endif

    return std::make_unique<GenericBackend>(size, mode);
}

} // namespace labneura