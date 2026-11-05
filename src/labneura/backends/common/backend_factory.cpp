#include "labneura/backends/common/backend_factory.h"
#include "backend_selection.h"
#include "labneura/backends/common/cpu_features.h"
#include "labneura/backends/common/dtype_validation.h"
#include "labneura/backends/cpu/generic.h"
#include <cstdlib>
#if defined(LABNEURA_ENABLE_AVX2)
#include "labneura/backends/cpu/avx2.h"
#endif

#if defined(LABNEURA_ENABLE_NEON)
#include "labneura/backends/cpu/neon.h"
#endif

namespace labneura {

namespace {

detail::BackendKind selected_backend() {
    const CPUFeatures features = detect_cpu_features();
    const bool neon_supported = cpu_supports_neon();
#if defined(LABNEURA_ENABLE_AVX2)
    constexpr bool avx2_compiled = true;
#else
    constexpr bool avx2_compiled = false;
#endif
#if defined(LABNEURA_ENABLE_NEON)
    constexpr bool neon_compiled = true;
#else
    constexpr bool neon_compiled = false;
#endif
    return detail::select_backend(std::getenv("LABNEURA_BACKEND"),
                                  avx2_compiled, features.avx2,
                                  neon_compiled, neon_supported);
}

}  // namespace

std::string detect_backend() {
    return detail::backend_name(selected_backend());
}

std::unique_ptr<TensorBackend>
create_best_backend(std::size_t size, QuantizationMode mode) {
    validate_tensor_dtype_supported(mode);
    switch (selected_backend()) {
        case detail::BackendKind::GENERIC:
            return std::make_unique<GenericBackend>(size, mode);
        case detail::BackendKind::NEON:
#if defined(LABNEURA_ENABLE_NEON)
            return std::make_unique<NEONBackend>(size, mode);
#else
            break;
#endif
        case detail::BackendKind::AVX2:
#if defined(LABNEURA_ENABLE_AVX2)
            return std::make_unique<AVX2Backend>(size, mode);
#else
            break;
#endif
    }
    return std::make_unique<GenericBackend>(size, mode);
}

} // namespace labneura