#pragma once
#include "labneura/backends/common/base.h"

namespace labneura {

std::unique_ptr<TensorBackend>
create_best_backend(std::size_t size, QuantizationMode mode);

// Return the backend selected by runtime CPU features and LABNEURA_BACKEND.
// Possible values: "AVX2", "NEON", "GENERIC"
// Throws std::invalid_argument for an invalid or unavailable explicit override.
std::string detect_backend();

} // namespace labneura