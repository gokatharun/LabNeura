#pragma once
#include "labneura/backends/base.h"

namespace labneura {

std::unique_ptr<TensorBackend>
create_best_backend(std::size_t size, QuantizationMode mode);

// Return the preferred backend name based on runtime CPU features
// Possible values: "AVX2", "NEON", "GENERIC"
std::string detect_backend();

} // namespace labneura