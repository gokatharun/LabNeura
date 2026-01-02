#pragma once
#include "labneura/backends/base.h"

namespace labneura {

std::unique_ptr<TensorBackend>
create_best_backend(std::size_t size, QuantizationMode mode);

} // namespace labneura