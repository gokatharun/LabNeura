#pragma once

#include "labneura/tensor.h"
#include <cstddef>
#include <cstdint>

namespace labneura {

// Implemented in avx2_ops_fp16_f16c.cpp, the only translation unit built with
// -mf16c. Callers must have verified F16C support at runtime.
void avx2_fp16_f16c_kernel(int16_t* dst, const int16_t* src, std::size_t size,
                           OperationType op_type);

}  // namespace labneura
