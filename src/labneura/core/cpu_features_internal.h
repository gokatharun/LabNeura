#pragma once

#include <cstddef>

namespace labneura::detail {

// Internal test seam (not installed): how many times hardware feature
// detection has actually executed in this process.
std::size_t cpu_feature_detection_count();

}  // namespace labneura::detail
