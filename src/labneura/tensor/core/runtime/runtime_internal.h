#pragma once

#include <cstddef>

namespace labneura::detail {

std::size_t resolve_cpu_count(std::size_t detected,
                              std::size_t fallback) noexcept;

}  // namespace labneura::detail
