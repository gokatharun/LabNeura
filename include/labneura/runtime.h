#pragma once

#include <cstddef>
#include <functional>

namespace labneura {

// Number of CPUs available to the current process, including OS affinity and
// CPU quota limits where the platform exposes them.
std::size_t available_cpu_count();

// Number of threads configured for LabNeura parallel_for execution. The
// initial value is one. Tensor operations do not yet use parallel_for.
std::size_t execution_thread_count() noexcept;

// Set the execution thread count. Counts must be in [1, available_cpu_count()];
// larger requests throw std::invalid_argument and leave the setting unchanged.
void set_execution_thread_count(std::size_t count);

// Select all CPUs reported as available to the current process.
void set_execution_thread_count_auto();

// Invoke function on non-overlapping [chunk_begin, chunk_end) ranges covering
// [begin, end). A zero grain size or reversed range throws std::invalid_argument.
// Empty ranges do no work. Ranges no larger than grain_size, or configurations
// with one thread, execute synchronously on the calling thread.
// Parallel callbacks may run concurrently and in any order. Concurrent callers
// are safe but their top-level parallel_for invocations are serialized. The
// configured count is sampled at call entry; later changes affect later calls.
void parallel_for(
    std::size_t begin, std::size_t end, std::size_t grain_size,
    const std::function<void(std::size_t, std::size_t)>& function);

}  // namespace labneura
