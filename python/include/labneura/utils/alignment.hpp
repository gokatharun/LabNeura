#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <new>
#include <stdexcept>

#if defined(_MSC_VER)
#include <malloc.h>
#else
#include <cerrno>
#endif

namespace labneura {
namespace util {

inline std::size_t checked_round_up(std::size_t value, std::size_t multiple) {
    if (multiple == 0) {
        throw std::invalid_argument("Alignment multiple must be non-zero");
    }

    const std::size_t remainder = value % multiple;
    if (remainder == 0) {
        return value;
    }

    const std::size_t increment = multiple - remainder;
    if (value > std::numeric_limits<std::size_t>::max() - increment) {
        throw std::bad_array_new_length();
    }
    return value + increment;
}

inline bool is_aligned(const void* p, std::size_t alignment) {
    return reinterpret_cast<std::uintptr_t>(p) % alignment == 0;
}

// Allocate 'count' elements of type T with the given alignment.
// Throws std::bad_array_new_length if the byte count overflows and
// std::bad_alloc if the aligned allocation cannot be satisfied.
template <typename T>
inline T* allocate_aligned(std::size_t count, std::size_t alignment = 64) {
    if (alignment < sizeof(void*) || (alignment & (alignment - 1)) != 0) {
        throw std::invalid_argument(
            "Alignment must be a power of two and at least sizeof(void*)");
    }
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
        throw std::bad_array_new_length();
    }

    // Keep zero-sized backend buffers non-null so empty-range pointer arithmetic
    // remains well-defined.
    const std::size_t bytes = count == 0 ? sizeof(T) : count * sizeof(T);
#if defined(_MSC_VER)
    void* p = _aligned_malloc(bytes, alignment);
    if (p == nullptr) {
        throw std::bad_alloc();
    }
#else
    void* p = nullptr;
    const int result = posix_memalign(&p, alignment, bytes);
    if (result == EINVAL) {
        throw std::invalid_argument("Invalid aligned allocation alignment");
    }
    if (result != 0) {
        throw std::bad_alloc();
    }
#endif
    return static_cast<T*>(p);
}

inline void aligned_free(void* p) {
#if defined(_MSC_VER)
    if (p) _aligned_free(p);
#else
    if (p) std::free(p);
#endif
}

// Typed helpers for unique_ptr function-pointer deleters
inline void free_aligned_double(double* p) { aligned_free(p); }
inline void free_aligned_float(float* p) { aligned_free(p); }
inline void free_aligned_int16(int16_t* p) { aligned_free(p); }
inline void free_aligned_int32(int32_t* p) { aligned_free(p); }
inline void free_aligned_int8(int8_t* p) { aligned_free(p); }

// Backend alignment constants
constexpr std::size_t AVX2_ALIGN = 32; // 256-bit
constexpr std::size_t SSE_ALIGN  = 16; // 128-bit
constexpr std::size_t NEON_ALIGN = 16; // 128-bit (safe)
constexpr std::size_t DEFAULT_ALIGN = 64; // Preferred heap alignment

} // namespace util
} // namespace labneura
