#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>

namespace labneura {
namespace util {

inline bool is_aligned(const void* p, std::size_t alignment) {
    return reinterpret_cast<std::uintptr_t>(p) % alignment == 0;
}

// Allocate 'count' elements of type T with the given alignment.
// Returns nullptr on failure.
template <typename T>
inline T* allocate_aligned(std::size_t count, std::size_t alignment = 64) {
#if defined(_MSC_VER)
    void* p = _aligned_malloc(count * sizeof(T), alignment);
    return static_cast<T*>(p);
#else
    void* p = nullptr;
    if (posix_memalign(&p, alignment, count * sizeof(T)) == 0) {
        return static_cast<T*>(p);
    }
    // Fallback to malloc (not guaranteed alignment)
    return static_cast<T*>(std::malloc(count * sizeof(T)));
#endif
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
