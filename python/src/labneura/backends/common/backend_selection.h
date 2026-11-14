#pragma once

#include <stdexcept>
#include <string>

namespace labneura::detail {

enum class BackendKind {
    GENERIC,
    AVX2,
    NEON,
};

inline BackendKind select_backend(const char* requested,
                                 bool avx2_compiled,
                                 bool avx2_supported,
                                 bool neon_compiled,
                                 bool neon_supported) {
    if (requested != nullptr) {
        const std::string name(requested);
        if (name == "GENERIC") {
            return BackendKind::GENERIC;
        }
        if (name == "AVX2") {
            if (!avx2_compiled) {
                throw std::invalid_argument(
                    "Requested backend AVX2 is not available in this build");
            }
            if (!avx2_supported) {
                throw std::invalid_argument(
                    "Requested backend AVX2 is not supported by this CPU");
            }
            return BackendKind::AVX2;
        }
        if (name == "NEON") {
            if (!neon_compiled) {
                throw std::invalid_argument(
                    "Requested backend NEON is not available in this build");
            }
            if (!neon_supported) {
                throw std::invalid_argument(
                    "Requested backend NEON is not supported by this platform");
            }
            return BackendKind::NEON;
        }
        throw std::invalid_argument(
            "Invalid LABNEURA_BACKEND value: " + name +
            " (expected GENERIC, AVX2, or NEON)");
    }

    if (avx2_compiled && avx2_supported) {
        return BackendKind::AVX2;
    }
    if (neon_compiled && neon_supported) {
        return BackendKind::NEON;
    }
    return BackendKind::GENERIC;
}

inline const char* backend_name(BackendKind backend) {
    switch (backend) {
        case BackendKind::GENERIC: return "GENERIC";
        case BackendKind::AVX2: return "AVX2";
        case BackendKind::NEON: return "NEON";
    }
    throw std::logic_error("Unknown backend kind");
}

}  // namespace labneura::detail
