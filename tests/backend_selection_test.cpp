#include "labneura/backends/common/backend_factory.h"
#include "labneura/backends/common/cpu_features.h"
#include "labneura/backends/cpu/generic.h"
#include "backend_selection.h"
#include "cpu_features_internal.h"
#include "test_environment.h"

#if defined(LABNEURA_TEST_AVX2)
#include "labneura/backends/cpu/avx2.h"
#endif

#if defined(LABNEURA_TEST_NEON)
#include "labneura/backends/cpu/neon.h"
#endif

#include <atomic>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using labneura::detail::BackendKind;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_selection_error(const char* requested,
                             bool avx2_compiled,
                             bool avx2_supported,
                             bool neon_compiled,
                             bool neon_supported,
                             const std::string& expected) {
    try {
        labneura::detail::select_backend(requested, avx2_compiled,
                                         avx2_supported, neon_compiled,
                                         neon_supported);
    } catch (const std::invalid_argument& error) {
        require(error.what() == expected,
                "Unexpected backend selection error: " +
                    std::string(error.what()));
        return;
    }
    throw std::runtime_error("Expected backend selection to reject " +
                             std::string(requested));
}

void check_simulated_feature_detection() {
    require(labneura::detail::select_backend("AVX2", true, true, false, false) ==
                BackendKind::AVX2,
            "Supported explicit AVX2 selection failed");
    require(labneura::detail::select_backend("NEON", false, false, true, true) ==
                BackendKind::NEON,
            "Supported explicit NEON selection failed");

    require_selection_error("AVX2", true, false, false, false,
                            "Requested backend AVX2 is not supported by this CPU");
    require_selection_error("NEON", false, false, true, false,
                            "Requested backend NEON is not supported by this platform");
    require_selection_error("AVX2", false, false, false, false,
                            "Requested backend AVX2 is not available in this build");
    require_selection_error("OTHER", true, true, true, true,
                            "Invalid LABNEURA_BACKEND value: OTHER "
                            "(expected GENERIC, AVX2, or NEON)");

    require(labneura::detail::select_backend(nullptr, true, true, true, true) ==
                BackendKind::AVX2,
            "Default backend did not prefer AVX2");
    require(labneura::detail::select_backend(nullptr, false, false, true, true) ==
                BackendKind::NEON,
            "Default backend did not select NEON");
    require(labneura::detail::select_backend(nullptr, false, false, false, false) ==
                BackendKind::GENERIC,
            "Default backend did not fall back to Generic");
    require(labneura::detail::select_backend("GENERIC", true, true, true, true) ==
                BackendKind::GENERIC,
            "Explicit Generic selection failed");
}

std::string concrete_backend_name(const labneura::TensorBackend& backend) {
    if (dynamic_cast<const labneura::GenericBackend*>(&backend) != nullptr) {
        return "GENERIC";
    }
#if defined(LABNEURA_TEST_AVX2)
    if (dynamic_cast<const labneura::AVX2Backend*>(&backend) != nullptr) {
        return "AVX2";
    }
#endif
#if defined(LABNEURA_TEST_NEON)
    if (dynamic_cast<const labneura::NEONBackend*>(&backend) != nullptr) {
        return "NEON";
    }
#endif
    throw std::runtime_error("Factory returned an unrecognized backend");
}

void check_actual_factory() {
    using labneura::test::ScopedEnvironmentVariable;
    const auto mode = labneura::QuantizationMode::INT8;

    {
        ScopedEnvironmentVariable clear_override("LABNEURA_BACKEND", std::nullopt);
        const auto backend = labneura::create_best_backend(4, mode);
        require(labneura::detect_backend() == concrete_backend_name(*backend),
                "detect_backend disagrees with default backend factory result");
    }

    {
        ScopedEnvironmentVariable generic_override("LABNEURA_BACKEND", "GENERIC");
        const auto backend = labneura::create_best_backend(4, mode);
        require(concrete_backend_name(*backend) == "GENERIC" &&
                    labneura::detect_backend() == "GENERIC",
                "Generic override was not reflected in selection reporting");
    }

    const auto features = labneura::detect_cpu_features();
    require(features.avx2 == labneura::cpu_supports_avx2(),
            "AVX2 feature query helpers disagree");
#if defined(LABNEURA_TEST_AVX2)
    if (features.avx2) {
        ScopedEnvironmentVariable avx2_override("LABNEURA_BACKEND", "AVX2");
        const auto backend = labneura::create_best_backend(4, mode);
        require(concrete_backend_name(*backend) == "AVX2" &&
                    labneura::detect_backend() == "AVX2",
                "Available AVX2 override did not select AVX2");
    } else {
        ScopedEnvironmentVariable avx2_override("LABNEURA_BACKEND", "AVX2");
        bool rejected = false;
        try {
            static_cast<void>(labneura::create_best_backend(4, mode));
        } catch (const std::invalid_argument& error) {
            require(error.what() ==
                        std::string("Requested backend AVX2 is not supported by this CPU"),
                    "Unexpected error for unsupported AVX2 override");
            rejected = true;
        }
        require(rejected, "Unsupported AVX2 override was not rejected");
        rejected = false;
        try {
            static_cast<void>(labneura::detect_backend());
        } catch (const std::invalid_argument& error) {
            require(error.what() ==
                        std::string("Requested backend AVX2 is not supported by this CPU"),
                    "Backend reporting returned an unexpected AVX2 error");
            rejected = true;
        }
        require(rejected, "Backend reporting accepted unsupported AVX2");
    }
#else
    {
        ScopedEnvironmentVariable avx2_override("LABNEURA_BACKEND", "AVX2");
        bool rejected = false;
        try {
            static_cast<void>(labneura::create_best_backend(4, mode));
        } catch (const std::invalid_argument& error) {
            require(error.what() ==
                        std::string("Requested backend AVX2 is not available in this build"),
                    "Unexpected error for unavailable AVX2 backend");
            rejected = true;
        }
        require(rejected, "Unavailable AVX2 backend was not rejected");
    }
#endif

#if defined(LABNEURA_TEST_NEON)
    if (labneura::cpu_supports_neon()) {
        ScopedEnvironmentVariable neon_override("LABNEURA_BACKEND", "NEON");
        const auto backend = labneura::create_best_backend(4, mode);
        require(concrete_backend_name(*backend) == "NEON" &&
                    labneura::detect_backend() == "NEON",
                "Available NEON override did not select NEON");
    } else {
        ScopedEnvironmentVariable neon_override("LABNEURA_BACKEND", "NEON");
        bool rejected = false;
        try {
            static_cast<void>(labneura::create_best_backend(4, mode));
        } catch (const std::invalid_argument& error) {
            require(error.what() ==
                        std::string("Requested backend NEON is not supported by this platform"),
                    "Unexpected error for unsupported NEON override");
            rejected = true;
        }
        require(rejected, "Unsupported NEON override was not rejected");
        rejected = false;
        try {
            static_cast<void>(labneura::detect_backend());
        } catch (const std::invalid_argument& error) {
            require(error.what() ==
                        std::string("Requested backend NEON is not supported by this platform"),
                    "Backend reporting returned an unexpected NEON error");
            rejected = true;
        }
        require(rejected, "Backend reporting accepted unsupported NEON");
    }
#else
    {
        ScopedEnvironmentVariable neon_override("LABNEURA_BACKEND", "NEON");
        bool rejected = false;
        try {
            static_cast<void>(labneura::create_best_backend(4, mode));
        } catch (const std::invalid_argument& error) {
            require(error.what() ==
                        std::string("Requested backend NEON is not available in this build"),
                    "Unexpected error for unavailable NEON backend");
            rejected = true;
        }
        require(rejected, "Unavailable NEON backend was not rejected");
    }
#endif

    {
        ScopedEnvironmentVariable invalid_override("LABNEURA_BACKEND", "OTHER");
        try {
            static_cast<void>(labneura::create_best_backend(4, mode));
        } catch (const std::invalid_argument& error) {
            require(error.what() ==
                        std::string("Invalid LABNEURA_BACKEND value: OTHER "
                                    "(expected GENERIC, AVX2, or NEON)"),
                    "Unexpected error for invalid backend override");
            return;
        }
        throw std::runtime_error("Invalid backend override was not rejected");
    }
}

void check_feature_detection_is_cached() {
    using labneura::test::ScopedEnvironmentVariable;
    using labneura::detail::cpu_feature_detection_count;

    // This runs first, so detection is triggered here at most once per process.
    const auto first = labneura::detect_cpu_features();
    require(cpu_feature_detection_count() == 1,
            "Feature detection should have run exactly once");

    const bool avx2 = labneura::cpu_supports_avx2();
    const bool f16c = labneura::cpu_supports_f16c();
    const bool neon = labneura::cpu_supports_neon();
    require(avx2 == first.avx2 && f16c == first.f16c,
            "Cached feature helpers disagree with detect_cpu_features()");

    // Selecting a backend repeatedly, with changing overrides, must not re-detect
    // and must still honour the current LABNEURA_BACKEND value.
    for (int i = 0; i < 50; ++i) {
        {
            ScopedEnvironmentVariable generic("LABNEURA_BACKEND", "GENERIC");
            require(labneura::detect_backend() == "GENERIC",
                    "Override ignored after features were cached");
            const auto backend = labneura::create_best_backend(
                4, labneura::QuantizationMode::INT8);
            require(concrete_backend_name(*backend) == "GENERIC",
                    "Factory ignored GENERIC override after caching");
        }
        {
            ScopedEnvironmentVariable invalid("LABNEURA_BACKEND", "OTHER");
            bool rejected = false;
            try {
                static_cast<void>(labneura::detect_backend());
            } catch (const std::invalid_argument&) {
                rejected = true;
            }
            require(rejected, "Invalid override accepted after caching");
        }
        {
            ScopedEnvironmentVariable none("LABNEURA_BACKEND", std::nullopt);
            const auto name = labneura::detect_backend();
            require(name == "GENERIC" || name == "AVX2" || name == "NEON",
                    "Unexpected default backend name");
        }
    }
    require(cpu_feature_detection_count() == 1,
            "Backend selection re-ran CPU feature detection");
    require(labneura::cpu_supports_neon() == neon &&
                labneura::cpu_supports_avx2() == avx2,
            "Cached features changed between calls");
}

void check_concurrent_selection() {
    using labneura::test::ScopedEnvironmentVariable;
    using labneura::detail::cpu_feature_detection_count;

    // The environment is changed before threads start and restored after they
    // join: concurrent setenv/getenv would itself be a data race.
    ScopedEnvironmentVariable generic("LABNEURA_BACKEND", "GENERIC");
    const auto expected = labneura::detect_cpu_features();
    std::atomic<int> failures{0};
    std::vector<std::thread> threads;
    for (int t = 0; t < 8; ++t) {
        threads.emplace_back([&] {
            for (int i = 0; i < 500; ++i) {
                const auto features = labneura::detect_cpu_features();
                if (features.avx2 != expected.avx2 || features.f16c != expected.f16c ||
                    features.sse41 != expected.sse41 ||
                    labneura::detect_backend() != "GENERIC") {
                    ++failures;
                }
                const auto backend = labneura::create_best_backend(
                    8, labneura::QuantizationMode::FP32);
                if (!backend || backend->size() != 8) {
                    ++failures;
                }
            }
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }
    require(failures.load() == 0, "Concurrent backend selection produced inconsistent results");
    require(cpu_feature_detection_count() == 1,
            "Concurrent selection re-ran CPU feature detection");
}

}  // namespace

int main() {
    try {
        check_feature_detection_is_cached();
        check_simulated_feature_detection();
        check_actual_factory();
        check_concurrent_selection();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    std::cout << "Backend selection tests passed\n";
    return 0;
}
