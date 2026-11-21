#include "labneura/runtime.h"

#include "runtime_internal.h"

#include <algorithm>
#include <atomic>
#include <charconv>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <sched.h>
#include <unistd.h>
#endif

namespace labneura {
namespace {

std::atomic<std::size_t> configured_thread_count{1};

#if defined(__APPLE__)
std::size_t sysctl_cpu_count(const char* name) noexcept {
    int value = 0;
    std::size_t length = sizeof(value);
    if (sysctlbyname(name, &value, &length, nullptr, 0) != 0 ||
        length != sizeof(value) || value <= 0) {
        return 0;
    }
    return static_cast<std::size_t>(value);
}
#elif defined(__linux__)
bool parse_positive_integer(const std::string& text,
                            std::uint64_t& value) noexcept {
    if (text.empty()) return false;
    const auto result =
        std::from_chars(text.data(), text.data() + text.size(), value);
    return result.ec == std::errc{} && result.ptr == text.data() + text.size() &&
           value > 0;
}

std::size_t cgroup_v2_cpu_limit() {
    std::ifstream input("/sys/fs/cgroup/cpu.max");
    std::string quota_text;
    std::string period_text;
    if (!(input >> quota_text >> period_text) || quota_text == "max") return 0;

    std::uint64_t quota = 0;
    std::uint64_t period = 0;
    if (!parse_positive_integer(quota_text, quota) ||
        !parse_positive_integer(period_text, period)) {
        return 0;
    }
    return static_cast<std::size_t>(std::max<std::uint64_t>(1, quota / period));
}

std::size_t cgroup_v1_cpu_limit() {
    std::ifstream quota_file("/sys/fs/cgroup/cpu/cpu.cfs_quota_us");
    std::ifstream period_file("/sys/fs/cgroup/cpu/cpu.cfs_period_us");
    std::int64_t quota = 0;
    std::uint64_t period = 0;
    if (!(quota_file >> quota) || !(period_file >> period) || quota <= 0 ||
        period == 0) {
        return 0;
    }
    return static_cast<std::size_t>(
        std::max<std::uint64_t>(1, static_cast<std::uint64_t>(quota) / period));
}

std::size_t linux_cpu_count() {
    std::size_t count = 0;

    cpu_set_t affinity;
    CPU_ZERO(&affinity);
    if (sched_getaffinity(0, sizeof(affinity), &affinity) == 0) {
        const int affinity_count = CPU_COUNT(&affinity);
        if (affinity_count > 0) {
            count = static_cast<std::size_t>(affinity_count);
        }
    }

    if (count == 0) {
        const long online = sysconf(_SC_NPROCESSORS_ONLN);
        if (online > 0) count = static_cast<std::size_t>(online);
    }

    const std::size_t quota = cgroup_v2_cpu_limit();
    const std::size_t legacy_quota =
        quota == 0 ? cgroup_v1_cpu_limit() : quota;
    if (legacy_quota > 0) count = count == 0 ? legacy_quota
                                              : std::min(count, legacy_quota);
    return count;
}
#endif

std::size_t detect_available_cpu_count() {
    std::size_t detected = 0;
#if defined(__APPLE__)
    detected = sysctl_cpu_count("hw.activecpu");
    if (detected == 0) detected = sysctl_cpu_count("hw.logicalcpu");
    if (detected == 0) detected = sysctl_cpu_count("hw.ncpu");
#elif defined(__linux__)
    detected = linux_cpu_count();
#endif
    return detail::resolve_cpu_count(detected,
                                     static_cast<std::size_t>(
                                         std::thread::hardware_concurrency()));
}

}  // namespace

namespace detail {

std::size_t resolve_cpu_count(std::size_t detected,
                              std::size_t fallback) noexcept {
    if (detected > 0) return detected;
    return fallback > 0 ? fallback : 1;
}

}  // namespace detail

std::size_t available_cpu_count() {
    static const std::size_t count = detect_available_cpu_count();
    return count;
}

std::size_t execution_thread_count() noexcept {
    return configured_thread_count.load(std::memory_order_relaxed);
}

void set_execution_thread_count(std::size_t count) {
    if (count == 0 || count > available_cpu_count()) {
        throw std::invalid_argument(
            "Execution thread count must be between 1 and available_cpu_count()");
    }
    configured_thread_count.store(count, std::memory_order_relaxed);
}

void set_execution_thread_count_auto() {
    configured_thread_count.store(available_cpu_count(),
                                  std::memory_order_relaxed);
}

}  // namespace labneura
