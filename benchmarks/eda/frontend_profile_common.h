#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace labneura::eda::benchmark {

struct TimingSummary {
    double minimum_seconds = 0.0;
    double median_seconds = 0.0;
    double p95_seconds = 0.0;
    double iqr_seconds = 0.0;
};

inline double percentile(const std::vector<double>& sorted, double fraction) {
    const auto index = static_cast<std::size_t>(
        std::ceil(fraction * static_cast<double>(sorted.size()))) - 1;
    return sorted[std::min(index, sorted.size() - 1)];
}

template <typename Function>
TimingSummary measure(std::size_t warmups, std::size_t repetitions,
                      Function&& function) {
    std::vector<double> samples;
    samples.reserve(repetitions);
    for (std::size_t iteration = 0; iteration < warmups + repetitions;
         ++iteration) {
        const auto start = std::chrono::steady_clock::now();
        function();
        const auto stop = std::chrono::steady_clock::now();
        if (iteration >= warmups) {
            samples.push_back(
                std::chrono::duration<double>(stop - start).count());
        }
    }
    std::sort(samples.begin(), samples.end());
    return {samples.front(), percentile(samples, 0.50),
            percentile(samples, 0.95),
            percentile(samples, 0.75) - percentile(samples, 0.25)};
}

inline std::string json_escape(std::string_view value) {
    std::ostringstream result;
    for (const unsigned char character : value) {
        switch (character) {
            case '\\': result << "\\\\"; break;
            case '"': result << "\\\""; break;
            case '\n': result << "\\n"; break;
            case '\r': result << "\\r"; break;
            case '\t': result << "\\t"; break;
            default:
                if (character < 0x20)
                    result << "\\u" << std::hex << std::setw(4)
                           << std::setfill('0') << static_cast<int>(character)
                           << std::dec;
                else
                    result << character;
        }
    }
    return result.str();
}

// Stable non-cryptographic content identity. The workload generator also
// records SHA-256; this hash lets the standalone C++ runner detect accidental
// input changes without adding a mandatory crypto dependency.
inline std::string fnv1a64(std::string_view value) {
    std::uint64_t hash = 14695981039346656037ULL;
    for (const unsigned char character : value) {
        hash ^= character;
        hash *= 1099511628211ULL;
    }
    std::ostringstream result;
    result << std::hex << std::setw(16) << std::setfill('0') << hash;
    return result.str();
}

inline void emit_result(std::string_view category, std::string_view frontend,
                        std::string_view backend, std::string_view workload,
                        std::string_view content_hash, std::size_t source_bytes,
                        std::size_t token_count, std::size_t module_count,
                        std::size_t process_count, std::size_t signal_count,
                        std::optional<std::size_t> port_count,
                        std::optional<std::size_t> net_count,
                        std::optional<std::size_t> instance_count,
                        std::size_t warmups, std::size_t repetitions,
                        const TimingSummary& timing,
                        std::string_view correctness_status,
                        std::string_view build_configuration,
                        std::string_view execution_backend = "cpu",
                        std::optional<std::size_t> semantic_reference_count = {},
                        std::optional<std::size_t> index_construction_count = {},
                        std::optional<std::uint64_t> index_construction_nanoseconds = {},
                        std::optional<std::size_t> index_storage_bytes = {},
                        std::optional<std::size_t> duplicate_reference_locations = {}) {
    std::cout << std::setprecision(12)
              << "{\"schema_version\":1,\"category\":\""
              << json_escape(category) << "\",\"frontend\":\""
              << json_escape(frontend) << "\",\"backend\":\""
              << json_escape(backend) << "\",\"execution_backend\":\""
              << json_escape(execution_backend) << "\",\"workload\":\""
              << json_escape(workload) << "\",\"content_hash_fnv1a64\":\""
              << content_hash << "\",\"source_bytes\":" << source_bytes
              << ",\"token_count\":" << token_count
              << ",\"module_count\":" << module_count
              << ",\"process_count\":" << process_count
              << ",\"signal_count\":" << signal_count << ",\"port_count\":";
    if (port_count)
        std::cout << *port_count;
    else
        std::cout << "null";
    std::cout << ",\"net_count\":";
    if (net_count)
        std::cout << *net_count;
    else
        std::cout << "null";
    std::cout << ",\"instance_count\":";
    if (instance_count)
        std::cout << *instance_count;
    else
        std::cout << "null";
    std::cout << ",\"warmups\":" << warmups
              << ",\"repetitions\":" << repetitions
              << ",\"minimum_seconds\":" << timing.minimum_seconds
              << ",\"median_seconds\":" << timing.median_seconds
              << ",\"p95_seconds\":" << timing.p95_seconds
              << ",\"iqr_seconds\":" << timing.iqr_seconds
              << ",\"cpu_time_seconds\":null,\"peak_rss_bytes\":null"
                 ",\"allocation_count\":null,\"allocation_bytes\":null"
              << ",\"semantic_reference_count\":";
    if (semantic_reference_count) std::cout << *semantic_reference_count;
    else std::cout << "null";
    std::cout << ",\"index_construction_count\":";
    if (index_construction_count) std::cout << *index_construction_count;
    else std::cout << "null";
    std::cout << ",\"index_construction_nanoseconds\":";
    if (index_construction_nanoseconds)
        std::cout << *index_construction_nanoseconds;
    else std::cout << "null";
    std::cout << ",\"index_storage_bytes\":";
    if (index_storage_bytes) std::cout << *index_storage_bytes;
    else std::cout << "null";
    std::cout << ",\"duplicate_reference_locations\":";
    if (duplicate_reference_locations)
        std::cout << *duplicate_reference_locations;
    else std::cout << "null";
    std::cout
              << ",\"correctness_status\":\""
              << json_escape(correctness_status)
              << "\",\"build_configuration\":\""
              << json_escape(build_configuration) << "\"}\n";
}

} // namespace labneura::eda::benchmark
