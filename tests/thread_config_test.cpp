#include "labneura/runtime.h"
#include "runtime_internal.h"

#include <atomic>
#include <cstddef>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

}  // namespace

int main() {
    using labneura::available_cpu_count;
    using labneura::execution_thread_count;
    using labneura::set_execution_thread_count;
    using labneura::set_execution_thread_count_auto;

    require(labneura::detail::resolve_cpu_count(6, 4) == 6,
            "detected CPU count was not preferred");
    require(labneura::detail::resolve_cpu_count(0, 4) == 4,
            "CPU count fallback was not used");
    require(labneura::detail::resolve_cpu_count(0, 0) == 1,
            "zero CPU counts did not resolve safely");

    const std::size_t available = available_cpu_count();
    require(available >= 1, "available CPU count must be positive");
    require(execution_thread_count() == 1,
            "default execution thread count must be one");

    bool rejected = false;
    try {
        set_execution_thread_count(0);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "zero thread count was accepted");

    if (available < std::numeric_limits<std::size_t>::max()) {
        rejected = false;
        try {
            set_execution_thread_count(available + 1);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        require(rejected, "count above available CPUs was accepted");
        require(execution_thread_count() == 1,
                "rejected thread count changed the setting");
    }

    set_execution_thread_count(available);
    require(execution_thread_count() == available,
            "explicit thread count was not stored");
    set_execution_thread_count_auto();
    require(execution_thread_count() == available,
            "automatic thread count did not select available CPUs");

    set_execution_thread_count(1);
    std::atomic<bool> failed{false};
    std::vector<std::thread> readers;
    std::thread writer([available] {
        for (int iteration = 0; iteration < 10000; ++iteration) {
            set_execution_thread_count(iteration % 2 == 0 ? 1 : available);
        }
    });
    for (int i = 0; i < 8; ++i) {
        readers.emplace_back([&failed, available] {
            for (int iteration = 0; iteration < 10000; ++iteration) {
                const std::size_t count = execution_thread_count();
                if (count != 1 && count != available) {
                    failed.store(true, std::memory_order_relaxed);
                    return;
                }
            }
        });
    }
    writer.join();
    for (auto& reader : readers) reader.join();
    require(!failed.load(std::memory_order_relaxed),
            "concurrent thread count access produced an invalid value");

    set_execution_thread_count(1);
    std::cout << "Thread configuration tests passed (available CPUs: "
              << available << ")\n";
    return 0;
}
