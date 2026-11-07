#include "labneura/runtime.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <iostream>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

std::vector<std::atomic<unsigned>> make_counts(std::size_t size) {
    std::vector<std::atomic<unsigned>> counts(size);
    for (auto& count : counts) count.store(0, std::memory_order_relaxed);
    return counts;
}

void verify_coverage(std::vector<std::atomic<unsigned>>& counts) {
    for (const auto& count : counts) {
        require(count.load(std::memory_order_relaxed) == 1,
                "range coverage was missing or overlapping");
    }
}

}  // namespace

int main() {
    using labneura::execution_thread_count;
    using labneura::parallel_for;
    using labneura::set_execution_thread_count;

    const std::size_t available = labneura::available_cpu_count();

    set_execution_thread_count(1);
    const auto caller = std::this_thread::get_id();
    bool stayed_on_caller = true;
    parallel_for(0, 100, 7, [&stayed_on_caller, caller](std::size_t, std::size_t) {
        stayed_on_caller = stayed_on_caller &&
                           std::this_thread::get_id() == caller;
    });
    require(stayed_on_caller, "one-thread execution was not synchronous");

    std::atomic<unsigned> empty_calls{0};
    parallel_for(4, 4, 1, [&empty_calls](std::size_t, std::size_t) {
        empty_calls.fetch_add(1, std::memory_order_relaxed);
    });
    require(empty_calls.load(std::memory_order_relaxed) == 0,
            "empty range invoked callback");

    set_execution_thread_count(available > 1 ? 2 : 1);
    bool small_range_on_caller = true;
    parallel_for(0, 3, 3, [&small_range_on_caller, caller](std::size_t, std::size_t) {
        small_range_on_caller = small_range_on_caller &&
                                std::this_thread::get_id() == caller;
    });
    require(small_range_on_caller, "small range was unnecessarily scheduled");

    constexpr std::size_t length = 4097;
    auto counts = make_counts(length);
    std::mutex ids_mutex;
    std::set<std::thread::id> thread_ids;
    parallel_for(0, length, 32, [&counts, &ids_mutex, &thread_ids](
                                     std::size_t begin, std::size_t end) {
        {
            std::lock_guard<std::mutex> lock(ids_mutex);
            thread_ids.insert(std::this_thread::get_id());
        }
        for (std::size_t i = begin; i < end; ++i) {
            counts[i].fetch_add(1, std::memory_order_relaxed);
        }
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    });
    verify_coverage(counts);
    if (available > 1) {
        require(thread_ids.size() > 1, "multi-thread execution used one thread");
    }

    bool invalid_grain_rejected = false;
    try {
        parallel_for(0, 1, 0, [](std::size_t, std::size_t) {});
    } catch (const std::invalid_argument&) {
        invalid_grain_rejected = true;
    }
    require(invalid_grain_rejected, "zero grain size was accepted");

    bool reversed_range_rejected = false;
    try {
        parallel_for(2, 1, 1, [](std::size_t, std::size_t) {});
    } catch (const std::invalid_argument&) {
        reversed_range_rejected = true;
    }
    require(reversed_range_rejected, "reversed range was accepted");

    bool exception_propagated = false;
    try {
        parallel_for(0, 1000, 1, [](std::size_t begin, std::size_t) {
            if (begin == 0) throw std::runtime_error("parallel callback failure");
        });
    } catch (const std::runtime_error& error) {
        exception_propagated =
            std::string(error.what()) == "parallel callback failure";
    }
    require(exception_propagated, "parallel callback exception was not propagated");

    for (int submission = 0; submission < 100; ++submission) {
        std::atomic<std::size_t> visited{0};
        parallel_for(0, 257, 5, [&visited](std::size_t begin, std::size_t end) {
            visited.fetch_add(end - begin, std::memory_order_relaxed);
        });
        require(visited.load(std::memory_order_relaxed) == 257,
                "repeated submission lost range elements");
    }

    if (available > 2) {
        set_execution_thread_count(3);
        std::atomic<std::size_t> visited{0};
        parallel_for(0, 1000, 10, [&visited](std::size_t begin, std::size_t end) {
            visited.fetch_add(end - begin, std::memory_order_relaxed);
        });
        require(visited.load(std::memory_order_relaxed) == 1000,
                "pool reconfiguration failed");
    }

    const std::size_t concurrent_callers = 4;
    std::atomic<bool> concurrent_failed{false};
    std::vector<std::thread> callers;
    for (std::size_t caller_index = 0; caller_index < concurrent_callers;
         ++caller_index) {
        callers.emplace_back([&concurrent_failed] {
            for (int submission = 0; submission < 20; ++submission) {
                std::atomic<std::size_t> visited{0};
                parallel_for(0, 1001, 11,
                             [&visited](std::size_t begin, std::size_t end) {
                                 visited.fetch_add(end - begin,
                                                   std::memory_order_relaxed);
                             });
                if (visited.load(std::memory_order_relaxed) != 1001) {
                    concurrent_failed.store(true, std::memory_order_relaxed);
                }
            }
        });
    }
    for (auto& thread : callers) thread.join();
    require(!concurrent_failed.load(std::memory_order_relaxed),
            "concurrent callers produced incomplete work");

    std::atomic<std::size_t> nested_visits{0};
    parallel_for(0, 40, 4, [&nested_visits](std::size_t, std::size_t) {
        parallel_for(0, 3, 1, [&nested_visits](std::size_t, std::size_t) {
            nested_visits.fetch_add(1, std::memory_order_relaxed);
        });
    });
    require(nested_visits.load(std::memory_order_relaxed) == 30,
            "nested execution did not use the synchronous fallback");

    set_execution_thread_count(1);
    require(execution_thread_count() == 1,
            "thread count did not restore to one");
    std::cout << "parallel_for tests passed (available CPUs: " << available
              << "); pool shutdown occurs at process exit\n";
    return 0;
}
