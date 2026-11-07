#include "labneura/runtime.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace labneura {
namespace {

thread_local unsigned parallel_callback_depth = 0;

class CallbackScope {
public:
    CallbackScope() { ++parallel_callback_depth; }
    ~CallbackScope() { --parallel_callback_depth; }

    CallbackScope(const CallbackScope&) = delete;
    CallbackScope& operator=(const CallbackScope&) = delete;
};

void run_synchronously(
    std::size_t begin, std::size_t end, std::size_t grain_size,
    const std::function<void(std::size_t, std::size_t)>& function) {
    for (std::size_t chunk_begin = begin; chunk_begin < end;) {
        const std::size_t chunk_length =
            std::min(grain_size, end - chunk_begin);
        CallbackScope scope;
        function(chunk_begin, chunk_begin + chunk_length);
        chunk_begin += chunk_length;
    }
}

struct Job {
    Job(std::size_t range_begin, std::size_t range_end, std::size_t chunk_size,
        const std::function<void(std::size_t, std::size_t)>& callback,
        std::size_t worker_count)
        : begin(range_begin),
          length(range_end - range_begin),
          grain_size(chunk_size),
          chunks(1 + (length - 1) / grain_size),
          function(callback),
          workers_remaining(worker_count) {}

    void run_chunks() noexcept {
        while (!cancelled.load(std::memory_order_relaxed)) {
            const std::size_t chunk =
                next_chunk.fetch_add(1, std::memory_order_relaxed);
            if (chunk >= chunks) return;

            const std::size_t offset = chunk * grain_size;
            const std::size_t chunk_begin = begin + offset;
            const std::size_t chunk_length =
                std::min(grain_size, length - offset);
            try {
                CallbackScope scope;
                function(chunk_begin, chunk_begin + chunk_length);
            } catch (...) {
                {
                    std::lock_guard<std::mutex> lock(exception_mutex);
                    if (!exception) exception = std::current_exception();
                }
                cancelled.store(true, std::memory_order_relaxed);
                return;
            }
        }
    }

    void worker_finished() {
        std::lock_guard<std::mutex> lock(completion_mutex);
        if (--workers_remaining == 0) completion_cv.notify_one();
    }

    void wait_for_workers() {
        std::unique_lock<std::mutex> lock(completion_mutex);
        completion_cv.wait(lock, [this] { return workers_remaining == 0; });
    }

    void rethrow_if_failed() const {
        std::lock_guard<std::mutex> lock(exception_mutex);
        if (exception) std::rethrow_exception(exception);
    }

    const std::size_t begin;
    const std::size_t length;
    const std::size_t grain_size;
    const std::size_t chunks;
    const std::function<void(std::size_t, std::size_t)>& function;
    std::atomic<std::size_t> next_chunk{0};
    std::atomic<bool> cancelled{false};
    mutable std::mutex exception_mutex;
    std::exception_ptr exception;
    std::mutex completion_mutex;
    std::condition_variable completion_cv;
    std::size_t workers_remaining;
};

class ThreadPool {
public:
    ~ThreadPool() { stop_workers(); }

    void execute(std::size_t begin, std::size_t end, std::size_t grain_size,
                 const std::function<void(std::size_t, std::size_t)>& function,
                 std::size_t thread_count) {
        std::unique_lock<std::mutex> submission_lock(submission_mutex);

        if (thread_count <= 1 || end - begin <= grain_size ||
            parallel_callback_depth != 0) {
            run_synchronously(begin, end, grain_size, function);
            return;
        }

        ensure_workers(thread_count - 1);
        auto job = std::make_shared<Job>(begin, end, grain_size, function,
                                         workers.size());
        {
            std::lock_guard<std::mutex> lock(state_mutex);
            current_job = job;
            ++generation;
        }
        work_cv.notify_all();

        job->run_chunks();
        job->wait_for_workers();
        job->rethrow_if_failed();
    }

private:
    void ensure_workers(std::size_t desired_count) {
        if (workers.size() == desired_count) return;
        stop_workers();
        try {
            workers.reserve(desired_count);
            for (std::size_t i = 0; i < desired_count; ++i) {
                workers.emplace_back([this] { worker_loop(); });
            }
        } catch (...) {
            stop_workers();
            throw;
        }
    }

    void stop_workers() noexcept {
        {
            std::lock_guard<std::mutex> lock(state_mutex);
            stopping = true;
        }
        work_cv.notify_all();
        for (auto& worker : workers) {
            if (worker.joinable()) worker.join();
        }
        workers.clear();
        {
            std::lock_guard<std::mutex> lock(state_mutex);
            current_job.reset();
            generation = 0;
            stopping = false;
        }
    }

    void worker_loop() {
        std::uint64_t observed_generation = 0;
        std::unique_lock<std::mutex> lock(state_mutex);
        while (true) {
            work_cv.wait(lock, [this, observed_generation] {
                return stopping || generation != observed_generation;
            });
            if (stopping) return;

            observed_generation = generation;
            const auto job = current_job;
            lock.unlock();
            job->run_chunks();
            job->worker_finished();
            lock.lock();
        }
    }

    std::mutex submission_mutex;
    std::mutex state_mutex;
    std::condition_variable work_cv;
    std::vector<std::thread> workers;
    std::shared_ptr<Job> current_job;
    std::uint64_t generation{0};
    bool stopping{false};
};

ThreadPool& thread_pool() {
    static ThreadPool pool;
    return pool;
}

}  // namespace

void parallel_for(
    std::size_t begin, std::size_t end, std::size_t grain_size,
    const std::function<void(std::size_t, std::size_t)>& function) {
    if (grain_size == 0) {
        throw std::invalid_argument("parallel_for grain_size must be positive");
    }
    if (end < begin) {
        throw std::invalid_argument("parallel_for end must not precede begin");
    }
    if (begin == end) return;
    if (!function) {
        throw std::invalid_argument("parallel_for function must be callable");
    }

    const std::size_t thread_count = execution_thread_count();
    if (thread_count <= 1 || end - begin <= grain_size ||
        parallel_callback_depth != 0) {
        run_synchronously(begin, end, grain_size, function);
        return;
    }
    thread_pool().execute(begin, end, grain_size, function, thread_count);
}

}  // namespace labneura
