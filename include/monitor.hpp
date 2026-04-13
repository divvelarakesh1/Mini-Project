#pragma once
#include <mutex>
#include <chrono>
#include <vector>
#include <atomic>

struct Config;

/**
 * @class Monitor
 * @brief Thread-safe logging module tracking convergence loss and throughput over time.
 * 
 * Aggregates mathematical loss metrics emitted securely from parallel worker threads.
 * Uses high-resolution physical time bounds to compute hardware processing speeds 
 * without clogging stdout streams, batching readouts dynamically.
 */
class Monitor {
public:
    Monitor(const Config& config);

    /**
     * @brief Pushes batch statistics into the centralized hardware monitor.
     * 
     * @param thread_id Topological thread ID running the batch.
     * @param global_step Identifying step number bounding this cycle.
     * @param batch_loss Floating exact loss calculated dynamically off network.
     * @param batch_size Amount of images pushed this cycle for hardware throughput.
     */
    void record_step(int thread_id, int global_step, double batch_loss, int batch_size);

private:
    const Config& config_;
    
    struct alignas(64) ThreadStats {
        long long steps{0};
        double accumulated_loss{0.0};
        long long images{0};
        double min_loss{1e9};
        double max_loss{-1e9};
    };

    std::vector<ThreadStats> active_stats_;
    std::vector<ThreadStats> snapshot_stats_;
    
    std::atomic<long long> global_tick_{0};
    std::mutex print_mtx_; // Exclusively for console stdout formatting

    using TimePoint = std::chrono::time_point<std::chrono::high_resolution_clock>;
    TimePoint start_time_;
    TimePoint last_log_time_;
};
