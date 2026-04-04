#pragma once
#include <mutex>
#include <chrono>

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
     * @param global_step Identifying step number bounding this cycle.
     * @param batch_loss Floating exact loss calculated dynamically off network.
     * @param batch_size Amount of images pushed this cycle for hardware throughput.
     */
    void record_step(int global_step, double batch_loss, int batch_size);

private:
    const Config& config_;
    
    std::mutex mtx_;
    int accumulated_steps_{0};
    double accumulated_loss_{0.0};
    int accumulated_images_{0};

    using TimePoint = std::chrono::time_point<std::chrono::high_resolution_clock>;
    TimePoint start_time_;
    TimePoint last_log_time_;
};
