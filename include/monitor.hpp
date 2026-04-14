#pragma once
#include <atomic>
#include <chrono>
#include <mutex>
#include <vector>

struct Config;

/**
 * @class Monitor
 * @brief Thread-safe logging module tracking convergence loss and throughput
 * over time.
 *
 * Aggregates mathematical loss metrics emitted securely from parallel worker
 * threads. Uses high-resolution physical time bounds to compute hardware
 * processing speeds without clogging stdout streams, batching readouts
 * dynamically. Progress is tracked purely by epochs, derived from the total
 * images processed across all threads.
 */
class Monitor {
public:
  Monitor(const Config &config, int total_samples);

  /**
   * @brief Pushes batch statistics into the centralized hardware monitor.
   *
   * @param thread_id Topological thread ID running the batch.
   * @param batch_loss Floating exact loss calculated dynamically off network.
   * @param batch_size Amount of images pushed this cycle for hardware
   * throughput.
   */
  void record(int thread_id, double batch_loss, int batch_size);

private:
  const Config &config_;
  int total_samples_;

  struct alignas(64) ThreadStats {
    long long images{0};
    double accumulated_loss{0.0};
    long long batches{0};
    double min_loss{1e9};
    double max_loss{-1e9};
  };

  std::vector<ThreadStats> active_stats_;
  std::vector<ThreadStats> snapshot_stats_;

  double last_logged_epoch_{0.0};

  std::mutex print_mtx_; // Exclusively for console stdout formatting

  using TimePoint = std::chrono::time_point<std::chrono::high_resolution_clock>;
  TimePoint start_time_;
  TimePoint last_log_time_;
};
