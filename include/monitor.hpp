#pragma once
#include <atomic>
#include <chrono>
#include <mutex>
#include <vector>

struct Config;
class Prober;

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
  bool should_stop() const;
  void set_prober(Prober *prober) { prober_ = prober; }

private:
  const Config &config_;
  int total_samples_;

  struct alignas(64) ThreadStats {
    long long images{0};
    double accumulated_loss{0.0};
    long long batches{0};
  };

  std::vector<ThreadStats> active_stats_;
  std::vector<ThreadStats> snapshot_stats_;

  std::atomic<long long> global_images_processed_{0};
  std::atomic<long long> global_batches_processed_{0};

  double last_logged_epoch_{0.0};
  std::atomic<bool> stop_requested_{false};
  std::atomic<bool> final_log_emitted_{false};

  std::mutex print_mtx_;

  using Clock = std::chrono::steady_clock;
  using TimePoint = std::chrono::time_point<Clock>;
  TimePoint start_time_;
  TimePoint last_log_time_;

  // UI Helpers
  std::string get_progress_bar(double progress, int width = 30) const;
  std::string format_time(double seconds) const;

  Prober *prober_{nullptr};
};
