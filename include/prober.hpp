#pragma once
#include "config.hpp"
#include <atomic>
#include <chrono>
#include <mutex>
#include <vector>

class Dispatcher;

/**
 * @class Prober
 * @brief Orchestrates performance tuning via Execution/Probing phases for both Threads and Intervals.
 */
class Prober {
public:
  enum class Phase {
    INITIAL_SWEEP_THREADS,
    INITIAL_SWEEP_INTERVAL,
    EXECUTION_PHASE,
    PROBING_THREADS,
    PROBING_INTERVAL
  };

  Prober(const Config &config, Dispatcher &dispatcher);

  /**
   * @brief Update the prober with a new batch loss.
   */
  void update(double batch_loss);

  Phase current_phase() const { return phase_.load(std::memory_order_acquire); }

private:
  void advance_phase();
  void evaluate_current_window();
  
  // State Machine Transitions
  void start_initial_thread_sweep();
  void start_initial_interval_sweep();
  void start_execution_phase();
  void start_probing_threads();
  void start_probing_interval();

  // Background Adjustment
  void apply_y_decay();

  const Config &config_;
  Dispatcher &dispatcher_;

  std::atomic<Phase> phase_{Phase::EXECUTION_PHASE};
  std::mutex mtx_;

  // --- Background y-Decay State ---
  std::atomic<int> decay_counter_{0};

  // --- Optimization Pipeline State ---
  std::atomic<int> step_counter_{0};
  std::atomic<int> execution_steps_{0};
  std::atomic<long long> accumulated_loss_{0};
  
  // Robust Baseline tracking
  std::atomic<long long> baseline_accum_loss_{0};
  double stabilized_baseline_{0.0};
  std::chrono::time_point<std::chrono::steady_clock> window_start_time_;

  // Current Best Scores
  double best_rate_{-1e9};
  int best_threads_{-1};
  int best_interval_{-1};

  // Currently being tested
  int current_test_threads_;
  int current_test_interval_;

  std::vector<int> search_queue_;
  bool next_probe_is_threads_{true}; // Used to alternate between thread and interval probing
};