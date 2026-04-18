#pragma once
#include "config.hpp"
#include <atomic>
#include <chrono>
#include <mutex>
#include <vector>

class Dispatcher;

/**
 * @class Prober
 * @brief Orchestrates performance tuning via Bounded Neighborhood Search.
 */
class Prober {
public:
  enum class Phase {
    INITIAL_SWEEP_THREADS,
    INITIAL_SWEEP_INTERVAL,
    STEADY_STATE,
    NEIGHBORHOOD_THREADS,
    NEIGHBORHOOD_INTERVAL
  };

  Prober(const Config &config, Dispatcher &dispatcher);

  /**
   * @brief Update the prober with a new batch loss.
   */
  void update(double batch_loss);

  Phase current_phase() const { return phase_.load(); }

private:
  void advance_phase();
  void evaluate_current_window();
  
  // Transition helpers
  void start_initial_thread_sweep();
  void start_initial_interval_sweep();
  void start_steady_state();
  void start_neighborhood_threads();
  void start_neighborhood_interval();

  const Config &config_;
  Dispatcher &dispatcher_;

  std::atomic<Phase> phase_{Phase::INITIAL_SWEEP_THREADS};
  std::mutex mtx_;

  // Metrics tracking
  std::atomic<int> step_counter_{0};
  std::atomic<int> steady_state_steps_{0};
  std::atomic<long long> accumulated_loss_{0};
  
  // Robust Baseline tracking
  std::atomic<long long> baseline_accum_loss_{0};
  double stabilized_baseline_{0.0};
  std::chrono::time_point<std::chrono::steady_clock> window_start_time_;

  // Optimization state
  double best_rate_{-1e9};
  int best_threads_{-1};
  int best_interval_{-1};

  int current_test_threads_;
  int current_test_interval_;

  std::vector<int> search_queue_;
};
