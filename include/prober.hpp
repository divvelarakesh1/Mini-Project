#pragma once
#include "config.hpp"
#include <atomic>
#include <chrono>
#include <mutex>
#include <queue>

class Dispatcher;

/**
 * @class Prober
 * @brief Orchestrates performance tuning via Execution and Probing phases.
 *
 * Uses Continuous Micro-Tracking: tests exactly 3 tight, safe neighborhood 
 * values around the current state and immediately locks the best one. 
 * Completely stateless, preventing extreme jumps that damage model convergence.
 */
class Prober {
public:
  enum class Phase {
    EXECUTION_PHASE,
    PROBING_THREADS,
    PROBING_INTERVAL
  };

  Prober(const Config &config, Dispatcher &dispatcher);

  void update(double batch_loss);

  Phase current_phase() const { return phase_.load(std::memory_order_acquire); }

private:
  void advance_phase();
  void evaluate_current_window();
  void handle_execution_phase();
  void handle_probing_phase(double batch_loss);

  void start_execution_phase();
  void start_probing_threads();
  void start_probing_interval();

  void apply_y_decay();

  static constexpr double kEpsilon               = 1e-9;
  static constexpr double kLossMultiplier        = 1e6;
  static constexpr double kRateMultiplier        = 10000.0;

  const Config       &config_;
  Dispatcher         &dispatcher_;
  std::atomic<Phase>  phase_{Phase::EXECUTION_PHASE};
  std::mutex          mtx_;

  std::atomic<int> decay_counter_{0};

  // --- Shared window measurement state ---
  std::atomic<int>       step_counter_{0};
  std::atomic<int>       execution_steps_{0};
  std::atomic<long long> accumulated_loss_{0};
  std::atomic<long long> baseline_accum_loss_{0};
  double                 stabilized_baseline_{0.0};
  std::chrono::time_point<std::chrono::steady_clock> window_start_time_;
  int                    baseline_limit_{0};

  // --- Parameter State ---
  int current_test_threads_;
  int current_test_interval_;

  // --- Micro-Tracking Queue ---
  std::queue<int> search_queue_;
  double best_rate_{-1e9};
  int    best_val_{-1};

  inline void add_atomic_double(std::atomic<long long> &target, double value) {
    target.fetch_add(static_cast<long long>(value * kLossMultiplier), std::memory_order_relaxed);
  }
  inline double get_atomic_double(const std::atomic<long long> &target) const {
    return target.load(std::memory_order_relaxed) / kLossMultiplier;
  }
};