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
 * Uses a Dual-Stage Architecture:
 * 1. Initial Global Binary Search: Rapidly brackets the global optimum from min to max.
 * 2. Neighborhood Search (Micro-Tracking): Continuously tests 3 tight, safe neighborhood 
 * values around the current state to safely track the optimum over time.
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
  static constexpr double kSignificanceThreshold = 1.02; // Candidate must beat anchor by 2% to win

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

  // --- Initial Global Binary Search State ---
  bool   initial_thread_search_done_{false};
  int    thread_bs_low_{-1};
  int    thread_bs_high_{-1};
  int    thread_anchor_{-1};
  double thread_anchor_rate_{-1e9};
  int    thread_probe_slot_{0};
  int    thread_candidate_{-1};

  bool   initial_interval_search_done_{false};
  int    interval_bs_low_{-1};
  int    interval_bs_high_{-1};
  int    interval_anchor_{-1};
  double interval_anchor_rate_{-1e9};
  int    interval_probe_slot_{0};
  int    interval_candidate_{-1};

  // --- Neighborhood Search (Micro-Tracking) State ---
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