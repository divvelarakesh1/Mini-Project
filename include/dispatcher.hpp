#pragma once
#include <atomic>
#include <tuple>
#include "config.hpp"

/**
 * @class Dispatcher
 * @brief Controls per-step interval-asynchrony gating for parallel threads.
 *
 * In highly concurrent environments like Hogwild!, it is sometimes optimal
 * to conditionally pause or synchronize threads. The Dispatcher intercepts
 * threads before they compute a step (`try_start_step`) and after they
 * calculate gradients (`finish_step`), dictating whether the global weights
 * should physically be updated.
 */
class Dispatcher {
public:
  Dispatcher(const Config& config) 
      : counter_(0), accepted_steps_(0), 
        interval_size_(config.interval_size), 
        decay_freq_(config.interval_decay_freq),
        mode_(config.exec_mode) {}

  // Returns {can_start, captured_step_index}
  std::tuple<bool, int> try_start_step() {
    int idx = counter_.fetch_add(1, std::memory_order_relaxed);
    return {true, idx};
  }

  // Returns true if caller should apply the gradient update
  bool finish_step(int step_index) { 
    if (mode_ == ExecutionMode::INTERVAL_ASYNC) {
      int current_idx = counter_.load(std::memory_order_relaxed);
      int current_interval_size = interval_size_.load(std::memory_order_relaxed);
      int start_interval = step_index / current_interval_size;
      int current_interval = current_idx / current_interval_size;
      
      // If the global step has progressed beyond the interval where this step started,
      // the gradient computations have crossed interval boundaries. Drop it.
      if (current_interval > start_interval) {
        return false;
      }

      // Track accepted gradient steps to apply y-decay strategy dynamically
      if (decay_freq_ > 0) {
        int accepted = accepted_steps_.fetch_add(1, std::memory_order_relaxed) + 1;
        if (accepted % decay_freq_ == 0) {
          int current_y = interval_size_.load(std::memory_order_relaxed);
          if (current_y > 1) {
            // Decay interval size by 1 (y-decay)
            interval_size_.compare_exchange_weak(current_y, current_y - 1, std::memory_order_relaxed);
          }
        }
      }
    }
    return true; 
  }

private:
  std::atomic<int> counter_;
  std::atomic<int> accepted_steps_;
  std::atomic<int> interval_size_;
  int decay_freq_;
  ExecutionMode mode_;
};
