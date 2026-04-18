#pragma once
#include "config.hpp"
#include <atomic>
#include <tuple>

/**
 * @class Dispatcher
 * @brief Controls per-step interval-asynchrony gating for parallel threads.
 */
class Dispatcher {
public:
  Dispatcher(const Config &config);

  // Returns {can_start, captured_step_index}
  std::tuple<bool, int> try_start_step(int thread_id);

  // Returns true if caller should apply the gradient update
  bool finish_step(int step_index, double batch_loss = 0.0);

  // Thread Balancing
  bool is_thread_active(int thread_id) const;
  int active_threads() const;
  void set_active_threads(int count);
  void set_interval_size(int size);

private:
  std::atomic<int> accepted_steps_;
  std::atomic<int> interval_size_;
  std::atomic<int> active_threads_;
  ExecutionMode mode_;
};
