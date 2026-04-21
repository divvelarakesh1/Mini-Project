#pragma once
#include "config.hpp"
#include <atomic>
#include <tuple>
#include <mutex>
#include <condition_variable>

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

  // --- Sleep/Wake Synchronization ---
  void wait_for_active(int thread_id);
  void stop_all();
  bool is_running() const { return is_running_.load(std::memory_order_acquire); }

private:
  std::atomic<int> accepted_steps_{0};
  std::atomic<int> interval_size_;
  
  // Thread Status
  int active_threads_; 
  std::atomic<bool> is_running_{true};
  
  // OS-level Synchronization Primitives
  mutable std::mutex mtx_;
  std::condition_variable cv_;
  
  ExecutionMode mode_;
};