#include "dispatcher.hpp"
#include <iostream>
#include <algorithm>

Dispatcher::Dispatcher(const Config &config)
    : interval_size_(std::max(1, config.interval_size)),
      active_threads_(config.num_threads),
      mode_(config.exec_mode) {
  accepted_steps_.store(0);
}

bool Dispatcher::is_thread_active(int thread_id) const {
  return thread_id < active_threads_.load(std::memory_order_relaxed);
}

int Dispatcher::active_threads() const {
  return active_threads_.load(std::memory_order_relaxed);
}

void Dispatcher::set_active_threads(int count) {
  active_threads_.store(count, std::memory_order_release);
}

void Dispatcher::set_interval_size(int size) {
  interval_size_.store(size, std::memory_order_release);
}

std::tuple<bool, int> Dispatcher::try_start_step(int thread_id) {
  if (!is_thread_active(thread_id)) {
    return {false, -1};
  }
  int accepted_snapshot = accepted_steps_.load(std::memory_order_acquire);
  return {true, accepted_snapshot};
}

bool Dispatcher::finish_step(int step_index, double batch_loss) {
  (void)batch_loss;

  if (mode_ == ExecutionMode::INTERVAL_ASYNC) {
    int current_interval_size =
        std::max(1, interval_size_.load(std::memory_order_acquire));
    int start_interval = step_index / current_interval_size;

    int accepted = accepted_steps_.load(std::memory_order_acquire);
    while (true) {
      int current_interval = accepted / current_interval_size;
      if (current_interval > start_interval) {
        return false;
      }

      if (accepted_steps_.compare_exchange_weak(accepted, accepted + 1,
                                                std::memory_order_acq_rel,
                                                std::memory_order_acquire)) {
        break;
      }
    }
  } else {
    accepted_steps_.fetch_add(1, std::memory_order_relaxed);
  }
  return true;
}
