#include "prober.hpp"
#include "dispatcher.hpp"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>

Prober::Prober(const Config &config, Dispatcher &dispatcher)
    : config_(config),
      dispatcher_(dispatcher),
      current_test_threads_(
          config.thread_mode == ThreadMode::PROBING
              ? std::max(config.thread_min_count, (config.thread_min_count + config.num_threads) / 2)
              : config.num_threads),
      current_test_interval_(
          config.interval_mode == IntervalMode::PROBING
              ? std::max(std::max(1, current_test_threads_ / 2), config.interval_size / 2)
              : config.interval_size) {

  dispatcher_.set_interval_size(current_test_interval_);
  dispatcher_.set_active_threads(current_test_threads_);

  if (config_.thread_mode == ThreadMode::PROBING) {
    start_probing_threads();
  } else if (config_.interval_mode == IntervalMode::PROBING) {
    start_probing_interval();
  } else {
    start_execution_phase();
  }
}

void Prober::apply_y_decay() {
  if (config_.interval_mode != IntervalMode::DECAY) return;

  int current_decay = decay_counter_.fetch_add(1, std::memory_order_relaxed) + 1;
  if (current_decay == config_.decay_steps) {
    decay_counter_.store(0, std::memory_order_relaxed);

    std::lock_guard<std::mutex> lock(mtx_);
    int dynamic_min_interval = std::max(1, current_test_threads_ / 2);
    int next_interval = std::max(dynamic_min_interval, current_test_interval_ - config_.decay_amount);

    if (current_test_interval_ != next_interval) {
      current_test_interval_ = next_interval;
      dispatcher_.set_interval_size(current_test_interval_);
      std::cout << "[Prober] y-Decay -> New Interval: " << current_test_interval_ << "\n";
    }
  }
}

void Prober::update(double batch_loss) {
  apply_y_decay();

  Phase current = phase_.load(std::memory_order_relaxed);
  if (current == Phase::EXECUTION_PHASE) {
    handle_execution_phase();
  } else {
    handle_probing_phase(batch_loss);
  }
}

void Prober::handle_execution_phase() {
  int exec_count = execution_steps_.fetch_add(1, std::memory_order_relaxed) + 1;
  if (exec_count == config_.probe_exec_steps) {
    if (config_.thread_mode == ThreadMode::PROBING || config_.interval_mode == IntervalMode::PROBING) {
      advance_phase();
    }
  }
}

void Prober::handle_probing_phase(double batch_loss) {
  int current_count = step_counter_.fetch_add(1, std::memory_order_relaxed) + 1;

  if (current_count <= baseline_limit_) {
    add_atomic_double(baseline_accum_loss_, batch_loss);
    if (current_count == baseline_limit_) {
      std::lock_guard<std::mutex> lock(mtx_);
      stabilized_baseline_ = get_atomic_double(baseline_accum_loss_) / baseline_limit_;
    }
  } else {
    add_atomic_double(accumulated_loss_, batch_loss);
  }

  if (current_count == config_.probe_test_steps) {
    std::lock_guard<std::mutex> lock(mtx_);
    evaluate_current_window();

    step_counter_.store(0, std::memory_order_relaxed);
    accumulated_loss_.store(0, std::memory_order_relaxed);
    baseline_accum_loss_.store(0, std::memory_order_relaxed);
  }
}

void Prober::evaluate_current_window() {
  auto now = std::chrono::steady_clock::now();
  auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - window_start_time_).count();

  int measurement_steps = config_.probe_test_steps - baseline_limit_;
  double avg_loss = get_atomic_double(accumulated_loss_) / std::max(1, measurement_steps);
  double throughput_ips = (config_.probe_test_steps * config_.batch_size) / (elapsed_ms / 1000.0 + kEpsilon);

  if (avg_loss <= 0.0) avg_loss = kEpsilon;
  if (stabilized_baseline_ <= 0.0) stabilized_baseline_ = kEpsilon;

  double log_improvement = std::log(stabilized_baseline_ / avg_loss);
  double rate = log_improvement * throughput_ips * kRateMultiplier;

  Phase current = phase_.load(std::memory_order_relaxed);

  if (current == Phase::PROBING_THREADS) {
    std::cout << "[Prober] Test Threads: " << std::setw(2) << current_test_threads_
              << " | " << std::fixed << std::setprecision(0) << std::setw(6) << throughput_ips << " ips"
              << " | rate: " << std::showpos << std::fixed << std::setprecision(4) << rate << std::noshowpos << "\n";

    if (best_val_ == -1 || rate > best_rate_) {
      best_rate_ = rate;
      best_val_ = current_test_threads_;
    }

    if (!search_queue_.empty()) {
      current_test_threads_ = search_queue_.front();
      search_queue_.pop();
      dispatcher_.set_active_threads(current_test_threads_);
      window_start_time_ = std::chrono::steady_clock::now();
    } else {
      std::cout << "[Prober] Thread Micro-Track Locked -> " << best_val_ << "\n";
      current_test_threads_ = best_val_;
      dispatcher_.set_active_threads(best_val_);
      advance_phase();
    }
  } 
  else if (current == Phase::PROBING_INTERVAL) {
    std::cout << "[Prober] Test Interval: " << std::setw(4) << current_test_interval_
              << " | " << std::fixed << std::setprecision(0) << std::setw(6) << throughput_ips << " ips"
              << " | rate: " << std::showpos << std::fixed << std::setprecision(4) << rate << std::noshowpos << "\n";

    if (best_val_ == -1 || rate > best_rate_) {
      best_rate_ = rate;
      best_val_ = current_test_interval_;
    }

    if (!search_queue_.empty()) {
      current_test_interval_ = search_queue_.front();
      search_queue_.pop();
      dispatcher_.set_interval_size(current_test_interval_);
      window_start_time_ = std::chrono::steady_clock::now();
    } else {
      std::cout << "[Prober] Interval Micro-Track Locked -> " << best_val_ << "\n";
      current_test_interval_ = best_val_;
      dispatcher_.set_interval_size(best_val_);
      advance_phase();
    }
  }
}

void Prober::advance_phase() {
  Phase current = phase_.load(std::memory_order_relaxed);

  if (current == Phase::EXECUTION_PHASE) {
    if (config_.thread_mode == ThreadMode::PROBING) {
      start_probing_threads();
    } else if (config_.interval_mode == IntervalMode::PROBING) {
      start_probing_interval();
    } else {
      start_execution_phase();
    }
  } else if (current == Phase::PROBING_THREADS) {
    if (config_.interval_mode == IntervalMode::PROBING) {
      start_probing_interval();
    } else {
      start_execution_phase();
    }
  } else if (current == Phase::PROBING_INTERVAL) {
    start_execution_phase();
  }
}

void Prober::start_probing_threads() {
  std::cout << "\n[Prober] Phase -> PROBING_THREADS (" << config_.probe_test_steps << " steps/test)\n";
  phase_.store(Phase::PROBING_THREADS, std::memory_order_release);
  std::queue<int>().swap(search_queue_);

  baseline_limit_    = std::max(1, config_.probe_test_steps / 5);
  window_start_time_ = std::chrono::steady_clock::now();
  best_rate_         = -1e9;
  best_val_          = -1;

  // Tiny, safe 10% micro-step (minimum of 1)
  int base = current_test_threads_;
  int delta = std::max(1, base / 10); 

  int left_bound = std::max(config_.thread_min_count, base - delta);
  int right_bound = std::min(config_.num_threads, base + delta);

  if (left_bound != base) search_queue_.push(left_bound);
  search_queue_.push(base);
  if (right_bound != base) search_queue_.push(right_bound);

  current_test_threads_ = search_queue_.front();
  search_queue_.pop();
  dispatcher_.set_active_threads(current_test_threads_);
}

void Prober::start_probing_interval() {
  std::cout << "\n[Prober] Phase -> PROBING_INTERVAL (" << config_.probe_test_steps << " steps/test)\n";
  phase_.store(Phase::PROBING_INTERVAL, std::memory_order_release);
  std::queue<int>().swap(search_queue_);

  baseline_limit_    = std::max(1, config_.probe_test_steps / 5);
  window_start_time_ = std::chrono::steady_clock::now();
  best_rate_         = -1e9;
  best_val_          = -1;

  // Tiny, safe 15% micro-step 
  int base = current_test_interval_;
  int delta = std::max(2, base / 6); 

  int dynamic_min = std::max(1, current_test_threads_ / 2);
  int dynamic_max = std::max(dynamic_min, std::min(config_.interval_size, current_test_threads_ * 3));

  int left_bound = std::max(dynamic_min, base - delta);
  int right_bound = std::min(dynamic_max, base + delta);

  if (left_bound != base) search_queue_.push(left_bound);
  search_queue_.push(base);
  if (right_bound != base) search_queue_.push(right_bound);

  current_test_interval_ = search_queue_.front();
  search_queue_.pop();
  dispatcher_.set_interval_size(current_test_interval_);
}

void Prober::start_execution_phase() {
  std::cout << "\n[Prober] Phase -> EXECUTION (" << config_.probe_exec_steps << " steps)\n";
  phase_.store(Phase::EXECUTION_PHASE, std::memory_order_release);
  execution_steps_.store(0, std::memory_order_relaxed);
}