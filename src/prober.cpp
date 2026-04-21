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

  // ---------------------------------------------------------------------------
  // THREAD EVALUATION
  // ---------------------------------------------------------------------------
  if (current == Phase::PROBING_THREADS) {
    std::cout << "[Prober] Test Threads: " << std::setw(3) << current_test_threads_
              << " | " << std::fixed << std::setprecision(0) << std::setw(6) << throughput_ips << " ips"
              << " | rate: " << std::showpos << std::fixed << std::setprecision(4) << rate << std::noshowpos << "\n";

    // PHASE 1: Global Binary Search
    if (!initial_thread_search_done_) {
      if (thread_probe_slot_ == 0) {
        thread_anchor_rate_ = rate;
        thread_anchor_ = current_test_threads_;
        
        int step = std::max(1, (thread_bs_high_ - thread_bs_low_) / 4);
        thread_candidate_ = (thread_anchor_ + step <= thread_bs_high_) ? thread_anchor_ + step : thread_anchor_ - step;
        
        thread_probe_slot_ = 1;
        current_test_threads_ = thread_candidate_;
        dispatcher_.set_active_threads(current_test_threads_);
        window_start_time_ = std::chrono::steady_clock::now();
      } else {
        // We have measured both anchor and candidate, figure out which side the peak is on
        if (thread_candidate_ > thread_anchor_) {
          if (rate > thread_anchor_rate_ * kSignificanceThreshold) {
            thread_bs_low_ = thread_anchor_; 
            best_val_ = thread_candidate_;
          } else {
            thread_bs_high_ = thread_candidate_; 
            best_val_ = thread_anchor_;
          }
        } else {
          if (rate > thread_anchor_rate_ * kSignificanceThreshold) {
            thread_bs_high_ = thread_anchor_; 
            best_val_ = thread_candidate_;
          } else {
            thread_bs_low_ = thread_candidate_; 
            best_val_ = thread_anchor_;
          }
        }

        if (thread_bs_low_ >= thread_bs_high_ || (thread_bs_high_ - thread_bs_low_) <= 2) {
          initial_thread_search_done_ = true;
          std::cout << "[Prober] Thread Global Binary Search Complete -> Locked at " << best_val_ << "\n";
        } else {
          std::cout << "[Prober] Thread Binary Search narrowed to [" << thread_bs_low_ << ", " << thread_bs_high_ << "]\n";
        }
        
        current_test_threads_ = best_val_;
        dispatcher_.set_active_threads(current_test_threads_);
        advance_phase();
      }
      return;
    }

    // PHASE 2: Neighborhood Search
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
      std::cout << "[Prober] Thread Neighborhood Search Winner -> " << best_val_ << "\n";
      current_test_threads_ = best_val_;
      dispatcher_.set_active_threads(best_val_);
      advance_phase();
    }
  } 
  
  // ---------------------------------------------------------------------------
  // INTERVAL EVALUATION
  // ---------------------------------------------------------------------------
  else if (current == Phase::PROBING_INTERVAL) {
    std::cout << "[Prober] Test Interval: " << std::setw(4) << current_test_interval_
              << " | " << std::fixed << std::setprecision(0) << std::setw(6) << throughput_ips << " ips"
              << " | rate: " << std::showpos << std::fixed << std::setprecision(4) << rate << std::noshowpos << "\n";

    // PHASE 1: Global Binary Search
    if (!initial_interval_search_done_) {
      if (interval_probe_slot_ == 0) {
        interval_anchor_rate_ = rate;
        interval_anchor_ = current_test_interval_;
        
        int step = std::max(1, (interval_bs_high_ - interval_bs_low_) / 4);
        interval_candidate_ = (interval_anchor_ + step <= interval_bs_high_) ? interval_anchor_ + step : interval_anchor_ - step;
        
        interval_probe_slot_ = 1;
        current_test_interval_ = interval_candidate_;
        dispatcher_.set_interval_size(current_test_interval_);
        window_start_time_ = std::chrono::steady_clock::now();
      } else {
        if (interval_candidate_ > interval_anchor_) {
          if (rate > interval_anchor_rate_ * kSignificanceThreshold) {
            interval_bs_low_ = interval_anchor_;
            best_val_ = interval_candidate_;
          } else {
            interval_bs_high_ = interval_candidate_;
            best_val_ = interval_anchor_;
          }
        } else {
          if (rate > interval_anchor_rate_ * kSignificanceThreshold) {
            interval_bs_high_ = interval_anchor_;
            best_val_ = interval_candidate_;
          } else {
            interval_bs_low_ = interval_candidate_;
            best_val_ = interval_anchor_;
          }
        }

        if (interval_bs_low_ >= interval_bs_high_ || (interval_bs_high_ - interval_bs_low_) <= 2) {
          initial_interval_search_done_ = true;
          std::cout << "[Prober] Interval Global Binary Search Complete -> Locked at " << best_val_ << "\n";
        } else {
          std::cout << "[Prober] Interval Binary Search narrowed to [" << interval_bs_low_ << ", " << interval_bs_high_ << "]\n";
        }
        
        current_test_interval_ = best_val_;
        dispatcher_.set_interval_size(current_test_interval_);
        advance_phase();
      }
      return;
    }

    // PHASE 2: Neighborhood Search
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
      std::cout << "[Prober] Interval Neighborhood Search Winner -> " << best_val_ << "\n";
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

  if (!initial_thread_search_done_) {
    std::cout << "[Prober] Starting Global Binary Search for Threads\n";
    if (thread_bs_low_ == -1) {
      thread_bs_low_ = config_.thread_min_count;
      thread_bs_high_ = config_.num_threads;
    }
    thread_probe_slot_ = 0;
    current_test_threads_ = thread_bs_low_ + (thread_bs_high_ - thread_bs_low_) / 2;
    dispatcher_.set_active_threads(current_test_threads_);
    return;
  }

  // Set up 3-point Neighborhood Micro-step
  int base = current_test_threads_;
  int delta = std::max(1, base / 10); 

  int left_bound = std::max(config_.thread_min_count, base - delta);
  int right_bound = std::min(config_.num_threads, base + delta);

  std::cout << "[Prober] Starting Neighborhood Search around " << base 
            << " (Testing: " << left_bound << ", " << base << ", " << right_bound << ")\n";

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

  int dynamic_min = std::max(1, current_test_threads_ / 2);
  int dynamic_max = std::max(dynamic_min, std::min(config_.interval_size, current_test_threads_ * 3));

  if (!initial_interval_search_done_) {
    std::cout << "[Prober] Starting Global Binary Search for Interval\n";
    if (interval_bs_low_ == -1) {
      interval_bs_low_ = dynamic_min;
      interval_bs_high_ = dynamic_max;
    } else {
      interval_bs_low_ = std::max(interval_bs_low_, dynamic_min);
      interval_bs_high_ = std::min(interval_bs_high_, dynamic_max);
    }
    
    interval_probe_slot_ = 0;
    current_test_interval_ = interval_bs_low_ + (interval_bs_high_ - interval_bs_low_) / 2;
    dispatcher_.set_interval_size(current_test_interval_);
    return;
  }

  // Set up 3-point Neighborhood Micro-step
  int base = current_test_interval_;
  int delta = std::max(2, base / 6); 

  int left_bound = std::max(dynamic_min, base - delta);
  int right_bound = std::min(dynamic_max, base + delta);

  std::cout << "[Prober] Starting Neighborhood Search around " << base 
            << " (Testing: " << left_bound << ", " << base << ", " << right_bound << ")\n";

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