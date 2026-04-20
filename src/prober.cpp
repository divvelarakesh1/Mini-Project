#include "prober.hpp"
#include "dispatcher.hpp"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>

Prober::Prober(const Config &config, Dispatcher &dispatcher)
    : config_(config), 
      dispatcher_(dispatcher), 
      current_test_threads_(config.num_threads),
      current_test_interval_(config.interval_size) {
  
  dispatcher_.set_interval_size(current_test_interval_);
  dispatcher_.set_active_threads(current_test_threads_);
  
  // Initialize the correct starting state based on config parameters
  if (config_.thread_mode == ThreadMode::PROBING) {
    start_initial_thread_sweep();
  } else if (config_.interval_mode == IntervalMode::PROBING) {
    start_initial_interval_sweep();
  } else {
    start_execution_phase();
  }
}

void Prober::apply_y_decay() {
  if (config_.interval_mode != IntervalMode::DECAY) return;

  // Lock-free check for the decay threshold
  int current_decay = decay_counter_.fetch_add(1, std::memory_order_relaxed) + 1;
  if (current_decay >= config_.decay_steps) {
    if (decay_counter_.exchange(0, std::memory_order_acquire) >= config_.decay_steps) {
      std::lock_guard<std::mutex> lock(mtx_);
      int next_interval = std::max(config_.min_interval, current_test_interval_ - config_.decay_amount);
      if (current_test_interval_ != next_interval) {
        current_test_interval_ = next_interval;
        dispatcher_.set_interval_size(current_test_interval_);
        std::cout << "[Prober] y-Decay -> New Interval: " << current_test_interval_ << "\n";
      }
    }
  }
}

void Prober::update(double batch_loss) {
  // 1. Background Decay (If enabled, runs independently of phases)
  apply_y_decay();

  Phase current = phase_.load(std::memory_order_relaxed);

  // ---------------------------------------------------------
  // FAST PATH: EXECUTION PHASE 
  // ---------------------------------------------------------
  if (current == Phase::EXECUTION_PHASE) {
    int exec_count = execution_steps_.fetch_add(1, std::memory_order_relaxed) + 1;
    if (exec_count >= config_.probe_exec_steps) {
      if (execution_steps_.exchange(0, std::memory_order_acquire) >= config_.probe_exec_steps) {
        // Only transition out if probing is actually enabled
        if (config_.thread_mode == ThreadMode::PROBING || config_.interval_mode == IntervalMode::PROBING) {
          advance_phase();
        }
      }
    }
    return;
  }

  // ---------------------------------------------------------
  // PROBING PATH (Thread or Interval)
  // ---------------------------------------------------------
  int current_count = step_counter_.fetch_add(1, std::memory_order_relaxed) + 1;
  int warmup_limit = std::max(1, config_.probe_test_steps / 10);
  int baseline_limit = warmup_limit + std::max(1, config_.probe_test_steps / 5);

  // 1) Warmup Phase
  if (current_count <= warmup_limit) {
    if (current_count == warmup_limit) {
      std::lock_guard<std::mutex> lock(mtx_);
      window_start_time_ = std::chrono::steady_clock::now();
    }
  } 
  // 2) Baseline Phase
  else if (current_count <= baseline_limit) {
    baseline_accum_loss_.fetch_add(static_cast<long long>(batch_loss * 1e6), std::memory_order_relaxed);
    if (current_count == baseline_limit) {
      std::lock_guard<std::mutex> lock(mtx_);
      stabilized_baseline_ = (baseline_accum_loss_.load(std::memory_order_relaxed) / 1e6) / (baseline_limit - warmup_limit);
    }
  }
  // 3) Measurement Phase
  else {
    accumulated_loss_.fetch_add(static_cast<long long>(batch_loss * 1e6), std::memory_order_relaxed);
  }

  // Evaluate Window
  if (current_count >= config_.probe_test_steps) {
    std::lock_guard<std::mutex> lock(mtx_);
    if (step_counter_.load(std::memory_order_relaxed) >= config_.probe_test_steps) {
      evaluate_current_window();
      
      step_counter_.store(0, std::memory_order_relaxed);
      accumulated_loss_.store(0, std::memory_order_relaxed);
      baseline_accum_loss_.store(0, std::memory_order_relaxed);
    }
  }
}

void Prober::evaluate_current_window() {
  auto now = std::chrono::steady_clock::now();
  auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - window_start_time_).count();
  
  int warmup_limit = std::max(1, config_.probe_test_steps / 10);
  int baseline_limit = warmup_limit + std::max(1, config_.probe_test_steps / 5);
  int evaluated_steps = config_.probe_test_steps - warmup_limit;
  int measurement_steps = config_.probe_test_steps - baseline_limit;

  double avg_loss = (accumulated_loss_.load(std::memory_order_relaxed) / 1e6) / std::max(1, measurement_steps);
  double throughput_ips = (evaluated_steps * config_.batch_size) / (elapsed_ms / 1000.0 + 1e-9);

  // Safety against negative/zero loss math errors
  if (avg_loss <= 0.0) avg_loss = 1e-9;
  if (stabilized_baseline_ <= 0.0) stabilized_baseline_ = 1e-9;
  
  // THE GOLD STANDARD LOGARITHMIC RATE
  // Treats a 10% loss drop early in training exactly the same as a 10% drop late in training
  double log_improvement = std::log(stabilized_baseline_ / avg_loss);
  double rate = log_improvement * throughput_ips * 10000.0; 

  const double SIGNIFICANCE_THRESHOLD = 1.02; // Require 2% improvement to switch
  Phase current = phase_.load(std::memory_order_relaxed);

  // --- THREAD EVALUATION ---
  if (current == Phase::INITIAL_SWEEP_THREADS || current == Phase::PROBING_THREADS) {
    std::cout << "[Prober] Test Threads: " << std::setw(2) << current_test_threads_ 
              << " | " << std::fixed << std::setprecision(0) << std::setw(6) << throughput_ips << " ips"
              << " | rate: " << std::showpos << std::fixed << std::setprecision(4) << rate << std::noshowpos << "\n";
    
    if (best_threads_ == -1 || rate > (best_rate_ * SIGNIFICANCE_THRESHOLD)) {
      best_rate_ = rate;
      best_threads_ = current_test_threads_;
    }

    if (!search_queue_.empty()) {
      current_test_threads_ = search_queue_.front();
      search_queue_.erase(search_queue_.begin());
      dispatcher_.set_active_threads(current_test_threads_);
    } else {
      std::cout << "[Prober] Threads Locked -> " << best_threads_ << "\n";
      current_test_threads_ = best_threads_;
      dispatcher_.set_active_threads(best_threads_);
      advance_phase();
    }
  } 
  // --- INTERVAL EVALUATION ---
  else if (current == Phase::INITIAL_SWEEP_INTERVAL || current == Phase::PROBING_INTERVAL) {
    std::cout << "[Prober] Test Interval: " << std::setw(3) << current_test_interval_ 
              << " | " << std::fixed << std::setprecision(0) << std::setw(6) << throughput_ips << " ips"
              << " | rate: " << std::showpos << std::fixed << std::setprecision(4) << rate << std::noshowpos << "\n";
    
    // Always take the true best — re-centers neighbourhood search on the real winner
    if (best_interval_ == -1 || rate > best_rate_) {
      best_rate_ = rate;
      best_interval_ = current_test_interval_;
    }

    if (!search_queue_.empty()) {
      current_test_interval_ = search_queue_.front();
      search_queue_.erase(search_queue_.begin());
      dispatcher_.set_interval_size(current_test_interval_);
    } else {
      std::cout << "[Prober] Interval Locked -> " << best_interval_ << "\n";
      current_test_interval_ = best_interval_;
      dispatcher_.set_interval_size(best_interval_);
      advance_phase();
    }
  }
}

void Prober::advance_phase() {
  Phase current = phase_.load(std::memory_order_relaxed);
  
  if (current == Phase::INITIAL_SWEEP_THREADS) {
    if (config_.interval_mode == IntervalMode::PROBING) start_initial_interval_sweep();
    else start_execution_phase();
  } 
  else if (current == Phase::INITIAL_SWEEP_INTERVAL) {
    start_execution_phase();
  } 
  else if (current == Phase::EXECUTION_PHASE) {
    // Determine which probe to run next based on Config
    if (config_.thread_mode == ThreadMode::PROBING && config_.interval_mode == IntervalMode::PROBING) {
      if (next_probe_is_threads_) start_probing_threads();
      else start_probing_interval();
      next_probe_is_threads_ = !next_probe_is_threads_; // Toggle for next time
    } 
    else if (config_.thread_mode == ThreadMode::PROBING) {
      start_probing_threads();
    } 
    else if (config_.interval_mode == IntervalMode::PROBING) {
      start_probing_interval();
    } 
    else {
      start_execution_phase();
    }
  } 
  else if (current == Phase::PROBING_THREADS || current == Phase::PROBING_INTERVAL) {
    start_execution_phase();
  }
}

// ---------------------------------------------------------
// STATE INITIALIZATION & QUEUE LOADING
// ---------------------------------------------------------

void Prober::start_initial_thread_sweep() {
  std::cout << "\n[Prober] Phase -> INITIAL_SWEEP_THREADS\n";
  phase_.store(Phase::INITIAL_SWEEP_THREADS, std::memory_order_release);
  search_queue_.clear();
  best_rate_ = -1e9;
  
  int step = std::max(1, config_.num_threads / 4);
  for (int t = config_.thread_min_count; t <= config_.num_threads; t += step) {
    search_queue_.push_back(t);
  }
  
  current_test_threads_ = search_queue_.front();
  search_queue_.erase(search_queue_.begin());
  dispatcher_.set_active_threads(current_test_threads_);
}

void Prober::start_initial_interval_sweep() {
  std::cout << "\n[Prober] Phase -> INITIAL_SWEEP_INTERVAL\n";
  phase_.store(Phase::INITIAL_SWEEP_INTERVAL, std::memory_order_release);
  search_queue_.clear();
  best_rate_ = -1e9;
  
  int p = config_.interval_size;
  while (p >= config_.min_interval) {
    search_queue_.push_back(p);
    int next = p / 2;
    if (next < config_.min_interval && p != config_.min_interval) {
      search_queue_.push_back(config_.min_interval);
      break;
    }
    if (next < config_.min_interval) break;
    p = next;
  }
  
  current_test_interval_ = search_queue_.front();
  search_queue_.erase(search_queue_.begin());
  dispatcher_.set_interval_size(current_test_interval_);
}

void Prober::start_probing_threads() {
  std::cout << "\n[Prober] Phase -> PROBING_THREADS (" << config_.probe_test_steps << " steps/test)\n";
  phase_.store(Phase::PROBING_THREADS, std::memory_order_release);
  search_queue_.clear();
  
  int base = best_threads_ != -1 ? best_threads_ : config_.num_threads;
  int delta = std::max(1, base / 4); 
  
  if (base - delta >= config_.thread_min_count) search_queue_.push_back(base - delta);
  search_queue_.push_back(base);
  if (base + delta <= config_.num_threads) search_queue_.push_back(base + delta);

  current_test_threads_ = search_queue_.front();
  search_queue_.erase(search_queue_.begin());
  dispatcher_.set_active_threads(current_test_threads_);
}

void Prober::start_probing_interval() {
  std::cout << "\n[Prober] Phase -> PROBING_INTERVAL (" << config_.probe_test_steps << " steps/test)\n";
  phase_.store(Phase::PROBING_INTERVAL, std::memory_order_release);
  search_queue_.clear();

  int base  = best_interval_ != -1 ? best_interval_ : config_.interval_size;
  int delta = std::max(1, base / 4);  // Fixed: was std::max(config_.min_interval, base / 4)
                                       // which caused delta=32 when min_interval=32, base=32
                                       // giving queue [24, 32, 64] instead of [24, 32, 40]

  if (base - delta >= config_.min_interval)  search_queue_.push_back(base - delta);
  search_queue_.push_back(base);
  if (base + delta <= config_.interval_size) search_queue_.push_back(base + delta);

  current_test_interval_ = search_queue_.front();
  search_queue_.erase(search_queue_.begin());
  dispatcher_.set_interval_size(current_test_interval_);
}

void Prober::start_execution_phase() {
  std::cout << "\n[Prober] Phase -> EXECUTION (" << config_.probe_exec_steps << " steps)\n";
  phase_.store(Phase::EXECUTION_PHASE, std::memory_order_release);
  execution_steps_.store(0, std::memory_order_relaxed);
  
  // NOTE: Reset best_rate so the next probe cycle finds fresh local optima.
  best_rate_ = -1e9; 
}