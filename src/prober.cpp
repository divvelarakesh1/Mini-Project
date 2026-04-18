#include "prober.hpp"
#include "dispatcher.hpp"
#include <iostream>
#include <iomanip>
#include <algorithm>

Prober::Prober(const Config &config, Dispatcher &dispatcher)
    : config_(config), dispatcher_(dispatcher), 
      current_test_threads_(config.num_threads),
      current_test_interval_(config.probe_initial_interval) {
  
  if (config_.use_thread_probing) {
    start_initial_thread_sweep();
  } else if (config_.use_probing) {
    start_initial_interval_sweep();
  } else {
    start_steady_state();
  }
}

void Prober::update(double batch_loss) {
  Phase current = phase_.load(std::memory_order_relaxed);

  // ---------------------------------------------------------
  // FAST PATH: STEADY STATE (Zero-overhead, lock-free execution)
  // ---------------------------------------------------------
  if (current == Phase::STEADY_STATE) {
    int ss_count = steady_state_steps_.fetch_add(1, std::memory_order_relaxed) + 1;
    if (ss_count >= config_.probe_exec_steps) {
      // Only transition if we haven't already (prevent multi-thread spam)
      if (steady_state_steps_.exchange(0, std::memory_order_acquire) >= config_.probe_exec_steps) {
        advance_phase();
      }
    }
    return;
  }

  // ---------------------------------------------------------
  // PROBING PATH
  // ---------------------------------------------------------
  int current_count = step_counter_.fetch_add(1, std::memory_order_relaxed) + 1;

  int warmup_limit = std::max(1, config_.probe_test_steps / 10);
  int baseline_limit = warmup_limit + std::max(1, config_.probe_test_steps / 5);

  // 1) Warmup Phase (Discard first 10% of window to let CPU caches settle)
  if (current_count <= warmup_limit) {
    if (current_count == warmup_limit) {
      std::lock_guard<std::mutex> lock(mtx_);
      window_start_time_ = std::chrono::steady_clock::now();
    }
  } 
  // 2) Baseline Phase (Next 20% to establish a noise-resistant initial loss)
  else if (current_count <= baseline_limit) {
    baseline_accum_loss_.fetch_add(static_cast<long long>(batch_loss * 1e6), std::memory_order_relaxed);
    if (current_count == baseline_limit) {
      std::lock_guard<std::mutex> lock(mtx_);
      stabilized_baseline_ = (baseline_accum_loss_.load(std::memory_order_relaxed) / 1e6) / (baseline_limit - warmup_limit);
    }
  }
  // 3) Measurement Phase (The rest of the window)
  else {
    accumulated_loss_.fetch_add(static_cast<long long>(batch_loss * 1e6), std::memory_order_relaxed);
  }

  // ---------------------------------------------------------
  // WINDOW EVALUATION & RACE CONDITION PREVENTION
  // ---------------------------------------------------------
  if (current_count >= config_.probe_test_steps) {
    std::lock_guard<std::mutex> lock(mtx_);
    
    // Double-check to ensure only one thread triggers the evaluation
    if (step_counter_.load(std::memory_order_relaxed) >= config_.probe_test_steps) {
      
      evaluate_current_window();
      
      // Safely reset accumulators for the *next* window while still holding the lock
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
  
  // FIX: Only count the steps that happened *after* the warmup timer started
  int evaluated_steps = config_.probe_test_steps - warmup_limit;
  int measurement_steps = config_.probe_test_steps - baseline_limit;

  double avg_loss = (accumulated_loss_.load(std::memory_order_relaxed) / 1e6) / std::max(1, measurement_steps);
  
  // FIX: Throughput accurately reflects the evaluated steps
  double throughput_ips = (evaluated_steps * config_.batch_size) / (elapsed_ms / 1000.0 + 1e-9);
  
  double loss_improvement = stabilized_baseline_ - avg_loss;
  
  // Rate = (Baseline - Current) * Throughput (Scaled for readability)
  double rate = loss_improvement * throughput_ips * 1000.0;

  Phase current = phase_.load(std::memory_order_relaxed);

  if (current == Phase::INITIAL_SWEEP_THREADS || current == Phase::NEIGHBORHOOD_THREADS) {
    std::cout << "[Prober] Test Threads: " << std::setw(2) << current_test_threads_ 
              << " | " << std::fixed << std::setprecision(0) << std::setw(6) << throughput_ips << " ips"
              << " | rate: " << std::showpos << std::fixed << std::setprecision(4) << rate << std::noshowpos << "\n";
    
    if (rate > best_rate_ || best_threads_ == -1) {
      best_rate_ = rate;
      best_threads_ = current_test_threads_;
    }

    if (!search_queue_.empty()) {
      current_test_threads_ = search_queue_.front();
      search_queue_.erase(search_queue_.begin());
      dispatcher_.set_active_threads(current_test_threads_);
    } else {
      std::cout << "[Prober] Winner (Threads): " << best_threads_ << "\n";
      dispatcher_.set_active_threads(best_threads_);
      advance_phase();
    }
  } 
  else if (current == Phase::INITIAL_SWEEP_INTERVAL || current == Phase::NEIGHBORHOOD_INTERVAL) {
    std::cout << "[Prober] Test Interval: " << std::setw(3) << current_test_interval_ 
              << " | " << std::fixed << std::setprecision(0) << std::setw(6) << throughput_ips << " ips"
              << " | rate: " << std::showpos << std::fixed << std::setprecision(4) << rate << std::noshowpos << "\n";
    
    if (rate > best_rate_ || best_interval_ == -1) {
      best_rate_ = rate;
      best_interval_ = current_test_interval_;
    }

    if (!search_queue_.empty()) {
      current_test_interval_ = search_queue_.front();
      search_queue_.erase(search_queue_.begin());
      dispatcher_.set_interval_size(current_test_interval_);
    } else {
      std::cout << "[Prober] Winner (Interval): " << best_interval_ << "\n";
      dispatcher_.set_interval_size(best_interval_);
      advance_phase();
    }
  }
}

// ---------------------------------------------------------
// STATE MACHINE TRANSITIONS
// ---------------------------------------------------------
void Prober::advance_phase() {
  Phase current = phase_.load(std::memory_order_relaxed);
  
  if (current == Phase::INITIAL_SWEEP_THREADS) {
    if (config_.use_probing) start_initial_interval_sweep();
    else start_steady_state();
  } 
  else if (current == Phase::INITIAL_SWEEP_INTERVAL) {
    start_steady_state();
  } 
  else if (current == Phase::STEADY_STATE) {
    start_neighborhood_threads();
  } 
  else if (current == Phase::NEIGHBORHOOD_THREADS) {
    start_neighborhood_interval();
  } 
  else if (current == Phase::NEIGHBORHOOD_INTERVAL) {
    start_steady_state();
  }
}

void Prober::start_initial_thread_sweep() {
  std::cout << "\n[Prober] Transition -> INITIAL_SWEEP_THREADS\n";
  phase_.store(Phase::INITIAL_SWEEP_THREADS, std::memory_order_release);
  search_queue_.clear();
  
  int t = config_.num_threads;
  while (t >= config_.thread_min_count) {
    search_queue_.push_back(t);
    int next = t / 2;
    if (next < config_.thread_min_count) break;
    t = next;
  }
  std::reverse(search_queue_.begin(), search_queue_.end()); // Ascending test order
  
  if (!search_queue_.empty()) {
    current_test_threads_ = search_queue_.front();
    search_queue_.erase(search_queue_.begin());
    dispatcher_.set_active_threads(current_test_threads_);
  }
  best_rate_ = -1e9;
}

void Prober::start_initial_interval_sweep() {
  std::cout << "\n[Prober] Transition -> INITIAL_SWEEP_INTERVAL\n";
  phase_.store(Phase::INITIAL_SWEEP_INTERVAL, std::memory_order_release);
  search_queue_.clear();
  
  int p = config_.probe_initial_interval;
  while (p >= config_.probe_min_interval) {
    search_queue_.push_back(p);
    int next = p / 2;
    if (next < config_.probe_min_interval) break;
    p = next;
  }
  if (search_queue_.empty()) search_queue_.push_back(config_.probe_initial_interval);
  
  current_test_interval_ = search_queue_.front();
  search_queue_.erase(search_queue_.begin());
  dispatcher_.set_interval_size(current_test_interval_);
  best_rate_ = -1e9;
}

void Prober::start_neighborhood_threads() {
  std::cout << "\n[Prober] Transition -> NEIGHBORHOOD_SWEEP_THREADS\n";
  phase_.store(Phase::NEIGHBORHOOD_THREADS, std::memory_order_release);
  search_queue_.clear();
  
  int base = best_threads_ != -1 ? best_threads_ : config_.num_threads;
  if (base / 2 >= config_.thread_min_count) search_queue_.push_back(base / 2);
  search_queue_.push_back(base);
  if (base * 2 <= config_.num_threads) search_queue_.push_back(base * 2);

  std::sort(search_queue_.begin(), search_queue_.end());
  search_queue_.erase(std::unique(search_queue_.begin(), search_queue_.end()), search_queue_.end());

  current_test_threads_ = search_queue_.front();
  search_queue_.erase(search_queue_.begin());
  dispatcher_.set_active_threads(current_test_threads_);
  best_rate_ = -1e9;
}

void Prober::start_neighborhood_interval() {
  std::cout << "\n[Prober] Transition -> NEIGHBORHOOD_SWEEP_INTERVAL\n";
  phase_.store(Phase::NEIGHBORHOOD_INTERVAL, std::memory_order_release);
  search_queue_.clear();

  int base = best_interval_ != -1 ? best_interval_ : config_.probe_initial_interval;
  if (base / 2 >= config_.probe_min_interval) search_queue_.push_back(base / 2);
  search_queue_.push_back(base);
  search_queue_.push_back(base * 2);

  std::sort(search_queue_.begin(), search_queue_.end());
  search_queue_.erase(std::unique(search_queue_.begin(), search_queue_.end()), search_queue_.end());

  current_test_interval_ = search_queue_.front();
  search_queue_.erase(search_queue_.begin());
  dispatcher_.set_interval_size(current_test_interval_);
  best_rate_ = -1e9;
}

void Prober::start_steady_state() {
  std::cout << "\n[Prober] Transition -> STEADY_STATE (" << config_.probe_exec_steps << " batches)\n";
  phase_.store(Phase::STEADY_STATE, std::memory_order_release);
  steady_state_steps_.store(0, std::memory_order_relaxed);
}