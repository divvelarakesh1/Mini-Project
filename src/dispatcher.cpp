#include "dispatcher.hpp"

Dispatcher::Dispatcher(const Config &config)
    : interval_size_(config.interval_size),
      active_threads_(config.num_threads),
      mode_(config.exec_mode) {
}

std::tuple<bool, int> Dispatcher::try_start_step(int thread_id) {
    // Add your interval gating logic here if INTERVAL_ASYNC is used
    int current_step = accepted_steps_.load(std::memory_order_relaxed);
    return {true, current_step}; 
}

bool Dispatcher::finish_step(int step_index, double batch_loss) {
    accepted_steps_.fetch_add(1, std::memory_order_relaxed);
    return true;
}

bool Dispatcher::is_thread_active(int thread_id) const {
    std::lock_guard<std::mutex> lock(mtx_);
    return thread_id < active_threads_;
}

int Dispatcher::active_threads() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return active_threads_;
}

void Dispatcher::set_interval_size(int size) {
    interval_size_.store(size, std::memory_order_release);
}

// --- Sleep/Wake Implementation ---

void Dispatcher::set_active_threads(int count) {
    {
        std::lock_guard<std::mutex> lock(mtx_);
        active_threads_ = count;
    }
    // Instantly wake all sleeping threads to re-evaluate their state
    cv_.notify_all(); 
}

void Dispatcher::wait_for_active(int thread_id) {
    std::unique_lock<std::mutex> lock(mtx_);
    // Puts the thread to sleep (0% CPU usage) UNTIL this lambda evaluates to true
    cv_.wait(lock, [this, thread_id] {
        return thread_id < active_threads_ || !is_running_.load(std::memory_order_acquire);
    });
}

void Dispatcher::stop_all() {
    is_running_.store(false, std::memory_order_release);
    cv_.notify_all(); // Wake up any sleeping threads so they can gracefully exit
}