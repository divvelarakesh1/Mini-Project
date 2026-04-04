#include "monitor.hpp"
#include "config.hpp"
#include <iostream>
#include <iomanip>

Monitor::Monitor(const Config& config) 
    : config_(config), 
      start_time_(std::chrono::high_resolution_clock::now()),
      last_log_time_(start_time_) 
{}

void Monitor::record_step(int global_step, double batch_loss, int batch_size) {
    std::lock_guard<std::mutex> lock(mtx_);

    accumulated_loss_ += batch_loss;
    accumulated_steps_++;
    accumulated_images_ += batch_size;

    // Output stats only every config.log_interval steps.
    // Notice global_step is usually bounded independently by threads, 
    // so checking accumulated_steps_ >= interval is more thread-reliable.
    if (accumulated_steps_ >= config_.log_interval) {
        auto now = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> total_elapsed = now - start_time_;
        std::chrono::duration<double> interval_elapsed = now - last_log_time_;

        double avg_loss = accumulated_loss_ / accumulated_steps_;
        double throughput = accumulated_images_ / interval_elapsed.count();

        std::cout << std::fixed << std::setprecision(4)
                  << "[Monitor] Step: " << std::setw(6) << global_step 
                  << " | Elapsed: " << std::setw(6) << total_elapsed.count() << "s"
                  << " | Avg Loss: " << std::setw(8) << avg_loss
                  << " | Speed: " << std::setprecision(0) << throughput << " images/sec" << "\n";

        // Reset accumulators
        accumulated_loss_ = 0.0;
        accumulated_steps_ = 0;
        accumulated_images_ = 0;
        last_log_time_ = now;
    }
}
