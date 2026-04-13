#include "monitor.hpp"
#include "config.hpp"
#include <iostream>
#include <iomanip>

Monitor::Monitor(const Config& config) 
    : config_(config), 
      start_time_(std::chrono::high_resolution_clock::now()),
      last_log_time_(start_time_) 
{
    active_stats_.resize(config_.num_threads);
    snapshot_stats_.resize(config_.num_threads);
}

void Monitor::record_step(int thread_id, int global_step, double batch_loss, int batch_size) {
    // 1) 100% Lock-Free Array Write
    auto& ts = active_stats_[thread_id];
    ts.steps++;
    ts.accumulated_loss += batch_loss;
    ts.images += batch_size;
    if (batch_loss < ts.min_loss) ts.min_loss = batch_loss;
    if (batch_loss > ts.max_loss) ts.max_loss = batch_loss;

    // 2) Atomic gate check
    long long current_tick = global_tick_.fetch_add(1, std::memory_order_relaxed) + 1;
    
    if (current_tick % config_.log_interval == 0) {
        // 3) Only ONE thread enters this section per log_interval 
        std::lock_guard<std::mutex> lock(print_mtx_);

        auto now = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> total_elapsed = now - start_time_;
        std::chrono::duration<double> interval_elapsed = now - last_log_time_;

        long long interval_steps = 0;
        long long interval_images = 0;
        double interval_loss = 0.0;
        double global_min = 1e9;
        double global_max = -1e9;
        long long total_lifetime_images = 0;

        // Snapshot and delta math! (Safely reading other threads without locking them)
        for(size_t i = 0; i < config_.num_threads; i++) {
            long long s = active_stats_[i].steps;
            double l = active_stats_[i].accumulated_loss;
            long long img = active_stats_[i].images;
            
            interval_steps += (s - snapshot_stats_[i].steps);
            interval_loss += (l - snapshot_stats_[i].accumulated_loss);
            interval_images += (img - snapshot_stats_[i].images);
            total_lifetime_images += img;

            if (active_stats_[i].min_loss < global_min) global_min = active_stats_[i].min_loss;
            if (active_stats_[i].max_loss > global_max) global_max = active_stats_[i].max_loss;

            snapshot_stats_[i].steps = s;
            snapshot_stats_[i].accumulated_loss = l;
            snapshot_stats_[i].images = img;
        }

        double avg_loss = interval_steps > 0 ? (interval_loss / interval_steps) : 0.0;
        double throughput = interval_images / interval_elapsed.count();

        std::cout << std::fixed << std::setprecision(4)
                  << "[Monitor] Step: " << std::setw(6) << global_step 
                  << " | Elapsed: " << std::setw(6) << total_elapsed.count() << "s\n"
                  << "          Avg Loss: " << std::setw(8) << avg_loss
                  << " | Min: " << std::setw(8) << global_min 
                  << " | Max: " << std::setw(8) << global_max << "\n"
                  << "          Speed: " << std::setprecision(0) << throughput << " images/sec" 
                  << " | Total Img: " << total_lifetime_images << "\n\n";

        last_log_time_ = now;
    }
}
