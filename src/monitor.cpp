#include "monitor.hpp"
#include "config.hpp"
#include <iomanip>
#include <iostream>

Monitor::Monitor(const Config &config, int total_samples)
    : config_(config), total_samples_(total_samples), start_time_(std::chrono::high_resolution_clock::now()),
      last_log_time_(start_time_) {
  active_stats_.resize(config_.num_threads);
  snapshot_stats_.resize(config_.num_threads);
}

void Monitor::record(int thread_id, double batch_loss, int batch_size) {
  // 1) 100% Lock-Free Array Write — each thread writes only to its own slot
  auto &ts = active_stats_[thread_id];
  ts.batches++;
  ts.accumulated_loss += batch_loss;
  ts.images += batch_size;
  // if (batch_loss < ts.min_loss)
  //   ts.min_loss = batch_loss;
  // if (batch_loss > ts.max_loss)
  //   ts.max_loss = batch_loss;

  // 2) Compute current epoch from total images across ALL threads
  long long total_images_quick = 0;
  for (size_t i = 0; i < config_.num_threads; i++) {
    total_images_quick += active_stats_[i].images;
  }
  double current_epoch = (total_samples_ > 0)
                             ? (static_cast<double>(total_images_quick) / total_samples_)
                             : 0.0;

  // 3) Log every 0.1 epochs (10 times per epoch)
  double log_epoch_interval = 0.1;
  if (current_epoch - last_logged_epoch_ < log_epoch_interval) {
    return;
  }

  // 4) Only ONE thread enters the print section at a time
  std::lock_guard<std::mutex> lock(print_mtx_);

  // Re-check after acquiring lock (another thread may have just logged)
  total_images_quick = 0;
  for (size_t i = 0; i < config_.num_threads; i++) {
    total_images_quick += active_stats_[i].images;
  }
  current_epoch = (total_samples_ > 0)
                      ? (static_cast<double>(total_images_quick) / total_samples_)
                      : 0.0;

  if (current_epoch - last_logged_epoch_ < log_epoch_interval) {
    return;
  }

  auto now = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> total_elapsed = now - start_time_;
  std::chrono::duration<double> interval_elapsed = now - last_log_time_;

  long long interval_batches = 0;
  long long interval_images = 0;
  double interval_loss = 0.0;
  // double global_min = 1e9;
  // double global_max = -1e9;

  // Snapshot and delta math (safely reading other threads' slots)
  for (size_t i = 0; i < config_.num_threads; i++) {
    long long b = active_stats_[i].batches;
    double l = active_stats_[i].accumulated_loss;
    long long img = active_stats_[i].images;

    interval_batches += (b - snapshot_stats_[i].batches);
    interval_loss += (l - snapshot_stats_[i].accumulated_loss);
    interval_images += (img - snapshot_stats_[i].images);

    // if (active_stats_[i].min_loss < global_min)
    //   global_min = active_stats_[i].min_loss;
    // if (active_stats_[i].max_loss > global_max)
    //   global_max = active_stats_[i].max_loss;

    snapshot_stats_[i].batches = b;
    snapshot_stats_[i].accumulated_loss = l;
    snapshot_stats_[i].images = img;
  }

  double avg_loss =
      interval_batches > 0 ? (interval_loss / interval_batches) : 0.0;
  double throughput = interval_elapsed.count() > 0
                          ? (interval_images / interval_elapsed.count())
                          : 0.0;

  // Per-thread image breakdown
  std::cout << std::fixed << std::setprecision(4)
            << "[Monitor] Epoch: " << std::setw(6) << std::setprecision(2) << current_epoch
            << " | Elapsed: " << std::setw(6) << total_elapsed.count()
            << "s\n"
            << "          Avg Loss: " << std::setw(8) << std::setprecision(4) << avg_loss << "\n"
            // << " | Min: " << std::setw(8) << global_min
            // << " | Max: " << std::setw(8) << global_max << "\n"
            << "          Speed: " << std::setprecision(0) << throughput
            << " images/sec"
            << " | Total Img: " << total_images_quick << "\n\n";

  // // Print per-thread image counts
  // std::cout << "          Per-Thread Images: [";
  // for (size_t i = 0; i < config_.num_threads; i++) {
  //   if (i > 0) std::cout << ", ";
  //   std::cout << "T" << i << ": " << active_stats_[i].images;
  // }
  // std::cout << "]\n\n";

  last_logged_epoch_ = current_epoch;
  last_log_time_ = now;
}
