#include "monitor.hpp"
#include "config.hpp"
#include "prober.hpp"
#include <iomanip>
#include <iostream>

namespace {

// ANSI Colors for Premium UI
const char *RESET = "\033[0m";
const char *BOLD = "\033[1m";
const char *CYAN = "\033[36m";
const char *GREEN = "\033[32m";
const char *YELLOW = "\033[33m";
const char *MAGENTA = "\033[35m";
const char *GREY = "\033[90m";

bool update_stop_state(const Config &config, double current_epoch,
                       double elapsed_seconds,
                       std::atomic<bool> &stop_requested) {
  bool stop_now = stop_requested.load(std::memory_order_relaxed);

  if (!stop_now && config.stop_mode == StopMode::EPOCHS &&
      config.target_epochs > 0.0 && current_epoch >= config.target_epochs) {
    stop_requested.store(true, std::memory_order_relaxed);
    stop_now = true;
  }

  if (!stop_now && config.stop_mode == StopMode::TIME &&
      config.target_time_seconds > 0.0 &&
      elapsed_seconds >= config.target_time_seconds) {
    stop_requested.store(true, std::memory_order_relaxed);
    stop_now = true;
  }

  return stop_now;
}

} // namespace

Monitor::Monitor(const Config &config, int total_samples)
    : config_(config), total_samples_(total_samples), start_time_(Clock::now()),
      last_log_time_(start_time_) {
  active_stats_.resize(config_.num_threads);
  snapshot_stats_.resize(config_.num_threads);
}

bool Monitor::should_stop() const {
  return stop_requested_.load(std::memory_order_relaxed);
}

std::string Monitor::get_progress_bar(double progress, int width) const {
  progress = std::max(0.0, std::min(1.0, progress));
  int pos = static_cast<int>(width * progress);

  std::string bar = "[";
  for (int i = 0; i < width; ++i) {
    if (i < pos)
      bar += "■";
    else if (i == pos)
      bar += "■"; // Subtle leading edge
    else
      bar += " ";
  }
  bar += "]";
  return bar;
}

std::string Monitor::format_time(double seconds) const {
  if (seconds < 0)
    return "--:--";
  int m = static_cast<int>(seconds) / 60;
  int s = static_cast<int>(seconds) % 60;
  char buf[16];
  std::snprintf(buf, sizeof(buf), "%02d:%02d", m, s);
  return std::string(buf);
}

void Monitor::record(int thread_id, double batch_loss, int batch_size) {
  // 1) HOT PATH: 100% Lock-Free Atomic Updates
  auto &ts = active_stats_[thread_id];
  ts.batches++;
  ts.accumulated_loss += batch_loss;
  ts.images += batch_size;

  if (prober_) {
    prober_->update(batch_loss);
  }

  long long current_total_images = global_images_processed_.fetch_add(
                                       batch_size, std::memory_order_relaxed) +
                                   batch_size;
  global_batches_processed_.fetch_add(1, std::memory_order_relaxed);

  // 2) FAST PROGRESS CHECK: Compute current epoch and check throttle
  double current_epoch =
      (total_samples_ > 0)
          ? (static_cast<double>(current_total_images) / total_samples_)
          : 0.0;

  auto now = Clock::now();
  std::chrono::duration<double> total_elapsed = now - start_time_;

  bool stop_now = update_stop_state(config_, current_epoch,
                                    total_elapsed.count(), stop_requested_);

  // Throttling logic: Log every 0.1 epochs or on force-stop
  double log_epoch_interval = 0.1;
  bool should_log =
      (current_epoch - last_logged_epoch_ >= log_epoch_interval) || stop_now;

  if (!should_log)
    return;

  // 3) SYNC ZONE: Only one thread format and print
  std::lock_guard<std::mutex> lock(print_mtx_);

  // Re-check state inside lock (protection against redundant logs from trailing
  // threads)
  if (!stop_now && (current_epoch - last_logged_epoch_ < log_epoch_interval))
    return;
  if (stop_now && final_log_emitted_.load(std::memory_order_relaxed))
    return;

  std::chrono::duration<double> interval_elapsed = now - last_log_time_;

  long long interval_batches = 0;
  long long interval_images = 0;
  double interval_loss = 0.0;

  // Snapshot delta math
  for (size_t i = 0; i < active_stats_.size(); i++) {
    long long b = active_stats_[i].batches;
    double l = active_stats_[i].accumulated_loss;
    long long img = active_stats_[i].images;

    interval_batches += (b - snapshot_stats_[i].batches);
    interval_loss += (l - snapshot_stats_[i].accumulated_loss);
    interval_images += (img - snapshot_stats_[i].images);

    snapshot_stats_[i].batches = b;
    snapshot_stats_[i].accumulated_loss = l;
    snapshot_stats_[i].images = img;
  }

  double avg_loss =
      (interval_batches > 0) ? (interval_loss / interval_batches) : 0.0;
  double throughput = (interval_elapsed.count() > 0)
                          ? (interval_images / interval_elapsed.count())
                          : 0.0;

  // 4) ETA Calculation
  double progress_ratio = 0.0;
  if (config_.stop_mode == StopMode::EPOCHS && config_.target_epochs > 0) {
    progress_ratio = current_epoch / config_.target_epochs;
  } else if (config_.stop_mode == StopMode::TIME &&
             config_.target_time_seconds > 0) {
    progress_ratio = total_elapsed.count() / config_.target_time_seconds;
  }

  double eta_seconds = -1.0;
  if (progress_ratio > 0.01 && progress_ratio < 1.0 && throughput > 1.0) {
    if (config_.stop_mode == StopMode::EPOCHS) {
      double remaining_images =
          (config_.target_epochs - current_epoch) * total_samples_;
      eta_seconds = remaining_images / throughput;
    } else {
      eta_seconds = config_.target_time_seconds - total_elapsed.count();
    }
  }

  // 5) PREMIERE OUTPUT FORMATTING
  std::cout << "\r" << std::flush; // Clean status line start
  std::cout << BOLD << CYAN << "[Monitor] " << RESET << "Epoch: " << BOLD
            << std::fixed << std::setprecision(2) << std::setw(6)
            << current_epoch << RESET << " | " << GREY
            << get_progress_bar(progress_ratio) << RESET << " | " << BOLD
            << YELLOW << "Loss: " << std::scientific << std::setprecision(4)
            << avg_loss << RESET << "\n";

  std::cout << "          " << GREEN << std::fixed << std::setprecision(0)
            << std::setw(6) << throughput << " img/s" << RESET << GREY << " | "
            << RESET << "Elapsed: " << format_time(total_elapsed.count())
            << GREY << " | " << RESET << "ETA: " << BOLD
            << (eta_seconds >= 0 ? format_time(eta_seconds) : "--:--") << RESET
            << " | Total: " << MAGENTA << current_total_images << RESET
            << "\n\n"
            << std::flush;

  last_logged_epoch_ = current_epoch;
  last_log_time_ = now;

  if (stop_now) {
    final_log_emitted_.store(true, std::memory_order_relaxed);
    std::cout << BOLD << GREEN
              << ">> Stopping criteria met. Finalizing run...\n"
              << RESET << std::endl;
  }
}
