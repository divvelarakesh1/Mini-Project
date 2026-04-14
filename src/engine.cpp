#include "engine.hpp"
#include "worker.hpp"
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

TrainingEngine::TrainingEngine(const Config &config, DataLoader &loader)
    : config_(config), loader_(loader), monitor_(config, loader.num_samples()),
      global_model_(config) {
  w_global_ = global_model_.get_weights();

  // Prepare accumulator for sync mode
  if (config_.exec_mode == ExecutionMode::SYNC_PARALLEL) {
    g_global_accum_ = global_model_.get_weights();
    for (auto &layer : g_global_accum_) {
      std::fill(layer.begin(), layer.end(), 0.0);
    }
  }

#ifdef _OPENMP
  omp_set_num_threads(config_.num_threads);
#endif
}

void TrainingEngine::run() {
  std::cout << "\n=====================================\n";
  std::cout << "[Engine] Booting Parallel Engine\n";
  std::cout << "[Engine] Threads Requested: " << config_.num_threads << "\n";
  std::cout << "[Engine] Momentum factor:   " << config_.momentum << "\n";

  switch (config_.exec_mode) {
  case ExecutionMode::SEQUENTIAL:
    std::cout << "[Engine] Mode: SEQUENTIAL (Single Threaded)\n";
    std::cout << "=====================================\n\n";
    run_sequential();
    break;
  case ExecutionMode::SYNC_PARALLEL:
    std::cout << "[Engine] Mode: SYNC_PARALLEL\n";
    std::cout << "=====================================\n\n";
    run_parallel_sync();
    break;
  case ExecutionMode::ASYNC_HOGWILD:
    std::cout << "[Engine] Mode: ASYNC_HOGWILD (Hogwild!)\n";
    std::cout << "=====================================\n\n";
    run_parallel_async();
    break;
  }

  std::cout << "\n=====================================\n";
  std::cout << "[Engine] Training Terminated Safely!\n";
  std::cout << "=====================================\n";
}

void TrainingEngine::run_sequential() {
  Worker::run_sequential(w_global_, loader_, config_, monitor_);
}

void TrainingEngine::run_parallel_sync() {
#pragma omp parallel
  {
    int thread_id = 0;
#ifdef _OPENMP
    thread_id = omp_get_thread_num();
#endif
    Worker::run_sync(thread_id, w_global_, g_global_accum_, loader_, config_,
                     monitor_);
  }
}

void TrainingEngine::run_parallel_async() {
#pragma omp parallel
  {
    int thread_id = 0;
#ifdef _OPENMP
    thread_id = omp_get_thread_num();
#endif
    Worker::run_async(thread_id, w_global_, loader_, dispatcher_, config_,
                      monitor_);
  }
}