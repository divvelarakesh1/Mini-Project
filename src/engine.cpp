#include "engine.hpp"
#include "worker.hpp"
#include "prober.hpp"
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

TrainingEngine::TrainingEngine(const Config &config, DataLoader &loader)
    : config_(config), loader_(loader), monitor_(config, loader.num_samples()),
      dispatcher_(config), global_model_(config) {
  
  if (config_.use_probing || config_.use_thread_probing) {
    prober_ = std::make_unique<Prober>(config_, dispatcher_);
    monitor_.set_prober(prober_.get());
  }
  
  w_global_ = global_model_.get_weights();

#ifdef _OPENMP
  omp_set_num_threads(config_.num_threads);
#endif
}

TrainingEngine::~TrainingEngine() = default;

void TrainingEngine::run() {
  std::cout << "\n=====================================\n";
  std::cout << "[Engine] Booting Parallel Engine\n";
  std::cout << "[Engine] Threads Requested: " << config_.num_threads << "\n";
  std::cout << "[Engine] Momentum factor:   " << config_.momentum << "\n";

  // Print optimizer mode details
  if (config_.opt_mode == OptimizerMode::DC_ASGD_C) {
    std::cout << "[Engine] Optimizer: DC_ASGD_C (constant lambda="
              << config_.lambda << ")\n";
  } else if (config_.opt_mode == OptimizerMode::DC_ASGD_A) {
    std::cout << "[Engine] Optimizer: DC_ASGD_A (adaptive lambda0="
              << config_.lambda << ", m=" << config_.dc_asgd_rms_momentum
              << ", eps=" << config_.dc_asgd_rms_epsilon << ")\n";
  } else {
    std::cout << "[Engine] Optimizer: STANDARD_SGD\n";
  }

  switch (config_.exec_mode) {
  case ExecutionMode::SEQUENTIAL:
    std::cout << "[Engine] Mode: SEQUENTIAL (Single Threaded)\n";
    std::cout << "=====================================\n\n";
    run_sequential();
    break;
    break;
  case ExecutionMode::ASYNC_HOGWILD:
    std::cout << "[Engine] Mode: ASYNC_HOGWILD (Hogwild!)\n";
    std::cout << "=====================================\n\n";
    run_parallel_async();
    break;
  case ExecutionMode::INTERVAL_ASYNC:
    std::cout << "[Engine] Mode: INTERVAL_ASYNC\n";
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