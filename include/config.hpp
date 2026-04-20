#pragma once

#include <string>

/**
 * @enum DatasetType
 * @brief Represents the targeted visual dataset format.
 */
enum class DatasetType {
  MNIST, ///< 28x28 grayscale images (784 features bounds).
  CIFAR  ///< 32x32 color images (3072 features bounds).
};

/**
 * @enum ExecutionMode
 * @brief Represents the parallelism mode for the training engine.
 */
enum class ExecutionMode {
  SEQUENTIAL,    ///< Pure single-threaded synchronous SGD
  ASYNC_HOGWILD, ///< Fully asynchronous Lock-free Hogwild! workers
  INTERVAL_ASYNC ///< Interval-Based Asynchronous SGD
};

/**
 * @enum OptimizerMode
 * @brief Represents modifications to typical SGD updating procedures.
 */
enum class OptimizerMode {
  STANDARD_SGD, ///< Vanilla gradient subtraction
  DC_ASGD_C,    ///< Delay-Compensated ASGD with constant λ
  DC_ASGD_A     ///< Delay-Compensated ASGD with adaptive λ (RMSProp-style)
};

/**
 * @enum StopMode
 * @brief Represents the primary training termination criterion.
 */
enum class StopMode {
  EPOCHS, ///< Stop after target_epochs total data epochs
  TIME    ///< Stop after target_time_seconds wall-clock seconds
};

/**
 * @enum IntervalMode
 * @brief Represents how the synchronization interval is adjusted.
 */
enum class IntervalMode {
  STATIC,  ///< Single fixed interval for duration of run.
  DECAY,   ///< Gradually reduces the interval periodically.
  PROBING  ///< Dynamically optimizes interval using probing.
};

/**
 * @enum ThreadMode
 * @brief Represents how the thread count is adjusted.
 */
enum class ThreadMode {
  STATIC,  ///< Use num_threads for the duration of the run.
  PROBING  ///< Dynamically optimizes thread count using probing.
};

/**
 * @struct Config
 * @brief The globally scoped configuration struct for the ML Parallel Engine.
 */
struct Config {
  // 1. Core Hyperparameters
  double eta = 0.1;                      // Learning rate
  double momentum = 0.9;                 // Momentum coefficient
  int batch_size = 64;                   // Mini-batch size
  double lambda = 0.04;                  // DC-ASGD base λ₀
  double dc_asgd_rms_momentum = 0.95;    // Adaptive λ moving average decay (m)
  double dc_asgd_rms_epsilon = 1e-7;     // Adaptive λ numerical stability (ε)

  // 2. Synchronization Strategy
  ExecutionMode exec_mode = ExecutionMode::ASYNC_HOGWILD;
  OptimizerMode opt_mode = OptimizerMode::DC_ASGD_A;
  IntervalMode interval_mode = IntervalMode::STATIC;
  ThreadMode thread_mode = ThreadMode::STATIC;
  int interval_size = 128;               // baseline sync boundary
  int min_interval = 4;                  // absolute floor for any adjustment

  // 3. Adjustment Dynamics (Probing / Decay)
  int decay_steps = 1000;                // Interval decrease frequency
  int decay_amount = 4;                  // Amount to subtract from interval
  int probe_test_steps = 100;            // Steps to run each probe test
  int probe_exec_steps = 5000;           // Steps of fixed execution between probes
  int thread_min_count = 1;              // Minimum thread count for probing

  // 4. Resources & Termination
  int num_threads = 32;                  // Max threads to spawn
  DatasetType dataset = DatasetType::CIFAR; 
  StopMode stop_mode = StopMode::EPOCHS;
  double target_epochs = 0.0;
  double target_time_seconds = 0.0;
};


Config load_config(const std::string &path);
