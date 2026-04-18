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
 * @struct Config
 * @brief The globally scoped configuration struct for the ML Parallel Engine.
 */
struct Config {
  double target_epochs = 0.0;
  double target_time_seconds = 0.0;
  double eta = 0.1;                      // Learning rate
  double momentum = 0.9;                 // Momentum coefficient
  double lambda = 0.04;                  // DC-ASGD base λ₀
  double dc_asgd_rms_momentum = 0.95;    // Adaptive λ moving average decay (m)
  double dc_asgd_rms_epsilon = 1e-7;     // Adaptive λ numerical stability (ε)
  int batch_size = 64;            // Mini-batch size
  int interval_size = 128;        // Default static Interval boundary size

  // -------------------------------------------------------------------------
  // Probing strategy for Interval Async and Thread Balancing
  // -------------------------------------------------------------------------
  bool use_probing = false;              // -c probe (Interval Probing)
  bool use_thread_probing = false;       // Enable dynamic thread count optimization
  int probe_initial_interval = 64;       // -y (o_semisync_period)
  int probe_min_interval = 4;            // -m (o_semisync_period_min)
  int probe_test_steps = 100;            // Number of steps to run each probe test
  int probe_exec_steps = 5000;          // Steps of fixed execution between probes
  int thread_min_count = 1;              // Minimum thread count for probing
  // -------------------------------------------------------------------------

  ExecutionMode exec_mode = ExecutionMode::ASYNC_HOGWILD;
  OptimizerMode opt_mode = OptimizerMode::DC_ASGD_A;

  StopMode stop_mode = StopMode::EPOCHS;

  int num_threads = 32;   // Number of threads
  DatasetType dataset = DatasetType::CIFAR; // Target dataset
};


Config load_config(const std::string &path);
