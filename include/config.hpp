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
  SYNC_PARALLEL, ///< Synchronous parallel batch-accumulated SGD
  INTERVAL_ASYNC ///< Interval-Based Asynchronous SGD
};

/**
 * @enum OptimizerMode
 * @brief Represents modifications to typical SGD updating procedures.
 */
enum class OptimizerMode {
  STANDARD_SGD, ///< Vanilla gradient subtraction
  DC_ASGD       ///< Delay-Compensated ASGD penalty logic
};

/**
 * @enum OptimizerAlgorithm
 * @brief Represents the core optimization algorithm.
 */
enum class OptimizerAlgorithm {
  SGD,     ///< Stochastic Gradient Descent (with momentum)
  ADAM,    ///< Adam Optimization
  RMSPROP  ///< RMSProp Optimization
};

/**
 * @struct Config
 * @brief The globally scoped configuration struct for the ML Parallel Engine.
 */
struct Config {
  int total_steps = 1000;
  double eta = 0.1;       // Learning rate
  double momentum = 0.9;  // Momentum coefficient (NEW)
  double lambda = 0.04;   // DC-ASGD penalty
  int batch_size = 64;    // Mini-batch size
  int interval_size = 50; // Interval boundary size
  int interval_decay_freq = 4096; // Steps per interval decay (0 to disable)
  
  // Adam/RMSProp hyperparams
  double beta1 = 0.9;
  double beta2 = 0.999;
  double epsilon = 1e-8;
  
  ExecutionMode exec_mode = ExecutionMode::ASYNC_HOGWILD;
  OptimizerMode opt_mode = OptimizerMode::STANDARD_SGD;
  OptimizerAlgorithm opt_algo = OptimizerAlgorithm::SGD;
  
  int num_threads = 4;    // Number of threads
  int log_interval = 100; // Interval at which the Monitor prints stats
  DatasetType dataset = DatasetType::CIFAR; // Target dataset
};

Config load_config(const std::string& path);