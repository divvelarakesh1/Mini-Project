#pragma once

/**
 * @enum DatasetType
 * @brief Represents the targeted visual dataset format.
 */
enum class DatasetType {
    MNIST,  ///< 28x28 grayscale images (784 features bounds).
    CIFAR   ///< 32x32 color images (3072 features bounds).
};

/**
 * @struct Config
 * @brief The globally scoped configuration struct for the ML Parallel Engine.
 * 
 * Instead of routing individual parameters sequentially, instantiate this struct 
 * downstream to dramatically adjust internal asynchronous locking modes, neural 
 * architectures, learning rates, and delay compensations dynamically.
 */
struct Config {
    int total_steps = 1000;
    double eta = 0.05;            // Learning rate
    double lambda = 0.04;         // DC-ASGD penalty
    int batch_size = 64;          // Mini-batch size
    bool use_dc_asgd = false;      // Enable Delay Compensation (DC-ASGD)
    bool use_sync_mode = true;   // Enable Synchronous Mode
    int num_threads = 4;          // Number of threads
    int log_interval = 100;       // Interval at which the Monitor prints aggregated stats
    DatasetType dataset = DatasetType::CIFAR; // Target dataset
};
