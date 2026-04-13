#pragma once

#include <Eigen/Core>
#include <atomic>
#include <mutex>
#include <string>
#include <vector>

/**
 * @class DataLoader
 * @brief Thread-safe provider for mini-batch data of CIFAR-10 and MNIST datasets.
 *
 * This class handles reading binary dataset formats, normalizes images 
 * to [0, 1] range in contiguous column-major forms for accelerated Eigen 
 * matrix multiplication, and dispenses data robustly across concurrent threads.
 *
 * Usage:
 * @code
 *   DataLoader loader;
 *   loader.load_cifar10("/path/to/data");
 *   auto [X, Y] = loader.next_batch(64); 
 * @endcode
 */
class DataLoader {
public:
    DataLoader();

    // -----------------------------------------------------------------------
    // Dataset Loaders 
    // Load data from disk into memory. Call once before starting worker threads.
    // -----------------------------------------------------------------------

    /**
     * @brief Load the CIFAR-10 training set.
     * @param dir Directory containing data_batch_1.bin … data_batch_5.bin
     */
    void load_cifar10(const std::string& dir);

    /**
     * @brief Load the MNIST dataset.
     * @param dir Directory containing train-images-idx3-ubyte and train-labels-idx1-ubyte
     */
    void load_mnist(const std::string& dir);

    // -----------------------------------------------------------------------
    // Batch Retrieval
    // Safe to call concurrently from multiple threads.
    // -----------------------------------------------------------------------

    /**
     * @brief Fetches the next mini-batch in a thread-safe manner.
     * 
     * Uses atomic operations to allocate a batch range. If the end of the dataset
     * is reached, it seamlessly maps indices via modulo arithmetic.
     * 
     * @param batch_size The number of samples to fetch in the batch.
     * @return A pair of Eigen matrices:
     *         - First:  Feature matrix (num_features x batch_size), values in [0, 1].
     *         - Second: One-hot encoded labels matrix (num_classes x batch_size).
     */
    std::pair<Eigen::MatrixXd, Eigen::MatrixXd> next_batch(int batch_size);

    // -----------------------------------------------------------------------
    // Properties and Accessors
    // -----------------------------------------------------------------------
    int  num_samples()  const { return static_cast<int>(labels_.size()); }
    int  num_features() const { return num_features_; }
    int  num_classes()  const { return num_classes_;  }
    bool is_loaded()    const { return !labels_.empty(); }

    /**
     * @brief Reset the data iterator to the beginning of the dataset.
     */
    void reset_cursor() { cursor_.store(0, std::memory_order_relaxed); }

    /**
     * @brief Shuffles the entire dataset in-place to promote robust learning.
     * @note This acquires a lock and should NOT be called concurrently with next_batch.
     */
    void shuffle();

private:
    // Internal helper to read a single CIFAR-10 data slice from disk.
    void load_cifar10_file(const std::string& path);

    // -----------------------------------------------------------------------
    // Storage Details (Column-Major Matrix Format)
    // - images_: (num_features × num_samples), float64, mapped pixel / 255.0
    // - labels_: (num_samples), integer class ids
    // -----------------------------------------------------------------------
    Eigen::MatrixXd      images_;
    std::vector<uint8_t> labels_;
    
    int num_features_{0};
    int num_classes_{0};

    // Atomic cursor ensures deterministic, collision-free reads across threads.
    // Making it size_t prevents overflow issues over extremely long training jobs.
    std::atomic<size_t> cursor_{0};

    // Mutex locking load and shuffle primitives
    std::mutex mtx_;
};
