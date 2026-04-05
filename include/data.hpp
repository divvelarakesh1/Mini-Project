#pragma once

#include <Eigen/Core>
#include <atomic>
#include <mutex>
#include <string>
#include <vector>

// ============================================================================
//  DataLoader — Thread-safe mini-batch provider for CIFAR-10 and MNIST
//
//  Binary file formats (little-endian):
//
//  CIFAR-10  (data_batch_1.bin … data_batch_5.bin / test_batch.bin)
//    Each record: 1 byte label  +  3072 bytes (R plane, G plane, B plane)
//    50 000 training samples  across 5 files of 10 000 each (32×32×3)
//
//  Usage (one shared instance, created before threads are spawned):
//
//    DataLoader loader;
//    loader.load_cifar10("/path/to/cifar-10-batches-bin");
//
//    // Each thread calls:
//    auto [X, Y] = loader.next_batch(64);   // returns Eigen matrices
// ============================================================================

class DataLoader {
public:
    // ctor – all state initialised to "empty"
    DataLoader();

    // -----------------------------------------------------------------------
    // Loaders – call exactly one of these before spawning threads.
    // The images are normalised to [0, 1] and stored column-major so that
    // each *column* is one sample (matches MiniDNN / Eigen convention).
    // -----------------------------------------------------------------------

    /**
     * Load the CIFAR-10 training set.
     * @param dir  Directory containing data_batch_1.bin … data_batch_5.bin
     */
    void load_cifar10(const std::string& dir);

    /**
     * Load the MNIST dataset.
     * @param dir  Directory containing train-images-idx3-ubyte and train-labels-idx1-ubyte
     */
    void load_mnist(const std::string& dir);

    // -----------------------------------------------------------------------
    // Batch access – safe to call concurrently from multiple threads.
    //
    // Each call atomically claims the next `batch_size` samples.  When the
    // internal cursor reaches the end of the dataset it wraps around to 0
    // (epoch boundary), so callers never get an empty batch.
    //
    // Returns:
    //   batch_X  – shape (num_features × batch_size)   float64
    //   batch_Y  – shape (num_classes  × batch_size)   float64  (one-hot)
    // -----------------------------------------------------------------------
    std::pair<Eigen::MatrixXd, Eigen::MatrixXd> next_batch(int batch_size);

    // -----------------------------------------------------------------------
    // Informational helpers
    // -----------------------------------------------------------------------
    int  num_samples()  const { return static_cast<int>(labels_.size()); }
    int  num_features() const { return num_features_; }
    int  num_classes()  const { return num_classes_;  }
    bool is_loaded()    const { return !labels_.empty(); }

    // Reset the internal cursor to the beginning of the dataset
    // (not normally needed – next_batch wraps automatically).
    void reset_cursor() { cursor_.store(0, std::memory_order_relaxed); }

    // Shuffle samples in-place (acquire lock – not safe to call while
    // threads are already calling next_batch).
    void shuffle();

private:
    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    // Read one CIFAR-10 binary file (10 000 records) into internal storage
    void load_cifar10_file(const std::string& path);

    // -----------------------------------------------------------------------
    // Storage (column-major: each column = one sample)
    // images_ : num_features × num_samples   (float64, pixel / 255.0)
    // labels_ : num_samples                  (integer class index)
    // -----------------------------------------------------------------------
    Eigen::MatrixXd      images_;          // (3072, N) – columns = samples
    std::vector<uint8_t> labels_;          // (N)       – integer class ids
    int                  num_features_{0}; // 3072 for all CIFAR variants
    int                  num_classes_{0};  // 10 or 100

    // Atomic cursor – each thread claims the next slice
    std::atomic<int> cursor_{0};

    // Mutex protecting shuffle() and initial load
    std::mutex mtx_;
};
