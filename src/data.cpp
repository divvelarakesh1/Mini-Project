/**
 * data.cpp  —  DataLoader implementation
 *
 * Thread-safe CIFAR-10 / MNIST dataset provider for concurrent learning models.
 */

#include "data.hpp"

#include <Eigen/Core>
#include <algorithm>
#include <cassert>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>

// ============================================================================
// Constants & Utility functions
// ============================================================================

namespace {
    // CIFAR-10 Constants
    constexpr int kCifarImageBytes    = 3072;  // 32 * 32 * 3
    constexpr int kCifarRecordBytes   = 1 + kCifarImageBytes; // 1 label byte + pixel data
    constexpr int kCifar10Samples     = 10000; // samples per training file
    constexpr int kCifar10FilesCount  = 5;     // data_batch_1 … data_batch_5
    constexpr int kCifar10ClassesCount = 10;

    // MNIST Constants
    constexpr uint32_t kMnistImageMagic = 2051; // 0x0803
    constexpr uint32_t kMnistLabelMagic = 2049; // 0x0801
    constexpr int kMnistExpectedRows    = 28;
    constexpr int kMnistExpectedCols    = 28;
    constexpr int kMnistClassesCount    = 10;

    /**
     * @brief Reverses byte order for unsigned 32-bit integers.
     * MNIST files store dataset statistics natively using the Big Endian format. 
     * However, standard Intel/ARM processors are Little Endian. This utility transforms 
     * parsed header numbers directly into native little-endian equivalents.
     */
    uint32_t swap_endianness_32(uint32_t i) {
        return ((i & 0x000000FF) << 24) |
               ((i & 0x0000FF00) <<  8) |
               ((i & 0x00FF0000) >>  8) |
               ((i & 0xFF000000) >> 24);
    }
}

// ============================================================================
// Constructor
// ============================================================================

DataLoader::DataLoader()
    : num_features_(0), num_classes_(0), cursor_(0) {}

// ============================================================================
// Public loaders
// ============================================================================

void DataLoader::load_cifar10(const std::string& dir)
{
    std::lock_guard<std::mutex> lock(mtx_);

    num_features_ = kCifarImageBytes;
    num_classes_  = kCifar10ClassesCount;

    // Pre-allocate space for optimal memory allocation
    images_.resize(kCifarImageBytes, kCifar10Samples * kCifar10FilesCount);
    labels_.reserve(kCifar10Samples * kCifar10FilesCount);

    for (int i = 1; i <= kCifar10FilesCount; ++i) {
        std::string path = dir + "/data_batch_" + std::to_string(i) + ".bin";
        load_cifar10_file(path);
    }

    // Shrink matrix bounds closely to actual populated elements
    int n = static_cast<int>(labels_.size());
    images_.conservativeResize(kCifarImageBytes, n);

    cursor_.store(0, std::memory_order_relaxed);
    
    std::cout << "[DataLoader] CIFAR-10 train loaded: "
              << n << " samples, features=" << num_features_
              << ", classes=" << num_classes_ << "\n";
}

// ============================================================================
// Thread-Safe Mini-Batch Logic
// ============================================================================

std::pair<Eigen::MatrixXd, Eigen::MatrixXd>
DataLoader::next_batch(int batch_size)
{
    const size_t N = labels_.size();
    assert(N > 0 && "DataLoader: dataset not loaded yet");
    assert(batch_size > 0 && "DataLoader: Requested an empty batch!");

    // -----------------------------------------------------------------------
    // Atomically claim the slice boundaries [start, start + batch_size).
    // The underlying atomic size_t provides near infinite sequence ranges. 
    // This allows fast increment operations devoid of compare-exchange retries.
    // Wrap around boundaries dynamically mapped during offset mapping.
    // -----------------------------------------------------------------------
    size_t start = cursor_.fetch_add(static_cast<size_t>(batch_size), std::memory_order_relaxed);

    Eigen::MatrixXd batch_X(num_features_, batch_size);
    Eigen::MatrixXd batch_Y = Eigen::MatrixXd::Zero(num_classes_, batch_size);

    for (int i = 0; i < batch_size; ++i) {
        size_t idx = (start + i) % N;

        batch_X.col(i)               = images_.col(idx);
        batch_Y(labels_[idx], i)     = 1.0;  // one-hot encode target class
    }

    return {batch_X, batch_Y};
}

// ============================================================================
// Dataset Shuffle Routine
// ============================================================================

void DataLoader::shuffle()
{
    std::lock_guard<std::mutex> lock(mtx_);
    const int N = static_cast<int>(labels_.size());
    if (N == 0) return;

    // Apply permutations directly inplace using standardized sequence mappings
    std::vector<int> perm(N);
    std::iota(perm.begin(), perm.end(), 0);
    std::mt19937 rng{std::random_device{}()};
    std::shuffle(perm.begin(), perm.end(), rng);

    Eigen::MatrixXd      tmp_images(num_features_, N);
    std::vector<uint8_t> tmp_labels(N);

    for (int i = 0; i < N; ++i) {
        tmp_images.col(i) = images_.col(perm[i]);
        tmp_labels[i]     = labels_[perm[i]];
    }

    images_ = std::move(tmp_images);
    labels_ = std::move(tmp_labels);

    cursor_.store(0, std::memory_order_relaxed);
}

// ============================================================================
// Private: CIFAR-10 Parser Helper
// ============================================================================

void DataLoader::load_cifar10_file(const std::string& path)
{
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open())
        throw std::runtime_error("[DataLoader] Cannot open: " + path);
        
    // Assert boundary specifications match tightly
    auto size = f.tellg();
    if (size != static_cast<std::streampos>(kCifar10Samples * kCifarRecordBytes)) {
        throw std::runtime_error("[DataLoader] Invalid CIFAR-10 file sizes at: " + path);
    }
    
    f.seekg(0, std::ios::beg);

    // Read full slice bulk directly into managed memory vector buffer
    std::vector<uint8_t> buffer(size);
    if (!f.read(reinterpret_cast<char*>(buffer.data()), size)) {
        throw std::runtime_error("[DataLoader] Read constraint failure in file: " + path);
    }

    int col = static_cast<int>(labels_.size());
    size_t offset = 0;

    for (int i = 0; i < kCifar10Samples; ++i) {
        labels_.push_back(buffer[offset]);

        // CIFAR encodes red mapping initially, then green blocks followed by blue arrays. 
        // We unpack sequentially into dense column-major target blocks scaling to [0.0, 1.0].
        for (int c = 0; c < 3; ++c) {
            for (int y = 0; y < 32; ++y) {
                for (int x = 0; x < 32; ++x) {
                    images_(c * 1024 + x * 32 + y, col) = buffer[offset + 1 + c * 1024 + y * 32 + x] / 255.0;
                }
            }
        }

        offset += kCifarRecordBytes;
        ++col;
    }
}

// ============================================================================
// Public: MNIST Binary Parser Driver
// ============================================================================

void DataLoader::load_mnist(const std::string& dir)
{
    std::lock_guard<std::mutex> lock(mtx_);

    std::string images_path = dir + "/train-images-idx3-ubyte";
    std::string labels_path = dir + "/train-labels-idx1-ubyte";

    std::ifstream fImages(images_path, std::ios::binary);
    std::ifstream fLabels(labels_path, std::ios::binary);

    if (!fImages.is_open()) throw std::runtime_error("[DataLoader] Cannot open MNIST images: " + images_path);
    if (!fLabels.is_open()) throw std::runtime_error("[DataLoader] Cannot open MNIST labels: " + labels_path);

    // ========================================================================
    // Header Identification Check 
    // ========================================================================
    uint32_t magic_img = 0, magic_lbl = 0;
    fImages.read(reinterpret_cast<char*>(&magic_img), 4);
    fLabels.read(reinterpret_cast<char*>(&magic_lbl), 4);

    magic_img = swap_endianness_32(magic_img);
    magic_lbl = swap_endianness_32(magic_lbl);

    if (magic_img != kMnistImageMagic) throw std::runtime_error("[DataLoader] Invalid MNIST image magic byte format.");
    if (magic_lbl != kMnistLabelMagic) throw std::runtime_error("[DataLoader] Invalid MNIST label magic byte format.");

    uint32_t num_images = 0, num_labels = 0;
    uint32_t rows = 0, cols = 0;

    fImages.read(reinterpret_cast<char*>(&num_images), 4);
    fLabels.read(reinterpret_cast<char*>(&num_labels), 4);
    fImages.read(reinterpret_cast<char*>(&rows), 4);
    fImages.read(reinterpret_cast<char*>(&cols), 4);

    num_images = swap_endianness_32(num_images);
    num_labels = swap_endianness_32(num_labels);
    rows = swap_endianness_32(rows);
    cols = swap_endianness_32(cols);

    if (num_images != num_labels) throw std::runtime_error("[DataLoader] Divergent sequence dimensions between labels and metrics.");
    if (rows != kMnistExpectedRows || cols != kMnistExpectedCols) throw std::runtime_error("[DataLoader] Data geometry validation failed expected sizes.");

    num_features_ = rows * cols; // Should equal 784 linearly mapped
    num_classes_  = kMnistClassesCount;

    images_.resize(num_features_, num_images);
    labels_.reserve(num_labels);

    // ========================================================================
    // Dense Matrix Class Decoding
    // ========================================================================
    std::vector<uint8_t> lbl_buffer(num_labels);
    if (!fLabels.read(reinterpret_cast<char*>(lbl_buffer.data()), num_labels)) {
        throw std::runtime_error("[DataLoader] Exhaustive dataset extraction interrupted.");
    }
    for (uint32_t i = 0; i < num_labels; ++i) {
        labels_.push_back(lbl_buffer[i]);
    }

    // ========================================================================
    // Image Layout Rendering Routine
    // ========================================================================
    const int img_size = num_features_;
    std::vector<uint8_t> img_buffer(img_size);
    for (uint32_t i = 0; i < num_images; ++i) {
        if (!fImages.read(reinterpret_cast<char*>(img_buffer.data()), img_size)) {
            throw std::runtime_error("[DataLoader] Exhaustive dataset parsing interrupted.");
        }
        
        // Iterating uniformly matching hardware optimized dense sequences over matrices natively.
        for (uint32_t y = 0; y < rows; ++y) {
            for (uint32_t x = 0; x < cols; ++x) {
                images_(x * rows + y, i) = img_buffer[y * cols + x] / 255.0;
            }
        }
    }

    cursor_.store(0, std::memory_order_relaxed);
    
    std::cout << "[DataLoader] MNIST testnet loaded: "
              << num_images << " samples, features=" << num_features_
              << ", classes=" << num_classes_ << "\n";
}
