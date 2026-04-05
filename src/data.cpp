/**
 * data.cpp  —  DataLoader implementation
 *
 * Thread-safe CIFAR-10 / CIFAR-100 mini-batch provider.
 *
 * CIFAR binary formats:
 *  CIFAR-10:  [1-byte label | 3072-byte image] × N  (N = 10 000 per file)
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
// Constants
// ============================================================================

static constexpr int kImageBytes   = 3072;  // 32 * 32 * 3
static constexpr int kCifar10N    = 10000;  // samples per training file
static constexpr int kCifar10Files = 5;     // data_batch_1 … data_batch_5
static constexpr int kCifar10Classes  = 10;

// ============================================================================
// ctor
// ============================================================================

DataLoader::DataLoader()
    : num_features_(0), num_classes_(0), cursor_(0) {}

// ============================================================================
// Public loaders
// ============================================================================

void DataLoader::load_cifar10(const std::string& dir)
{
    std::lock_guard<std::mutex> lock(mtx_);

    num_features_ = kImageBytes;
    num_classes_  = kCifar10Classes;

    // Pre-allocate: 50 000 columns
    images_.resize(kImageBytes, kCifar10N * kCifar10Files);
    labels_.reserve(kCifar10N * kCifar10Files);

    for (int i = 1; i <= kCifar10Files; ++i) {
        std::string path = dir + "/data_batch_" + std::to_string(i) + ".bin";
        load_cifar10_file(path);
    }

    // Trim to actual loaded size (in case any file was short)
    int n = static_cast<int>(labels_.size());
    images_.conservativeResize(kImageBytes, n);

    cursor_.store(0, std::memory_order_relaxed);
    std::cout << "[DataLoader] CIFAR-10 train loaded: "
              << n << " samples, features=" << kImageBytes
              << ", classes=" << kCifar10Classes << "\n";
}

// ============================================================================
// next_batch — thread-safe mini-batch fetch
// ============================================================================

std::pair<Eigen::MatrixXd, Eigen::MatrixXd>
DataLoader::next_batch(int batch_size)
{
    const int N = static_cast<int>(labels_.size());
    assert(N > 0 && "DataLoader: dataset not loaded yet");
    assert(batch_size > 0);

    // -----------------------------------------------------------------------
    // 1.  Atomically claim [start, end) in the shuffled sample index space.
    //     We use a simple fetch_add; the claimed window wraps around mod N
    //     so callers always receive exactly batch_size columns.
    // -----------------------------------------------------------------------
    int start = cursor_.fetch_add(batch_size, std::memory_order_relaxed);

    // If this thread wrapped past N, reset the cursor.
    // Multiple threads may race here; that's fine — only one write "wins"
    // and the actual indices below are taken mod N so nothing out-of-bounds.
    if (start >= N) {
        // Soft-reset: many threads may CAS, only one succeeds — doesn't matter
        int expected = start + batch_size;
        int desired  = batch_size;
        cursor_.compare_exchange_weak(expected, desired,
                                      std::memory_order_relaxed);
        start = start % N;
    }

    // -----------------------------------------------------------------------
    // 2.  Gather columns (wrap-around safe)
    // -----------------------------------------------------------------------
    Eigen::MatrixXd batch_X(num_features_, batch_size);
    Eigen::MatrixXd batch_Y = Eigen::MatrixXd::Zero(num_classes_, batch_size);

    for (int i = 0; i < batch_size; ++i) {
        int idx = (start + i) % N;

        batch_X.col(i)               = images_.col(idx);
        batch_Y(labels_[idx], i)     = 1.0;  // one-hot
    }

    return {batch_X, batch_Y};
}

// ============================================================================
// shuffle  (hold mutex — do not call while threads are running)
// ============================================================================

void DataLoader::shuffle()
{
    std::lock_guard<std::mutex> lock(mtx_);
    const int N = static_cast<int>(labels_.size());
    assert(N > 0);

    // Build a permutation and apply it in-place
    std::vector<int> perm(N);
    std::iota(perm.begin(), perm.end(), 0);
    std::mt19937 rng{std::random_device{}()};
    std::shuffle(perm.begin(), perm.end(), rng);

    // Apply permutation to images_ and labels_
    Eigen::MatrixXd   tmp_images(num_features_, N);
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
// Private: load one CIFAR-10 binary file (10 000 records)
// ============================================================================

void DataLoader::load_cifar10_file(const std::string& path)
{
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open())
        throw std::runtime_error("[DataLoader] Cannot open: " + path);
        
    // Robustness: ensure exact strict bounds for CIFAR files
    static constexpr int kRecordBytes = 1 + kImageBytes;
    auto size = f.tellg();
    if (size != static_cast<std::streampos>(kCifar10N * kRecordBytes)) {
        throw std::runtime_error("[DataLoader] Invalid CIFAR-10 file sizes at: " + path);
    }
    
    // Return to start
    f.seekg(0, std::ios::beg);

    std::vector<uint8_t> record(kRecordBytes);

    int col = static_cast<int>(labels_.size());  // append after any prior data

    while (f.read(reinterpret_cast<char*>(record.data()), kRecordBytes)) {
        labels_.push_back(record[0]);

        // Reshape [0, 255] row-major -> [0, 1] col-major (as preferred by MiniDNN)
        for (int c = 0; c < 3; ++c) {
            for (int y = 0; y < 32; ++y) {
                for (int x = 0; x < 32; ++x) {
                    images_(c * 1024 + x * 32 + y, col) = record[1 + c * 1024 + y * 32 + x] / 255.0;
                }
            }
        }

        ++col;
    }
}

// ============================================================================
// Private: Endianness swapping for MNIST format
// ============================================================================
static uint32_t reverseInt(uint32_t i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((uint32_t)c1 << 24) + ((uint32_t)c2 << 16) + ((uint32_t)c3 << 8) + c4;
}

// ============================================================================
// Public: Load MNIST dataset
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

    // Read magic numbers
    uint32_t magic_img = 0, magic_lbl = 0;
    fImages.read(reinterpret_cast<char*>(&magic_img), 4);
    fLabels.read(reinterpret_cast<char*>(&magic_lbl), 4);

    magic_img = reverseInt(magic_img);
    magic_lbl = reverseInt(magic_lbl);

    if (magic_img != 2051) throw std::runtime_error("[DataLoader] Invalid MNIST image file magic number.");
    if (magic_lbl != 2049) throw std::runtime_error("[DataLoader] Invalid MNIST label file magic number.");

    // Read headers
    uint32_t num_images = 0, num_labels = 0;
    uint32_t rows = 0, cols = 0;

    fImages.read(reinterpret_cast<char*>(&num_images), 4);
    fLabels.read(reinterpret_cast<char*>(&num_labels), 4);
    fImages.read(reinterpret_cast<char*>(&rows), 4);
    fImages.read(reinterpret_cast<char*>(&cols), 4);

    num_images = reverseInt(num_images);
    num_labels = reverseInt(num_labels);
    rows = reverseInt(rows);
    cols = reverseInt(cols);

    if (num_images != num_labels) throw std::runtime_error("[DataLoader] Mismatch between MNIST images and labels count.");

    // Robustness: ensure sizes matches our expectations.
    if (rows != 28 || cols != 28) throw std::runtime_error("[DataLoader] Expected 28x28 MNIST images.");

    num_features_ = rows * cols; // 784
    num_classes_  = 10;

    images_.resize(num_features_, num_images);
    labels_.reserve(num_labels);

    // Read labels
    std::vector<uint8_t> lbl_buffer(num_labels);
    if (!fLabels.read(reinterpret_cast<char*>(lbl_buffer.data()), num_labels)) {
        throw std::runtime_error("[DataLoader] Failed to read all MNIST labels.");
    }
    for (uint32_t i = 0; i < num_labels; ++i) {
        labels_.push_back(lbl_buffer[i]);
    }

    // Read images
    const int img_size = num_features_;
    std::vector<uint8_t> img_buffer(img_size);
    for (uint32_t i = 0; i < num_images; ++i) {
        if (!fImages.read(reinterpret_cast<char*>(img_buffer.data()), img_size)) {
            throw std::runtime_error("[DataLoader] Failed to read all MNIST images.");
        }
        // Reshape [0, 255] row-major -> [0, 1] col-major (as preferred by MiniDNN)
        for (uint32_t y = 0; y < rows; ++y) {
            for (uint32_t x = 0; x < cols; ++x) {
                images_(x * rows + y, i) = img_buffer[y * cols + x] / 255.0;
            }
        }
    }

    cursor_.store(0, std::memory_order_relaxed);
    std::cout << "[DataLoader] MNIST loaded: "
              << num_images << " samples, features=" << num_features_
              << ", classes=" << num_classes_ << "\n";
}
