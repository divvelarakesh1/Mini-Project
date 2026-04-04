# Mini-Project: Parallel Deep Learning Engine (Hogwild! & Sync SGD)

This project provides a highly customizable, multithreaded backend built from scratch in C++ to train **MiniDNN** deep learning models concurrently. It bypasses conventional frameworks to directly implement thread-safe interval asynchrony, Delay Compensated Asynchronous Stochastic Gradient Descent (DC-ASGD), and strict Synchronous Parallel training modes.

## Architecture

The system wraps the lightweight, header-only `MiniDNN` tensor library and manually manages execution pipelines across OpenMP thread pools. The core functionalities are:
- **`Worker`**: Spawns multiple isolation modes for threading. 
  - *Async Mode*: Leverages lock-free `Hogwild!` updates and evaluates gradient staleness via DC-ASGD penalties to ensure mathematical convergence during uncoordinated writes. 
  - *Sync Mode*: Strict barrier-governed synchronous parallel SGD loops over dynamic dataset chunks.
- **`DataLoader`**: Atomically distributes disjoint samples to threads from large binary datasets in memory, completely eliminating CPU data-loading bottlenecks. Supports standard datasets (CIFAR and MNIST).
- **`Dispatcher`**: Controls interval-asynchrony, dynamically waking and sleeping threads logically to maximize hardware occupancy.

## Setup Instructions

### 1. Requirements
- C++17 compliant compiler (GCC/Clang)
- `CMake` (>= 3.16)
- `OpenMP` (Optional but mandated for threading features. macOS: `brew install libomp`)
- *Third-Party libs*: `Eigen` and `MiniDNN` are included natively entirely as headers.

### 2. Download Datasets
You do **not** need to manually download or parse data formats. Execute the included script from the project root to fetch canonical `CIFAR-10` and `MNIST` binary datasets into the `/data` folder:
```bash
./scripts/get_datasets.sh
```

### 3. Build Project
Generate and execute the build via CMake:
```bash
mkdir -p build && cd build
cmake ..
make -j4
```

## Running & Configuration

Configuration settings are globally scoped within `include/config.hpp`. You can modify the struct at compile time to drastically alter the engine's behavior:
```cpp
struct Config {
    int total_steps = 1000;
    double eta = 0.01;            // Learning rate
    double lambda = 0.04;         // DC-ASGD compensation penalty param
    int batch_size = 64;          
    bool use_dc_asgd = true;      // true = DC-ASGD, false = vanilla Hogwild!
    int num_threads = 4;        
    DatasetType dataset = DatasetType::CIFAR; // Swap out your dataset dynamically here
};
```
By simply toggling `config.dataset = DatasetType::MNIST;`, the system automatically provisions different computational graphs matching MNIST sizing properties (`28x28x1`) versus CIFAR models (`32x32x3`).

## Robustness Features
The data parsers heavily strictly validate Endian architectures. Since historical `MNIST` files are compiled identically in `Big-Endian` format, the C++ pipelines here natively perform bit-shifting to ensure hardware compatibility across standard Little-Endian CPU instances, allowing direct parsing without pre-preparation.
