# Mini-Project: Parallel Deep Learning Engine (Hogwild! & Sync SGD)

This project provides a highly customizable, multithreaded backend built from scratch in C++ to train **MiniDNN** deep learning models concurrently. It bypasses conventional frameworks to directly implement thread-safe interval asynchrony, Delay Compensated Asynchronous Stochastic Gradient Descent (DC-ASGD), and strict Synchronous Parallel training modes.

## Architecture Pipeline

The system wraps the lightweight, header-only `MiniDNN` tensor library and manually manages execution pipelines across OpenMP thread pools. The runtime flow is structured into three primary components:

- **`TrainingEngine`**: The core graph manager. It initializes the threads, routes configurations, and binds the data loaders to the neural network processors.
- **`Worker`**: Spawns multithreaded isolation models for the neural network. 
  - *Async Mode (Hogwild!)*: Leverages lock-free local gradient updates and evaluates staleness via DC-ASGD (Delay-Compensated ASGD) penalties to ensure mathematical convergence during uncoordinated writes to global memory.
  - *Sync Mode*: Strict barrier-governed synchronous parallel SGD loops over dynamic dataset chunks using centralized gradient accumulators.
  - *Sequential Mode*: A standard, single-threaded stochastic gradient descent loop designed for benchmarking.
- **`DataLoader`**: Atomically distributes disjoint dataset samples to the workers via thread-safe read locks. Completely eliminates CPU data-loading bottlenecks by natively caching the datasets into fast matrix abstractions in memory. 

## Datasets Supported

By simply toggling `"dataset"` in `config.json`, the system automatically provisions different computational graphs matching specific data formats:
* **MNIST**: Evaluates `28x28x1` (784 features) grayscale digits. Natively swaps Endian architectures on-the-fly for modern little-endian CPUs.
* **CIFAR-10**: Evaluates `32x32x3` (3072 features) standard color imagery.

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

## Configuration Management

Configuration settings are now managed via `config.json` in the project root. This allows for runtime adjustments without requiring recompilation. If the file is missing or invalid, the system automatically falls back to safe default parameters.

### Example `config.json`
```json
{
  "total_steps": 1000,
  "eta": 0.1,
  "momentum": 0.9,
  "beta1": 0.9,
  "beta2": 0.999,
  "epsilon": 1e-8,
  "lambda": 0.04,
  "batch_size": 64,
  "exec_mode": "ASYNC_HOGWILD",
  "opt_algo": "SGD",
  "opt_mode": "STANDARD_SGD",
  "num_threads": 4,
  "log_interval": 100,
  "dataset": "CIFAR"
}
```

### Parameter Breakdown
*   **`total_steps`**: Total number of training iterations.
*   **`eta`**: The learning rate for gradient updates.
*   **`momentum`**: Momentum coefficient for the optimizer (Standard SGD).
*   **`beta1` / `beta2` / `epsilon`**: Hyperparameters specifically used by the Adam and RMSProp optimizers.
*   **`lambda`**: Penalty parameter specifically for DC-ASGD delay compensation.
*   **`batch_size`**: Number of samples processed per batch.
*   **`exec_mode`**: Parallelism strategy (`SEQUENTIAL`, `SYNC_PARALLEL`, or `ASYNC_HOGWILD`).
*   **`opt_algo`**: The structural optimization algorithm to use (`SGD`, `ADAM`, or `RMSPROP`).
*   **`opt_mode`**: Gradient descent behavior mode (`STANDARD_SGD` or `DC_ASGD`).
*   **`num_threads`**: Number of parallel worker threads. Set to `0` to auto-detect hardware concurrency.
*   **`log_interval`**: Frequency (in steps) of printing training metrics to the console.
*   **`dataset`**: Target dataset to load (`MNIST` or `CIFAR`).

