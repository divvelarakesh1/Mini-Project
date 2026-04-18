# Parallel Deep Learning Training Engine

A multithreaded deep learning training backend built from scratch in C++ using OpenMP. Trains convolutional neural networks on CIFAR-10 and MNIST datasets with support for multiple execution modes (Hogwild!, Sequential, Interval-Asynchronous) and multiple optimizer algorithms (SGD with Momentum, Adam, RMSProp).

## Architecture Overview

```
┌────────────────────────────────────────────────────────┐
│                      main.cpp                          │
│         Load config.json → DataLoader → Engine         │
└──────────────────────┬─────────────────────────────────┘
                       │
              ┌────────▼────────┐
              │ TrainingEngine  │
              │  Routes config  │
              │  Spawns threads │
              └──┬─────┬─────┬──┘
                 │     │     │
        ┌────────┘     │     └────────┐
        ▼              ▼              ▼
   ┌─────────┐   ┌──────────┐   ┌──────────┐
   │ Worker  │   │  Worker  │   │  Worker  │
   │ Thread 0│   │ Thread 1 │   │ Thread N │
   └────┬────┘   └────┬─────┘   └────┬─────┘
        │              │              │
        └──────────┬───┘──────────────┘
                   ▼
          ┌────────────────┐
          │ Global Weights │  (shared memory)
          └────────────────┘
                   │
          ┌────────▼────────┐
          │     Monitor     │
          │  Epoch tracking │
          │  Loss & Speed   │
          └─────────────────┘
```

### Core Components

| Component | File(s) | Purpose |
|-----------|---------|---------|
| **TrainingEngine** | `engine.hpp/cpp` | Central orchestrator. Initializes threads, routes execution modes, binds the data loader and monitor to workers. |
| **Worker** | `worker.hpp/cpp` | The compute kernel. Each thread runs a Worker loop that pulls batches, computes gradients, and updates global weights. Multiple modes are supported (see below). |
| **DataLoader** | `data.hpp/cpp` | Thread-safe dataset provider. Loads CIFAR-10/MNIST from binary files, normalizes pixels to `[0, 1]`, and distributes batches via atomic cursors. |
| **Monitor** | `monitor.hpp/cpp` | Lock-free metrics aggregator. Tracks training progress by **epochs** (total images processed / dataset size), logs average loss and throughput every 0.1 epochs. |
| **Optimizer** | `optimizer.hpp/cpp` | Polymorphic optimizer with factory pattern. Supports SGD+Momentum, Adam, and RMSProp. Each worker thread owns a local optimizer instance. |
| **Dispatcher** | `dispatcher.hpp` | Step gate for parallel threads. Enforces interval-asynchrony boundaries and dynamic y-decay gating to control gradient staleness in async modes. |
| **MiniDNNModel** | `model.hpp` | Wraps the MiniDNN header-only library. Builds dataset-specific CNN architectures and provides forward/backward pass + gradient extraction. |
| **Config** | `config.hpp/cpp` | Loads all hyperparameters from `config.json` at runtime. Falls back to safe defaults if file is missing. |

## Execution Modes

### 1. Asynchronous — Hogwild! (`ASYNC_HOGWILD`)
Each thread independently reads global weights, computes a local forward/backward pass, and writes gradients back to shared memory **without locks**. Supports optional **DC-ASGD** (Delay-Compensated ASGD) penalty to stabilize convergence under high staleness.

### 2. Interval-Asynchronous (`INTERVAL_ASYNC`)
Combines Hogwild! with interval gating. Threads are allowed to run asynchronously within an "interval window" (set by `interval_size`). If a thread's gradient computation crosses an interval boundary (i.e., the global step has progressed too far), the update is dropped to prevent high staleness. Supports dynamic **Interval Decay** (y-decay) to gradually tighten synchronization.

### 3. Sequential (`SEQUENTIAL`)
Standard single-threaded SGD loop. Useful as a baseline for benchmarking parallel speedup and validating convergence behavior.

## Optimizer Algorithms

| Algorithm | Config Value | Key Params | Typical `eta` |
|-----------|-------------|------------|---------------|
| **SGD + Momentum** | `"SGD"` | `eta`, `momentum` | `0.1` |
| **Adam** | `"ADAM"` | `eta`, `beta1`, `beta2`, `epsilon` | `0.001` |
| **RMSProp** | `"RMSPROP"` | `eta`, `beta2`, `epsilon` | `0.001` |

> **Important:** Adam and RMSProp require a much smaller learning rate (~0.001) than SGD (~0.1). Using `eta: 0.1` with Adam will cause `nan` loss due to weight explosion.

## Neural Network Architectures

### CIFAR-10 (32×32×3 color images)
```
Conv(3×3, 3→16) → ReLU → MaxPool(2×2)
Conv(3×3, 16→32) → ReLU → MaxPool(2×2)
FC(1152→128) → ReLU
FC(128→10) → Softmax
```

### MNIST (28×28×1 grayscale digits)
```
Conv(5×5, 1→8) → ReLU → MaxPool(2×2)
FC(1152→64) → ReLU
FC(64→10) → Softmax
```

Both use **Multi-Class Cross-Entropy** loss.

## Datasets

| Dataset | Samples | Features | Classes |
|---------|---------|----------|---------|
| **CIFAR-10** | 50,000 | 3,072 (32×32×3) | 10 |
| **MNIST** | 60,000 | 784 (28×28×1) | 10 |

Toggle between them by setting `"dataset"` to `"CIFAR"` or `"MNIST"` in `config.json`.

## Setup

### Prerequisites
- C++17 compiler (GCC or Clang)
- CMake ≥ 3.16
- OpenMP (macOS: `brew install libomp`)
- Third-party libraries (Eigen, MiniDNN, nlohmann/json) are bundled in `third-party/`

### 1. Install Dependencies
```bash
./scripts/install_deps.sh
```
This fetches Eigen, MiniDNN, and nlohmann/json into the `third-party/` directory.

### 2. Download Datasets
```bash
./scripts/get_datasets.sh
```
This fetches CIFAR-10 and MNIST binary files into the `data/` directory.

### 3. Build
```bash
mkdir -p build && cd build
cmake ..
make -j$(nproc)    # Linux
make -j$(sysctl -n hw.ncpu)  # macOS
```

### 3. Run
```bash
cd build
./mini_project
```
The engine reads `config.json` from the project root (or parent directory if run from `build/`).

## Configuration

All hyperparameters are set in `config.json` — no recompilation needed.

### Example `config.json`
```json
{
  "total_steps": 10000,
  "target_epochs": 0.0,
  "target_time_seconds": 0.0,
  "eta": 0.001,
  "momentum": 0.9,
  "lambda": 0.04,
  "batch_size": 128,
  "interval_size": 1024,
  "interval_decay_freq": 4096,
  "exec_mode": "INTERVAL_ASYNC",
  "opt_algo": "ADAM",
  "opt_mode": "STANDARD_SGD",
  "stop_mode": "STEPS",
  "beta1": 0.9,
  "beta2": 0.999,
  "epsilon": 1e-8,
  "num_threads": 32,
  "log_interval": 100,
  "dataset": "CIFAR"
}
```

### Parameter Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `total_steps` | int | `1000` | Total training iterations per thread when `stop_mode` is `STEPS`. |
| `target_epochs` | float | `0.0` | Epoch target used when `stop_mode` is `EPOCHS`. |
| `target_time_seconds` | float | `0.0` | Wall-clock time target used when `stop_mode` is `TIME`. |
| `eta` | float | `0.1` | Learning rate. Use `~0.1` for SGD, `~0.001` for Adam/RMSProp. |
| `momentum` | float | `0.9` | Momentum coefficient (SGD only). |
| `beta1` | float | `0.9` | First moment decay rate (Adam only). |
| `beta2` | float | `0.999` | Second moment decay rate (Adam and RMSProp). |
| `epsilon` | float | `1e-8` | Numerical stability constant (Adam and RMSProp). |
| `lambda` | float | `0.04` | DC-ASGD delay compensation penalty. |
| `batch_size` | int | `64` | Samples per mini-batch. |
| `interval_size` | int | `1024` | Initial interval window size for `INTERVAL_ASYNC`. |
| `interval_decay_freq` | int | `0` | Frequency (in steps) to decay `interval_size` by 1. Set to `0` to disable. |
| `exec_mode` | string | `"ASYNC_HOGWILD"` | Execution mode: `SEQUENTIAL`, `ASYNC_HOGWILD`, or `INTERVAL_ASYNC`. |
| `opt_algo` | string | `"SGD"` | Optimizer algorithm: `SGD`, `ADAM`, or `RMSPROP`. |
| `opt_mode` | string | `"STANDARD_SGD"` | Gradient mode: `STANDARD_SGD` or `DC_ASGD` (async only). |
| `stop_mode` | string | `"STEPS"` | Training stop criterion: `STEPS`, `EPOCHS`, or `TIME`. |
| `num_threads` | int | `32` | Number of OpenMP threads. Set to `0` for auto-detect. |
| `log_interval` | int | `100` | Frequency of logging metrics to console. |
| `dataset` | string | `"CIFAR"` | Dataset: `MNIST` or `CIFAR`. |

## Monitor Output

Training progress is tracked by **epochs** (total images processed across all threads ÷ dataset size). The monitor logs every ~0.1 epochs:

```
[Monitor] Epoch:   1.02 | Elapsed:  10.98s
          Avg Loss:   2.3045
          Speed: 4613 images/sec | Total Img: 51200

[Monitor] Epoch:   1.13 | Elapsed:  12.06s
          Avg Loss:   2.2856
          Speed: 4763 images/sec | Total Img: 56320
```

## Running Experiments

An automated benchmark script is provided to compare all training modes:

```bash
python3 scripts/run_experiments.py
```

You can also force a shared epoch-based or time-based stopping rule across all runs:

```bash
python3 scripts/run_experiments.py --epochs 5
python3 scripts/run_experiments.py --seconds 300
```

By default, the script also parses the logs it generated and saves:
- `metrics_summary.csv`
- `loss_vs_epoch.png`
- `loss_vs_time.png`
- `epoch_vs_time.png`
- `speed_vs_epoch.png`

If you only want the raw logs, you can skip plot generation:

```bash
python3 scripts/run_experiments.py --epochs 5 --no-plots
```

This script sequentially runs the following configurations and saves logs to the `logs/` directory:
1.  **Sequential**: Baseline single-threaded training.
2.  **Plain Async**: Unrestricted Hogwild! training.
3.  **Async + Penalty**: Hogwild! with DC-ASGD delay compensation.
4.  **Interval Async**: Interval-based asynchronous training.
5.  **Interval Async + Penalty**: Combining interval gating with delay compensation.

Logs can be analyzed to compare convergence speed and throughput across different staleness control strategies. For fair cross-mode comparisons, prefer `--epochs` for equal data exposure and `--seconds` for equal wall-clock budgets.

## Project Structure

```
Mini-Project/
├── config.json              # Runtime configuration
├── CMakeLists.txt           # Build system
├── scripts/
│   ├── get_datasets.sh      # Dataset download script
│   └── install_deps.sh      # Dependency installer
├── include/
│   ├── config.hpp           # Config struct & enums
│   ├── data.hpp             # DataLoader interface
│   ├── dispatcher.hpp       # Async step dispatcher
│   ├── engine.hpp           # TrainingEngine interface
│   ├── model.hpp            # MiniDNN model wrapper
│   ├── monitor.hpp          # Epoch-based monitor
│   ├── optimizer.hpp        # Optimizer base + SGD/Adam/RMSProp
│   └── worker.hpp           # Worker interface
├── src/
│   ├── main.cpp             # Entry point
│   ├── config.cpp           # JSON config parser
│   ├── data.cpp             # Dataset loaders (CIFAR-10, MNIST)
│   ├── engine.cpp           # Engine execution logic
│   ├── monitor.cpp          # Monitor implementation
│   ├── optimizer.cpp        # Optimizer implementations
│   └── worker.cpp           # Worker training loops
├── third-party/
│   ├── eigen/               # Eigen linear algebra library
│   ├── MiniDNN/             # MiniDNN neural network library
│   └── nlohmann/            # JSON parsing library
└── data/
    ├── cifar-10/            # CIFAR-10 binary files
    └── mnist/               # MNIST binary files
```
