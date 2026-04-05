#include <iostream>
#include <string>
#include <cstdlib>
#include <algorithm>

#include <thread>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "config.hpp"
#include "data.hpp"
#include "dispatcher.hpp"
#include "monitor.hpp"
#include "worker.hpp"
#include "model.hpp"

int main(int argc, char** argv) {
    Config config;
    
    // Gracefully handle passed CLI arguments
    if (argc > 1) {
        try {
            int threads = std::stoi(argv[1]);
            config.num_threads = (threads > 0) ? threads : std::thread::hardware_concurrency();
        } catch (const std::invalid_argument&) {
            std::cerr << "[Warning] Invalid thread count provided. Defaulting to: " << config.num_threads << "\n";
        }
    } else {
        // Automatically default to maximum hardware throughput if not specified
        config.num_threads = std::thread::hardware_concurrency();
        if (config.num_threads == 0) config.num_threads = 4;
    }
    
    std::cout << "=====================================\n";
    std::cout << "[Main] Booting Parallel Engine\n";
    std::cout << "[Main] Threads Requested: " << config.num_threads << "\n";
    std::cout << "=====================================\n";
    
    // 1. Storage & Parsers
    DataLoader loader;
    try {
        if (config.dataset == DatasetType::MNIST) {
            std::cout << "[Main] Target dataset is MNIST. Loading stream formats...\n";
            loader.load_mnist("../data/mnist");
        } else {
            std::cout << "[Main] Target dataset is CIFAR-10. Loading stream formats...\n";
            loader.load_cifar10("../data/cifar-10");
        }
    } catch (const std::exception& e) {
        std::cerr << "\n[CRITICAL ERROR] Failed to load datasets: " << e.what() << "\n";
        std::cerr << "Did you run `./scripts/get_datasets.sh` first to populate the `data/` folder?\n";
        return 1;
    }
    
    // 2. Global Scoped Modules
    MiniDNNModel global_model(config);
    ParameterList w_global = global_model.get_weights();
    
    ParameterList g_global_accum;
    if (config.use_sync_mode) {
        g_global_accum = global_model.get_weights();
        for (auto& vec : g_global_accum) {
            std::fill(vec.begin(), vec.end(), 0.0);
        }
    }
    
    Monitor monitor(config);
    Dispatcher dispatcher;
    
    // 3. Thread Spawning & Concurrency Bounds
#ifdef _OPENMP
    omp_set_num_threads(config.num_threads);
#endif

    std::cout << "[Main] Entering computational parallel graph.\n";

    #pragma omp parallel
    {
        int thread_id = 0;
#ifdef _OPENMP
        thread_id = omp_get_thread_num();
#endif
        
        if (config.use_sync_mode) {
            Worker::run_sync(thread_id, w_global, g_global_accum, loader, config, monitor);
        } else {
            // Boot asynchronous logic per thread independently 
            Worker::run_async(thread_id, w_global, loader, dispatcher, config, monitor);
        }
    }

    std::cout << "=====================================\n";
    std::cout << "[Main] Training Terminated Safely!\n";
    return 0;
}
