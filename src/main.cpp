#include <iostream>
#include <string>
#include <cstdlib>
#include <thread>

#include "config.hpp"
#include "data.hpp"
#include "engine.hpp"

int main() {
    // Load configuration directly from config.json, abandoning command-line arguments.
    Config config = load_config("config.json");
    
    // Automatically default to maximum hardware throughput if not validly specified in config
    if (config.num_threads <= 0) {
        config.num_threads = std::thread::hardware_concurrency();
        if (config.num_threads == 0) config.num_threads = 4;
    }
    
    std::cout << "=====================================\n";
    std::cout << "[Main] Initializing Parallel Engine Setup\n";
    std::cout << "=====================================\n";
    
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
    
    TrainingEngine engine(config, loader);
    engine.run();

    return 0;
}