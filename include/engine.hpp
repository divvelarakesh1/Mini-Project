#pragma once
#include "config.hpp"
#include "data.hpp"
#include "dispatcher.hpp"
#include "monitor.hpp"
#include "model.hpp"
#include <memory>

class Prober;

class TrainingEngine {
public:
    TrainingEngine(const Config& config, DataLoader& loader, DataLoader& test_loader);
    ~TrainingEngine();

    // Main entry point for the engine, manages threading internally
    void run();

private:
    const Config& config_;
    DataLoader& loader_;
    DataLoader& test_loader_;
    Monitor monitor_;
    Dispatcher dispatcher_;
    std::unique_ptr<Prober> prober_;
    
    // Global model instances
    MiniDNNModel global_model_;
    ParameterList w_global_;         // Global Weights

    void run_sequential();
    void run_parallel_async();
};