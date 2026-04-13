#pragma once
#include "config.hpp"
#include "data.hpp"
#include "dispatcher.hpp"
#include "monitor.hpp"
#include "model.hpp"

class TrainingEngine {
public:
    TrainingEngine(const Config& config, DataLoader& loader);

    // Main entry point for the engine, manages threading internally
    void run();

private:
    const Config& config_;
    DataLoader& loader_;
    Monitor monitor_;
    Dispatcher dispatcher_;
    
    // Global model instances
    MiniDNNModel global_model_;
    ParameterList w_global_;         // Global Weights
    ParameterList g_global_accum_;   // Gradient Accumulator (Sync Mode)

    void run_sequential();
    void run_parallel_sync();
    void run_parallel_async();
};