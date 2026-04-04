#pragma once
#include "dispatcher.hpp"
#include "model.hpp"
#include "data.hpp"
#include "config.hpp"
#include "monitor.hpp"
#include <vector>

// Type aliases for cleaner code
using ParameterList = std::vector<std::vector<double>>;

class Worker {
public:
    /**
     * @brief Mode 1: Interval-Asynchrony + Delay Compensation (Hogwild! / DC-ASGD)
     * 
     * In this mode, threads run completely decoupled via lock-free loops. 
     * Threads read global weights freely and write back calculated gradients.
     * If `config.use_dc_asgd` is true, a dynamic mathematical penalty is applied 
     * measuring how "stale" the gradient became while the local thread was computing.
     * 
     * @param thread_id The unique integer mapped identifier for the executing thread.
     * @param w_global A concurrent reference to the global neural network weights.
     * @param loader The thread-safe Atomic Data provider dispensing batches.
     * @param dispatcher Centralized traffic control for algorithmic interval bounds.
     * @param config The primary config instance spanning neural networks settings.
     * @param monitor Thread-safe telemetry extractor mapping steps and dynamic loss arrays.
     */
    static void run_async(
        int thread_id,
        ParameterList& w_global,
        DataLoader& loader,
        Dispatcher& dispatcher,
        const Config& config,
        Monitor& monitor
    );

    /**
     * @brief Mode 2: Strict Synchronous Parallel SGD
     * 
     * In stark contrast to `run_async`, this method forces threads to compute
     * homogeneously. Every thread securely reads identical `w_global` states, 
     * computes uncoupled fractional gradients against distinct `DataLoader` batches, 
     * and rigorously halts against an OpenMP #pragma barrier. Thread 0 then applies
     * the aggregated sum across the model, preserving exact mathematical consistency.
     * 
     * @param thread_id The unique integer mapped identifier. Thread 0 computes the aggregate.
     * @param w_global A concurrent reference to the global neural network weights.
     * @param g_global_accum The isolated, thread-safe gradient accumulator block.
     * @param loader The thread-safe Atomic Data provider dispensing unique chunks.
     * @param config The primary config instance spanning neural network setups.
     * @param monitor Thread-safe telemetry extractor tracking algorithmic convergence.
     */
    static void run_sync(
        int thread_id,
        ParameterList& w_global,
        ParameterList& g_global_accum, // Shared accumulator for gradients
        DataLoader& loader,
        const Config& config,
        Monitor& monitor
    );
};