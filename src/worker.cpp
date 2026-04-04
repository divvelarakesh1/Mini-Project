#include "worker.hpp"
#include <Eigen/Core>
#ifdef _OPENMP
#  include <omp.h>
#endif

using EigenArrayXS = Eigen::Array<double, Eigen::Dynamic, 1>;

// ==============================================================================
// MODE 1: ASYNCHRONOUS (Hogwild! + DC-ASGD + Interval-Asynchrony)
// ==============================================================================
void Worker::run_async(
    int thread_id,
    ParameterList& w_global,
    DataLoader& loader,
    Dispatcher& dispatcher,
    const Config& config,
    Monitor& monitor) 
{
    (void)thread_id; // Explicitly suppress unused parameter warnings
    
    // 1. Private isolated network
    MiniDNNModel local_model(config);

    for (int step = 0; step < config.total_steps; ++step) {
        
        // 2. Interval-Asynchrony Gate
        auto [can_start, start_idx] = dispatcher.try_start_step();
        if (!can_start) continue;

        // 3. Sync & Compute
        local_model.set_weights(w_global);

        // Load actual mini-batch (batch size config.batch_size)
        auto [batch_X, batch_Y] = loader.next_batch(config.batch_size);

        auto [g_local, batch_loss] = local_model.compute_gradients(batch_X, batch_Y);
        ParameterList w_local = local_model.get_weights();

        // Feed stats asynchronously tracking current loss
        monitor.record_step(step, batch_loss, config.batch_size);

        // 4. DC-ASGD & Hogwild! Update
        if (dispatcher.finish_step(start_idx)) {
            for (size_t layer = 0; layer < w_global.size(); ++layer) {
                if (w_global[layer].empty()) continue;

                Eigen::Map<EigenArrayXS> w_glob(w_global[layer].data(), w_global[layer].size());
                Eigen::Map<EigenArrayXS> w_loc(w_local[layer].data(), w_local[layer].size());
                Eigen::Map<EigenArrayXS> g_loc(g_local[layer].data(), g_local[layer].size());

                if (config.use_dc_asgd) {
                    EigenArrayXS delay = w_glob - w_loc;
                    EigenArrayXS g_dc = g_loc + config.lambda * g_loc.square() * delay;
                    // Lock-free write
                    w_glob -= config.eta * g_dc; 
                } else {
                    // Standard Hogwild! Update
                    w_glob -= config.eta * g_loc;
                }
            }
        }
    }
}

// ==============================================================================
// MODE 2: SYNCHRONOUS (Strict Thread Barriers)
// ==============================================================================
void Worker::run_sync(
    int thread_id,
    ParameterList& w_global,
    ParameterList& g_global_accum,
    DataLoader& loader,
    const Config& config,
    Monitor& monitor) 
{
    // 1. Private isolated network
    MiniDNNModel local_model(config);

    for (int step = 0; step < config.total_steps; ++step) {
        
        // BARRIER 1: Ensure all threads start the step at the exact same time
        #ifdef _OPENMP
        #pragma omp barrier
        #endif

        // 2. Sync local model with master model weights
        local_model.set_weights(w_global);

        // Load atomic batch size to ensure each thread gets different slices
        auto [batch_X, batch_Y] = loader.next_batch(config.batch_size);

        // 3. Compute gradients privately
        auto [g_local, batch_loss] = local_model.compute_gradients(batch_X, batch_Y);

        // Store loss securely into global monitor
        monitor.record_step(step, batch_loss, config.batch_size);

        // 4. Safely accumulate gradients into the global pool
        #ifdef _OPENMP
        #pragma omp critical
        #endif // (block below always executes; guard only suppresses the pragma)
        {
            for (size_t layer = 0; layer < g_global_accum.size(); ++layer) {
                if (g_global_accum[layer].empty()) continue;
                
                Eigen::Map<EigenArrayXS> g_accum(g_global_accum[layer].data(), g_global_accum[layer].size());
                Eigen::Map<EigenArrayXS> g_loc(g_local[layer].data(), g_local[layer].size());
                
                g_accum += g_loc;
            }
        }

        // BARRIER 2: Wait for all threads to finish accumulating their math
        #ifdef _OPENMP
        #pragma omp barrier
        #endif

        // 5. Thread 0 applies the averaged gradients to the global model
        if (thread_id == 0) {
            for (size_t layer = 0; layer < w_global.size(); ++layer) {
                if (w_global[layer].empty()) continue;

                Eigen::Map<EigenArrayXS> w_glob(w_global[layer].data(), w_global[layer].size());
                Eigen::Map<EigenArrayXS> g_accum(g_global_accum[layer].data(), g_global_accum[layer].size());

                // Apply gradient descent (averaging across num_threads)
                w_glob -= config.eta * (g_accum / config.num_threads);

                // Zero out the accumulator for the next step
                g_accum.setZero();
            }
        }

        // BARRIER 3: Stop threads from starting the next step until Thread 0 is done updating
        #pragma omp barrier
    }
}