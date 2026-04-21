#include "worker.hpp"
#include "optimizer.hpp"
#include <Eigen/Core>
#include <algorithm>

#include <thread>
#ifdef _OPENMP
#include <omp.h>
#endif

using EigenArrayXS = Eigen::Array<double, Eigen::Dynamic, 1>;

namespace {

} // namespace

// ==============================================================================
// MODE 1: ASYNCHRONOUS (Hogwild! + DC-ASGD + Momentum)
// ==============================================================================
void Worker::run_async(int thread_id, ParameterList &global_weights,
                       DataLoader &loader, Dispatcher &dispatcher,
                       const Config &config, Monitor &monitor) {
  (void)thread_id;

  MiniDNNModel local_model(config);
  auto local_opt = Optimizer::create(config, global_weights);

  // Check both the monitor (training status) and dispatcher (engine status)
  while (!monitor.should_stop() && dispatcher.is_running()) {
    
    // 1. SLEEP / SYNCHRONIZATION
    // If the Prober scaled down the thread count, extra threads halt here at 0% CPU.
    dispatcher.wait_for_active(thread_id);

    // FAILSAFE: If woke up because the engine is shutting down, exit loop immediately.
    if (!dispatcher.is_running()) {
      break;
    }

    // 2. GATING: Get permission to start the step
    auto [can_start, start_idx] = dispatcher.try_start_step(thread_id);
    if (!can_start) {
      std::this_thread::yield(); // Yield if interval-sync is holding this thread back
      continue;
    }

    // 3. MACHINE LEARNING LOGIC
    local_model.set_weights(global_weights);
    auto [batch_X, batch_Y] = loader.next_batch(config.batch_size);
    auto [local_gradients, batch_loss] =
        local_model.compute_gradients(batch_X, batch_Y);

    // Get weights after backward pass processing (Zero-copy)
    const ParameterList &local_weights = local_model.get_cached_weights();

    monitor.record(thread_id, batch_loss, config.batch_size);

    // 4. COMMIT & UPDATE
    if (dispatcher.finish_step(start_idx, batch_loss)) {
      for (size_t layer = 0; layer < global_weights.size(); ++layer) {
        if (global_weights[layer].empty())
          continue;

        Eigen::Map<EigenArrayXS> mapped_w(global_weights[layer].data(),
                                          global_weights[layer].size());
        Eigen::Map<EigenArrayXS> mapped_local_g(local_gradients[layer].data(),
                                                local_gradients[layer].size());

        // Apply DC-ASGD Delay Compensation if requested
        if (config.opt_mode == OptimizerMode::DC_ASGD_C) {
          // DC-ASGD-c: constant λ throughout training
          Eigen::Map<const EigenArrayXS> mapped_local_w(
              local_weights[layer].data(), local_weights[layer].size());
          EigenArrayXS delay = mapped_w - mapped_local_w;
          EigenArrayXS active_gradient =
              mapped_local_g + config.lambda * mapped_local_g.square() * delay;
          local_opt->step(layer, mapped_w, active_gradient);

        } else if (config.opt_mode == OptimizerMode::DC_ASGD_A) {
          // DC-ASGD-a: adaptive per-element λ via RMSProp-style variance
          auto *sgdm_opt =
              static_cast<SGDMomentumOptimizer *>(local_opt.get());
          EigenArrayXS lambda_t =
              sgdm_opt->compute_adaptive_lambda(layer, mapped_local_g);

          Eigen::Map<const EigenArrayXS> mapped_local_w(
              local_weights[layer].data(), local_weights[layer].size());
          EigenArrayXS delay = mapped_w - mapped_local_w;
          EigenArrayXS active_gradient =
              mapped_local_g + lambda_t * mapped_local_g.square() * delay;
          local_opt->step(layer, mapped_w, active_gradient);

        } else {
          local_opt->step(layer, mapped_w, mapped_local_g);
        }
      }
    }
  }

  // Signal all other threads (including sleeping ones) to wake up and exit
  // if this thread detected that the stopping criteria were met.
  if (monitor.should_stop()) {
    dispatcher.stop_all();
  }
}

// ==============================================================================
// MODE 3: SEQUENTIAL (Standard single-threaded SGD + Momentum)
// ==============================================================================
void Worker::run_sequential(ParameterList &global_weights, DataLoader &loader,
                            const Config &config, Monitor &monitor) {

  MiniDNNModel local_model(config);
  auto local_opt = Optimizer::create(config, global_weights);
  int steps_per_epoch = std::max(1, loader.num_samples() / config.batch_size);

  for (int step = 0; !monitor.should_stop(); ++step) {
    if (step > 0 && (step % steps_per_epoch == 0)) {
      loader.shuffle();
    }

    local_model.set_weights(global_weights);
    auto [batch_X, batch_Y] = loader.next_batch(config.batch_size);
    auto [local_gradients, batch_loss] =
        local_model.compute_gradients(batch_X, batch_Y);

    monitor.record(0, batch_loss, config.batch_size);

    for (size_t layer = 0; layer < global_weights.size(); ++layer) {
      if (global_weights[layer].empty())
        continue;

      Eigen::Map<EigenArrayXS> mapped_w(global_weights[layer].data(),
                                        global_weights[layer].size());
      Eigen::Map<EigenArrayXS> mapped_local_g(local_gradients[layer].data(),
                                              local_gradients[layer].size());

      // Apply using optimizer
      local_opt->step(layer, mapped_w, mapped_local_g);
    }
  }
}