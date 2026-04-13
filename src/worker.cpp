#include "worker.hpp"
#include "optimizer.hpp"
#include <Eigen/Core>

#ifdef _OPENMP
#include <omp.h>
#endif

using EigenArrayXS = Eigen::Array<double, Eigen::Dynamic, 1>;

// ==============================================================================
// MODE 1: ASYNCHRONOUS (Hogwild! + DC-ASGD + Momentum)
// ==============================================================================
void Worker::run_async(int thread_id, ParameterList &global_weights,
                       DataLoader &loader, Dispatcher &dispatcher,
                       const Config &config, Monitor &monitor) {
  (void)thread_id;

  MiniDNNModel local_model(config);
  auto local_opt = Optimizer::create(config, global_weights);

  for (int step = 0; step < config.total_steps; ++step) {
    auto [can_start, start_idx] = dispatcher.try_start_step();
    if (!can_start)
      continue;

    local_model.set_weights(global_weights);
    auto [batch_X, batch_Y] = loader.next_batch(config.batch_size);
    auto [local_gradients, batch_loss] =
        local_model.compute_gradients(batch_X, batch_Y);

    // Get weights after backward pass processing (Zero-copy)
    const ParameterList &local_weights = local_model.get_cached_weights();

    monitor.record_step(thread_id, step, batch_loss, config.batch_size);

    if (dispatcher.finish_step(start_idx)) {
      for (size_t layer = 0; layer < global_weights.size(); ++layer) {
        if (global_weights[layer].empty())
          continue;

        Eigen::Map<EigenArrayXS> mapped_w(global_weights[layer].data(),
                                          global_weights[layer].size());
        Eigen::Map<EigenArrayXS> mapped_local_g(local_gradients[layer].data(),
                                                local_gradients[layer].size());

        // Apply DC-ASGD Delay Compensation if requested
        if (config.opt_mode == OptimizerMode::DC_ASGD) {
          Eigen::Map<const EigenArrayXS> mapped_local_w(
              local_weights[layer].data(), local_weights[layer].size());
          EigenArrayXS delay = mapped_w - mapped_local_w;
          EigenArrayXS active_gradient =
              mapped_local_g + config.lambda * mapped_local_g.square() * delay;
          local_opt->step(layer, mapped_w, active_gradient);
        } else {
          local_opt->step(layer, mapped_w, mapped_local_g);
        }
      }
    }
  }
}

// ==============================================================================
// MODE 2: SYNCHRONOUS (Strict Thread Barriers with Epoch Shuffling + Momentum)
// ==============================================================================
void Worker::run_sync(int thread_id, ParameterList &global_weights,
                      ParameterList &global_gradient_accumulator,
                      DataLoader &loader, const Config &config,
                      Monitor &monitor) {

  MiniDNNModel local_model(config);
  auto local_opt = Optimizer::create(config, global_weights);
  int steps_per_epoch = std::max(1, loader.num_samples() / config.batch_size);

  for (int step = 0; step < config.total_steps; ++step) {

    // Shuffle data at the beginning of an epoch, protected by thread barriers
    if (step > 0 && (step % steps_per_epoch == 0)) {
#ifdef _OPENMP
#pragma omp barrier
#endif
      if (thread_id == 0) {
        loader.shuffle();
      }
#ifdef _OPENMP
#pragma omp barrier
#endif
    }

#ifdef _OPENMP
#pragma omp barrier
#endif

    local_model.set_weights(global_weights);
    auto [batch_X, batch_Y] = loader.next_batch(config.batch_size);
    auto [local_gradients, batch_loss] =
        local_model.compute_gradients(batch_X, batch_Y);

    monitor.record_step(thread_id, step, batch_loss, config.batch_size);

    // Accumulate gradients synchronously
#ifdef _OPENMP
#pragma omp critical
#endif
    {
      for (size_t layer = 0; layer < global_gradient_accumulator.size();
           ++layer) {
        if (global_gradient_accumulator[layer].empty())
          continue;
        Eigen::Map<EigenArrayXS> mapped_accum(
            global_gradient_accumulator[layer].data(),
            global_gradient_accumulator[layer].size());
        Eigen::Map<EigenArrayXS> mapped_local_g(local_gradients[layer].data(),
                                                local_gradients[layer].size());
        mapped_accum += mapped_local_g;
      }
    }

#ifdef _OPENMP
#pragma omp barrier
#endif

    // Thread 0 averages the accumulator and applies the momentum update
    if (thread_id == 0) {
      for (size_t layer = 0; layer < global_weights.size(); ++layer) {
        if (global_weights[layer].empty())
          continue;

        Eigen::Map<EigenArrayXS> mapped_w(global_weights[layer].data(),
                                          global_weights[layer].size());
        Eigen::Map<EigenArrayXS> mapped_accum(
            global_gradient_accumulator[layer].data(),
            global_gradient_accumulator[layer].size());

        // Calculate average gradient across all threads in-place
        mapped_accum /= config.num_threads;

        // Apply using thread 0's optimizer
        local_opt->step(layer, mapped_w, mapped_accum);

        // Zero out accumulator for the next step
        mapped_accum.setZero();
      }
    }

#ifdef _OPENMP
#pragma omp barrier
#endif
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

  for (int step = 0; step < config.total_steps; ++step) {
    if (step > 0 && (step % steps_per_epoch == 0)) {
      loader.shuffle();
    }

    local_model.set_weights(global_weights);
    auto [batch_X, batch_Y] = loader.next_batch(config.batch_size);
    auto [local_gradients, batch_loss] =
        local_model.compute_gradients(batch_X, batch_Y);

    monitor.record_step(0, step, batch_loss, config.batch_size);

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