#include "worker.hpp"
#include "optimizer.hpp"
#include <algorithm>
#include <Eigen/Core>

#ifdef _OPENMP
#include <omp.h>
#endif

using EigenArrayXS = Eigen::Array<double, Eigen::Dynamic, 1>;

namespace {

bool uses_dynamic_stop(const Config &config) {
  if (config.stop_mode == StopMode::EPOCHS) {
    return config.target_epochs > 0.0;
  }

  if (config.stop_mode == StopMode::TIME) {
    return config.target_time_seconds > 0.0;
  }

  return false;
}

bool should_start_step(int step, const Config &config, const Monitor &monitor) {
  if (uses_dynamic_stop(config)) {
    return !monitor.should_stop();
  }

  return step < config.total_steps;
}

int sync_steps_per_epoch(const DataLoader &loader, const Config &config) {
  int effective_threads = std::max(1, config.num_threads);
  int global_batch = std::max(1, config.batch_size * effective_threads);
  return std::max(1, loader.num_samples() / global_batch);
}

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

  for (int step = 0; should_start_step(step, config, monitor); ++step) {
    auto [can_start, start_idx] = dispatcher.try_start_step();
    if (!can_start)
      continue;

    local_model.set_weights(global_weights);
    auto [batch_X, batch_Y] = loader.next_batch(config.batch_size);
    auto [local_gradients, batch_loss] =
        local_model.compute_gradients(batch_X, batch_Y);

    // Get weights after backward pass processing (Zero-copy)
    const ParameterList &local_weights = local_model.get_cached_weights();

    monitor.record(thread_id, batch_loss, config.batch_size);

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
  int steps_per_epoch = sync_steps_per_epoch(loader, config);

  for (int step = 0;; ++step) {
#ifdef _OPENMP
#pragma omp barrier
#endif
    if (!should_start_step(step, config, monitor)) {
      break;
    }

    // Shuffle data at the beginning of an epoch, protected by thread barriers
    if (step > 0 && (step % steps_per_epoch == 0)) {
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

    monitor.record(thread_id, batch_loss, config.batch_size);

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

    if (uses_dynamic_stop(config) && monitor.should_stop()) {
      break;
    }
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

  for (int step = 0; should_start_step(step, config, monitor); ++step) {
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
