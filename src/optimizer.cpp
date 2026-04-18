#include "optimizer.hpp"

std::unique_ptr<Optimizer>
Optimizer::create(const Config &config, const ParameterList &initial_weights) {
  return std::make_unique<SGDMomentumOptimizer>(config, initial_weights);
}

SGDMomentumOptimizer::SGDMomentumOptimizer(const Config &config,
                                           const ParameterList &initial_weights)
    : config_(config) {
  // Initialize velocity to zero with the exact same shapes as the model weights
  velocity_ = initial_weights;
  for (auto &layer : velocity_) {
    std::fill(layer.begin(), layer.end(), 0.0);
  }

  // Allocate mean_square_ buffer for DC-ASGD-a (adaptive λ) mode
  if (config_.opt_mode == OptimizerMode::DC_ASGD_A) {
    mean_square_ = initial_weights;
    for (auto &layer : mean_square_) {
      std::fill(layer.begin(), layer.end(), 0.0);
    }
  }
}

void SGDMomentumOptimizer::step(size_t layer_idx, Eigen::Map<EigenArrayXS> &w,
                                const EigenArrayXS &grad) {
  if (velocity_.size() <= layer_idx || velocity_[layer_idx].empty()) {
    w -= config_.eta * grad; // fallback to standard SGD
    return;
  }

  Eigen::Map<EigenArrayXS> v(velocity_[layer_idx].data(),
                             velocity_[layer_idx].size());

  // Update velocity: v = momentum * v + eta * gradient
  v = config_.momentum * v + config_.eta * grad;

  // Apply to weights: w = w - v
  w -= v;
}

EigenArrayXS
SGDMomentumOptimizer::compute_adaptive_lambda(size_t layer_idx,
                                              const EigenArrayXS &grad) {
  Eigen::Map<EigenArrayXS> ms(mean_square_[layer_idx].data(),
                               mean_square_[layer_idx].size());

  // Update moving average: ms = m * ms + (1 - m) * grad²
  ms = config_.dc_asgd_rms_momentum * ms +
       (1.0 - config_.dc_asgd_rms_momentum) * grad.square();

  // Return per-element λ_t = λ₀ / (ms + ε)
  return config_.lambda / (ms + config_.dc_asgd_rms_epsilon);
}
