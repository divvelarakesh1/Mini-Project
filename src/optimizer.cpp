#include "optimizer.hpp"

std::unique_ptr<Optimizer> Optimizer::create(const Config& config, const ParameterList& initial_weights) {
    if (config.opt_algo == OptimizerAlgorithm::ADAM) {
        return std::make_unique<AdamOptimizer>(config, initial_weights);
    } else if (config.opt_algo == OptimizerAlgorithm::RMSPROP) {
        return std::make_unique<RMSPropOptimizer>(config, initial_weights);
    }
    return std::make_unique<SGDMomentumOptimizer>(config, initial_weights);
}

SGDMomentumOptimizer::SGDMomentumOptimizer(const Config& config, const ParameterList& initial_weights)
    : config_(config) 
{
    // Initialize velocity to zero with the exact same shapes as the model weights
    velocity_ = initial_weights;
    for (auto& layer : velocity_) {
        std::fill(layer.begin(), layer.end(), 0.0);
    }
}

void SGDMomentumOptimizer::step(size_t layer_idx, Eigen::Map<EigenArrayXS>& w, const EigenArrayXS& grad) {
    if (velocity_.size() <= layer_idx || velocity_[layer_idx].empty()) {
        w -= config_.eta * grad; // fallback to standard SGD
        return;
    }

    Eigen::Map<EigenArrayXS> v(velocity_[layer_idx].data(), velocity_[layer_idx].size());
    
    // Update velocity: v = momentum * v + eta * gradient
    v = config_.momentum * v + config_.eta * grad;
    
    // Apply to weights: w = w - v
    w -= v;
}

// -----------------------------------------------------------------------------
// Adam Optimizer
// -----------------------------------------------------------------------------
AdamOptimizer::AdamOptimizer(const Config& config, const ParameterList& initial_weights)
    : config_(config)
{
    m_ = initial_weights;
    v_ = initial_weights;
    for (auto& layer : m_) std::fill(layer.begin(), layer.end(), 0.0);
    for (auto& layer : v_) std::fill(layer.begin(), layer.end(), 0.0);
    t_.resize(initial_weights.size(), 1.0);
}

void AdamOptimizer::step(size_t layer_idx, Eigen::Map<EigenArrayXS>& w, const EigenArrayXS& grad) {
    if (m_.size() <= layer_idx || m_[layer_idx].empty()) {
        w -= config_.eta * grad;
        return;
    }

    Eigen::Map<EigenArrayXS> m(m_[layer_idx].data(), m_[layer_idx].size());
    Eigen::Map<EigenArrayXS> v(v_[layer_idx].data(), v_[layer_idx].size());

    m = config_.beta1 * m + (1.0 - config_.beta1) * grad;
    v = config_.beta2 * v + (1.0 - config_.beta2) * grad.square();

    EigenArrayXS m_hat = m / (1.0 - std::pow(config_.beta1, t_[layer_idx]));
    EigenArrayXS v_hat = v / (1.0 - std::pow(config_.beta2, t_[layer_idx]));

    w -= config_.eta * (m_hat / (v_hat.sqrt() + config_.epsilon));

    t_[layer_idx] += 1.0;
}

// -----------------------------------------------------------------------------
// RMSProp Optimizer
// -----------------------------------------------------------------------------
RMSPropOptimizer::RMSPropOptimizer(const Config& config, const ParameterList& initial_weights)
    : config_(config)
{
    v_ = initial_weights;
    for (auto& layer : v_) std::fill(layer.begin(), layer.end(), 0.0);
}

void RMSPropOptimizer::step(size_t layer_idx, Eigen::Map<EigenArrayXS>& w, const EigenArrayXS& grad) {
    if (v_.size() <= layer_idx || v_[layer_idx].empty()) {
        w -= config_.eta * grad;
        return;
    }

    Eigen::Map<EigenArrayXS> v(v_[layer_idx].data(), v_[layer_idx].size());

    v = config_.beta2 * v + (1.0 - config_.beta2) * grad.square();

    w -= config_.eta * (grad / (v.sqrt() + config_.epsilon));
}
