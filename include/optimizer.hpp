#pragma once
#include "config.hpp"
#include <Eigen/Core>
#include <memory>
#include <vector>

using ParameterList = std::vector<std::vector<double>>;
using EigenArrayXS = Eigen::Array<double, Eigen::Dynamic, 1>;

class Optimizer {
public:
  virtual ~Optimizer() = default;

  // Abstract method to step the optimizer state and apply gradients to weights
  virtual void step(size_t layer_idx, Eigen::Map<EigenArrayXS> &w,
                    const EigenArrayXS &grad) = 0;

  // Factory method wrapper
  static std::unique_ptr<Optimizer>
  create(const Config &config, const ParameterList &initial_weights);
};

class SGDMomentumOptimizer : public Optimizer {
public:
  SGDMomentumOptimizer(const Config &config,
                       const ParameterList &initial_weights);

  void step(size_t layer_idx, Eigen::Map<EigenArrayXS> &w,
            const EigenArrayXS &grad) override;

  /**
   * @brief Compute per-element adaptive λ using RMSProp-style moving average.
   *
   * Updates the internal mean_square_ buffer and returns:
   *   λ_t = λ₀ / (MeanSquare_t + ε)
   *
   * Only valid when opt_mode == DC_ASGD_A. The mean_square_ buffer must have
   * been allocated in the constructor.
   */
  EigenArrayXS compute_adaptive_lambda(size_t layer_idx,
                                       const EigenArrayXS &grad);

private:
  const Config &config_;
  ParameterList velocity_;
  ParameterList mean_square_; // Per-layer moving average of g² (DC-ASGD-a only)
};
