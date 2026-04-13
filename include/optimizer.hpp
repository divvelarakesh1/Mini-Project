#pragma once
#include "config.hpp"
#include <vector>
#include <memory>
#include <Eigen/Core>

using ParameterList = std::vector<std::vector<double>>;
using EigenArrayXS = Eigen::Array<double, Eigen::Dynamic, 1>;

class Optimizer {
public:
    virtual ~Optimizer() = default;
    
    // Abstract method to step the optimizer state and apply gradients to weights
    virtual void step(size_t layer_idx, Eigen::Map<EigenArrayXS>& w, const EigenArrayXS& grad) = 0;

    // Factory method wrapper
    static std::unique_ptr<Optimizer> create(const Config& config, const ParameterList& initial_weights);
};

class SGDMomentumOptimizer : public Optimizer {
public:
    SGDMomentumOptimizer(const Config& config, const ParameterList& initial_weights);
    
    void step(size_t layer_idx, Eigen::Map<EigenArrayXS>& w, const EigenArrayXS& grad) override;

private:
    const Config& config_;
    ParameterList velocity_;
};

class AdamOptimizer : public Optimizer {
public:
    AdamOptimizer(const Config& config, const ParameterList& initial_weights);
    void step(size_t layer_idx, Eigen::Map<EigenArrayXS>& w, const EigenArrayXS& grad) override;

private:
    const Config& config_;
    ParameterList m_;
    ParameterList v_;
    std::vector<double> t_;
};

class RMSPropOptimizer : public Optimizer {
public:
    RMSPropOptimizer(const Config& config, const ParameterList& initial_weights);
    void step(size_t layer_idx, Eigen::Map<EigenArrayXS>& w, const EigenArrayXS& grad) override;

private:
    const Config& config_;
    ParameterList v_;
};
