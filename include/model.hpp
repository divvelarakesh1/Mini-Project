#pragma once
#include <vector>
#include <Eigen/Core>
#include <MiniDNN.h>
#include "config.hpp"

using ParameterList = std::vector<std::vector<double>>;

/**
 * MiniDNNModel — thin wrapper around a MiniDNN network.
 *
 * Exposes the three operations used by Worker:
 *   set_weights()       — load global weights into the local network
 *   get_weights()       — read current weights out of the local network
 *   compute_gradients() — run a forward+backward pass on a mini-batch, returning {gradients, loss}
 */
class MiniDNNModel {
public:
    MiniDNNModel(const Config& config) {
        if (config.dataset == DatasetType::CIFAR) {
            // Layer 1: Conv, input 32x32x3 (CIFAR), 16 output channels, 5x5 filter
            MiniDNN::Layer* layer1 = new MiniDNN::Convolutional<MiniDNN::ReLU>(32, 32, 3, 16, 5, 5);
            // Layer 2: Max pooling, input 28x28x16, pooling window 2x2
            MiniDNN::Layer* layer2 = new MiniDNN::MaxPooling<MiniDNN::ReLU>(28, 28, 16, 2, 2);
            // Layer 3: FCN, input 14x14x16 = 3136, output 10 (classes)
            MiniDNN::Layer* layer3 = new MiniDNN::FullyConnected<MiniDNN::Identity>(14 * 14 * 16, 10);
            
            net_.add_layer(layer1);
            net_.add_layer(layer2);
            net_.add_layer(layer3);
        } else if (config.dataset == DatasetType::MNIST) {
            // Layer 1: Conv, input 28x28x1 (MNIST), 3 output channels, 5x5 filter
            MiniDNN::Layer* layer1 = new MiniDNN::Convolutional<MiniDNN::ReLU>(28, 28, 1, 3, 5, 5);
            // Layer 2: Max pooling, input 24x24x3, pooling window 2x2
            MiniDNN::Layer* layer2 = new MiniDNN::MaxPooling<MiniDNN::ReLU>(24, 24, 3, 2, 2);
            // Layer 3: FCN, input 12x12x3 = 432, output 10 (classes)
            MiniDNN::Layer* layer3 = new MiniDNN::FullyConnected<MiniDNN::Identity>(12 * 12 * 3, 10);
            
            net_.add_layer(layer1);
            net_.add_layer(layer2);
            net_.add_layer(layer3);
        }
        
        net_.set_output(new MiniDNN::MultiClassEntropy());
        
        // Initialize weights
        net_.init(0, 0.01, 123);
        
        // Grab initial weights
        weights_ = net_.get_parameters();
    }

    void set_weights(const ParameterList& weights) {
        weights_ = weights;
        net_.set_parameters(weights_);
    }

    ParameterList get_weights() const {
        return net_.get_parameters();
    }

    // Returns {per-layer gradient vectors, batch loss}
    std::pair<ParameterList, double> compute_gradients(
        const Eigen::MatrixXd& batch_X,
        const Eigen::MatrixXd& batch_Y)
    {
        net_.forward(batch_X);
        net_.backprop(batch_X, batch_Y);
        // We can safely extract the mathematical loss dynamically from the output branch
        double loss = net_.get_output()->loss();
        return {net_.get_derivatives(), loss};
    }

private:
    MiniDNN::Network net_;
    ParameterList weights_;
};
