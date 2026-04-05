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
 * set_weights()       — load global weights into the local network
 * get_weights()       — read current weights out of the local network
 * compute_gradients() — run a forward+backward pass on a mini-batch, returning {gradients, loss}
 */
class MiniDNNModel {
public:
    MiniDNNModel(const Config& config) {
        if (config.dataset == DatasetType::CIFAR) {
            // ---------------------------------------------------------
            // CIFAR-10 ARCHITECTURE (Mini-VGG Style)
            // Input: 32x32x3
            // ---------------------------------------------------------
            // Block 1: Conv -> Pool
            // Conv 1: 3x3 filter, 16 channels. Output -> 30x30x16
            MiniDNN::Layer* conv1 = new MiniDNN::Convolutional<MiniDNN::ReLU>(32, 32, 3, 16, 3, 3);
            // Pool 1: 2x2 window. Output -> 15x15x16
            MiniDNN::Layer* pool1 = new MiniDNN::MaxPooling<MiniDNN::Identity>(30, 30, 16, 2, 2);
            
            // Block 2: Conv -> Pool
            // Conv 2: 3x3 filter, 32 channels. Output -> 13x13x32
            MiniDNN::Layer* conv2 = new MiniDNN::Convolutional<MiniDNN::ReLU>(15, 15, 16, 32, 3, 3);
            // Pool 2: 2x2 window. Output -> 6x6x32
            MiniDNN::Layer* pool2 = new MiniDNN::MaxPooling<MiniDNN::Identity>(13, 13, 32, 2, 2);
            
            // Block 3: Fully Connected Classifier
            // FC 1: Flattened 6*6*32 (1152) -> 128 hidden nodes (ReLU)
            MiniDNN::Layer* fc1 = new MiniDNN::FullyConnected<MiniDNN::ReLU>(6 * 6 * 32, 128);
            // FC 2: 128 hidden nodes -> 10 output classes (Softmax)
            MiniDNN::Layer* fc2 = new MiniDNN::FullyConnected<MiniDNN::Softmax>(128, 10);
            
            net_.add_layer(conv1);
            net_.add_layer(pool1);
            net_.add_layer(conv2);
            net_.add_layer(pool2);
            net_.add_layer(fc1);
            net_.add_layer(fc2);
            
        } else if (config.dataset == DatasetType::MNIST) {
            // ---------------------------------------------------------
            // MNIST ARCHITECTURE (Standard LeNet Style)
            // Input: 28x28x1
            // ---------------------------------------------------------
            // Conv 1: 5x5 filter, 8 channels. Output -> 24x24x8
            MiniDNN::Layer* conv1 = new MiniDNN::Convolutional<MiniDNN::ReLU>(28, 28, 1, 8, 5, 5);
            // Pool 1: 2x2 window. Output -> 12x12x8
            MiniDNN::Layer* pool1 = new MiniDNN::MaxPooling<MiniDNN::Identity>(24, 24, 8, 2, 2);
            
            // FC 1: Flattened 12*12*8 (1152) -> 64 hidden nodes (ReLU)
            MiniDNN::Layer* fc1 = new MiniDNN::FullyConnected<MiniDNN::ReLU>(12 * 12 * 8, 64);
            // FC 2: 64 hidden nodes -> 10 output classes (Softmax)
            MiniDNN::Layer* fc2 = new MiniDNN::FullyConnected<MiniDNN::Softmax>(64, 10);
            
            net_.add_layer(conv1);
            net_.add_layer(pool1);
            net_.add_layer(fc1);
            net_.add_layer(fc2);
        }
        
        // Final Output Loss Calculation
        net_.set_output(new MiniDNN::MultiClassEntropy());
        
        // Initialize weights (mean 0, variance 0.01, random seed)
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