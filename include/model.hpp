#pragma once

#include "config.hpp"
#include <Eigen/Core>
#include <MiniDNN.h>
#include <vector>

/**
 * @typedef ParameterList
 * @brief Represents the weight and bias parameters of a neural network.
 * 
 * Stored as a vector of vectors (one per layer) of double-precision values.
 * This format is used to sync weights across parallel worker threads.
 */
using ParameterList = std::vector<std::vector<double>>;

/**
 * @class MiniDNNModel
 * @brief High-level wrapper for the MiniDNN neural network.
 *
 * This class encapsulates the creation, weight management, and gradient computation
 * of a neural network. It supports different architectures (CIFAR-10, MNIST) 
 * dynamically based on the provided configuration.
 */
class MiniDNNModel {
public:
    /**
     * @brief Constructs a network architecture based on the selected dataset.
     * @param config The global configuration object containing dataset and hyperparameter settings.
     */
    MiniDNNModel(const Config &config) {
        if (config.dataset == DatasetType::CIFAR) {
            setup_cifar_architecture();
        } else if (config.dataset == DatasetType::MNIST) {
            setup_mnist_architecture();
        }

        // Use Multi-Class Cross Entropy for classification tasks
        net_.set_output(new MiniDNN::MultiClassEntropy());

        // Initialize weights with a normal distribution (mean=0, sigma=0.01)
        // seed 123 provides deterministic initialization for debugging
        net_.init(0, 0.01, 123);

        // Cache the initial weight state
        weights_ = net_.get_parameters();
    }

    /**
     * @brief Loads a set of global weights into the local network instance.
     * @param weights The source weight list to copy into the network.
     */
    void set_weights(const ParameterList &weights) {
        weights_ = weights;
        net_.set_parameters(weights_);
    }

    /**
     * @brief Retrieves the current weights from the network.
     * @return A deep copy of the network parameter list.
     */
    ParameterList get_weights() const { 
        return net_.get_parameters(); 
    }

    /**
     * @brief Retrieves the last set global weights cached internally.
     * @return A constant reference to avoid a deep copy allocation.
     */
    const ParameterList& get_cached_weights() const { 
        return weights_; 
    }

    /**
     * @brief Computes gradients and loss for a single mini-batch.
     * 
     * Executes a full Forward-Backward pass.
     * 
     * @param batch_X Normalized input feature matrix (num_features x batch_size).
     * @param batch_Y One-hot encoded target label matrix (num_classes x batch_size).
     * @return A pair containing:
     *         - First: Calculated gradients per layer (ParameterList format).
     *         - Second: Scalar batch loss value.
     */
    std::pair<ParameterList, double> compute_gradients(const Eigen::MatrixXd &batch_X,
                                                       const Eigen::MatrixXd &batch_Y) {
        // Forward propagation: Calculate layer activations
        net_.forward(batch_X);
        
        // Backward propagation: Calculate error terms and gradients
        net_.backprop(batch_X, batch_Y);
        
        // Extract the loss value from the output layer
        double loss = net_.get_output()->loss();
        
        return {net_.get_derivatives(), loss};
    }

    /**
     * @brief Accessor for the underlying MiniDNN network object.
     */
    MiniDNN::Network *get_network() { return &net_; }

private:
    /**
     * @brief Configures a Convolutional Neural Network optimized for color CIFAR-10 images.
     * 
     * Topology:
     * - Conv(3x3x3, 16) -> ReLU -> MaxPool(2x2)
     * - Conv(3x3x16, 32) -> ReLU -> MaxPool(2x2)
     * - FC(1152, 128) -> ReLU
     * - FC(128, 10) -> Softmax
     */
    void setup_cifar_architecture() {
        // Block 1: Feature Extraction
        net_.add_layer(new MiniDNN::Convolutional<MiniDNN::ReLU>(32, 32, 3, 16, 3, 3));
        net_.add_layer(new MiniDNN::MaxPooling<MiniDNN::Identity>(30, 30, 16, 2, 2));

        // Block 2: Higher-level representation
        net_.add_layer(new MiniDNN::Convolutional<MiniDNN::ReLU>(15, 15, 16, 32, 3, 3));
        net_.add_layer(new MiniDNN::MaxPooling<MiniDNN::Identity>(13, 13, 32, 2, 2));

        // Block 3: Classification
        net_.add_layer(new MiniDNN::FullyConnected<MiniDNN::ReLU>(6 * 6 * 32, 128));
        net_.add_layer(new MiniDNN::FullyConnected<MiniDNN::Softmax>(128, 10));
    }

    /**
     * @brief Configures a LeNet-style architecture for grayscale MNIST digits.
     * 
     * Topology:
     * - Conv(5x5x1, 8) -> ReLU -> MaxPool(2x2)
     * - FC(1152, 64) -> ReLU
     * - FC(64, 10) -> Softmax
     */
    void setup_mnist_architecture() {
        net_.add_layer(new MiniDNN::Convolutional<MiniDNN::ReLU>(28, 28, 1, 8, 5, 5));
        net_.add_layer(new MiniDNN::MaxPooling<MiniDNN::Identity>(24, 24, 8, 2, 2));

        net_.add_layer(new MiniDNN::FullyConnected<MiniDNN::ReLU>(12 * 12 * 8, 64));
        net_.add_layer(new MiniDNN::FullyConnected<MiniDNN::Softmax>(64, 10));
    }

    MiniDNN::Network net_;
    ParameterList weights_;
};