#pragma once

#include "config.hpp"
#include <Eigen/Core>
#include <MiniDNN.h>
#include <vector>

class DataLoader;

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
 * This class encapsulates the creation, weight management, and gradient
 * computation of a neural network. It supports different architectures
 * (CIFAR-10, MNIST) dynamically based on the provided configuration.
 */
class MiniDNNModel {
public:
  /**
   * @brief Constructs a network architecture based on the selected dataset.
   * @param config The global configuration object containing dataset and
   * hyperparameter settings.
   */
  MiniDNNModel(const Config &config);

  void set_weights(const ParameterList &weights);
  ParameterList get_weights() const;
  const ParameterList &get_cached_weights() const;

  std::pair<ParameterList, double> compute_gradients(const Eigen::MatrixXd &batch_X, const Eigen::MatrixXd &batch_Y);

  /**
   * @brief Evaluates the model on a given dataset and returns the accuracy.
   * @param loader The DataLoader containing the evaluation dataset.
   * @return A double-precision value representing the accuracy ratio (0.0 to 1.0).
   */
  double evaluate(const DataLoader &loader);

  MiniDNN::Network *get_network();

private:
  void setup_cifar_architecture();
  void setup_mnist_architecture();
  void apply_he_initialization(DatasetType dataset);

  MiniDNN::Network net_;
  ParameterList weights_;
};