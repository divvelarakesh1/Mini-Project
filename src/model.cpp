#include "model.hpp"
#include "data.hpp"
#include <random>

MiniDNNModel::MiniDNNModel(const Config &config) {
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
  
  // Apply He initialization iteratively to existing layers
  apply_he_initialization(config.dataset);

  // Cache the initial weight state
  weights_ = net_.get_parameters();
}

void MiniDNNModel::set_weights(const ParameterList &weights) {
  weights_ = weights;
  net_.set_parameters(weights_);
}

ParameterList MiniDNNModel::get_weights() const { 
  return net_.get_parameters(); 
}

const ParameterList &MiniDNNModel::get_cached_weights() const { 
  return weights_; 
}

std::pair<ParameterList, double> MiniDNNModel::compute_gradients(const Eigen::MatrixXd &batch_X, const Eigen::MatrixXd &batch_Y) {
  // Forward propagation: Calculate layer activations
  net_.forward(batch_X);

  // Backward propagation: Calculate error terms and gradients
  net_.backprop(batch_X, batch_Y);

  // Extract the loss value from the output layer
  double loss = net_.get_output()->loss();

  return {net_.get_derivatives(), loss};
}

double MiniDNNModel::evaluate(const DataLoader &loader) {
  const Eigen::MatrixXd &X = loader.get_images();
  const std::vector<uint8_t> &Y = loader.get_labels();

  if (X.cols() == 0 || Y.empty()) {
    return 0.0;
  }

  // Predict outputs for the entire dataset
  Eigen::MatrixXd pred_scores = net_.predict(X);

  int correct = 0;
  int n = static_cast<int>(Y.size());

  for (int i = 0; i < n; ++i) {
    int pred_idx;
    pred_scores.col(i).maxCoeff(&pred_idx);
    if (pred_idx == static_cast<int>(Y[i])) {
      correct++;
    }
  }

  return static_cast<double>(correct) / n;
}

MiniDNN::Network *MiniDNNModel::get_network() { 
  return &net_; 
}

void MiniDNNModel::setup_cifar_architecture() {
  // Block 1: Feature Extraction
  net_.add_layer(new MiniDNN::Convolutional<MiniDNN::ReLU>(32, 32, 3, 6, 5, 5));
  net_.add_layer(new MiniDNN::MaxPooling<MiniDNN::Identity>(28, 28, 6, 2, 2));

  // Block 2: Higher-level representation
  net_.add_layer(new MiniDNN::Convolutional<MiniDNN::ReLU>(14, 14, 6, 16, 5, 5));
  net_.add_layer(new MiniDNN::MaxPooling<MiniDNN::Identity>(10, 10, 16, 2, 2));

  // Block 3: Classification
  net_.add_layer(new MiniDNN::FullyConnected<MiniDNN::ReLU>(5 * 5 * 16, 120));
  net_.add_layer(new MiniDNN::FullyConnected<MiniDNN::Softmax>(120, 10));
}

void MiniDNNModel::setup_mnist_architecture() {
  net_.add_layer(new MiniDNN::Convolutional<MiniDNN::ReLU>(28, 28, 1, 8, 5, 5));
  net_.add_layer(new MiniDNN::MaxPooling<MiniDNN::Identity>(24, 24, 8, 2, 2));

  net_.add_layer(new MiniDNN::FullyConnected<MiniDNN::ReLU>(12 * 12 * 8, 64));
  net_.add_layer(new MiniDNN::FullyConnected<MiniDNN::Softmax>(64, 10));
}

void MiniDNNModel::apply_he_initialization(DatasetType dataset) {
  auto params = net_.get_parameters();
  
  // fan_in values for each TRAINABLE layer only (Conv and FC layers).
  // MaxPooling layers return empty parameter vectors and must be skipped.
  std::vector<int> fan_ins;
  std::vector<int> bias_counts;
  
  if (dataset == DatasetType::CIFAR) {
    // Conv(3×5×5), Conv(6×5×5), FC(400→120), FC(120→10)
    fan_ins     = {3 * 5 * 5, 6 * 5 * 5, 400, 120};
    bias_counts = {6, 16, 120, 10};
  } else if (dataset == DatasetType::MNIST) {
    // Conv(1×5×5), FC(1152→64), FC(64→10)
    fan_ins     = {1 * 5 * 5, 1152, 64};
    bias_counts = {8, 64, 10};
  } else {
    return;
  }

  std::mt19937 rng(123);
  size_t trainable_idx = 0;

  for (size_t i = 0; i < params.size() && trainable_idx < fan_ins.size(); ++i) {
    // Skip non-trainable layers (e.g. MaxPooling returns empty params)
    if (params[i].empty()) {
      continue;
    }

    double stddev = std::sqrt(2.0 / fan_ins[trainable_idx]);
    std::normal_distribution<double> dist(0.0, stddev);
    
    int num_biases  = bias_counts[trainable_idx];
    int num_weights = static_cast<int>(params[i].size()) - num_biases;
    
    // He-scale the weights
    for (int j = 0; j < num_weights; ++j) {
      params[i][j] = dist(rng);
    }
    // Zero-initialize biases
    for (size_t j = static_cast<size_t>(num_weights); j < params[i].size(); ++j) {
      params[i][j] = 0.0;
    }
    
    ++trainable_idx;
  }
  net_.set_parameters(params);
}
