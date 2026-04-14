#include "config.hpp"
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

Config load_config(const std::string &path) {
  Config config;
  std::ifstream file(path);

  // Fallback: Check parent directory if not found (vibrant for build/ folders)
  if (!file.is_open()) {
    file.open("../" + path);
  }

  if (!file.is_open()) {
    std::cerr << "[Warning] Could not open config file: " << path
              << " (checked current and parent dir). Using defaults.\n";
    return config;
  }

  json j;
  try {
    file >> j;
  } catch (json::parse_error &e) {
    std::cerr << "[Error] JSON parse error in " << path << ": " << e.what()
              << "\n";
    return config;
  }

  if (j.contains("total_steps"))
    config.total_steps = j["total_steps"];
  if (j.contains("eta"))
    config.eta = j["eta"];
  if (j.contains("momentum"))
    config.momentum = j["momentum"];
  if (j.contains("lambda"))
    config.lambda = j["lambda"];
  if (j.contains("batch_size"))
    config.batch_size = j["batch_size"];
  if (j.contains("interval_size"))
    config.interval_size = j["interval_size"];
  if (j.contains("interval_decay_freq"))
    config.interval_decay_freq = j["interval_decay_freq"];
  if (j.contains("num_threads"))
    config.num_threads = j["num_threads"];
  if (j.contains("log_interval"))
    config.log_interval = j["log_interval"];
  if (j.contains("beta1"))
    config.beta1 = j["beta1"];
  if (j.contains("beta2"))
    config.beta2 = j["beta2"];
  if (j.contains("epsilon"))
    config.epsilon = j["epsilon"];

  if (j.contains("exec_mode")) {
    std::string mode = j["exec_mode"];
    if (mode == "SEQUENTIAL")
      config.exec_mode = ExecutionMode::SEQUENTIAL;
    else if (mode == "ASYNC_HOGWILD")
      config.exec_mode = ExecutionMode::ASYNC_HOGWILD;
    else if (mode == "SYNC_PARALLEL")
      config.exec_mode = ExecutionMode::SYNC_PARALLEL;
    else if (mode == "INTERVAL_ASYNC")
      config.exec_mode = ExecutionMode::INTERVAL_ASYNC;
  }

  if (j.contains("opt_mode")) {
    std::string mode = j["opt_mode"];
    if (mode == "STANDARD_SGD")
      config.opt_mode = OptimizerMode::STANDARD_SGD;
    else if (mode == "DC_ASGD")
      config.opt_mode = OptimizerMode::DC_ASGD;
  }

  if (j.contains("opt_algo")) {
    std::string algo = j["opt_algo"];
    if (algo == "SGD")
      config.opt_algo = OptimizerAlgorithm::SGD;
    else if (algo == "ADAM")
      config.opt_algo = OptimizerAlgorithm::ADAM;
    else if (algo == "RMSPROP")
      config.opt_algo = OptimizerAlgorithm::RMSPROP;
  }

  if (j.contains("dataset")) {
    std::string ds = j["dataset"];
    if (ds == "MNIST")
      config.dataset = DatasetType::MNIST;
    else if (ds == "CIFAR")
      config.dataset = DatasetType::CIFAR;
  }

  return config;
}
