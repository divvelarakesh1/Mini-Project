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

  if (j.contains("target_epochs"))
    config.target_epochs = j["target_epochs"];
  if (j.contains("target_time_seconds"))
    config.target_time_seconds = j["target_time_seconds"];
  if (j.contains("eta"))
    config.eta = j["eta"];
  if (j.contains("momentum"))
    config.momentum = j["momentum"];
  if (j.contains("lambda"))
    config.lambda = j["lambda"];
  if (j.contains("dc_asgd_rms_momentum"))
    config.dc_asgd_rms_momentum = j["dc_asgd_rms_momentum"];
  if (j.contains("dc_asgd_rms_epsilon"))
    config.dc_asgd_rms_epsilon = j["dc_asgd_rms_epsilon"];
  if (j.contains("batch_size"))
    config.batch_size = j["batch_size"];
  if (j.contains("interval_size"))
    config.interval_size = j["interval_size"];
  if (j.contains("num_threads"))
    config.num_threads = j["num_threads"];

  // Parse probing parameters
  if (j.contains("use_probing"))
    config.use_probing = j["use_probing"];
  if (j.contains("use_thread_probing"))
    config.use_thread_probing = j["use_thread_probing"];
  if (j.contains("probe_initial_interval"))
    config.probe_initial_interval = j["probe_initial_interval"];
  if (j.contains("probe_min_interval"))
    config.probe_min_interval = j["probe_min_interval"];
  if (j.contains("probe_test_steps"))
    config.probe_test_steps = j["probe_test_steps"];
  if (j.contains("probe_exec_steps"))
    config.probe_exec_steps = j["probe_exec_steps"];
  if (j.contains("thread_min_count"))
    config.thread_min_count = j["thread_min_count"];

  if (j.contains("exec_mode")) {
    std::string mode = j["exec_mode"];
    if (mode == "SEQUENTIAL")
      config.exec_mode = ExecutionMode::SEQUENTIAL;
    else if (mode == "ASYNC_HOGWILD")
      config.exec_mode = ExecutionMode::ASYNC_HOGWILD;
    else if (mode == "INTERVAL_ASYNC")
      config.exec_mode = ExecutionMode::INTERVAL_ASYNC;
  }

  if (j.contains("opt_mode")) {
    std::string mode = j["opt_mode"];
    if (mode == "STANDARD_SGD")
      config.opt_mode = OptimizerMode::STANDARD_SGD;
    else if (mode == "DC_ASGD_C" || mode == "DC_ASGD")
      config.opt_mode = OptimizerMode::DC_ASGD_C;
    else if (mode == "DC_ASGD_A")
      config.opt_mode = OptimizerMode::DC_ASGD_A;
  }

  if (j.contains("stop_mode")) {
    std::string mode = j["stop_mode"];
    if (mode == "EPOCHS")
      config.stop_mode = StopMode::EPOCHS;
    else if (mode == "TIME")
      config.stop_mode = StopMode::TIME;
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
