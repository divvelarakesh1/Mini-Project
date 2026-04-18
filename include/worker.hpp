#pragma once
#include "config.hpp"
#include "data.hpp"
#include "dispatcher.hpp"
#include "model.hpp"
#include "monitor.hpp"
#include <vector>

using ParameterList = std::vector<std::vector<double>>;

namespace Worker {

void run_async(int thread_id, ParameterList &global_weights, DataLoader &loader,
               Dispatcher &dispatcher, const Config &config, Monitor &monitor);

void run_sequential(ParameterList &global_weights, DataLoader &loader,
                    const Config &config, Monitor &monitor);

} // namespace Worker