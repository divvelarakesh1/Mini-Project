#pragma once
#include <atomic>
#include <tuple>

/**
 * @class Dispatcher
 * @brief Controls per-step interval-asynchrony gating for parallel threads.
 *
 * In highly concurrent environments like Hogwild!, it is sometimes optimal
 * to conditionally pause or synchronize threads. The Dispatcher intercepts
 * threads before they compute a step (`try_start_step`) and after they calculate
 * gradients (`finish_step`), dictating whether the global weights should physically
 * be updated. Currently acts as a non-blocking pass-through but can be scaled 
 * to implement complex lock-free concurrency bounds.
 */
class Dispatcher {
public:
    // Returns {can_start, captured_step_index}
    std::tuple<bool, int> try_start_step() {
    }

    // Returns true if caller should apply the gradient update
    bool finish_step(int /*step_index*/) {
    }

private:
};
