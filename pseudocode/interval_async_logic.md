# Interval-Asynchronous SGD Architecture

```python
# ==========================================
# 1. GLOBAL STATE INITIALIZATION
# ==========================================
theta_global    # Shared global model parameters
M               # Total number of active worker threads
eta             # Learning rate
y               # The current asynchronous interval size (number of steps)

# ==========================================
# 2. THE DISPATCHER COMPONENT
# ==========================================
# Manages interval boundaries and lock-free thread coordination.
class Dispatcher:
    I_first   = 0    # The step index that initiated the current interval
    I_done    = 0    # Number of steps accepted so far in the current interval
    s_started = 0    # Total steps started across the entire execution
    s_done    = 0    # Total steps completed across the entire execution
    y         = y_0  # Initial interval size

    # Called by workers before they compute a gradient
    def TryStartStep(worker_id):
        if worker_id >= M:
            return (False, -1)  # Restricts active threads if M changes dynamically
        
        # Atomically increment s_started (Fetch-and-Add)
        step_index = FAA(s_started, 1)
        return (True, step_index)

    # Called by workers after computing a gradient
    def FinishStep(worker_id, step_index, loss):
        # 1. Check if the step crosses the boundary.
        #    If it started before the current interval, reject it to prevent staleness.
        if step_index < I_first:
            return False
        
        # 2. Use Compare-And-Swap (CAS) in a retry loop for lock-free updates
        while True:
            old_done = I_done
            
            # Check if the interval closed while we were computing or waiting
            if old_done >= y or step_index < I_first:
                return False
            
            new_done = old_done + 1
            if CAS(I_done, old_done, new_done):  # Retry until CAS succeeds
                break
        
        # 3. INTERVAL BOUNDARY REACHED: Advance to the next interval
        #    We only do this if WE were the specific thread that hit the limit (new_done == y)
        if new_done == y:
            y         = UpdateIntervalSize(y, loss)  # Tune the interval size
            I_first   = s_started                    # Reset the boundary marker
            I_done    = 0                            # Reset accepted count
        
        FAA(s_done, 1)
        return True

    # Parameter Tuning Logic: Adjusting the degree of asynchrony
    def UpdateIntervalSize(current_y, current_loss):
        # Strategy A: y-decay (Simple & Consistent)
        # Gradually decrease y over time because the model becomes more
        # sensitive to noise as it approaches convergence.
        return current_y - 1

        # -- OR --

        # Strategy B: Adaptive Window-Probing
        # Periodically test different interval sizes y' in the range
        # [y - kw/2, y + kw/2) for a set number of steps (p).
        # Track the loss for each candidate, and select the y' that yields
        # the steepest convergence rate for the next execution phase.
        # return optimal_y_found_by_probing

# ==========================================
# 3. THE WORKER THREAD COMPONENT
# ==========================================
def IntervalAsyncWorker(worker_id):
    while not ISFINISHED():
        
        # 1. Ask the Dispatcher for permission to start
        can_start, step_index = Dispatcher.TryStartStep(worker_id)
        
        if can_start:
            # 2. Make a local copy of the global model
            theta_local = theta_global
            
            # 3. Fetch data and compute gradient
            batch_subset      = GetBatchSubset(step_index)
            g_local, loss     = ComputeGradient(theta_local, batch_subset)
            
            # 4. Ask the Dispatcher if we are allowed to apply this gradient
            #    (It will be rejected if the interval window has already moved on)
            is_accepted = Dispatcher.FinishStep(worker_id, step_index, loss)
            
            if is_accepted:
                # 5. Apply gradient to the global model concurrently (HOGWILD semantics)
                theta_global = theta_global - eta * g_local
            # If rejected, the computed gradient is simply discarded.
```
