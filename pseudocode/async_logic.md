# Asynchronous Parallel SGD (ASGD / HOGWILD!) Architecture

```python
# Global State Initialization
theta_global    # The shared global model parameters
M               # Total number of active worker threads
eta             # Learning rate

def AsyncWorker(worker_id):
    while not ISFINISHED():
        # 1. Read the current global model state (No waiting at a barrier!)
        theta_local = theta_global
        
        # 2. Retrieve a specific subset of the mini-batch for this worker
        batch_subset = GetBatchSubset(worker_id)
        
        # 3. Compute the local gradient based on the local model and data subset
        #    CRITICAL NOTE: While this computation is happening, other workers
        #    are actively updating theta_global in the background!
        g_local = ComputeGradient(theta_local, batch_subset)
        
        # 4. Apply the computed gradient directly to the global model.
        #    This uses lock-free HOGWILD! semantics, meaning concurrent global
        #    model updates can be interleaved without locks.
        theta_global = theta_global - eta * g_local
        
        # 5. Immediately loop back to start the next step.
        #    Zero idle time. No barriers.
```
