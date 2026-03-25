# Synchronous Parallel SGD (SSGD) Architecture

```python
# Global State Initialization
theta_global    # The shared global model parameters
M               # Total number of active worker threads
eta             # Learning rate
barrier         # Synchronization barrier initialized for M threads

def SyncWorker(worker_id):
    while not ISFINISHED():
        # 1. Synchronize before starting a new step to ensure everyone is on the same page
        WaitAtBarrier(barrier)
        
        # 2. Make a local copy of the latest global model state
        theta_local = theta_global
        
        # 3. Retrieve a specific subset of the mini-batch for this worker
        batch_subset = GetBatchSubset(worker_id)
        
        # 4. Compute the local gradient based on the local model and data subset
        g_local = ComputeGradient(theta_local, batch_subset)
        
        # 5. Push the computed gradient to the central aggregator or parameter server
        PushGradientToServer(g_local)
        
        # 6. SYNCHRONIZATION BOTTLENECK: Wait for all M workers to finish computing
        WaitAtBarrier(barrier)
        
        # 7. Aggregate all gradients and apply the update to the global model
        #    (Usually handled by a designated master thread or the parameter server)
        if worker_id == 0:
            g_aggregated = AggregateAllGradients()
            theta_global = theta_global - eta * g_aggregated
        
        # 8. Wait for the global update to finish before pulling the new model
        WaitAtBarrier(barrier)
```
