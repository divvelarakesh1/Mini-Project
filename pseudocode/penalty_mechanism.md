# Delay Compensated Asynchronous SGD (DC-ASGD) Architecture

```python
# ==========================================
# 1. GLOBAL STATE (PARAMETER SERVER)
# ==========================================
theta_global    # The shared global model parameters (w_t in the paper)
M               # Total number of active worker threads
eta             # Learning rate
lambda_0        # Initial variance control parameter (the "penalty" scale)
theta_bak       # An array/map storing a backup of the model for each worker
MeanSquare      # Used for adaptive lambda tuning (moving average of gradients)

# ==========================================
# 2. THE WORKER THREAD COMPONENT
# ==========================================
# Workers are completely asynchronous and barrier-free.
def DC_AsyncWorker(worker_id):
    while not ISFINISHED():
        # 1. Pull the latest global model from the server
        theta_local = RequestModelFromServer(worker_id)
        
        # 2. Fetch data and compute the STANDARD gradient
        #    (The worker doesn't know about the delay compensation)
        batch_subset = GetBatchSubset(worker_id)
        g_local      = ComputeGradient(theta_local, batch_subset)
        
        # 3. Push the computed gradient to the parameter server
        PushGradientToServer(worker_id, g_local)

# ==========================================
# 3. THE PARAMETER SERVER COMPONENT
# ==========================================
# The server handles requests and mathematically compensates for delays.

# Event A: A worker asks for the latest model
def on_pull_request(worker_id):
    # 1. Backup the current global model state for THIS specific worker
    #    We need this later to calculate exactly how "delayed" they are.
    theta_bak[worker_id] = theta_global
    
    # 2. Send the current model to the worker
    return theta_global

# Event B: A worker submits a computed gradient
def on_gradient_push(worker_id, g_local):
    # 1. Calculate the delay gap (how much the global model changed
    #    while this specific worker was computing)
    delay_gap = theta_global - theta_bak[worker_id]
    
    # 2. Tune the penalty parameter lambda (DC-ASGD-a: Adaptive method)
    #    Uses a moving average to reduce variance among coordinates.
    m_decay    = 0.95
    MeanSquare = (m_decay * MeanSquare) + ((1 - m_decay) * (g_local * g_local))
    lambda_t   = lambda_0 / (MeanSquare + 1e-7)   # Adaptive lambda
    
    # (Note: For DC-ASGD-c (Constant), you would just use lambda_t = lambda_0)
    
    # 3. Compute the Delay-Compensated Gradient ("The Penalty")
    #    Formula: g_local + lambda * g_local ⊙ g_local ⊙ delay_gap
    #    (⊙ = element-wise multiplication)
    hessian_approx = lambda_t * (g_local * g_local)   # Diagonal Hessian approx
    penalty_term   = hessian_approx * delay_gap
    
    delay_compensated_gradient = g_local + penalty_term
    
    # 4. Update the global model with the compensated gradient
    theta_global = theta_global - eta * delay_compensated_gradient
```
