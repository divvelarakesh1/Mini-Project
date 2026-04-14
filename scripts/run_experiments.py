#!/usr/bin/env python3
import json
import subprocess
import os
import sys
import shutil

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.json")
BUILD_DIR = os.path.join(PROJECT_ROOT, "build")
EXECUTABLE = "./mini_project"
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")

EXPERIMENTS = [
    {"name": "1_Sequential", "exec_mode": "SEQUENTIAL", "opt_mode": "STANDARD_SGD"},
    {"name": "2_Synchronous", "exec_mode": "SYNC_PARALLEL", "opt_mode": "STANDARD_SGD"},
    {"name": "3_Plain_Async", "exec_mode": "ASYNC_HOGWILD", "opt_mode": "STANDARD_SGD"},
    {"name": "4_Async_Penalty", "exec_mode": "ASYNC_HOGWILD", "opt_mode": "DC_ASGD"},
    {"name": "5_Interval_Async", "exec_mode": "INTERVAL_ASYNC", "opt_mode": "STANDARD_SGD"},
    {"name": "6_Interval_Async_Penalty", "exec_mode": "INTERVAL_ASYNC", "opt_mode": "DC_ASGD"}
]

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)

def save_config(cfg):
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)

def main():
    if not os.path.exists(LOGS_DIR):
        os.makedirs(LOGS_DIR)

    original_config = load_config()
    
    # Base adjustments to make runs quick for initial testing.
    # Set this up based on your desired test epochs (total_steps).
    # Since total_steps = 1000 or 10000, we'll just run whatever is in config but override the target modes.
    
    for exp in EXPERIMENTS:
        print(f"===================================================")
        print(f"Starting Experiment: {exp['name']}")
        print(f"===================================================")
        
        cfg = load_config()
        cfg["exec_mode"] = exp["exec_mode"]
        cfg["opt_mode"] = exp["opt_mode"]
        save_config(cfg)
        
        log_file = os.path.join(LOGS_DIR, f"{exp['name']}.log")
        
        # Run binary from build directory
        os.chdir(BUILD_DIR)
        
        try:
            with open(log_file, "w") as out:
                subprocess.run([EXECUTABLE], stdout=out, stderr=subprocess.STDOUT, check=True)
            print(f"--> Saved output to logs/{os.path.basename(log_file)}")
        except subprocess.CalledProcessError as e:
            print(f"--> ERROR: Benchmark failed for {exp['name']}. Check {log_file} for details.")
        
        # return to original script directory
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        
    print()
    print("All experiments completed successfully. Restoring original configuration.")
    save_config(original_config)

if __name__ == "__main__":
    main()
