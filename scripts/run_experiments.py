#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import subprocess
import warnings

MPL_CONFIG_DIR = os.path.join("/tmp", "mini_project_mpl")
os.makedirs(MPL_CONFIG_DIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", MPL_CONFIG_DIR)
warnings.filterwarnings("ignore", message="Unable to import Axes3D.*")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MATPLOTLIB = True
except ImportError:
    plt = None
    HAVE_MATPLOTLIB = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.json")
BUILD_DIR = os.path.join(PROJECT_ROOT, "build")
EXECUTABLE = "./mini_project"
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")

# ANSI stripping regex
ANSI_ESCAPE = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

def strip_ansi(text):
    return ANSI_ESCAPE.sub('', text)

def duration_to_seconds(duration_str):
    """Converts MM:SS or HH:MM:SS to total seconds."""
    try:
        parts = list(map(int, duration_str.split(':')))
        if len(parts) == 2: # MM:SS
            return float(parts[0] * 60 + parts[1])
        if len(parts) == 3: # HH:MM:SS
            return float(parts[0] * 3600 + parts[1] * 60 + parts[2])
    except Exception:
        pass
    return 0.0

def downsample_points(xs, ys, max_points=100):
    """Reduces the number of points by averaging them into buckets."""
    n = len(xs)
    if n <= max_points:
        return xs, ys
    
    bucket_size = n / max_points
    new_xs = []
    new_ys = []
    
    for i in range(max_points):
        start = int(i * bucket_size)
        end = int((i + 1) * bucket_size) if i < max_points - 1 else n
        
        if start >= end:
            continue
            
        bucket_xs = xs[start:end]
        bucket_ys = ys[start:end]
        
        new_xs.append(sum(bucket_xs) / len(bucket_xs))
        new_ys.append(sum(bucket_ys) / len(bucket_ys))
        
    return new_xs, new_ys

MONITOR_PATTERN = re.compile(r"\[Monitor\]\s+Epoch:\s*([0-9.]+)")
LOSS_PATTERN = re.compile(r"Loss:\s*([-+0-9.eE]+)")
SPEED_PATTERN = re.compile(r"([0-9.]+)\s+img/s")
ELAPSED_PATTERN = re.compile(r"Elapsed:\s*([0-9:]+)")

# ==============================================================================
# EXPERIMENT DEFINITIONS
# ==============================================================================
# All experiments share: eta = 0.005, momentum = 0.5, use_probing = false
# Per-experiment overrides: exec_mode, opt_mode, batch_size, num_threads,
#                           interval_size, lambda
# ==============================================================================
EXPERIMENTS = [
    # 1. Baseline: single-threaded SGD. Batch 16, 1 thread.
    {"name": "1_Sequential", "exec_mode": "SEQUENTIAL", "opt_mode": "STANDARD_SGD",
     "batch_size": 16, "num_threads": 1,
     "lambda": 0.0, "interval_size": 0},

    # 2. Hogwild!: fully asynchronous lock-free SGD.
    {"name": "2_Plain_Async", "exec_mode": "ASYNC_HOGWILD", "opt_mode": "STANDARD_SGD",
     "batch_size": 16, "num_threads": 16,
     "lambda": 0.0, "interval_size": 0},

    # 3. DC-ASGD-a: Hogwild + adaptive delay-compensation (RMSProp-style λ).
    #    Paper recommends λ₀=2.0, m=0.95, ε=1e-7 for CIFAR-10.
    {"name": "3_Async_DC_ASGD_A", "exec_mode": "ASYNC_HOGWILD", "opt_mode": "DC_ASGD_A",
     "batch_size": 16, "num_threads": 16,
     "lambda": 2.0, "interval_size": 0,
     "dc_asgd_rms_momentum": 0.95, "dc_asgd_rms_epsilon": 1e-7},

    # 4. Interval-Async: bounded-staleness with y=128.
    {"name": "4_Interval_Async", "exec_mode": "INTERVAL_ASYNC", "opt_mode": "STANDARD_SGD",
     "batch_size": 16, "num_threads": 16,
     "lambda": 0.0, "interval_size": 128},

    # 5. Interval-Async + DC-ASGD-a: wider interval (256) is safe because
    #    the adaptive delay-compensation math corrects stale gradients.
    {"name": "5_Interval_DC_ASGD_A", "exec_mode": "INTERVAL_ASYNC", "opt_mode": "DC_ASGD_A",
     "batch_size": 16, "num_threads": 16,
     "lambda": 2.0, "interval_size": 256,
     "dc_asgd_rms_momentum": 0.95, "dc_asgd_rms_epsilon": 1e-7},

    # 6. Self-Balancing: Using the new Bounded Neighborhood Search (localized probing).
    {"name": "6_Self_Balancing", "exec_mode": "INTERVAL_ASYNC", "opt_mode": "DC_ASGD_A",
     "batch_size": 16, "num_threads": 64, # Initial thread limit
     "lambda": 2.0, "interval_size": 128, # Initial interval
     "use_probing": True, "use_thread_probing": True,
     "probe_test_steps": 1000, "probe_exec_steps": 10000},
]

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run all training-mode experiments with a shared stop criterion."
    )
    group = parser.add_mutually_exclusive_group()

    group.add_argument("--epochs", type=float, help="Run each mode until this many epochs are processed.")
    group.add_argument("--seconds", type=float, help="Run each mode for this many wall-clock seconds.")
    parser.add_argument("--no-plots", action="store_true", help="Skip CSV/plot generation after the experiment sweep.")
    return parser.parse_args()

def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_config(cfg):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")

def format_value(value):
    if isinstance(value, int):
        return str(value)
    if float(value).is_integer():
        return str(int(value))
    return str(value).replace(".", "p")

def apply_stop_override(cfg, args):

    if args.epochs is not None:
        cfg["stop_mode"] = "EPOCHS"
        cfg["target_epochs"] = args.epochs
        cfg["target_time_seconds"] = 0.0
        return f"epochs_{format_value(args.epochs)}"
    if args.seconds is not None:
        cfg["stop_mode"] = "TIME"
        cfg["target_epochs"] = 0.0
        cfg["target_time_seconds"] = args.seconds
        return f"time_{format_value(args.seconds)}s"

    stop_mode = cfg.get("stop_mode", "EPOCHS")
    if stop_mode == "TIME":
        return f"time_{format_value(cfg.get('target_time_seconds', 0.0))}s"
    return f"epochs_{format_value(cfg.get('target_epochs', 0.0))}"

def parse_log_metrics(log_path):
    metrics = []
    with open(log_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for idx, line in enumerate(lines):
        clean_line = strip_ansi(line)
        monitor_match = MONITOR_PATTERN.search(clean_line)
        if not monitor_match:
            continue

        epoch = float(monitor_match.group(1))
        
        # Loss is on the same line as Epoch
        loss_match = LOSS_PATTERN.search(clean_line)
        loss = float(loss_match.group(1)) if loss_match else None
        
        elapsed = None
        speed = None

        # Speed and Elapsed are on the next line
        if idx + 1 < len(lines):
            next_line = strip_ansi(lines[idx + 1])
            speed_match = SPEED_PATTERN.search(next_line)
            if speed_match:
                speed = float(speed_match.group(1))
            
            elapsed_match = ELAPSED_PATTERN.search(next_line)
            if elapsed_match:
                elapsed = duration_to_seconds(elapsed_match.group(1))

        if epoch is not None:
            metrics.append({
                "epoch": epoch,
                "elapsed_seconds": elapsed,
                "avg_loss": loss,
                "speed_images_per_sec": speed,
            })
    return metrics

def write_metrics_csv(logs_dir, series_by_name):
    csv_path = os.path.join(logs_dir, "metrics_summary.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["experiment", "point_index", "epoch", "elapsed_seconds", "avg_loss", "speed_images_per_sec"])
        for exp_name, metrics in series_by_name.items():
            for idx, point in enumerate(metrics):
                writer.writerow([exp_name, idx, point["epoch"], point["elapsed_seconds"], point["avg_loss"], point["speed_images_per_sec"]])
    return csv_path

def plot_metric(logs_dir, run_label, series_by_name, x_key, y_key, title, filename, xlabel, ylabel):
    if not HAVE_MATPLOTLIB:
        return None

    plt.figure(figsize=(10, 6))
    plotted_any = False

    for exp_name, metrics in series_by_name.items():
        xs = [point[x_key] for point in metrics if point.get(x_key) is not None and point.get(y_key) is not None]
        ys = [point[y_key] for point in metrics if point.get(x_key) is not None and point.get(y_key) is not None]
        if not xs or not ys:
            continue
            
        # Target ~100 points per line for clarity
        xs, ys = downsample_points(xs, ys, max_points=100)
        
        plt.plot(xs, ys, marker="o", linewidth=2, markersize=4, label=exp_name)
        plotted_any = True

    if not plotted_any:
        plt.close()
        return None

    plt.title(f"{title} ({run_label})")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()

    plot_path = os.path.join(logs_dir, filename)
    plt.savefig(plot_path, dpi=180)
    plt.close()
    return plot_path

def generate_reports(logs_dir, run_label):
    series_by_name = {}
    for exp in EXPERIMENTS:
        log_path = os.path.join(logs_dir, f"{exp['name']}.log")
        if os.path.exists(log_path):
            series_by_name[exp["name"]] = parse_log_metrics(log_path)

    if not any(series_by_name.values()):
        print("--> No monitor metrics found, skipping CSV/plot generation.")
        return

    csv_path = write_metrics_csv(logs_dir, series_by_name)
    print(f"--> Saved metrics CSV to {os.path.relpath(csv_path, PROJECT_ROOT)}")

    if not HAVE_MATPLOTLIB:
        print("--> matplotlib is not installed, skipping PNG plot generation.")
        return

    plot_specs = [
        ("elapsed_seconds", "avg_loss", "Average Loss vs Time", "loss_vs_time.png", "Elapsed Time (s)", "Average Loss"),
        ("elapsed_seconds", "speed_images_per_sec", "Throughput vs Time", "speed_vs_time.png", "Elapsed Time (s)", "Images / sec"),
        ("epoch", "avg_loss", "Average Loss vs Epoch", "loss_vs_epoch.png", "Epoch", "Average Loss"),
        ("epoch", "speed_images_per_sec", "Throughput vs Epoch", "speed_vs_epoch.png", "Epoch", "Images / sec"),
        ("elapsed_seconds", "epoch", "Epoch Progress vs Time", "epoch_vs_time.png", "Elapsed Time (s)", "Epoch"),
    ]

    for x_key, y_key, title, filename, xlabel, ylabel in plot_specs:
        plot_path = plot_metric(logs_dir, run_label, series_by_name, x_key, y_key, title, filename, xlabel, ylabel)
        if plot_path is not None:
            print(f"--> Saved plot to {os.path.relpath(plot_path, PROJECT_ROOT)}")

def main():
    args = parse_args()
    original_config = load_config()
    run_config = dict(original_config)
    
    run_label = apply_stop_override(run_config, args)
    logs_dir = os.path.join(LOGS_DIR, run_label)
    os.makedirs(logs_dir, exist_ok=True)
    save_config(run_config)

    try:
        for exp in EXPERIMENTS:
            print("===================================================")
            print(f"Starting Experiment: {exp['name']}")
            print(f"  Stop Criterion : {run_label}")
            print(f"  Mode           : {exp['exec_mode']} / {exp['opt_mode']}")
            print(f"  Batch Size     : {exp['batch_size']}  |  Threads: {exp['num_threads']}")
            print(f"  Lambda         : {exp['lambda']}  |  Interval: {exp['interval_size']}")
            print("===================================================")

            cfg = load_config()

            # Shared hyperparameters (uniform across all experiments)
            cfg["eta"] = 0.005
            cfg["momentum"] = 0.5
            cfg["use_probing"] = exp.get("use_probing", False)
            cfg["use_thread_probing"] = exp.get("use_thread_probing", False)

            # Per-experiment overrides
            cfg["exec_mode"] = exp["exec_mode"]
            cfg["opt_mode"] = exp["opt_mode"]
            cfg["batch_size"] = exp["batch_size"]
            cfg["num_threads"] = exp["num_threads"]
            cfg["lambda"] = exp["lambda"]
            cfg["interval_size"] = exp["interval_size"]

            # Prober tuning (optional overrides)
            if "probe_test_steps" in exp:
                cfg["probe_test_steps"] = exp["probe_test_steps"]
            if "probe_exec_steps" in exp:
                cfg["probe_exec_steps"] = exp["probe_exec_steps"]

            # DC-ASGD-a adaptive lambda parameters (only present for adaptive experiments)
            if "dc_asgd_rms_momentum" in exp:
                cfg["dc_asgd_rms_momentum"] = exp["dc_asgd_rms_momentum"]
            if "dc_asgd_rms_epsilon" in exp:
                cfg["dc_asgd_rms_epsilon"] = exp["dc_asgd_rms_epsilon"]

            save_config(cfg)

            log_file = os.path.join(logs_dir, f"{exp['name']}.log")
            os.chdir(BUILD_DIR)

            try:
                with open(log_file, "w", encoding="utf-8") as out:
                    subprocess.run([EXECUTABLE], stdout=out, stderr=subprocess.STDOUT, check=True)
                print(f"--> Saved output to {os.path.relpath(log_file, PROJECT_ROOT)}")
            except subprocess.CalledProcessError:
                print(f"--> ERROR: Benchmark failed for {exp['name']}. Check {log_file} for details.")
            finally:
                os.chdir(SCRIPT_DIR)

        if not args.no_plots:
            generate_reports(logs_dir, run_label)

    finally:
        print("\nAll experiments completed. Restoring original configuration.")
        save_config(original_config)

if __name__ == "__main__":
    main()