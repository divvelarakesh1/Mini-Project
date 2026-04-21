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
    try:
        parts = list(map(int, duration_str.split(':')))
        if len(parts) == 2: # MM:SS
            return float(parts[0] * 60 + parts[1])
        if len(parts) == 3: # HH:MM:SS
            return float(parts[0] * 3600 + parts[1] * 60 + parts[2])
    except Exception:
        pass
    return 0.0

def downsample_points(xs, ys, max_points=150):
    n = len(xs)
    if n <= max_points:
        return xs, ys
    
    bucket_size = n / max_points
    new_xs, new_ys = [], []
    
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
# DEEP SWEEP ARCHITECTURES & GRID SEARCH
# ==============================================================================
FAMILIES = {
    "1_Sequential": [
        {"exec_mode": "SEQUENTIAL", "opt_mode": "STANDARD_SGD", "batch_size": bs, "num_threads": 1, "lambda": 0.0, "interval_size": 0}
        for bs in [16, 32, 64]
    ],
    "2_Async": [
        {"exec_mode": "ASYNC_HOGWILD", "opt_mode": "STANDARD_SGD", "batch_size": 16, "num_threads": t, "lambda": 0.0, "interval_size": 0}
        for t in [16, 32, 64, 128] 
    ],
    "3_Async_DC_ASGD": [
        {"exec_mode": "ASYNC_HOGWILD", "opt_mode": "DC_ASGD_A", "batch_size": 16, "num_threads": t, "lambda": 2.0, "interval_size": 0}
        for t in [16, 32, 64, 128]
    ],
    "4_Async_Interval_Decay": [
        {"exec_mode": "INTERVAL_ASYNC", "interval_mode": "DECAY", "opt_mode": "STANDARD_SGD", "batch_size": 16, "num_threads": t, "lambda": 0.0, "interval_size": i, "decay_steps": ds, "decay_amount": da}
        for t in [16, 32, 64] for i in [64, 128, 256] for ds in [4096] for da in [1, 4]
    ],
    "5_Async_Interval_DC_ASGD_Decay": [
        {"exec_mode": "INTERVAL_ASYNC", "interval_mode": "DECAY", "opt_mode": "DC_ASGD_A", "batch_size": 16, "num_threads": t, "lambda": 2.0, "interval_size": i, "decay_steps": ds, "decay_amount": da}
        for t in [16, 32, 64] for i in [64, 128, 256] for ds in [4096] for da in [1, 4]
    ],
    "6_Fully_Auto_Probing": [
        {"exec_mode": "INTERVAL_ASYNC", "interval_mode": "PROBING", "thread_mode": "PROBING", "opt_mode": "DC_ASGD_A", "batch_size": 16, "num_threads": 128, "lambda": 2.0, "interval_size": 256, "min_interval": 16, "probe_test_steps": pts, "probe_exec_steps": pes}
        for pts in [256, 512, 1024] for pes in [1024, 2048, 4096]
    ]
}

EXPERIMENTS = []
for family, configs in FAMILIES.items():
    for cfg in configs:
        name_parts = [family]
        if "Sequential" in family:
            name_parts.append(f"bs{cfg['batch_size']}")
        elif "2_Async" == family or "3_Async_DC_ASGD" == family:
            name_parts.append(f"t{cfg['num_threads']}")
        elif "Interval" in family and "Probing" not in family:
            name_parts.append(f"t{cfg['num_threads']}_i{cfg['interval_size']}")
            if cfg.get("interval_mode") == "DECAY":
                name_parts.append(f"ds{cfg.get('decay_steps', 0)}_da{cfg.get('decay_amount', 0)}")
        elif "Probing" in family or "Auto" in family:
            name_parts.append(f"max_t{cfg['num_threads']}_max_i{cfg['interval_size']}")
            if "probe_test_steps" in cfg:
                name_parts.append(f"pt{cfg['probe_test_steps']}_pe{cfg['probe_exec_steps']}")
            
        name = "_".join(name_parts)
        
        exp = {"name": name, "family": family}
        exp.update(cfg)
        EXPERIMENTS.append(exp)

def parse_args():
    parser = argparse.ArgumentParser(description="Run all training-mode experiments with a shared stop criterion.")
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
    if isinstance(value, int): return str(value)
    if float(value).is_integer(): return str(int(value))
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
        if not monitor_match: continue

        epoch = float(monitor_match.group(1))
        loss_match = LOSS_PATTERN.search(clean_line)
        loss = float(loss_match.group(1)) if loss_match else None
        
        elapsed, speed = None, None
        if idx + 1 < len(lines):
            next_line = strip_ansi(lines[idx + 1])
            speed_match = SPEED_PATTERN.search(next_line)
            if speed_match: speed = float(speed_match.group(1))
            
            elapsed_match = ELAPSED_PATTERN.search(next_line)
            if elapsed_match: elapsed = duration_to_seconds(elapsed_match.group(1))

        if epoch is not None:
            metrics.append({"epoch": epoch, "elapsed_seconds": elapsed, "avg_loss": loss, "speed_images_per_sec": speed})
    return metrics

def plot_metric(logs_dir, run_label, series_by_name, x_key, y_key, title, filename, xlabel, ylabel):
    if not HAVE_MATPLOTLIB: return None
    
    # Make the figure slightly wider to accommodate large legends
    plt.figure(figsize=(12, 6))
    plotted_any = False

    for exp_name, metrics in series_by_name.items():
        xs = [pt[x_key] for pt in metrics if pt.get(x_key) is not None and pt.get(y_key) is not None]
        ys = [pt[y_key] for pt in metrics if pt.get(x_key) is not None and pt.get(y_key) is not None]
        if not xs or not ys: continue
            
        xs, ys = downsample_points(xs, ys, max_points=150)
        # Use thinner lines (1.5) so dense grids don't overlap too much
        plt.plot(xs, ys, marker=".", linewidth=1.5, markersize=3, label=exp_name)
        plotted_any = True

    if not plotted_any:
        plt.close()
        return None

    plt.title(f"{title} ({run_label})")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.35)
    
    # Push legend completely outside to the right
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0., fontsize='small')
    plt.tight_layout()

    plot_path = os.path.join(logs_dir, filename)
    plt.savefig(plot_path, dpi=180, bbox_inches='tight')
    plt.close()
    return plot_path

def generate_reports(logs_dir, run_label):
    series_by_name = {}
    for exp in EXPERIMENTS:
        log_path = os.path.join(logs_dir, f"{exp['name']}.log")
        if os.path.exists(log_path):
            series_by_name[exp["name"]] = parse_log_metrics(log_path)

    if not any(series_by_name.values()):
        print("--> No monitor metrics found, skipping plots.")
        return

    # Group series by family
    family_series = {fam: {} for fam in FAMILIES.keys()}
    for exp in EXPERIMENTS:
        if exp["name"] in series_by_name:
            family_series[exp["family"]][exp["name"]] = series_by_name[exp["name"]]

    best_of_each = {}
    plot_specs = [
        ("elapsed_seconds", "avg_loss", "Average Loss vs Time", "loss_vs_time.png", "Elapsed Time (s)", "Average Loss"),
        ("elapsed_seconds", "speed_images_per_sec", "Throughput vs Time", "speed_vs_time.png", "Elapsed Time (s)", "Images / sec"),
        ("epoch", "avg_loss", "Average Loss vs Epoch", "loss_vs_epoch.png", "Epoch", "Average Loss")
    ]

    # Plot Each Architecture Independently & Find the Best Setting
    for family, f_series in family_series.items():
        if not f_series: continue
        
        # Determine the best setting (Lowest average loss at the end of the run)
        best_name = None
        best_loss = float('inf')
        
        for name, metrics in f_series.items():
            valid_losses = [pt["avg_loss"] for pt in metrics if pt.get("avg_loss") is not None]
            if valid_losses:
                # Average the last 5 logs to ignore random spikes
                tail_avg = sum(valid_losses[-5:]) / len(valid_losses[-5:])
                if tail_avg < best_loss:
                    best_loss = tail_avg
                    best_name = name
                    
        if best_name:
            best_of_each[best_name] = f_series[best_name]
            
        # Plot family variations
        if HAVE_MATPLOTLIB:
            for x_key, y_key, title, filename, xlabel, ylabel in plot_specs:
                f_name = f"{family}_{filename}"
                plot_metric(logs_dir, run_label, f_series, x_key, y_key, f"{title} [{family}]", f_name, xlabel, ylabel)

    # Plot Final Comparison of the Best of Each
    if HAVE_MATPLOTLIB and best_of_each:
        print(f"\n--> Generating Final Comparison Plots from Best Candidates:")
        for name in best_of_each.keys():
            print(f"    Selected: {name}")
            
        for x_key, y_key, title, filename, xlabel, ylabel in plot_specs:
            f_name = f"FINAL_BEST_{filename}"
            path = plot_metric(logs_dir, run_label, best_of_each, x_key, y_key, f"FINAL COMPARISON: {title}", f_name, xlabel, ylabel)
            if path: print(f"--> Saved final summary to {os.path.relpath(path, PROJECT_ROOT)}")

def main():
    args = parse_args()
    original_config = load_config()
    run_config = dict(original_config)
    
    run_label = apply_stop_override(run_config, args)
    logs_dir = os.path.join(LOGS_DIR, run_label)
    os.makedirs(logs_dir, exist_ok=True)
    save_config(run_config)

    try:
        total_runs = len(EXPERIMENTS)
        for i, exp in enumerate(EXPERIMENTS, 1):
            print("===================================================")
            print(f"Starting Experiment [{i}/{total_runs}]: {exp['name']}")
            print(f"  Mode           : {exp['exec_mode']} / {exp['opt_mode']}")
            print(f"  Batch Size     : {exp['batch_size']}  |  Threads: {exp['num_threads']}")
            print(f"  Lambda         : {exp['lambda']}  |  Interval: {exp['interval_size']}")
            if exp.get("interval_mode") == "DECAY":
                print(f"  Decay Steps    : {exp['decay_steps']}  |  Decay Amt: {exp['decay_amount']}")
            if exp.get("interval_mode") == "PROBING" or exp.get("thread_mode") == "PROBING":
                print(f"  Probe Test     : {exp.get('probe_test_steps', 'N/A')} | Probe Exec: {exp.get('probe_exec_steps', 'N/A')}")
            print("===================================================")

            cfg = load_config()

            # Shared hyperparameters - specifically tuning eta to 0.005 as recommended by the paper for CIFAR
            cfg["eta"] = 0.005
            cfg["momentum"] = 0.5
            cfg["interval_mode"] = exp.get("interval_mode", "STATIC")
            cfg["thread_mode"] = exp.get("thread_mode", "STATIC")
            cfg["min_interval"] = exp.get("min_interval", 16)
            cfg["decay_steps"] = exp.get("decay_steps", 4096)
            cfg["decay_amount"] = exp.get("decay_amount", 1)
            cfg["thread_min_count"] = exp.get("thread_min_count", 1)

            # Core experiment overrides
            cfg["exec_mode"] = exp["exec_mode"]
            cfg["opt_mode"] = exp["opt_mode"]
            cfg["batch_size"] = exp["batch_size"]
            cfg["num_threads"] = exp["num_threads"]
            cfg["lambda"] = exp["lambda"]
            cfg["interval_size"] = exp["interval_size"]

            # Optional Prober & RMSProp Overrides
            if "probe_test_steps" in exp: cfg["probe_test_steps"] = exp["probe_test_steps"]
            if "probe_exec_steps" in exp: cfg["probe_exec_steps"] = exp["probe_exec_steps"]
            if "dc_asgd_rms_momentum" in exp: cfg["dc_asgd_rms_momentum"] = exp["dc_asgd_rms_momentum"]
            if "dc_asgd_rms_epsilon" in exp: cfg["dc_asgd_rms_epsilon"] = exp["dc_asgd_rms_epsilon"]

            save_config(cfg)

            log_file = os.path.join(logs_dir, f"{exp['name']}.log")
            os.chdir(BUILD_DIR)

            try:
                with open(log_file, "w", encoding="utf-8") as out:
                    subprocess.run([EXECUTABLE], stdout=out, stderr=subprocess.STDOUT, check=True)
                print(f"--> Saved output to {os.path.relpath(log_file, PROJECT_ROOT)}")
            except subprocess.CalledProcessError:
                print(f"--> ERROR: Benchmark failed for {exp['name']}. Check log for details.")
            finally:
                os.chdir(SCRIPT_DIR)

        if not args.no_plots:
            generate_reports(logs_dir, run_label)

    finally:
        print("\nAll experiments completed. Restoring original configuration.")
        save_config(original_config)

if __name__ == "__main__":
    main()