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
MONITOR_PATTERN = re.compile(r"\[Monitor\]\s+Epoch:\s*([0-9.]+)\s*\|\s*Elapsed:\s*([0-9.]+)s")
LOSS_PATTERN = re.compile(r"Avg Loss:\s*([-+0-9.eE]+)")
SPEED_PATTERN = re.compile(r"Speed:\s*([-+0-9.eE]+)\s+images/sec")

EXPERIMENTS = [
    {"name": "1_Sequential", "exec_mode": "SEQUENTIAL", "opt_mode": "STANDARD_SGD"},
    {"name": "2_Synchronous", "exec_mode": "SYNC_PARALLEL", "opt_mode": "STANDARD_SGD"},
    {"name": "3_Plain_Async", "exec_mode": "ASYNC_HOGWILD", "opt_mode": "STANDARD_SGD"},
    {"name": "4_Async_Penalty", "exec_mode": "ASYNC_HOGWILD", "opt_mode": "DC_ASGD"},
    {"name": "5_Interval_Async", "exec_mode": "INTERVAL_ASYNC", "opt_mode": "STANDARD_SGD"},
    {"name": "6_Interval_Async_Penalty", "exec_mode": "INTERVAL_ASYNC", "opt_mode": "DC_ASGD"},
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run all training-mode experiments with a shared stop criterion."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--steps",
        type=int,
        help="Run each mode for this many per-thread steps.",
    )
    group.add_argument(
        "--epochs",
        type=float,
        help="Run each mode until this many epochs are processed.",
    )
    group.add_argument(
        "--seconds",
        type=float,
        help="Run each mode for this many wall-clock seconds.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip CSV/plot generation after the experiment sweep.",
    )
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
    if args.steps is not None:
        cfg["stop_mode"] = "STEPS"
        cfg["total_steps"] = args.steps
        cfg["target_epochs"] = 0.0
        cfg["target_time_seconds"] = 0.0
        return f"steps_{format_value(args.steps)}"

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

    stop_mode = cfg.get("stop_mode", "STEPS")
    if stop_mode == "EPOCHS":
        return f"epochs_{format_value(cfg.get('target_epochs', 0.0))}"
    if stop_mode == "TIME":
        return f"time_{format_value(cfg.get('target_time_seconds', 0.0))}s"
    return f"steps_{format_value(int(cfg.get('total_steps', 0)))}"


def parse_log_metrics(log_path):
    metrics = []

    with open(log_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for idx, line in enumerate(lines):
        monitor_match = MONITOR_PATTERN.search(line)
        if not monitor_match:
            continue

        epoch = float(monitor_match.group(1))
        elapsed = float(monitor_match.group(2))

        loss = None
        speed = None

        if idx + 1 < len(lines):
            loss_match = LOSS_PATTERN.search(lines[idx + 1])
            if loss_match:
                loss = float(loss_match.group(1))

        if idx + 2 < len(lines):
            speed_match = SPEED_PATTERN.search(lines[idx + 2])
            if speed_match:
                speed = float(speed_match.group(1))

        metrics.append(
            {
                "epoch": epoch,
                "elapsed_seconds": elapsed,
                "avg_loss": loss,
                "speed_images_per_sec": speed,
            }
        )

    return metrics


def write_metrics_csv(logs_dir, series_by_name):
    csv_path = os.path.join(logs_dir, "metrics_summary.csv")

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "experiment",
                "point_index",
                "epoch",
                "elapsed_seconds",
                "avg_loss",
                "speed_images_per_sec",
            ]
        )

        for exp_name, metrics in series_by_name.items():
            for idx, point in enumerate(metrics):
                writer.writerow(
                    [
                        exp_name,
                        idx,
                        point["epoch"],
                        point["elapsed_seconds"],
                        point["avg_loss"],
                        point["speed_images_per_sec"],
                    ]
                )

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
        (
            "epoch",
            "avg_loss",
            "Average Loss vs Epoch",
            "loss_vs_epoch.png",
            "Epoch",
            "Average Loss",
        ),
        (
            "elapsed_seconds",
            "avg_loss",
            "Average Loss vs Time",
            "loss_vs_time.png",
            "Elapsed Time (s)",
            "Average Loss",
        ),
        (
            "elapsed_seconds",
            "epoch",
            "Epoch Progress vs Time",
            "epoch_vs_time.png",
            "Elapsed Time (s)",
            "Epoch",
        ),
        (
            "epoch",
            "speed_images_per_sec",
            "Throughput vs Epoch",
            "speed_vs_epoch.png",
            "Epoch",
            "Images / sec",
        ),
    ]

    for x_key, y_key, title, filename, xlabel, ylabel in plot_specs:
        plot_path = plot_metric(
            logs_dir,
            run_label,
            series_by_name,
            x_key,
            y_key,
            title,
            filename,
            xlabel,
            ylabel,
        )
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
            print(f"Stop Criterion: {run_label}")
            print("===================================================")

            cfg = load_config()
            cfg["exec_mode"] = exp["exec_mode"]
            cfg["opt_mode"] = exp["opt_mode"]
            save_config(cfg)

            log_file = os.path.join(logs_dir, f"{exp['name']}.log")

            os.chdir(BUILD_DIR)

            try:
                with open(log_file, "w", encoding="utf-8") as out:
                    subprocess.run(
                        [EXECUTABLE],
                        stdout=out,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
                print(f"--> Saved output to {os.path.relpath(log_file, PROJECT_ROOT)}")
            except subprocess.CalledProcessError:
                print(f"--> ERROR: Benchmark failed for {exp['name']}. Check {log_file} for details.")
            finally:
                os.chdir(SCRIPT_DIR)

        if not args.no_plots:
            generate_reports(logs_dir, run_label)
    finally:
        print()
        print("All experiments completed. Restoring original configuration.")
        save_config(original_config)


if __name__ == "__main__":
    main()
