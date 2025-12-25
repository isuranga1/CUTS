import os
import subprocess
from multiprocessing import Pool
from datetime import datetime

SEEDS = [1, 2, 3]
METHODS = ["brain_tumor"]

SCRIPT_DIR = "src/scripts_analysis"
CONFIG_DIR = "../../config"
LOG_DIR = "../../logs"

MAX_PROCESSES = 3


def timestamp():
    return datetime.now().strftime("%H:%M:%S")


def run_command(cmd, log_file, prefix):
    print(f"[{timestamp()}] {prefix} START", flush=True)

    process = subprocess.Popen(
        ["python", "-u"] + cmd,   # 🔥 unbuffered python
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=SCRIPT_DIR,
        text=True,
        bufsize=1
    )

    for line in process.stdout:
        log_file.write(line)
        log_file.flush()
        print(f"[{prefix}] {line}", end="")

    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"{prefix} FAILED")

    print(f"[{timestamp()}] {prefix} DONE", flush=True)


def run_experiment(task):
    method, seed = task

    os.makedirs(LOG_DIR, exist_ok=True)
    config_file = f"{CONFIG_DIR}/{method}_seed{seed}.yaml"
    log_path = f"{LOG_DIR}/{method}_seed{seed}.log"

    prefix = f"{method}|seed{seed}"

    print(f"[{timestamp()}] 🚀 STARTING {prefix}", flush=True)

    with open(log_path, "w", buffering=1) as log_file:  # line-buffered
        log_file.write(
            f"Method: {method}\nSeed: {seed}\nConfig: {config_file}\n\n"
        )

        run_command(
            ["generate_kmeans.py", "--config", config_file],
            log_file,
            f"{prefix}|kmeans"
        )

        run_command(
            ["generate_diffusion.py", "--config", config_file],
            log_file,
            f"{prefix}|diffusion"
        )

        run_command(
            ["generate_baselines.py", "--config", config_file],
            log_file,
            f"{prefix}|baselines"
        )

        run_command(
            ["run_metrics.py", "--config", config_file],
            log_file,
            f"{prefix}|metrics"
        )

    print(f"[{timestamp()}] ✅ FINISHED {prefix}", flush=True)


def main():
    tasks = [(m, s) for m in METHODS for s in SEEDS]

    with Pool(processes=MAX_PROCESSES) as pool:
        pool.map(run_experiment, tasks)


if __name__ == "__main__":
    main()
