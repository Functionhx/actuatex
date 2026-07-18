"""Compare PD / action-scale ablation variants.

Reads the baseline + variant evaluation summaries and TensorBoard reward curves,
emits evaluation/pd_ablation/{comparison_table.csv, reward_curves.png,
metric_bars.png}. Graceacefully skips variants whose data is missing.
"""

import csv
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from legged_gym import LEGGED_GYM_ROOT_DIR

ROOT = LEGGED_GYM_ROOT_DIR
# (name, logroot, summary path, run-name suffix for TB, label)
VARIANTS = [
    ("baseline", "logs/tinymal_baseline",
     "evaluation/tinymal/summary.json", "std0p3_seed1", "Kp20 Kd0.5 as0.25"),
    ("stiff", "logs/tinymal_pd_ablation",
     "evaluation/pd_ablation/kp40_kd1_as0p25/summary.json", "kp40_kd1_as0p25", "Kp40 Kd1 as0.25"),
    ("soft_scale", "logs/tinymal_pd_ablation",
     "evaluation/pd_ablation/kp20_kd0p5_as0p5/summary.json", "kp20_kd0p5_as0p5", "Kp20 Kd0.5 as0.5"),
]
OUT = os.path.join(ROOT, "evaluation", "pd_ablation")


def find_run(logroot, suffix):
    hits = glob.glob(os.path.join(ROOT, logroot, "*" + suffix))
    return sorted(hits)[-1] if hits else None


def load_scalar(run_dir, tag):
    if not run_dir:
        return None
    ea = EventAccumulator(run_dir, size_guidance={"scalars": 10000})
    ea.Reload()
    if tag in ea.Tags().get("scalars", []):
        ev = ea.Scalars(tag)
        return np.array([e.step for e in ev]), np.array([e.value for e in ev])
    return None


def main():
    os.makedirs(OUT, exist_ok=True)
    summaries = {}
    for name, _, spath, _, _ in VARIANTS:
        p = os.path.join(ROOT, spath)
        summaries[name] = json.load(open(p)) if os.path.exists(p) else None

    # --- reward / episode-length curves ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, logroot, _, suffix, label in VARIANTS:
        run = find_run(logroot, suffix)
        for ax, tag, ylabel in [
            (axes[0], "Train/mean_reward", "mean reward"),
            (axes[1], "Train/mean_episode_length", "mean episode length"),
        ]:
            series = load_scalar(run, tag)
            if series is not None:
                ax.plot(series[0], series[1], label=name)
    for ax, t in zip(axes, ["Training reward", "Episode length"]):
        ax.set_title(t); ax.set_xlabel("iteration"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "reward_curves.png"), dpi=130)
    plt.close(fig)

    # --- comparison table + metric bars (forward_0p3 segment) ---
    segs = ["stand", "forward_0p3", "forward_0p6", "backward_0p3", "lateral_0p2", "yaw_0p5"]
    metrics = ["vx_rmse", "torque_rms", "action_rms", "abs_pitch_mean"]
    present = [n for n, *_ in VARIANTS if summaries.get(n)]
    with open(os.path.join(OUT, "comparison_table.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant"] + [f"{seg}/{m}" for seg in segs for m in metrics])
        for name in present:
            row = [name]
            for seg in segs:
                sd = summaries[name].get(seg, {})
                for m in metrics:
                    row.append(round(sd.get(m, float("nan")), 4) if sd else "")
            w.writerow(row)

    if present:
        fig, axes = plt.subplots(1, len(metrics), figsize=(14, 3.5))
        x = np.arange(len(present))
        for ax, m in zip(axes, metrics):
            vals = [summaries[n].get("forward_0p3", {}).get(m, 0) for n in present]
            ax.bar(x, vals)
            ax.set_xticks(x); ax.set_xticklabels(present, rotation=30, fontsize=8)
            ax.set_title(f"forward_0p3\n{m}"); ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "metric_bars.png"), dpi=130)
        plt.close(fig)
    print(f"comparison_table={os.path.join(OUT, 'comparison_table.csv')}")
    print(f"reward_curves={os.path.join(OUT, 'reward_curves.png')}")
    print(f"metric_bars={os.path.join(OUT, 'metric_bars.png')}")


if __name__ == "__main__":
    main()
