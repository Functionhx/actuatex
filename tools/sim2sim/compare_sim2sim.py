"""3-way sim2sim comparison: Isaac Gym vs Isaac Sim vs MuJoCo.

Reads the per-segment evaluation summaries from each backend and emits
evaluation/sim2sim_compare/{comparison.csv, tracking_overlay.png}. Backends
whose summary is absent are skipped (so this can run incrementally).
"""

import csv
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ARTIFACTS_ROOT = os.environ.get(
    "ACTUATEX_ARTIFACTS", os.path.join(REPO_ROOT, "artifacts")
)
OUT = os.path.join(ARTIFACTS_ROOT, "sim2sim_compare")
SEGS = ["stand", "forward_0p3", "forward_0p6", "backward_0p3", "lateral_0p2", "yaw_0p5"]
EP_LEN = 30.0  # Isaac Gym eval episode length (no fall => survives full episode)

# (label, summary path, color)
BACKENDS = [
    ("Isaac Gym (PhysX)", os.path.join(ARTIFACTS_ROOT, "isaac_gym", "summary.json"), "#1f77b4"),
    ("Isaac Sim (PhysX 5)", os.path.join(ARTIFACTS_ROOT, "isaac_lab", "summary.json"), "#2ca02c"),
    ("MuJoCo", os.path.join(ARTIFACTS_ROOT, "mujoco", "summary.json"), "#d62728"),
]


def load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def seg_rmse(sd, seg):
    if not sd or seg not in sd:
        return np.nan
    return sd[seg].get("vx_rmse", np.nan)


def seg_survival(sd, seg):
    if not sd or seg not in sd:
        return np.nan
    if "survival_time_s" in sd[seg]:
        return sd[seg]["survival_time_s"]
    if sd[seg].get("fallen"):
        return 0.0
    return EP_LEN  # Isaac Gym: no survival field, never fell => full episode


def main():
    os.makedirs(OUT, exist_ok=True)
    loaded = [(lbl, load(p), c) for lbl, p, c in BACKENDS]
    present = [(lbl, sd, c) for lbl, sd, c in loaded if sd]

    # --- comparison table ---
    with open(os.path.join(OUT, "comparison.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["backend"] + [f"{s}_vx_rmse" for s in SEGS]
                    + [f"{s}_survival_s" for s in SEGS])
        for lbl, sd, _ in present:
            row = [lbl]
            row += [round(seg_rmse(sd, s), 4) for s in SEGS]
            row += [round(seg_survival(sd, s), 3) for s in SEGS]
            w.writerow(row)

    # --- survival bar chart ---
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(SEGS))
    width = 0.8 / max(1, len(present))
    for i, (lbl, sd, c) in enumerate(present):
        vals = [min(seg_survival(sd, s), EP_LEN) for s in SEGS]
        ax.bar(x + (i - len(present) / 2) * width + width / 2, vals, width, label=lbl, color=c)
    ax.set_xticks(x); ax.set_xticklabels(SEGS, rotation=20, fontsize=9)
    ax.set_ylabel("survival time (s)")
    ax.set_title("Sim2Sim survival per command segment (higher = more stable)")
    ax.set_ylim(0, EP_LEN * 1.05); ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "survival_overlay.png"), dpi=140)
    plt.close(fig)

    # --- vx_rmse bar chart (forward_0p3 headline) ---
    fig, ax = plt.subplots(figsize=(5, 4))
    labels = [lbl for lbl, _, _ in present]
    vals = [seg_rmse(sd, "forward_0p3") for _, sd, _ in present]
    ax.bar(labels, vals, color=[c for _, _, c in present])
    ax.set_ylabel("forward 0.3 m/s tracking RMSE (m/s)")
    ax.set_title("Sim2Sim forward-tracking gap")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "tracking_rmse_bars.png"), dpi=140)
    plt.close(fig)

    print(f"backends compared: {[lbl for lbl,_,_ in present]}")
    print(f"comparison.csv / figures -> {OUT}")


if __name__ == "__main__":
    main()
