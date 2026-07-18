"""Small throughput smoke benchmark for the vectorized MuJoCo environment."""

import cProfile
import io
import pstats
import time

import torch

from actuatex_paths import ROBOT_URDF
from mujoco_vec_env import MjTinyMalEnv


def main():
    env = MjTinyMalEnv(str(ROBOT_URDF), num_envs=64, seed=1)
    env.reset()
    actions = torch.zeros(64, 12)

    for _ in range(3):
        env.step(actions)

    start = time.time()
    for _ in range(10):
        _, _, rewards, resets, _ = env.step(actions)
    elapsed = time.time() - start
    print(f"10 steps (64 envs): {elapsed:.2f}s -> {elapsed / 10 * 1000:.1f} ms/step")
    print(f"reward mean: {rewards.mean():.4f}, resets: {resets.sum().item()}")

    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(5):
        env.step(actions)
    profiler.disable()
    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).sort_stats("cumulative").print_stats(12)
    print(stream.getvalue())


if __name__ == "__main__":
    main()
