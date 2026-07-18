"""Train or safely resume the final TinyMal domain-randomized policy.

The bundled rsl_rl v1.0.2 stores ``current_learning_iteration`` in checkpoints,
but only updates that field after a complete ``learn`` call.  Consequently an
intermediate ``model_250.pt`` says ``iter=0`` internally.  This launcher parses
the authoritative iteration from the filename so a recovered run still ends at
``model_1500.pt`` instead of silently performing 1500 additional updates.

Set ``TINYMAL_RESUME_CHECKPOINT`` to resume after a simulator/driver failure.
"""

import os
import re

import isaacgym  # noqa: F401  # Must precede torch.

from legged_gym.envs import *  # noqa: F401,F403  # Registers tasks.
from legged_gym.utils import get_args, task_registry


def _iteration_from_checkpoint(path):
    match = re.search(r"model_(\d+)\.pt$", os.path.basename(path))
    if match is None:
        raise ValueError(
            "Cannot infer completed iterations from checkpoint filename: " + path
        )
    return int(match.group(1))


def train(args):
    if args.task != "tinymal_robust":
        raise ValueError("train_tinymal_robust.py requires --task=tinymal_robust")

    env, _ = task_registry.make_env(name=args.task, args=args)
    runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args
    )

    completed_iterations = 0
    checkpoint = os.environ.get("TINYMAL_RESUME_CHECKPOINT")
    if checkpoint:
        checkpoint = os.path.abspath(checkpoint)
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(checkpoint)
        completed_iterations = _iteration_from_checkpoint(checkpoint)
        print(
            f"Resuming {checkpoint} at iteration {completed_iterations}; "
            "loading model and optimizer state"
        )
        runner.load(checkpoint, load_optimizer=True)
        # Correct rsl_rl v1.0.2's stale intermediate-checkpoint metadata.
        runner.current_learning_iteration = completed_iterations

    target_iterations = int(train_cfg.runner.max_iterations)
    remaining_iterations = target_iterations - completed_iterations
    if remaining_iterations <= 0:
        raise ValueError(
            f"checkpoint iteration {completed_iterations} already reaches target "
            f"{target_iterations}"
        )
    print(
        f"Training iteration {completed_iterations} -> {target_iterations} "
        f"({remaining_iterations} updates remaining)"
    )
    runner.learn(
        num_learning_iterations=remaining_iterations,
        init_at_random_ep_len=True,
    )
    env.gym.destroy_sim(env.sim)


if __name__ == "__main__":
    train(get_args())
