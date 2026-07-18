"""Long inference rollout used to isolate Isaac Gym GPU-physics failures.

Environment variables:

``TINYMAL_STRESS_CHECKPOINT``
    Actor checkpoint.  If unset, zero actions are used.
``TINYMAL_STRESS_STEPS``
    Policy steps to execute (default 1000, i.e. 20 simulated seconds).
``TINYMAL_STRESS_TERRAIN``
    ``heightfield``, ``trimesh``, or ``plane``.
``TINYMAL_STRESS_PUSH``
    Set to ``0`` to disable sustained force pushes.
"""

import os

import isaacgym  # noqa: F401  # Must precede torch.
import torch

from legged_gym.envs import *  # noqa: F401,F403  # Registers tasks.
from legged_gym.utils import get_args, task_registry


def stress(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name="tinymal_robust")
    env_cfg.env.num_envs = args.num_envs or 4096
    env_cfg.terrain.mesh_type = os.environ.get(
        "TINYMAL_STRESS_TERRAIN", env_cfg.terrain.mesh_type
    )
    env_cfg.domain_rand.randomize_push_force = (
        os.environ.get("TINYMAL_STRESS_PUSH", "1") == "1"
    )
    env, _ = task_registry.make_env(
        name="tinymal_robust", args=args, env_cfg=env_cfg
    )

    checkpoint = os.environ.get("TINYMAL_STRESS_CHECKPOINT")
    if checkpoint:
        runner, _ = task_registry.make_alg_runner(
            env=env,
            name="tinymal_robust",
            args=args,
            train_cfg=train_cfg,
            log_root=None,
        )
        runner.load(os.path.abspath(checkpoint), load_optimizer=False)
        policy = runner.get_inference_policy(device=env.device)
    else:
        policy = None
        env.reset()

    # Match OnPolicyRunner's training initialization so pushes and timeouts are
    # spread across environments rather than synchronized into one burst.
    env.episode_length_buf = torch.randint_like(
        env.episode_length_buf, high=int(env.max_episode_length)
    )
    steps = int(os.environ.get("TINYMAL_STRESS_STEPS", "1000"))
    resets = 0
    with torch.inference_mode():
        for step in range(steps):
            actions = (
                policy(env.get_observations())
                if policy is not None
                else torch.zeros(
                    env.num_envs, env.num_actions, device=env.device
                )
            )
            obs, _, rewards, dones, _ = env.step(actions)
            resets += int(dones.sum().item())
            if not bool(
                torch.isfinite(obs).all()
                and torch.isfinite(rewards).all()
                and torch.isfinite(env.root_states).all()
                and torch.isfinite(env.dof_state).all()
            ):
                raise AssertionError(f"non-finite physics tensor at step {step}")
            if step % 100 == 0 or step + 1 == steps:
                print(
                    f"stress step={step + 1}/{steps} resets={resets} "
                    f"z_mean={env.base_pos[:, 2].mean().item():.4f} "
                    f"max_abs_torque={env.torques.abs().max().item():.3f}",
                    flush=True,
                )
    print(
        f"PASS terrain={env_cfg.terrain.mesh_type} "
        f"push={env_cfg.domain_rand.randomize_push_force} "
        f"envs={env.num_envs} steps={steps} resets={resets}"
    )
    env.gym.destroy_sim(env.sim)


if __name__ == "__main__":
    stress(get_args())
