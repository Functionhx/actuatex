"""Train TinyMal walking policy in MuJoCo using rsl_rl PPO.

Uses the same architecture/hparams as the Isaac Gym baseline for a controlled
sim2sim comparison: ActorCritic MLP 48->512->256->128->12, ELU, init_noise_std=0.3,
entropy_coef=0.005, lr=1e-3 adaptive, seed=1.

Run: python train_mujoco.py [--num_envs 64] [--max_iters 1500]
"""

import os
import sys
import time
import argparse
import copy
import shutil

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from actuatex_paths import ARTIFACTS_ROOT, ROBOT_URDF, RSL_RL_ROOT

if RSL_RL_ROOT.is_dir():
    sys.path.insert(0, str(RSL_RL_ROOT))

from mujoco_vec_env import MjTinyMalEnv
from rsl_rl.runners import OnPolicyRunner

URDF = str(ROBOT_URDF)
OUT_DIR = str(ARTIFACTS_ROOT / "mujoco")
CANONICAL_MODEL = str(ARTIFACTS_ROOT / "checkpoints" / "mujoco" / "model.pt")


def get_train_cfg():
    return {
        "seed": 1,
        "runner_class_name": "OnPolicyRunner",
        "policy": {
            "init_noise_std": 0.3,
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "activation": "elu",
        },
        "algorithm": {
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "entropy_coef": 0.005,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "learning_rate": 3e-4,
            # Fixed LR: the adaptive schedule gets stuck at 1e-5 with 1024 envs
            # (noisy gradients → high KL → min LR → no learning). The Isaac Gym
            # baseline's 4096-env batches have 4x lower variance, letting the
            # adaptive LR recover. Fixed lr=1e-3 bypasses this.
            "schedule": "fixed",
            "gamma": 0.99,
            "lam": 0.95,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
        },
        "runner": {
            "policy_class_name": "ActorCritic",
            "algorithm_class_name": "PPO",
            "num_steps_per_env": 24,
            "max_iterations": 1500,
            "save_interval": 50,
            "experiment_name": "mujoco_tinymal",
            "run_name": "mujoco_baseline",
            "resume": False,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--max_iters", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--log_dir", type=str,
                        default=os.path.join(OUT_DIR, "logs", "mujoco_tinymal"))
    parser.add_argument("--num_threads", type=int, default=8)
    parser.add_argument("--command_mode", choices=("forward", "omni"), default="omni")
    parser.add_argument("--dense_tracking", action="store_true")
    parser.add_argument("--no_obs_noise", action="store_true")
    parser.add_argument("--init_pose_noise", type=float, default=0.5)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--entropy_coef", type=float, default=0.005)
    parser.add_argument("--init_noise_std", type=float, default=0.3)
    parser.add_argument("--save_interval", type=int, default=50)
    parser.add_argument(
        "--init_checkpoint", type=str, default=None,
        help="load actor/critic weights but start a fresh MuJoCo optimizer",
    )
    parser.add_argument(
        "--reference_checkpoint", type=str, default=None,
        help="frozen actor used for policy-output distillation",
    )
    parser.add_argument("--reference_coef", type=float, default=0.0)
    parser.add_argument(
        "--freeze_actor_features", action="store_true",
        help="train only actor.6 plus critic/std to protect the established gait",
    )
    parser.add_argument(
        "--output_model", type=str, default=None,
        help="canonical output copy; defaults to mujoco_trained_model.pt",
    )
    parser.add_argument("--benchmark", action="store_true",
                        help="Run 5 iters and report timing, no save.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = MjTinyMalEnv(
        URDF,
        num_envs=args.num_envs,
        device="cpu",
        seed=args.seed,
        num_threads=args.num_threads,
        init_pose_noise=args.init_pose_noise,
        command_mode=args.command_mode,
        dense_tracking=args.dense_tracking,
        add_noise=not args.no_obs_noise,
    )
    print(f"[env] num_envs={env.num_envs} num_obs={env.num_obs} "
          f"num_actions={env.num_actions} dt={env.dt}")

    train_cfg = get_train_cfg()
    train_cfg["seed"] = args.seed
    train_cfg["runner"]["max_iterations"] = args.max_iters
    train_cfg["runner"]["save_interval"] = args.save_interval
    train_cfg["algorithm"]["learning_rate"] = args.learning_rate
    train_cfg["algorithm"]["entropy_coef"] = args.entropy_coef
    train_cfg["policy"]["init_noise_std"] = args.init_noise_std

    os.makedirs(args.log_dir, exist_ok=True)
    runner = OnPolicyRunner(env=env, train_cfg=train_cfg,
                            log_dir=args.log_dir, device="cpu")

    if args.init_checkpoint is not None:
        init_path = os.path.abspath(args.init_checkpoint)
        payload = torch.load(init_path, map_location="cpu")
        runner.alg.actor_critic.load_state_dict(payload["model_state_dict"], strict=True)
        with torch.no_grad():
            runner.alg.actor_critic.std.fill_(args.init_noise_std)
        print(
            f"[init] loaded fresh optimizer from {init_path}; "
            f"exploration std={args.init_noise_std}"
        )

    if args.freeze_actor_features:
        for name, parameter in runner.alg.actor_critic.actor.named_parameters():
            parameter.requires_grad_(name.startswith("6."))
        trainable = sum(
            parameter.numel()
            for parameter in runner.alg.actor_critic.actor.parameters()
            if parameter.requires_grad
        )
        print(f"[init] frozen actor feature extractor; trainable actor params={trainable}")

    if args.reference_coef > 0.0:
        reference_path = args.reference_checkpoint or args.init_checkpoint
        if reference_path is None:
            raise ValueError("--reference_coef requires a reference or init checkpoint")
        reference_payload = torch.load(os.path.abspath(reference_path), map_location="cpu")
        reference_actor = copy.deepcopy(runner.alg.actor_critic.actor)
        reference_actor.load_state_dict(
            {
                name[len("actor."):]: value
                for name, value in reference_payload["model_state_dict"].items()
                if name.startswith("actor.")
            },
            strict=True,
        )
        reference_actor.eval()
        for parameter in reference_actor.parameters():
            parameter.requires_grad_(False)
        runner.alg.reference_actor = reference_actor
        runner.alg.reference_loss_coef = args.reference_coef
        runner.alg.reference_mask_fn = lambda observations: torch.ones(
            observations.shape[0], dtype=torch.bool, device=observations.device
        )
        print(
            f"[init] frozen policy teacher coefficient={args.reference_coef:g} "
            f"checkpoint={os.path.abspath(reference_path)}"
        )

    if args.benchmark:
        print("[benchmark] running 5 iterations...")
        t0 = time.time()
        runner.learn(5)
        dt = time.time() - t0
        print(f"[benchmark] 5 iters in {dt:.1f}s -> {dt/5:.2f}s/iter "
              f"-> {args.max_iters*dt/5/60:.1f} min for {args.max_iters} iters")
        return

    n = args.max_iters
    print(f"[train] starting PPO for {n} iterations, {env.num_envs} envs...")
    t0 = time.time()
    runner.learn(n)
    wall = time.time() - t0
    print(f"[train] done in {wall/60:.1f} min")

    # Export the final actor weights to the canonical path.
    final_model = os.path.join(args.log_dir, f"model_{n}.pt")
    out_model = (
        os.path.abspath(args.output_model)
        if args.output_model is not None
        else CANONICAL_MODEL
    )
    os.makedirs(os.path.dirname(out_model), exist_ok=True)
    shutil.copy2(final_model, out_model)
    print(f"[train] saved actor state_dict to {out_model}")
    print(f"[train] final checkpoint: {final_model}")


if __name__ == "__main__":
    main()
