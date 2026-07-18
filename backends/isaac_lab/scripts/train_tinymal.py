#!/usr/bin/env python
"""Train the TinyMal Isaac Lab task with RSL-RL.

Run via Isaac Sim's Python launcher:
    "$ISAAC_SIM_PYTHON" train_tinymal.py \
        --task Isaac-Velocity-Flat-TinyMal-v0 --num_envs 4096 --max_iterations 1500 \
        --headless --seed 1
"""
import argparse
import os
import sys
from pathlib import Path

# Make the sibling tinymal_lab package importable.
BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-TinyMal-v0")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--experiment_name", type=str, default=None)
parser.add_argument("--run_name", type=str, default=None)
parser.add_argument("--save_interval", type=int, default=None)
parser.add_argument("--learning_rate", type=float, default=None)
parser.add_argument("--schedule", choices=("fixed", "adaptive"), default=None)
parser.add_argument("--init_noise_std", type=float, default=None)
parser.add_argument(
    "--init_checkpoint",
    type=str,
    default=None,
    help="warm-start actor MLP only from an old Gym or new Lab checkpoint",
)
parser.add_argument("--resume", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.headless = True

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import logging  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from datetime import datetime  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import isaaclab_tasks  # noqa: E402,F401  (registers built-in tasks)
import tinymal_lab  # noqa: E402,F401     (registers Isaac-Velocity-Flat-TinyMal-v0)
from isaaclab.envs import ManagerBasedRLEnvCfg  # noqa: E402
from isaaclab.utils.io import dump_yaml  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402
import importlib.metadata as _metadata  # noqa: E402

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("tinymal_train")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

LOG_ROOT = os.path.join(
    os.environ.get("ACTUATEX_ARTIFACTS", str(REPO_ROOT / "artifacts")),
    "isaac_lab",
    "logs",
    "rsl_rl",
)


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg):
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    if args_cli.experiment_name is not None:
        agent_cfg.experiment_name = args_cli.experiment_name
    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name
    if args_cli.save_interval is not None:
        agent_cfg.save_interval = args_cli.save_interval
    if args_cli.learning_rate is not None:
        agent_cfg.algorithm.learning_rate = args_cli.learning_rate
    if args_cli.schedule is not None:
        agent_cfg.algorithm.schedule = args_cli.schedule
    if args_cli.init_noise_std is not None:
        agent_cfg.policy.init_noise_std = args_cli.init_noise_std
    agent_cfg.seed = args_cli.seed
    env_cfg.seed = args_cli.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    # Migrate the Isaac Lab actor/critic cfg into the rsl-rl-lib 5.x model schema
    # (adds actor/critic class_name + RslRlMLPModelCfg that OnPolicyRunner expects).
    _rsl_ver = _metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, _rsl_ver)

    log_root_path = os.path.abspath(os.path.join(LOG_ROOT, agent_cfg.experiment_name))
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    env_cfg.log_dir = log_dir
    print(f"[INFO] log_dir = {log_dir}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

    if args_cli.init_checkpoint is not None:
        init_path = os.path.abspath(args_cli.init_checkpoint)
        payload = torch.load(init_path, weights_only=False, map_location="cpu")
        if "model_state_dict" in payload:
            actor_state = {
                "mlp." + key[len("actor.") :]: value
                for key, value in payload["model_state_dict"].items()
                if key.startswith("actor.")
            }
        elif "actor_state_dict" in payload:
            actor_state = {
                key: value
                for key, value in payload["actor_state_dict"].items()
                if key.startswith("mlp.")
            }
        else:
            raise KeyError(
                f"{init_path} has neither model_state_dict nor actor_state_dict"
            )
        incompatible = runner.alg.actor.load_state_dict(actor_state, strict=False)
        unexpected = list(incompatible.unexpected_keys)
        missing = [
            key for key in incompatible.missing_keys
            if not key.startswith("distribution.")
        ]
        if unexpected or missing:
            raise RuntimeError(
                f"actor warm-start mismatch: missing={missing}, unexpected={unexpected}"
            )
        target_std = (
            args_cli.init_noise_std
            if args_cli.init_noise_std is not None
            else float(agent_cfg.actor.distribution_cfg.init_std)
        )
        distribution = runner.alg.actor.distribution
        if hasattr(distribution, "std_param"):
            distribution.std_param.data.fill_(target_std)
        elif hasattr(distribution, "log_std_param"):
            distribution.log_std_param.data.fill_(math.log(target_std))
        print(
            f"[INFO] actor warm-started from {init_path}; exploration std={target_std}"
        )

    if args_cli.resume:
        from isaaclab_tasks.utils import get_checkpoint_path
        resume_path = get_checkpoint_path(log_root_path)
        print(f"[INFO] resume from {resume_path}")
        runner.load(resume_path)

    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    t0 = time.time()
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    print(f"Training time: {round(time.time() - t0, 2)} s")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
