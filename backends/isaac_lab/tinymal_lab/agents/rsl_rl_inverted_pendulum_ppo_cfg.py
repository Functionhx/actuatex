"""Shared PPO architecture for the 1 -> 2 -> 3 inverted-pendulum curriculum."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class InvertedPendulumPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    num_steps_per_env = 32
    max_iterations = 480
    save_interval = 40
    experiment_name = "inverted_pendulum_isaacsim6"
    run_name = "shared14d_seed1"
    clip_actions = 1.0

    actor = RslRlMLPModelCfg(
        hidden_dims=[128, 128, 64],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.20),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[128, 128, 64],
        activation="elu",
        obs_normalization=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
