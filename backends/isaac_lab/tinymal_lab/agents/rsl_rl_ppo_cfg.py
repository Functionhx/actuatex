# PPO config for the TinyMal Isaac Lab port.
# Mirrors legged_gym LeggedRobotCfgPPO defaults + tinymal_config overrides:
#   init_noise_std=0.3, entropy_coef=0.005, actor [512,256,128] elu, lr 1e-3 adaptive,
#   4096 envs, seed 1, 1500 iters. Actor arch matches the old model_1500.pt keys
#   (actor.{0,2,4,6}) so the checkpoint loads cleanly for sim2sim.

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class TinymalFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # RSL-RL 5 no longer infers observation-set mappings.  The baseline actor
    # and critic intentionally receive the same policy observations; the
    # robust runner below overrides the critic mapping with privileged terms.
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    num_steps_per_env = 24            # legged_gym runner.num_steps_per_env
    max_iterations = 1500             # tinymal runner.max_iterations
    save_interval = 50                # tinymal runner.save_interval
    experiment_name = "tinymal_flat_isaaclab"
    run_name = "port_seed1"
    clip_actions = 100.0             # legged_gym normalization.clip_actions

    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=0.3,
        ),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,           # tinymal override (base default is 0.01)
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
