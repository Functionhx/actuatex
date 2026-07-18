"""PPO settings for PhysX-5-native TinyMal fine-tuning."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlSymmetryCfg,
)

from .rsl_rl_ppo_cfg import TinymalFlatPPORunnerCfg
from ..symmetry import compute_symmetric_states


@configclass
class TinymalNativePPORunnerCfg(TinymalFlatPPORunnerCfg):
    max_iterations = 600
    save_interval = 25
    experiment_name = "tinymal_native_isaaclab"
    run_name = "physx5_forward"

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.15,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.003,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class TinymalRobustPPORunnerCfg(TinymalNativePPORunnerCfg):
    """PPO with privileged critic and left-right symmetry regularization."""

    obs_groups = {"actor": ["policy"], "critic": ["policy", "privileged"]}
    max_iterations = 500
    run_name = "physx5_robust"
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.002,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=RslRlSymmetryCfg(
            # The stair specialist is initially quite asymmetric.  Directly
            # augmenting PPO samples reuses the original action log-prob for a
            # mirrored action and produced a persistent ~0.10 surrogate loss.
            # Mirror consistency regularizes the mean policy progressively
            # without contaminating PPO's on-policy ratio.
            use_data_augmentation=False,
            use_mirror_loss=True,
            data_augmentation_func=compute_symmetric_states,
            mirror_loss_coeff=0.05,
        ),
    )
