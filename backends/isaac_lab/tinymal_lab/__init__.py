"""TinyMal Isaac Lab tasks for native PhysX-5 locomotion training.

Importing this package registers:
    Isaac-Velocity-Flat-TinyMal-v0
    Isaac-Velocity-Flat-TinyMal-Play-v0
    Isaac-Velocity-Native-Forward-TinyMal-v0
    Isaac-Velocity-Native-Omni-TinyMal-v0
    Isaac-Velocity-Native-Stairs-TinyMal-v0
    Isaac-Velocity-Native-Robust-TinyMal-v0
    Isaac-Velocity-Native-Robust-Stairs-TinyMal-v0
"""

import gymnasium as gym

from . import agents as agents  # noqa: F401  (ensures agents module path resolves)


def _register():
    gym.register(
        id="Isaac-Velocity-Flat-TinyMal-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.tinymal_flat_env_cfg:TinymalFlatEnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TinymalFlatPPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-Flat-TinyMal-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.tinymal_flat_env_cfg:TinymalFlatEnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TinymalFlatPPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-Native-Forward-TinyMal-v0",
        entry_point=f"{__name__}.native_env:TinymalNativeRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": (
                f"{__name__}.tinymal_native_env_cfg:TinymalNativeForwardEnvCfg"
            ),
            "rsl_rl_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_native_ppo_cfg:TinymalNativePPORunnerCfg"
            ),
        },
    )
    gym.register(
        id="Isaac-Velocity-Native-Omni-TinyMal-v0",
        entry_point=f"{__name__}.native_env:TinymalNativeRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": (
                f"{__name__}.tinymal_native_env_cfg:TinymalNativeOmniEnvCfg"
            ),
            "rsl_rl_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_native_ppo_cfg:TinymalNativePPORunnerCfg"
            ),
        },
    )
    gym.register(
        id="Isaac-Velocity-Native-Stairs-TinyMal-v0",
        entry_point=f"{__name__}.native_env:TinymalNativeRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": (
                f"{__name__}.tinymal_stair_env_cfg:TinymalStairEnvCfg"
            ),
            "rsl_rl_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_native_ppo_cfg:TinymalNativePPORunnerCfg"
            ),
        },
    )
    gym.register(
        id="Isaac-Velocity-Native-Robust-TinyMal-v0",
        entry_point=f"{__name__}.native_env:TinymalNativeRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": (
                f"{__name__}.tinymal_robust_env_cfg:TinymalRobustEnvCfg"
            ),
            "rsl_rl_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_native_ppo_cfg:TinymalRobustPPORunnerCfg"
            ),
        },
    )
    gym.register(
        id="Isaac-Velocity-Native-Robust-Stairs-TinyMal-v0",
        entry_point=f"{__name__}.native_env:TinymalNativeRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": (
                f"{__name__}.tinymal_robust_env_cfg:TinymalRobustStairEnvCfg"
            ),
            "rsl_rl_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_native_ppo_cfg:TinymalRobustPPORunnerCfg"
            ),
        },
    )


_register()
