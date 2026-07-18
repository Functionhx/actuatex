"""Stair fine-tuning config for the final domain-randomized TinyMal actor."""

from legged_gym.envs.tinymal.tinymal_robust_config import TinyMalRobustCfg
from legged_gym.envs.tinymal.tinymal_stairs_config import (
    TinyMalStairsCfg,
    TinyMalStairsCfgPPO,
)


class TinyMalRobustStairsCfg(TinyMalStairsCfg):
    # Keep TinyMalStairs' commands, geometry, and task rewards, but restore the
    # full physics distribution that the final flat actor was trained against.
    class domain_rand(TinyMalRobustCfg.domain_rand):
        pass

    class sim(TinyMalRobustCfg.sim):
        pass


class TinyMalRobustStairsCfgPPO(TinyMalStairsCfgPPO):
    class algorithm(TinyMalStairsCfgPPO.algorithm):
        learning_rate = 1.0e-4
        schedule = "fixed"

    class runner(TinyMalStairsCfgPPO.runner):
        experiment_name = "tinymal_sim2real_stairs"
        run_name = "robust_stair_transfer"
        max_iterations = 100
        save_interval = 25
