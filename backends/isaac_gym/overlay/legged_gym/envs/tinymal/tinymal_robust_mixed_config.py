"""Configuration for simultaneous robust-flat and stair rehearsal."""

from legged_gym.envs.tinymal.tinymal_robust_stairs_config import (
    TinyMalRobustStairsCfg,
    TinyMalRobustStairsCfgPPO,
)


class TinyMalRobustMixedCfg(TinyMalRobustStairsCfg):
    class commands(TinyMalRobustStairsCfg.commands):
        curriculum = False
        heading_command = False

        class ranges(TinyMalRobustStairsCfg.commands.ranges):
            lin_vel_x = [-0.6, 0.6]
            lin_vel_y = [-0.3, 0.3]
            ang_vel_yaw = [-0.8, 0.8]
            heading = [0.0, 0.0]

    class stairs(TinyMalRobustStairsCfg.stairs):
        flat_env_fraction = 0.5
        step_height = 0.02
        command_speed = 0.3
        # Optional command-channel marker presented to the policy on stair
        # cells.  Rewards and heading control continue to use command_speed.
        policy_command_speed = 0.3
        heading_gain = 0.5
        curriculum = False


class TinyMalRobustMixedCfgPPO(TinyMalRobustStairsCfgPPO):
    class algorithm(TinyMalRobustStairsCfgPPO.algorithm):
        learning_rate = 5.0e-5
        schedule = "fixed"

    class runner(TinyMalRobustStairsCfgPPO.runner):
        experiment_name = "tinymal_sim2real_mixed"
        run_name = "flat_stairs_rehearsal"
        max_iterations = 100
        save_interval = 25
