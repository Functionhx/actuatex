from legged_gym.envs.tinymal.tinymal_config import TinyMalCfg, TinyMalCfgPPO


class TinyMalStairsCfg(TinyMalCfg):
    """Forward stair-climbing task with robot-scale geometry."""

    class env(TinyMalCfg.env):
        num_envs = 4096
        env_spacing = 3.3
        # The accepted 15 mm policy needs about 8.3 s of commanded motion to
        # clear the top.  A shorter episode resets before a complete successful
        # trajectory can contribute to PPO.
        episode_length_s = 12

    class commands(TinyMalCfg.commands):
        curriculum = False
        # A zero yaw-rate target does not recover heading after a riser impact:
        # it merely asks the robot to stop rotating at its new orientation.
        # Keep the desired world heading at zero so collision-induced yaw is
        # converted into a corrective yaw-rate command.
        heading_command = True
        resampling_time = 10.0

        class ranges(TinyMalCfg.commands.ranges):
            lin_vel_x = [0.25, 0.40]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            heading = [0.0, 0.0]

    class domain_rand(TinyMalCfg.domain_rand):
        # Isolate terrain capability before adding external disturbances.
        push_robots = False

    class rewards(TinyMalCfg.rewards):
        # With sigma=0.25, standing still under a 0.30 m/s command still earns
        # exp(-0.3^2 / 0.25)=0.70, so a blocked policy has little incentive to
        # discover a climbing gait. Tighten the task-level acceptance basin.
        tracking_sigma = 0.05

        class scales(TinyMalCfg.rewards.scales):
            termination = -2.0
            lateral_position = -0.5
            # Dense task-frame progress prevents the positive-reward clipping
            # from creating a zero-return blind zone at a stair riser.
            world_forward_progress = 5.0
            tracking_lin_vel = 2.0
            # Stair impacts create a persistent heading error unless yaw
            # tracking is strong enough to turn the body back toward world x.
            tracking_ang_vel = 2.0
            # Rapid pitch/roll is expected during step negotiation; retain a
            # penalty, but do not let it erase every exploratory transition.
            ang_vel_xy = -0.01

    class stairs:
        start_x = 0.55
        step_width = 0.14
        step_height = 0.03
        num_steps = 5
        # Nearly fills the 3.3 m environment spacing, so going around the side
        # cannot be mistaken for climbing.
        total_width = 3.10
        top_length = 1.40
        curriculum = True
        min_step_height = 0.005
        max_step_height = 0.040
        curriculum_levels = 8
        record_video = False
        camera_width = 960
        camera_height = 540


class TinyMalStairsCfgPPO(TinyMalCfgPPO):
    class algorithm(TinyMalCfgPPO.algorithm):
        entropy_coef = 0.0

    class runner(TinyMalCfgPPO.runner):
        experiment_name = "tinymal_stairs"
        run_name = "mixed_15mm_to_40mm"
        max_iterations = 800
        save_interval = 50
