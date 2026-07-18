from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class TinyMalCfg(LeggedRobotCfg):
    """Baseline velocity-tracking task for the course TinyMal model."""

    class env(LeggedRobotCfg.env):
        num_envs = 4096

    class init_state(LeggedRobotCfg.init_state):
        # Forward kinematics puts the nominal foot centers about 0.225 m below
        # the base. The extra clearance prevents initial ground penetration.
        pos = [0.0, 0.0, 0.28]
        default_joint_angles = {
            "FL_hip_joint": -0.16,
            "FR_hip_joint": 0.16,
            "RL_hip_joint": -0.16,
            "RR_hip_joint": 0.16,
            "FL_thigh_joint": 0.68,
            "FR_thigh_joint": 0.68,
            "RL_thigh_joint": 0.68,
            "RR_thigh_joint": 0.68,
            "FL_calf_joint": 1.3,
            "FR_calf_joint": 1.3,
            "RL_calf_joint": 1.3,
            "RR_calf_joint": 1.3,
        }

    class commands(LeggedRobotCfg.commands):
        curriculum = True
        max_curriculum = 1.0

        class ranges(LeggedRobotCfg.commands.ranges):
            # Start below Go2 speeds because TinyMal is shorter and lighter.
            lin_vel_x = [-0.6, 0.6]
            lin_vel_y = [-0.3, 0.3]
            ang_vel_yaw = [-0.8, 0.8]
            heading = [-3.14, 3.14]

    class control(LeggedRobotCfg.control):
        control_type = "P"
        stiffness = {"joint": 20.0}
        damping = {"joint": 0.5}
        action_scale = 0.25
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/tinymal/urdf/tinymal.urdf"
        name = "tinymal"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        fix_base_link = False
        self_collisions = 1
        # The SolidWorks-exported STL files already use the expected frame.
        flip_visual_attachments = False

    class rewards(LeggedRobotCfg.rewards):
        base_height_target = 0.24
        soft_dof_pos_limit = 0.9

        class scales(LeggedRobotCfg.rewards.scales):
            orientation = -1.0
            base_height = -2.0
            torques = -0.0002
            dof_pos_limits = -10.0
            stand_still = -0.1


class TinyMalCfgPPO(LeggedRobotCfgPPO):
    class policy(LeggedRobotCfgPPO.policy):
        # std=1.0 perturbs each position target by roughly 0.25 rad at startup,
        # which destabilizes this small robot before PPO sees useful rewards.
        init_noise_std = 0.3

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.005

    class runner(LeggedRobotCfgPPO.runner):
        experiment_name = "tinymal_baseline"
        run_name = "course_default_angles"
        max_iterations = 1500
        save_interval = 50
