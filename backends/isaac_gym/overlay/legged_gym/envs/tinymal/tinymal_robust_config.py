"""Sim-to-real-oriented TinyMal training configuration.

The original ``tinymal`` task remains unchanged so that the course baseline and
the completed ablations stay reproducible.  This task deliberately broadens the
physics distribution used for the final policy.
"""

from legged_gym.envs.tinymal.tinymal_config import TinyMalCfg, TinyMalCfgPPO


class TinyMalRobustCfg(TinyMalCfg):
    """TinyMal task with physics, actuator, delay, push, and terrain variation."""

    class env(TinyMalCfg.env):
        num_envs = 4096

    class terrain(TinyMalCfg.terrain):
        # Preview 4's GPU heightfield narrowphase develops an MMU fault during
        # long 4096-env runs.  A single tiled triangle mesh carries the same
        # sampled heights through the stable, standard legged_gym rough path.
        mesh_type = "trimesh"
        curriculum = False
        horizontal_scale = 0.05
        vertical_scale = 0.002
        border_size = 5.0
        terrain_length = 4.0
        terrain_width = 4.0
        num_rows = 8
        num_cols = 8

        # Per-patch amplitudes are sampled uniformly.  Some patches are kept
        # exactly flat so the policy cannot forget the nominal deployment case.
        roughness_range = [0.0, 0.012]
        roughness_step = 0.002
        roughness_downsampled_scale = 0.15
        flat_patch_fraction = 0.125

    class domain_rand(TinyMalCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.5, 1.25]

        # Scale the 2.2657 kg base link rather than add a robot-independent
        # absolute payload.  This is exactly +/-15 percent around the URDF mass.
        randomize_base_mass = True
        base_mass_scale_range = [0.85, 1.15]

        # Isaac Gym applies these per environment and per joint.  Kp and Kd are
        # independently perturbed to cover motor/drive calibration error.
        randomize_motor_gains = True
        motor_kp_scale_range = [0.8, 1.2]
        motor_kd_scale_range = [0.8, 1.2]

        # Policy-rate latency: 1--3 samples at 50 Hz (20--60 ms).
        randomize_control_delay = True
        control_delay_range = [1, 3]

        # These actor properties are fixed within an episode but distributed
        # across the vectorized environments.
        randomize_joint_friction = True
        joint_friction_range = [0.0, 0.10]
        randomize_joint_armature = True
        joint_armature_range = [0.0, 0.02]

        # Use a sustained physical force instead of the legacy instantaneous
        # base-velocity overwrite.  The force is re-sampled every five seconds.
        push_robots = False
        push_interval_s = 5.0
        randomize_push_force = True
        push_force_range = [5.0, 30.0]
        push_duration_range_s = [0.10, 0.25]

    class noise(TinyMalCfg.noise):
        add_noise = True
        noise_level = 1.0

    class sim(TinyMalCfg.sim):
        class physx(TinyMalCfg.sim.physx):
            # Rough heightfields create more contact pairs than the historical
            # plane.  Leave headroom so the GPU narrowphase never approaches
            # Preview 4's default contact-buffer limit at 4096 environments.
            max_gpu_contact_pairs = 2**24
            default_buffer_size_multiplier = 10


class TinyMalRobustCfgPPO(TinyMalCfgPPO):
    class runner(TinyMalCfgPPO.runner):
        experiment_name = "tinymal_sim2real"
        run_name = "full_domain_rand_seed1"
        max_iterations = 1500
        save_interval = 50
