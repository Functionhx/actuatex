"""Collision-accurate stairs with the full TinyMal sim-to-real randomization."""

from legged_gym.envs.tinymal.tinymal_robust import TinyMalRobust
from legged_gym.envs.tinymal.tinymal_stairs import TinyMalStairs


class TinyMalRobustStairs(TinyMalStairs, TinyMalRobust):
    """Use stair geometry/rewards and robust actuator/delay/push dynamics.

    The MRO is intentional: ``TinyMalStairs.create_sim`` builds the explicit
    staircase, while its ``super()._init_buffers()`` flows through
    ``TinyMalRobust`` so motor gains, delays, and force buffers remain active.
    """

    pass
