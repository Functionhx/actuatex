from legged_gym.envs.tinymal.tinymal_config import TinyMalCfg, TinyMalCfgPPO


class TinyMalPushCfg(TinyMalCfg):
    """Flat-ground TinyMal for push-recovery evaluation. Physics identical to
    the baseline; the env subclass only adds external base-force injection."""

    pass


class TinyMalPushCfgPPO(TinyMalCfgPPO):
    class runner(TinyMalCfgPPO.runner):
        experiment_name = "tinymal_push"
        run_name = "push_eval"
