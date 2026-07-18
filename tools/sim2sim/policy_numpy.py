"""Load the trained TinyMal actor and run it in numpy (no Isaac Gym).

The checkpoint is a torch state_dict; we load it with torch (available in both
unitree-rl and Isaac Sim's bundled Python), then evaluate the MLP with numpy so
the sim2sim loop has no Isaac Gym dependency.

Actor MLP (actor_critic.py, keys actor.{0,2,4,6}):
    Linear(48->512), ELU, Linear(512->256), ELU, Linear(256->128), ELU,
    Linear(128->12)   <- final layer has NO activation (raw action mean)
Deployment uses the mean action (get_inference_policy -> act_inference).
"""

import numpy as np

try:
    import torch
except ImportError as e:  # pragma: no cover
    raise ImportError("policy_numpy needs torch to load the .pt checkpoint") from e


def _elu(x):
    # ELU(x) = x if x>0 else e^x - 1
    return np.where(x > 0.0, x, np.expm1(x))


class NumpyPolicy:
    """Mean-action inference of the trained actor."""

    def __init__(self, checkpoint_path):
        ck = torch.load(checkpoint_path, map_location="cpu")
        sd = ck["model_state_dict"]
        self.layers = [
            (sd["actor.0.weight"].numpy(), sd["actor.0.bias"].numpy()),
            (sd["actor.2.weight"].numpy(), sd["actor.2.bias"].numpy()),
            (sd["actor.4.weight"].numpy(), sd["actor.4.bias"].numpy()),
            (sd["actor.6.weight"].numpy(), sd["actor.6.bias"].numpy()),
        ]

    def __call__(self, obs48):
        x = np.asarray(obs48, dtype=np.float64)
        last = len(self.layers) - 1
        for i, (w, b) in enumerate(self.layers):
            x = x @ w.T + b
            if i != last:  # ELU after the first three linears, not the last
                x = _elu(x)
        return x
