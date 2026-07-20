"""Shared SAC and TD3 implementations for MuJoCo and Isaac Lab.

The module intentionally depends only on PyTorch. Both physics backends feed
the same task-specific observation/action contract into these agents, so
differences in results come from dynamics and data collection rather than two
unrelated algorithm implementations.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


CHECKPOINT_FORMAT = "actuatex_off_policy_v1"


@dataclass
class ReplayRatioScheduler:
    """Convert a backend-independent replay ratio into gradient updates.

    ``replay_sample_ratio`` is the number of replay rows sampled by optimizer
    batches per newly collected transition. Counting sampled rows, instead of
    vector-environment calls, keeps the training budget comparable when the
    number of environments or the minibatch size changes.

    ``updates_per_vector_step`` is a legacy escape hatch. When supplied it
    deliberately restores the old backend-dependent behavior and the achieved
    replay ratio remains observable through :attr:`achieved_ratio`.
    """

    batch_size: int
    replay_sample_ratio: float = 4.0
    updates_per_vector_step: int | None = None
    pending_update_credit: float = field(init=False, default=0.0)
    eligible_transitions: int = field(init=False, default=0)
    gradient_updates: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if (
            not math.isfinite(self.replay_sample_ratio)
            or self.replay_sample_ratio <= 0.0
        ):
            raise ValueError("replay_sample_ratio must be finite and positive")
        if (
            self.updates_per_vector_step is not None
            and self.updates_per_vector_step <= 0
        ):
            raise ValueError("updates_per_vector_step must be positive when set")

    def updates_for(self, new_transitions: int) -> int:
        """Return optimizer updates earned by newly trainable transitions."""

        if new_transitions < 0:
            raise ValueError("new_transitions cannot be negative")
        if new_transitions == 0:
            return 0
        self.eligible_transitions += new_transitions
        if self.updates_per_vector_step is not None:
            updates = self.updates_per_vector_step
        else:
            self.pending_update_credit += (
                new_transitions * self.replay_sample_ratio / self.batch_size
            )
            updates = math.floor(self.pending_update_credit + 1.0e-12)
            self.pending_update_credit -= updates
        self.gradient_updates += updates
        return updates

    @property
    def achieved_ratio(self) -> float:
        """Replay rows sampled per eligible transition so far."""

        if self.eligible_transitions == 0:
            return 0.0
        return self.gradient_updates * self.batch_size / self.eligible_transitions


def _mlp(
    input_dim: int,
    hidden_dims: tuple[int, ...],
    output_dim: int,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.extend((nn.Linear(previous_dim, hidden_dim), nn.ReLU()))
        previous_dim = hidden_dim
    layers.append(nn.Linear(previous_dim, output_dim))
    return nn.Sequential(*layers)


class ReplayBuffer:
    """Device-resident cyclic replay buffer with vectorized insertion."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        capacity: int,
        device: torch.device | str,
    ) -> None:
        if observation_dim <= 0 or action_dim <= 0 or capacity <= 0:
            raise ValueError("replay dimensions and capacity must be positive")
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.capacity = capacity
        self.device = torch.device(device)
        self.observations = torch.empty(
            (capacity, observation_dim), dtype=torch.float32, device=self.device
        )
        self.actions = torch.empty(
            (capacity, action_dim), dtype=torch.float32, device=self.device
        )
        self.rewards = torch.empty(
            (capacity, 1), dtype=torch.float32, device=self.device
        )
        self.next_observations = torch.empty_like(self.observations)
        self.dones = torch.empty_like(self.rewards)
        self.position = 0
        self.size = 0

    def add(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_observation: torch.Tensor,
        done: torch.Tensor,
    ) -> None:
        batch_size = observation.shape[0]
        if batch_size <= 0:
            return
        if batch_size > self.capacity:
            start = batch_size - self.capacity
            observation = observation[start:]
            action = action[start:]
            reward = reward[start:]
            next_observation = next_observation[start:]
            done = done[start:]
            batch_size = self.capacity
        indices = (
            torch.arange(batch_size, device=self.device, dtype=torch.long)
            + self.position
        ) % self.capacity
        self.observations[indices] = observation.detach().to(
            device=self.device, dtype=torch.float32
        )
        self.actions[indices] = action.detach().to(
            device=self.device, dtype=torch.float32
        )
        self.rewards[indices, 0] = (
            reward.detach().to(device=self.device, dtype=torch.float32).reshape(-1)
        )
        self.next_observations[indices] = next_observation.detach().to(
            device=self.device, dtype=torch.float32
        )
        self.dones[indices, 0] = (
            done.detach().to(device=self.device, dtype=torch.float32).reshape(-1)
        )
        self.position = (self.position + batch_size) % self.capacity
        self.size = min(self.size + batch_size, self.capacity)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.size < batch_size:
            raise ValueError(
                f"replay contains {self.size} transitions, needs {batch_size}"
            )
        indices = torch.randint(
            self.size,
            (batch_size,),
            device=self.device,
        )
        return (
            self.observations[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_observations[indices],
            self.dones[indices],
        )


class SquashedGaussianActor(nn.Module):
    """SAC actor with tanh reparameterization and corrected log density."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, ...],
        *,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
    ) -> None:
        super().__init__()
        if not hidden_dims:
            raise ValueError("actor needs at least one hidden layer")
        self.backbone = _mlp(
            observation_dim,
            hidden_dims[:-1],
            hidden_dims[-1],
        )
        self.mean = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def distribution_parameters(
        self, observation: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = F.relu(self.backbone(observation))
        mean = self.mean(features)
        log_std = self.log_std(features).clamp(self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.distribution_parameters(observation)
        standard_deviation = log_std.exp()
        normal = torch.distributions.Normal(mean, standard_deviation)
        pre_tanh = normal.rsample()
        action = torch.tanh(pre_tanh)
        log_probability = normal.log_prob(pre_tanh) - torch.log(
            1.0 - action.square() + 1.0e-6
        )
        return action, log_probability.sum(dim=-1, keepdim=True)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        mean, _ = self.distribution_parameters(observation)
        return torch.tanh(mean)


class DeterministicActor(nn.Module):
    """TD3 actor with bounded actions."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, ...],
    ) -> None:
        super().__init__()
        self.network = _mlp(observation_dim, hidden_dims, action_dim)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.network(observation))


class TwinQNetwork(nn.Module):
    """Two independent action-value networks for clipped double Q learning."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, ...],
    ) -> None:
        super().__init__()
        input_dim = observation_dim + action_dim
        self.q1 = _mlp(input_dim, hidden_dims, 1)
        self.q2 = _mlp(input_dim, hidden_dims, 1)

    def forward(
        self, observation: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = torch.cat((observation, action), dim=-1)
        return self.q1(features), self.q2(features)

    def first(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.q1(torch.cat((observation, action), dim=-1))


def _soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for target_parameter, source_parameter in zip(
            target.parameters(), source.parameters(), strict=True
        ):
            target_parameter.lerp_(source_parameter, tau)


@dataclass(frozen=True)
class OffPolicyConfig:
    observation_dim: int = 14
    action_dim: int = 1
    hidden_dims: tuple[int, ...] = (256, 256)
    gamma: float = 0.99
    tau: float = 0.005
    actor_learning_rate: float = 3.0e-4
    critic_learning_rate: float = 3.0e-4


class SACAgent:
    """Soft Actor-Critic with learned entropy temperature."""

    algorithm = "sac"

    def __init__(
        self,
        config: OffPolicyConfig,
        device: torch.device | str,
        *,
        initial_alpha: float = 0.2,
        alpha_learning_rate: float = 3.0e-4,
    ) -> None:
        if initial_alpha <= 0.0:
            raise ValueError("initial_alpha must be positive")
        self.config = config
        self.device = torch.device(device)
        self.actor = SquashedGaussianActor(
            config.observation_dim,
            config.action_dim,
            config.hidden_dims,
        ).to(self.device)
        self.critic = TwinQNetwork(
            config.observation_dim,
            config.action_dim,
            config.hidden_dims,
        ).to(self.device)
        self.target_critic = deepcopy(self.critic).requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=config.actor_learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=config.critic_learning_rate
        )
        self.log_alpha = (
            torch.tensor(
                initial_alpha,
                dtype=torch.float32,
                device=self.device,
            )
            .log_()
            .requires_grad_(True)
        )
        self.alpha_optimizer = torch.optim.Adam(
            (self.log_alpha,), lr=alpha_learning_rate
        )
        self.target_entropy = -float(config.action_dim)
        self.update_count = 0

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def act(self, observation: torch.Tensor, *, explore: bool) -> torch.Tensor:
        observation = observation.to(self.device, dtype=torch.float32)
        with torch.inference_mode():
            if explore:
                action, _ = self.actor.sample(observation)
                return action
            return self.actor(observation)

    def update(self, replay: ReplayBuffer, batch_size: int) -> dict[str, float]:
        observation, action, reward, next_observation, done = replay.sample(batch_size)
        with torch.no_grad():
            next_action, next_log_probability = self.actor.sample(next_observation)
            target_q1, target_q2 = self.target_critic(next_observation, next_action)
            target_value = torch.minimum(target_q1, target_q2) - (
                self.alpha.detach() * next_log_probability
            )
            target = reward + self.config.gamma * (1.0 - done) * target_value

        current_q1, current_q2 = self.critic(observation, action)
        critic_loss = F.mse_loss(current_q1, target) + F.mse_loss(current_q2, target)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic.requires_grad_(False)
        sampled_action, log_probability = self.actor.sample(observation)
        actor_q1, actor_q2 = self.critic(observation, sampled_action)
        actor_loss = (
            self.alpha.detach() * log_probability - torch.minimum(actor_q1, actor_q2)
        ).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()
        self.critic.requires_grad_(True)

        alpha_loss = -(
            self.log_alpha * (log_probability + self.target_entropy).detach()
        ).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()
        _soft_update(self.target_critic, self.critic, self.config.tau)
        self.update_count += 1
        return {
            "critic_loss": float(critic_loss.detach().item()),
            "actor_loss": float(actor_loss.detach().item()),
            "alpha": float(self.alpha.detach().item()),
            "alpha_loss": float(alpha_loss.detach().item()),
        }

    def checkpoint(self, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        return _checkpoint_payload(
            self.algorithm,
            self.config,
            self.actor.state_dict(),
            self.update_count,
            metadata,
        )


class TD3Agent:
    """Twin Delayed DDPG with target policy smoothing."""

    algorithm = "td3"

    def __init__(
        self,
        config: OffPolicyConfig,
        device: torch.device | str,
        *,
        target_policy_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        policy_delay: int = 2,
    ) -> None:
        if target_policy_noise < 0.0 or target_noise_clip < 0.0:
            raise ValueError("target noise scales must be non-negative")
        if policy_delay <= 0:
            raise ValueError("policy_delay must be positive")
        self.config = config
        self.device = torch.device(device)
        self.actor = DeterministicActor(
            config.observation_dim,
            config.action_dim,
            config.hidden_dims,
        ).to(self.device)
        self.target_actor = deepcopy(self.actor).requires_grad_(False)
        self.critic = TwinQNetwork(
            config.observation_dim,
            config.action_dim,
            config.hidden_dims,
        ).to(self.device)
        self.target_critic = deepcopy(self.critic).requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=config.actor_learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=config.critic_learning_rate
        )
        self.target_policy_noise = target_policy_noise
        self.target_noise_clip = target_noise_clip
        self.policy_delay = policy_delay
        self.update_count = 0

    def act(self, observation: torch.Tensor, *, explore: bool) -> torch.Tensor:
        del explore
        observation = observation.to(self.device, dtype=torch.float32)
        with torch.inference_mode():
            return self.actor(observation)

    def update(self, replay: ReplayBuffer, batch_size: int) -> dict[str, float]:
        observation, action, reward, next_observation, done = replay.sample(batch_size)
        with torch.no_grad():
            noise = torch.randn_like(action) * self.target_policy_noise
            noise.clamp_(-self.target_noise_clip, self.target_noise_clip)
            next_action = (self.target_actor(next_observation) + noise).clamp(-1.0, 1.0)
            target_q1, target_q2 = self.target_critic(next_observation, next_action)
            target = reward + self.config.gamma * (1.0 - done) * torch.minimum(
                target_q1, target_q2
            )
        current_q1, current_q2 = self.critic(observation, action)
        critic_loss = F.mse_loss(current_q1, target) + F.mse_loss(current_q2, target)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        self.update_count += 1
        metrics = {"critic_loss": float(critic_loss.detach().item())}
        if self.update_count % self.policy_delay == 0:
            self.critic.requires_grad_(False)
            actor_loss = -self.critic.first(observation, self.actor(observation)).mean()
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_optimizer.step()
            self.critic.requires_grad_(True)
            _soft_update(self.target_actor, self.actor, self.config.tau)
            _soft_update(self.target_critic, self.critic, self.config.tau)
            metrics["actor_loss"] = float(actor_loss.detach().item())
        return metrics

    def checkpoint(self, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        return _checkpoint_payload(
            self.algorithm,
            self.config,
            self.actor.state_dict(),
            self.update_count,
            metadata,
        )


def create_agent(
    algorithm: str,
    config: OffPolicyConfig,
    device: torch.device | str,
) -> SACAgent | TD3Agent:
    normalized = algorithm.lower()
    if normalized == "sac":
        return SACAgent(config, device)
    if normalized == "td3":
        return TD3Agent(config, device)
    raise ValueError(f"unsupported off-policy algorithm {algorithm!r}")


def _checkpoint_payload(
    algorithm: str,
    config: OffPolicyConfig,
    actor_state_dict: dict[str, torch.Tensor],
    update_count: int,
    metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "checkpoint_format": CHECKPOINT_FORMAT,
        "off_policy_algorithm": algorithm,
        "observation_dim": config.observation_dim,
        "action_dim": config.action_dim,
        "hidden_dims": list(config.hidden_dims),
        "actor_state_dict": actor_state_dict,
        "update_count": update_count,
        "metadata": metadata or {},
    }


def load_off_policy_actor(
    checkpoint: Path | str | dict[str, Any],
    *,
    device: torch.device | str = "cpu",
) -> nn.Module:
    """Reconstruct the deterministic inference actor from a saved checkpoint."""

    if isinstance(checkpoint, dict):
        payload = checkpoint
    else:
        payload = torch.load(
            checkpoint,
            map_location=device,
            weights_only=False,
        )
    if payload.get("checkpoint_format") != CHECKPOINT_FORMAT:
        raise ValueError("not an ActuateX off-policy checkpoint")
    observation_dim = int(payload["observation_dim"])
    action_dim = int(payload["action_dim"])
    hidden_dims = tuple(int(value) for value in payload["hidden_dims"])
    algorithm = str(payload["off_policy_algorithm"])
    if algorithm == "sac":
        actor: nn.Module = SquashedGaussianActor(
            observation_dim, action_dim, hidden_dims
        )
    elif algorithm == "td3":
        actor = DeterministicActor(observation_dim, action_dim, hidden_dims)
    else:
        raise ValueError(f"unsupported checkpoint algorithm {algorithm!r}")
    actor.load_state_dict(payload["actor_state_dict"], strict=True)
    return actor.to(device).eval()
