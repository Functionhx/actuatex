from __future__ import annotations

from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from tasks.inverted_pendulum.off_policy_rl import (  # noqa: E402
    OffPolicyConfig,
    ReplayBuffer,
    SACAgent,
    TD3Agent,
    load_off_policy_actor,
)


def filled_replay(capacity: int = 128) -> ReplayBuffer:
    replay = ReplayBuffer(14, 1, capacity, "cpu")
    for _ in range(3):
        observation = torch.randn(48, 14)
        replay.add(
            observation,
            torch.tanh(torch.randn(48, 1)),
            torch.randn(48),
            observation + torch.randn_like(observation) * 0.01,
            torch.randint(0, 2, (48,), dtype=torch.float32),
        )
    return replay


def test_replay_wraps_and_samples_expected_shapes() -> None:
    replay = filled_replay(100)
    assert replay.size == 100
    assert replay.position == 44
    batch = replay.sample(32)
    assert [tuple(value.shape) for value in batch] == [
        (32, 14),
        (32, 1),
        (32, 1),
        (32, 14),
        (32, 1),
    ]


def test_sac_and_td3_update_with_bounded_actions() -> None:
    torch.manual_seed(7)
    replay = filled_replay()
    config = OffPolicyConfig(hidden_dims=(32, 32))
    for agent in (SACAgent(config, "cpu"), TD3Agent(config, "cpu")):
        metrics = agent.update(replay, 32)
        assert torch.isfinite(torch.tensor(metrics["critic_loss"]))
        action = agent.act(torch.randn(16, 14), explore=False)
        assert action.shape == (16, 1)
        assert torch.all(action.abs() <= 1.0)


def test_checkpoint_round_trip_preserves_deterministic_actor() -> None:
    torch.manual_seed(11)
    observation = torch.randn(8, 14)
    for agent in (
        SACAgent(OffPolicyConfig(hidden_dims=(16, 16)), "cpu"),
        TD3Agent(OffPolicyConfig(hidden_dims=(16, 16)), "cpu"),
    ):
        restored = load_off_policy_actor(agent.checkpoint())
        torch.testing.assert_close(
            agent.act(observation, explore=False),
            restored(observation),
        )
