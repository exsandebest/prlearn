from unittest.mock import MagicMock

import pytest

from prlearn import Trainer
from prlearn.base.agent import Agent
from prlearn.base.agent_combiner import AgentCombiner
from prlearn.base.environment import Environment
from prlearn.base.experience import Experience
from prlearn.collection.agent_combiners import FixedAgentCombiner, RandomAgentCombiner


@pytest.fixture
def mock_agent() -> MagicMock:
    return MagicMock(Agent)


@pytest.fixture
def mock_env() -> MagicMock:
    return MagicMock(Environment)


@pytest.fixture
def mock_combiner() -> MagicMock:
    return MagicMock(AgentCombiner)


def test_invalid_n_workers(mock_agent: MagicMock, mock_env: MagicMock):
    """Test that Trainer raises ValueError for n_workers <= 0."""
    with pytest.raises(ValueError):
        Trainer(agent=mock_agent, env=mock_env, n_workers=0)


def test_invalid_mode(mock_agent: MagicMock, mock_env: MagicMock):
    """Test that Trainer raises ValueError for invalid mode."""
    with pytest.raises(ValueError):
        Trainer(agent=mock_agent, env=mock_env, mode="invalid_mode")


def test_invalid_sync_mode(mock_agent: MagicMock, mock_env: MagicMock):
    """Test that Trainer raises ValueError for invalid sync_mode."""
    with pytest.raises(ValueError):
        Trainer(agent=mock_agent, env=mock_env, sync_mode="invalid_sync_mode")


def test_multiple_agents_invalid_length(mock_agent: MagicMock, mock_env: MagicMock):
    """Test that Trainer raises ValueError if agent list length != n_workers."""
    with pytest.raises(ValueError):
        Trainer(agent=[mock_agent], env=mock_env, n_workers=2, mode="parallel_learning")


def test_multiple_agents_wrong_mode(mock_agent: MagicMock, mock_env: MagicMock):
    """Test that Trainer raises ValueError if multiple agents are used in wrong mode."""
    with pytest.raises(ValueError):
        Trainer(
            agent=[mock_agent, mock_agent],
            env=mock_env,
            n_workers=2,
            mode="parallel_collecting",
        )


def test_schedule_invalid_item(mock_agent: MagicMock, mock_env: MagicMock):
    """Test that Trainer raises ValueError for invalid schedule item."""
    with pytest.raises(ValueError):
        Trainer(
            agent=mock_agent,
            env=mock_env,
            schedule=[("train_agent", "invalid_interval", "seconds")],
        )


def test_parallel_learning_mode_without_combiner(
    mock_agent: MagicMock, mock_env: MagicMock
):
    """Test that Trainer auto-selects combiner if not provided in parallel_learning mode."""
    trainer = Trainer(
        agent=mock_agent, env=mock_env, mode="parallel_learning", n_workers=2
    )
    assert isinstance(trainer.combiner, RandomAgentCombiner)
    trainer = Trainer(
        agent=mock_agent, env=mock_env, mode="parallel_learning", n_workers=1
    )
    assert isinstance(trainer.combiner, FixedAgentCombiner)


def test_initialization(
    mock_agent: MagicMock, mock_env: MagicMock, mock_combiner: MagicMock
):
    """Test Trainer initialization with custom combiner."""
    trainer = Trainer(
        agent=mock_agent, env=mock_env, n_workers=1, combiner=mock_combiner
    )
    assert trainer.n_workers == 1
    assert trainer.agent == mock_agent
    assert trainer.env == mock_env
    assert trainer.combiner == mock_combiner
    assert isinstance(trainer.experience, Experience)


def test_update_worker_data(mock_agent: MagicMock, mock_env: MagicMock):
    """Test that _update_worker_data correctly updates worker stats."""
    trainer = Trainer(agent=mock_agent, env=mock_env, n_workers=1)
    data = MagicMock()
    data.n_total_episodes = 5
    data.n_total_steps = 10
    data.agent_version = 2
    data.stats = {"key": "value"}
    data.rewards = [1, 2, 3]
    trainer._update_worker_data(0, data)
    assert trainer.workers_episodes[0] == 5
    assert trainer.workers_steps[0] == 10
    assert trainer.workers_agent_versions[0] == 2
    assert trainer.workers_stats[0] == {"key": "value"}
    assert trainer.workers_rewards[0] == [1, 2, 3]


@pytest.mark.parametrize(
    "mode,sync_mode",
    [
        ("parallel_learning", "sync"),
        ("parallel_learning", "async"),
        ("parallel_collecting", "sync"),
        ("parallel_collecting", "async"),
    ],
)
def test_trainer_modes_and_sync(
    mock_agent: MagicMock, mock_env: MagicMock, mode, sync_mode
):
    """Test Trainer initialization with all combinations of mode and sync_mode."""
    trainer = Trainer(
        agent=mock_agent, env=mock_env, n_workers=2, mode=mode, sync_mode=sync_mode
    )
    assert trainer.mode.value == mode
    assert trainer.sync_mode.value == sync_mode


def test_trainer_custom_combiner_usage(mock_agent, mock_env):
    """Test that a custom AgentCombiner is used in parallel_learning mode."""

    class CustomCombiner(AgentCombiner):
        def combine(
            self, workers_agents, main_agent, workers_stats=None, main_agent_stats=None
        ):
            return main_agent

    combiner = CustomCombiner()
    trainer = Trainer(
        agent=mock_agent,
        env=mock_env,
        n_workers=2,
        mode="parallel_learning",
        combiner=combiner,
    )
    assert trainer.combiner is combiner


def test_trainer_schedule_variations(mock_agent, mock_env):
    """Test Trainer initialization with different schedule units (steps, episodes, seconds)."""
    schedules = [
        [("finish", 10, "steps"), ("train_agent", 2, "steps")],
        [("finish", 5, "episodes"), ("train_agent", 1, "episodes")],
        [("finish", 1, "seconds"), ("train_agent", 0.5, "seconds")],
    ]
    for schedule in schedules:
        trainer = Trainer(
            agent=mock_agent, env=mock_env, n_workers=1, schedule=schedule
        )
        assert trainer.schedule_config == schedule


if __name__ == "__main__":
    pytest.main()
