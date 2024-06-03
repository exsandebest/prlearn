import pytest
from unittest.mock import MagicMock
from prlearn.base.agent import Agent
from prlearn.base.agent_combiner import AgentCombiner
from prlearn.base.environment import Environment
from prlearn.base.experience import Experience
from prlearn import Trainer


@pytest.fixture
def mock_agent():
    return MagicMock(Agent)


@pytest.fixture
def mock_env():
    return MagicMock(Environment)


@pytest.fixture
def mock_combiner():
    return MagicMock(AgentCombiner)


def test_invalid_n_workers(mock_agent, mock_env):
    with pytest.raises(ValueError):
        Trainer(agent=mock_agent, env=mock_env, n_workers=0)


def test_invalid_mode(mock_agent, mock_env):
    with pytest.raises(ValueError):
        Trainer(agent=mock_agent, env=mock_env, mode="invalid_mode")


def test_invalid_sync_mode(mock_agent, mock_env):
    with pytest.raises(ValueError):
        Trainer(agent=mock_agent, env=mock_env, sync_mode="invalid_sync_mode")


def test_multiple_agents_invalid_length(mock_agent, mock_env):
    with pytest.raises(ValueError):
        Trainer(agent=[mock_agent], env=mock_env, n_workers=2, mode="parallel_learning")


def test_multiple_agents_wrong_mode(mock_agent, mock_env):
    with pytest.raises(ValueError):
        Trainer(
            agent=[mock_agent, mock_agent],
            env=mock_env,
            n_workers=2,
            mode="parallel_collecting",
        )


def test_schedule_invalid_item(mock_agent, mock_env):
    with pytest.raises(ValueError):
        Trainer(
            agent=mock_agent,
            env=mock_env,
            schedule=[("train_agent", "invalid_interval", "seconds")],
        )


def test_parallel_learning_mode_without_combiner(mock_agent, mock_env):
    with pytest.raises(ValueError):
        Trainer(agent=mock_agent, env=mock_env, mode="parallel_learning", n_workers=2)


def test_initialization(mock_agent, mock_env, mock_combiner):
    trainer = Trainer(
        agent=mock_agent, env=mock_env, n_workers=1, combiner=mock_combiner
    )
    assert trainer.n_workers == 1
    assert trainer.agent == mock_agent
    assert trainer.env == mock_env
    assert trainer.combiner == mock_combiner
    assert isinstance(trainer.experience, Experience)


def test_update_worker_data(mock_agent, mock_env):
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


if __name__ == "__main__":
    pytest.main()
