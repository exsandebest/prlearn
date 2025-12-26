from unittest.mock import MagicMock, patch

import pytest

from prlearn.base.agent import Agent
from prlearn.base.environment import Environment
from prlearn.base.experience import Experience
from prlearn.base.worker import Worker
from prlearn.common.dataclasses import Mode, SyncMode
from prlearn.common.pas import (
    PAS_ACTION_TRAIN_AGENT,
    ProcessActionScheduler,
)
from prlearn.utils.multiproc_lib import mp


@pytest.fixture
def mock_agent():
    agent = MagicMock(Agent)
    agent.action.return_value = 1
    return agent


@pytest.fixture
def mock_environment():
    env = MagicMock(Environment)
    env.reset.return_value = (0, {})
    env.step.return_value = (0, 1, True, False, {})
    return env


@pytest.fixture
def worker(mock_agent, mock_environment):
    conn_in = mp.Queue()
    conn_out = mp.Queue()
    global_params = {
        "mode": Mode.PARALLEL_LEARNING,
        "sync_mode": SyncMode.SYNCHRONOUS,
        "scheduler": ProcessActionScheduler([("finish", 3, "episodes")]),
    }
    return Worker(
        worker_id=0,
        env=mock_environment,
        agent=mock_agent,
        connection=(conn_in, conn_out),
        global_params=global_params,
    )


def test_worker_initialization(worker):
    """Test Worker initialization and attributes."""
    assert worker.worker_id == 0
    assert worker.env is not None
    assert worker.agent is not None
    assert isinstance(worker.experience, Experience)
    assert worker.wait_new_agent is False


def test_worker_update_stats(worker):
    """Test that _update_stats computes correct statistics."""
    worker.rewards = [1, 2, 3]
    worker._update_stats()
    assert worker.stats["max_reward"] == 3
    assert worker.stats["mean_reward"] == 2
    assert worker.stats["median_reward"] == 2
    assert worker.stats["min_reward"] == 1


def test_worker_get_new_agent(worker):
    """Test that _get_new_agent sets agent from agent_store if available."""
    worker.store_agent_available = True
    new_agent = MagicMock(Agent)
    worker.agent_store = new_agent
    worker.agent.set = MagicMock()
    worker._get_new_agent()
    worker.agent.set.assert_called_with(new_agent)


def test_worker_get_new_agent_no_set_method(worker):
    """Test _get_new_agent fallback if agent has no set() method."""
    worker.store_agent_available = True
    new_agent = MagicMock(Agent)
    worker.agent_store = new_agent
    # Remove set method
    del worker.agent.set
    worker._get_new_agent()
    assert worker.agent == new_agent


def test_worker_parallel_learning_step(worker):
    """Test that _parallel_learning_step calls agent.train when scheduled."""
    worker.scheduler.check = MagicMock(return_value={"steps": 1})
    worker.agent.train = MagicMock()
    worker.experience.get_experience_batch = MagicMock(return_value=MagicMock())
    worker._parallel_learning_step()
    worker.scheduler.check.assert_called_with(
        PAS_ACTION_TRAIN_AGENT, worker.total_steps, worker.total_episodes
    )
    worker.agent.train.assert_called()


def test_worker_run(worker):
    """Test that worker.run() calls all main methods in order (smoke test)."""
    with patch.object(worker, "_start", return_value=None) as mock_start, patch.object(
        worker, "_run_listener", return_value=None
    ) as mock_listener, patch.object(
        worker, "_run_simulation", return_value=None
    ) as mock_simulation, patch.object(
        worker, "_finish", return_value=None
    ) as mock_finish:
        worker.run()
        mock_start.assert_called_once()
        mock_listener.assert_called_once()
        mock_simulation.assert_called_once()
        mock_finish.assert_called_once()


def test_worker_message_handling(worker):
    """Test that _run_listener handles unexpected message types gracefully."""
    # Put a message with an unknown type
    from prlearn.common.dataclasses import MessageType, TrainerMessage

    worker.conn_in.put(TrainerMessage(type=MessageType.TRAINER_START))
    worker.stop_listener = True
    # Should not raise
    worker._run_listener()


def test_worker_error_in_env_step(monkeypatch, worker):
    """Test that errors in env.step do not crash the worker (simulate)."""

    def raise_error(*a, **k):
        raise RuntimeError("env error")

    worker.env.step = raise_error
    # Patch _run_simulation to call env.step and handle error
    with pytest.raises(RuntimeError):
        worker.env.step(None)


@pytest.mark.parametrize(
    "mode,sync_mode",
    [
        (Mode.PARALLEL_LEARNING, SyncMode.SYNCHRONOUS),
        (Mode.PARALLEL_LEARNING, SyncMode.ASYNCHRONOUS),
        (Mode.PARALLEL_COLLECTING, SyncMode.SYNCHRONOUS),
        (Mode.PARALLEL_COLLECTING, SyncMode.ASYNCHRONOUS),
    ],
)
def test_worker_modes_and_sync(mock_agent, mock_environment, mode, sync_mode):
    """Test Worker initialization with all combinations of Mode and SyncMode."""
    conn_in = mp.Queue()
    conn_out = mp.Queue()
    global_params = {
        "mode": mode,
        "sync_mode": sync_mode,
        "scheduler": ProcessActionScheduler([("finish", 3, "episodes")]),
    }
    worker = Worker(
        worker_id=0,
        env=mock_environment,
        agent=mock_agent,
        connection=(conn_in, conn_out),
        global_params=global_params,
    )
    assert worker.mode == mode
    assert worker.sync_mode == sync_mode


if __name__ == "__main__":
    pytest.main()
