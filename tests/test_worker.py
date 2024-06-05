import pytest
from unittest.mock import MagicMock, patch

from prlearn.common.pas import ProcessActionScheduler
from prlearn.utils.multiproc_lib import mp
from prlearn.base.worker import Worker
from prlearn.base.agent import Agent
from prlearn.base.environment import Environment
from prlearn.base.experience import Experience
from prlearn.common.dataclasses import (
    SyncMode,
    Mode,
)


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
        "scheduler": ProcessActionScheduler([("finish", 3, "episodes")])
    }
    return Worker(
        worker_id=0,
        env=mock_environment,
        agent=mock_agent,
        connection=(conn_in, conn_out),
        global_params=global_params,
    )


def test_worker_initialization(worker):
    assert worker.worker_id == 0
    assert worker.env is not None
    assert worker.agent is not None
    assert isinstance(worker.experience, Experience)


def test_worker_update_stats(worker):
    worker.rewards = [1, 2, 3]
    worker._update_stats()
    assert worker.stats["max_reward"] == 3
    assert worker.stats["mean_reward"] == 2
    assert worker.stats["median_reward"] == 2
    assert worker.stats["min_reward"] == 1


def test_worker_get_new_agent(worker):
    worker.store_agent_available = True
    new_agent = MagicMock(Agent)
    worker.agent_store = new_agent
    worker.agent.set = MagicMock()
    worker._get_new_agent()
    worker.agent.set.assert_called_with(new_agent)


def test_worker_parallel_learning_step(worker):
    worker.scheduler.check_agent_train = MagicMock(return_value={"steps": 1})
    worker.agent.train = MagicMock()
    worker.experience.get_experience_batch = MagicMock(return_value=MagicMock())
    worker._parallel_learning_step()
    worker.agent.train.assert_called()


def test_worker_run(worker):
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


if __name__ == "__main__":
    pytest.main()
