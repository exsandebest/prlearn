import time

import pytest

from prlearn.common.dataclasses import Mode
from prlearn.common.pas import ProcessActionScheduler


@pytest.fixture
def scheduler():
    config = [
        ("train_agent", 5, "seconds"),
        ("worker_send_data", 10, "steps"),
        ("finish", 3, "episodes"),
        ("combine_agents", 7, "seconds"),
    ]
    return ProcessActionScheduler(config)


def test_initial_state(scheduler):
    """Test initial state and config values of scheduler."""
    assert scheduler.state["train_agent"]["steps"] == 0
    assert scheduler.state["train_agent"]["episodes"] == 0
    assert "seconds" in scheduler.state["train_agent"]
    assert scheduler.config["train_agent"]["seconds_interval"] == 5
    assert scheduler.config["worker_send_data"]["steps_interval"] == 10
    assert scheduler.config["finish"]["episodes_interval"] == 3
    assert scheduler.config["combine_agents"]["seconds_interval"] == 7


def test_set_time(scheduler):
    """Test set_time updates the state for all actions or a specific action."""
    current_time = time.time()
    scheduler.set_time(current_time)
    assert scheduler.state["train_agent"]["seconds"] == current_time


def test_check_agent_train_seconds(scheduler):
    """Test check_agent_train triggers after enough seconds have passed."""
    scheduler.set_time(time.time() - 6, "train_agent")
    result = scheduler.check_agent_train(check_time=time.time())
    assert result is not None
    assert result["seconds"] >= 5


def test_check_worker_send_steps(scheduler):
    """Test check_worker_send triggers after enough steps have passed."""
    scheduler.state["worker_send_data"]["steps"] = 0
    result = scheduler.check_worker_send(n_steps=11)
    assert result is not None
    assert result["steps"] == 11


def test_check_train_finish_episodes(scheduler):
    """Test check_train_finish triggers after enough episodes have passed."""
    scheduler.state["finish"]["episodes"] = 0
    result = scheduler.check_train_finish(n_episodes=4)
    assert result is not None
    assert result["episodes"] == 4


def test_check_combine_agents_seconds(scheduler):
    """Test check_worker_send does not trigger if not enough steps have passed."""
    scheduler.state["worker_send_data"]["steps"] = 0
    result = scheduler.check_worker_send(n_steps=8)
    assert result is None


def test_check_worker_send_steps_not_complete(scheduler):
    """Test check_combine_agents triggers after enough seconds have passed."""
    scheduler.set_time(time.time() - 8, "combine_agents")
    result = scheduler.check_combine_agents(check_time=time.time())
    assert result is not None
    assert result["seconds"] >= 7


def test_invalid_action():
    """Test that check() raises ValueError for invalid action name."""
    scheduler = ProcessActionScheduler()
    with pytest.raises(ValueError):
        scheduler.check("invalid_action")


def test_invalid_config_item():
    """Test that invalid config item raises ValueError."""
    with pytest.raises(ValueError):
        ProcessActionScheduler([("train_agent", "invalid_interval", "seconds")])


def test_invalid_config_units():
    """Test that invalid config units raise ValueError."""
    with pytest.raises(ValueError):
        ProcessActionScheduler([("train_agent", 5, "invalid_units")])


def test_auto_worker_send_data():
    """Test that worker_send_data is auto-calculated for different n_workers and modes."""
    scheduler = ProcessActionScheduler(
        [
            ("train_agent", 5, "steps"),
            ("finish", 3, "episodes"),
            ("combine_agents", 7, "steps"),
        ],
        n_workers=2,
        mode=Mode.PARALLEL_LEARNING,
    )
    assert scheduler.config["worker_send_data"]["steps_interval"] == 3.5
    assert scheduler.config["worker_send_data"]["episodes_interval"] == 3


def test_empty_config():
    """Test that scheduler works with empty config (no actions scheduled)."""
    scheduler = ProcessActionScheduler()
    for action in scheduler.possible_actions:
        for key in ["seconds_interval", "steps_interval", "episodes_interval"]:
            assert scheduler.config[action][key] is None


def test_modes_variation():
    """Test that scheduler works with different Mode values."""
    for mode in [Mode.PARALLEL_COLLECTING, Mode.PARALLEL_LEARNING]:
        scheduler = ProcessActionScheduler(
            [("train_agent", 5, "steps")], n_workers=3, mode=mode
        )
        assert scheduler.config["train_agent"]["steps_interval"] == 5


def test_check_boundary_values():
    """Test check() with boundary values (exactly at the interval)."""
    scheduler = ProcessActionScheduler([("train_agent", 5, "steps")])
    scheduler.state["train_agent"]["steps"] = 0
    result = scheduler.check("train_agent", n_steps=5)
    assert result is not None
    assert result["steps"] == 5


if __name__ == "__main__":
    pytest.main()
