import time
import pytest
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
    assert scheduler.state["train_agent"]["steps"] == 0
    assert scheduler.state["train_agent"]["episodes"] == 0
    assert "seconds" in scheduler.state["train_agent"]
    assert scheduler.config["train_agent"]["seconds_interval"] == 5
    assert scheduler.config["worker_send_data"]["steps_interval"] == 10
    assert scheduler.config["finish"]["episodes_interval"] == 3
    assert scheduler.config["combine_agents"]["seconds_interval"] == 7


def test_set_time(scheduler):
    current_time = time.time()
    scheduler.set_time(current_time)
    assert scheduler.state["train_agent"]["seconds"] == current_time


def test_check_agent_train_seconds(scheduler):
    scheduler.set_time(time.time() - 6, "train_agent")
    result = scheduler.check_agent_train(check_time=time.time())
    assert result is not None
    assert result["seconds"] >= 5


def test_check_worker_send_steps(scheduler):
    scheduler.state["worker_send_data"]["steps"] = 0
    result = scheduler.check_worker_send(n_steps=11)
    assert result is not None
    assert result["steps"] == 11


def test_check_train_finish_episodes(scheduler):
    scheduler.state["finish"]["episodes"] = 0
    result = scheduler.check_train_finish(n_episodes=4)
    assert result is not None
    assert result["episodes"] == 4


def test_check_combine_agents_seconds(scheduler):
    scheduler.state["worker_send_data"]["steps"] = 0
    result = scheduler.check_worker_send(n_steps=8)
    assert result is None


def test_check_worker_send_steps_not_complete(scheduler):
    scheduler.set_time(time.time() - 8, "combine_agents")
    result = scheduler.check_combine_agents(check_time=time.time())
    assert result is not None
    assert result["seconds"] >= 7


def test_invalid_action():
    scheduler = ProcessActionScheduler()
    with pytest.raises(ValueError):
        scheduler.check("invalid_action")


def test_invalid_config_item():
    with pytest.raises(ValueError):
        ProcessActionScheduler([("train_agent", "invalid_interval", "seconds")])


def test_invalid_config_units():
    with pytest.raises(ValueError):
        ProcessActionScheduler([("train_agent", 5, "invalid_units")])


if __name__ == "__main__":
    pytest.main()
