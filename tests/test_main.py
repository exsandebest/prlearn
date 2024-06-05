import os
import random
from typing import Any, Dict, Tuple

import pytest
from prlearn import Trainer, Agent, Environment, Experience
from prlearn.common.dataclasses import Mode


class MyAgent(Agent):

    def __init__(self, max_action_value):
        self.max_action_value = max_action_value
        self.action_calls = 0
        self.train_calls = 0

    def action(self, state):
        self.action_calls += 1
        return random.randint(0, self.max_action_value)

    def train(self, exp: Experience):
        self.train_calls += 1
        assert isinstance(exp, Experience)


class MyEnv(Environment):
    def __init__(self, max_steps):
        self.max_steps = max_steps
        self.steps = 0

    def step(self, action: Any) -> Tuple[Any, Any, bool, bool, Dict[str, Any]]:
        self.steps += 1
        return [self.steps, self.max_steps], 1, self.steps >= self.max_steps, False, {}

    def reset(self):
        self.steps = 0
        return [self.steps, self.max_steps], {}


@pytest.fixture
def agent():
    return MyAgent(100)


@pytest.fixture
def env():
    return MyEnv(100)


def test_simple_learning(agent, env):
    trainer = Trainer(agent, env, schedule=[("finish", 100, "steps"), ("train_agent", 20, "steps")])

    trained_agent, result = trainer.run()

    assert isinstance(trained_agent, MyAgent)
    assert isinstance(result, dict)
    assert "workers" in result
    assert "trainer" in result
    assert result["trainer"] is trainer
    assert trainer.mode == Mode.PARALLEL_LEARNING

    assert id(agent) == id(trained_agent)

def test_observations_collecting(agent, env):
    trainer = Trainer(
        agent,
        env,
        n_workers=2,
        schedule=[("finish", 100, "steps"), ("worker_send_data", 100, "steps")]
    )

    trained_agent, result = trainer.run()

    assert isinstance(trained_agent, MyAgent)
    assert isinstance(result, dict)
    assert "workers" in result
    assert "trainer" in result


def test_learning_with_parallel_collecting(agent, env):
    trainer = Trainer(
        agent,
        env,
        n_workers=2,
        schedule=[("finish", 100, "steps"), ("train_agent", 10, "steps")],
        sync_mode="sync"
    )

    trained_agent, result = trainer.run()

    assert isinstance(trained_agent, MyAgent)
    assert isinstance(result, dict)
    assert "workers" in result
    assert "trainer" in result


def test_parallel_learning(agent, env):
    trainer = Trainer(
        agent,
        env,
        n_workers=2,
        schedule=[("finish", 200, "steps"), ("train_agent", 10, "steps"), ("combine_agents", 80, "steps")],
        mode="parallel_learning",
        sync_mode="sync"
    )

    trained_agent, result = trainer.run()

    assert isinstance(trained_agent, MyAgent)
    assert isinstance(result, dict)
    assert "workers" in result
    assert "trainer" in result


def test_multiple_parallel_learning(agent, env):
    trainer = Trainer(
        [agent, agent],
        env,
        n_workers=2,
        schedule=[("finish", 200, "steps"), ("train_agent", 10, "steps"), ("combine_agents", 80, "steps")],
        mode="parallel_learning",
        sync_mode="sync"
    )

    agent, result = trainer.run()

    assert isinstance(agent, MyAgent)
    assert isinstance(result, dict)


if __name__ == "__main__":
    os.environ["LOG_LEVEL"] = "WARNING"
    pytest.main()
