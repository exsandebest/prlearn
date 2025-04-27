import os
import random
from typing import Any, Dict, List, Optional, Tuple

import pytest

from prlearn import Agent, AgentCombiner, Environment, Experience, Trainer
from prlearn.collection.agent_combiners import RandomAgentCombiner
from prlearn.common.dataclasses import Mode, SyncMode


class MyAgent(Agent):
    def __init__(self, max_action_value):
        self.max_action_value = max_action_value
        self.action_calls = 0
        self.train_calls = 0
        self.exp_total_size = 0

    def action(self, state):
        self.action_calls += 1
        return random.randint(0, self.max_action_value)

    def train(self, exp: Experience):
        self.train_calls += 1
        self.exp_total_size += len(exp)
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


class MyAC(RandomAgentCombiner):
    def __init__(self, **kwargs):
        self.combine_calls = 0
        super().__init__(**kwargs)

    def combine(
        self,
        workers_agents: List[Agent],
        main_agent: Agent,
        workers_stats: Optional[List[Dict[str, Any]]] = None,
        main_agent_stats: Optional[Dict[str, Any]] = None,
    ) -> Agent:
        self.combine_calls += 1
        return random.choice(workers_agents)


@pytest.fixture
def agent():
    return MyAgent(100)


@pytest.fixture
def agent2():
    return MyAgent(50)


@pytest.fixture
def env():
    return MyEnv(100)


@pytest.fixture
def combiner():
    return MyAC(seed=0)


def test_simple_learning(agent, env):
    """Test simple learning scenario with steps and periodic training."""
    trainer = Trainer(
        agent, env, schedule=[("finish", 100, "steps"), ("train_agent", 20, "steps")]
    )

    trained_agent, result = trainer.run()

    assert isinstance(trained_agent, MyAgent)
    assert isinstance(result, dict)
    assert "workers" in result
    assert "trainer" in result
    assert result["trainer"] is trainer
    assert trainer.mode == Mode.PARALLEL_LEARNING
    assert trainer.sync_mode == SyncMode.ASYNCHRONOUS
    assert trained_agent.train_calls == 5
    assert trained_agent.action_calls == 100
    assert len(result["workers"][0]["experience"]) == 100
    assert result["workers"][0]["total_steps"] == 100


def test_observations_collecting(agent, env):
    """Test parallel data collection with multiple workers and no training."""
    trainer = Trainer(
        agent,
        env,
        n_workers=2,
        schedule=[("finish", 100, "steps")],
    )

    trained_agent, result = trainer.run()

    assert isinstance(trained_agent, MyAgent)
    assert isinstance(result, dict)
    assert "workers" in result
    assert "trainer" in result
    assert trainer.mode == Mode.PARALLEL_COLLECTING
    assert trainer.sync_mode == SyncMode.ASYNCHRONOUS
    assert trained_agent.train_calls == 0
    assert trained_agent.action_calls == 0
    assert len(result["workers"][0]["experience"]) == 100
    assert len(result["workers"][1]["experience"]) == 100
    assert result["workers"][0]["total_steps"] == 100
    assert result["workers"][1]["total_steps"] == 100


def test_learning_with_parallel_collecting(agent, env):
    """Test parallel data collection with periodic training and sync mode."""
    trainer = Trainer(
        agent,
        env,
        n_workers=2,
        schedule=[("finish", 100, "steps"), ("train_agent", 10, "steps")],
        sync_mode="sync",
    )

    trained_agent, result = trainer.run()

    assert isinstance(trained_agent, MyAgent)
    assert isinstance(result, dict)
    assert "workers" in result
    assert "trainer" in result
    assert trainer.mode == Mode.PARALLEL_COLLECTING
    assert trainer.sync_mode == SyncMode.SYNCHRONOUS
    assert trained_agent.train_calls == 20
    assert trained_agent.action_calls == 0
    assert len(result["workers"][0]["experience"]) == 100
    assert len(result["workers"][1]["experience"]) == 100
    assert result["workers"][0]["total_steps"] == 100
    assert result["workers"][1]["total_steps"] == 100
    assert result["workers"][0]["agent_version"] == 20
    assert result["workers"][1]["agent_version"] == 20
    assert trainer.agent_version == 20
    assert trained_agent.exp_total_size == 200


def test_parallel_learning(agent, env, combiner):
    """Test parallel learning with agent combination and sync mode."""
    trainer = Trainer(
        agent,
        env,
        n_workers=2,
        schedule=[
            ("finish", 200, "steps"),
            ("train_agent", 10, "steps"),
            ("combine_agents", 40, "steps"),
        ],
        mode="parallel_learning",
        sync_mode="sync",
        combiner=combiner,
    )

    trained_agent, result = trainer.run()

    assert isinstance(trained_agent, MyAgent)
    assert isinstance(result, dict)
    assert "workers" in result
    assert "trainer" in result
    assert trainer.mode == Mode.PARALLEL_LEARNING
    assert trainer.sync_mode == SyncMode.SYNCHRONOUS
    assert combiner.combine_calls == 10
    assert len(result["workers"][0]["experience"]) == 200
    assert len(result["workers"][1]["experience"]) == 200
    assert result["workers"][0]["total_steps"] == 200
    assert result["workers"][1]["total_steps"] == 200
    assert result["workers"][0]["agent_version"] == 30
    assert result["workers"][1]["agent_version"] == 30


def test_multiple_parallel_learning(agent, agent2, env):
    """Test parallel learning with multiple different agents."""
    trainer = Trainer(
        [agent, agent2],
        env,
        n_workers=2,
        schedule=[("finish", 200, "steps"), ("train_agent", 10, "steps")],
        mode="parallel_learning",
        sync_mode="sync",
    )

    trained_agent, result = trainer.run()

    assert isinstance(trained_agent, MyAgent)
    assert isinstance(result, dict)
    assert "workers" in result
    assert "trainer" in result
    assert trainer.mode == Mode.PARALLEL_LEARNING
    assert trainer.sync_mode == SyncMode.SYNCHRONOUS
    assert len(result["workers"][0]["experience"]) == 200
    assert len(result["workers"][1]["experience"]) == 200
    assert result["workers"][0]["agent"].max_action_value == agent.max_action_value
    assert result["workers"][1]["agent"].max_action_value == agent2.max_action_value
    assert result["workers"][0]["total_steps"] == 200
    assert result["workers"][1]["total_steps"] == 200
    assert result["workers"][0]["agent_version"] == 20
    assert result["workers"][1]["agent_version"] == 20


def test_custom_combiner(agent, env):
    """Test using a custom AgentCombiner with custom logic."""

    class CustomCombiner(AgentCombiner):
        def combine(
            self, workers_agents, main_agent, workers_stats=None, main_agent_stats=None
        ):
            # Always return the first agent
            return workers_agents[0]

    combiner = CustomCombiner()
    trainer = Trainer(
        agent,
        env,
        n_workers=2,
        schedule=[
            ("finish", 50, "steps"),
            ("train_agent", 10, "steps"),
            ("combine_agents", 20, "steps"),
        ],
        mode="parallel_learning",
        sync_mode="sync",
        combiner=combiner,
    )
    trained_agent, result = trainer.run()
    assert isinstance(trained_agent, MyAgent)
    assert trainer.combiner is combiner


def test_statistics_correctness(agent, env):
    """Test that statistics in result are correct and consistent."""
    trainer = Trainer(agent, env, schedule=[("finish", 10, "steps")])
    trained_agent, result = trainer.run()
    for worker in result["workers"]:
        assert worker["total_steps"] == 10
        assert len(worker["experience"]) == 10
        assert isinstance(worker["rewards"], list)


def test_schedule_variations(agent, env):
    """Test Trainer with different schedule units (steps, episodes, seconds)."""
    schedules = [
        [("finish", 10, "steps"), ("train_agent", 2, "steps")],
        [("finish", 5, "episodes"), ("train_agent", 1, "episodes")],
        [("finish", 1, "seconds"), ("train_agent", 0.5, "seconds")],
    ]
    for schedule in schedules:
        trainer = Trainer(agent, env, schedule=schedule)
        trained_agent, result = trainer.run()
        assert isinstance(trained_agent, MyAgent)
        assert isinstance(result, dict)
        assert "workers" in result


def test_parallel_learning_async(agent, env):
    """Test parallel learning in async mode."""
    trainer = Trainer(
        agent,
        env,
        n_workers=2,
        schedule=[
            ("finish", 50, "steps"),
            ("train_agent", 10, "steps"),
            ("combine_agents", 20, "steps"),
        ],
        mode="parallel_learning",
        sync_mode="async",
    )
    trained_agent, result = trainer.run()
    assert isinstance(trained_agent, MyAgent)
    assert trainer.sync_mode == SyncMode.ASYNCHRONOUS


if __name__ == "__main__":
    pytest.main()
