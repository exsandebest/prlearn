import pytest

from prlearn.base.agent import Agent
from prlearn.collection.agent_combiners import (
    FixedAgentCombiner,
    FixedStatAgentCombiner,
    RandomAgentCombiner,
)


class DummyAgent(Agent):
    def __init__(self, name: str):
        self.name = name

    def action(self, state):
        return self.name

    def train(self, experience):
        return None


@pytest.fixture
def agents():
    return [DummyAgent("a"), DummyAgent("b"), DummyAgent("c")]


def test_random_agent_combiner_seeded_choice_is_reproducible(agents):
    """RandomAgentCombiner should respect the provided seed."""
    combiner = RandomAgentCombiner(seed=0)
    selected = combiner.combine(agents, agents[0])
    assert selected is agents[1]
    combiner_again = RandomAgentCombiner(seed=0)
    assert combiner_again.combine(agents, agents[0]) is agents[1]


def test_fixed_agent_combiner_returns_requested_index(agents):
    """FixedAgentCombiner should return the agent at the configured index."""
    combiner = FixedAgentCombiner(idx=1)
    assert combiner.combine(agents, agents[0]) is agents[1]


def test_fixed_stat_agent_combiner_uses_stats_when_available(agents):
    """FixedStatAgentCombiner should select the agent with the best provided stat."""
    stats = [
        {"mean_reward": 1.0},
        {"mean_reward": 3.0},
        {"mean_reward": 2.0},
    ]
    combiner = FixedStatAgentCombiner(stat_name="mean_reward")
    assert combiner.combine(agents, agents[0], workers_stats=stats) is agents[1]


def test_fixed_stat_agent_combiner_fallback_without_complete_stats(agents):
    """If statistics are missing, the first agent should be returned."""
    stats = [{"mean_reward": 1.0}, None, {}]
    combiner = FixedStatAgentCombiner(stat_name="mean_reward")
    assert combiner.combine(agents, agents[0], workers_stats=stats) is agents[0]


if __name__ == "__main__":
    pytest.main()
