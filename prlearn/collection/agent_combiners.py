import random
from typing import Any, Dict, List, Optional

from prlearn.base.agent import Agent
from prlearn.base.agent_combiner import AgentCombiner


class RandomAgentCombiner(AgentCombiner):
    """
    Combiner that selects a random agent from the list of worker agents.

    Args:
        seed (Optional[int]): Optional seed for random number generator.
    """

    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)

    def combine(
        self,
        workers_agents: List[Agent],
        main_agent: Agent,
        workers_stats: Optional[List[Dict[str, Any]]] = None,
        main_agent_stats: Optional[Dict[str, Any]] = None,
    ) -> Agent:
        """
        Select a random agent from the list of worker agents.

        Args:
            workers_agents (List[Agent]): List of agents from workers.
            main_agent (Agent): Main agent to combine into.
            workers_stats (Optional[List[Dict[str, Any]]]): Optional statistics for worker agents.
            main_agent_stats (Optional[Dict[str, Any]]): Optional statistics for the main agent.

        Returns:
            Agent: A randomly selected agent from the list of worker agents.
        """
        return random.choice(workers_agents)


class FixedAgentCombiner(AgentCombiner):
    """
    Combiner that selects an agent based on a fixed index.

    Args:
        idx (int): Index of the agent to select.
    """

    def __init__(self, idx: int = 0):
        self.idx = idx

    def combine(
        self,
        workers_agents: List[Agent],
        main_agent: Agent,
        workers_stats: Optional[List[Dict[str, Any]]] = None,
        main_agent_stats: Optional[Dict[str, Any]] = None,
    ) -> Agent:
        """
        Select an agent based on a fixed index.

        Args:
            workers_agents (List[Agent]): List of agents from workers.
            main_agent (Agent): Main agent to combine into.
            workers_stats (Optional[List[Dict[str, Any]]]): Optional statistics for worker agents.
            main_agent_stats (Optional[Dict[str, Any]]): Optional statistics for the main agent.

        Returns:
            Agent: The agent at the specified index.
        """
        return workers_agents[self.idx]


class FixedStatAgentCombiner(AgentCombiner):
    """
    Combiner that selects an agent based on a specific statistic.

    Args:
        stat_name (str): The name of the statistic to use for selection.
    """

    def __init__(self, stat_name: str = "mean_reward"):
        self.stat_name = stat_name

    def combine(
        self,
        workers_agents: List[Agent],
        main_agent: Agent,
        workers_stats: Optional[List[Dict[str, Any]]] = None,
        main_agent_stats: Optional[Dict[str, Any]] = None,
    ) -> Agent:
        """
        Select an agent based on a specific statistic.

        Args:
            workers_agents (List[Agent]): List of agents from workers.
            main_agent (Agent): Main agent to combine into.
            workers_stats (Optional[List[Dict[str, Any]]]): Optional statistics for worker agents.
            main_agent_stats (Optional[Dict[str, Any]]): Optional statistics for the main agent.

        Returns:
            Agent: The agent with the highest specified statistic. If statistics are not available for all agents,
                   returns the first agent in the list.
        """
        scores = []

        for stat in workers_stats or []:
            if stat is not None and self.stat_name in stat:
                scores.append(stat[self.stat_name])

        if len(scores) == len(workers_agents):
            index = scores.index(max(scores))
            return workers_agents[index]
        else:
            return workers_agents[0]
