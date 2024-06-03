from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from prlearn.base.agent import Agent


class AgentCombiner(ABC):
    """
    Abstract base class for combining agents from multiple workers.

    Methods:
        combine(workers_agents, main_agent, workers_stats, main_agent_stats):
            Abstract method to combine worker agents into a main agent.
    """

    @abstractmethod
    def combine(
        self,
        workers_agents: List[Agent],
        main_agent: Agent,
        workers_stats: Optional[List[Dict[str, Any]]] = None,
        main_agent_stats: Optional[Dict[str, Any]] = None,
    ) -> Agent:
        """
        Combine worker agents into a main agent.

        Args:
            workers_agents (List[Agent]): List of agents from workers.
            main_agent (Agent): Main agent to combine into.
            workers_stats (Optional[List[Dict[str, Any]]]): Optional statistics for worker agents.
            main_agent_stats (Optional[Dict[str, Any]]): Optional statistics for the main agent.

        Returns:
            Agent: The combined agent.

        Raises:
            NotImplementedError: If the method is not implemented.
        """
        raise NotImplementedError(
            "Method 'combine' of class AgentCombiner is not implemented"
        )
