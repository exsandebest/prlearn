from typing import Any, Dict, List, Optional, Self, Tuple


class Experience:
    """
    Container for storing and manipulating experience tuples in RL.

    Args:
        observations (Optional[List[Any]]): List of observations.
        actions (Optional[List[Any]]): List of actions.
        rewards (Optional[List[Any]]): List of rewards.
        next_observations (Optional[List[Any]]): List of next observations.
        terminated (Optional[List[bool]]): List of termination flags.
        truncated (Optional[List[bool]]): List of truncation flags.
        info (Optional[List[Dict[str, Any]]]): List of info dicts.
        agent_versions (Optional[List[int]]): List of agent version numbers.
        worker_ids (Optional[List[int]]): List of worker IDs.
        episodes (Optional[List[int]]): List of episode numbers.
    """

    def __init__(
        self,
        observations: Optional[List[Any]] = None,
        actions: Optional[List[Any]] = None,
        rewards: Optional[List[Any]] = None,
        next_observations: Optional[List[Any]] = None,
        terminated: Optional[List[bool]] = None,
        truncated: Optional[List[bool]] = None,
        info: Optional[List[Dict[str, Any]]] = None,
        agent_versions: Optional[List[int]] = None,
        worker_ids: Optional[List[int]] = None,
        episodes: Optional[List[int]] = None,
    ):
        self.observations = observations or []
        self.actions = actions or []
        self.rewards = rewards or []
        self.next_observations = next_observations or []
        self.terminated = terminated or []
        self.truncated = truncated or []
        self.info = info or []
        self.agent_versions = agent_versions or []
        self.worker_ids = worker_ids or []
        self.episodes = episodes or []

    def __len__(self) -> int:
        """
        Returns the number of experience steps.

        Returns:
            int: Number of steps.
        """
        return len(self.observations)

    def add_step(
        self,
        observation: Any,
        action: Any,
        reward: Any,
        next_observation: Any,
        terminated: bool,
        truncated: bool,
        info: Dict[str, Any],
        agent_version: int,
        worker_id: int,
        episode: int,
    ) -> None:
        """
        Add a single step to the experience buffer.

        Args:
            observation (Any): Observation.
            action (Any): Action.
            reward (Any): Reward.
            next_observation (Any): Next observation.
            terminated (bool): Termination flag.
            truncated (bool): Truncation flag.
            info (Dict[str, Any]): Info dict.
            agent_version (int): Agent version.
            worker_id (int): Worker ID.
            episode (int): Episode number.
        """
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_observations.append(next_observation)
        self.terminated.append(terminated)
        self.truncated.append(truncated)
        self.info.append(info)
        self.agent_versions.append(agent_version)
        self.worker_ids.append(worker_id)
        self.episodes.append(episode)

    def clear(self) -> None:
        """
        Clear all stored experience.
        """
        for attr in (
            self.observations,
            self.actions,
            self.rewards,
            self.next_observations,
            self.terminated,
            self.truncated,
            self.info,
            self.agent_versions,
            self.worker_ids,
            self.episodes,
        ):
            attr.clear()

    def add_experience(self, exp: Self) -> None:
        """
        Add another Experience object to this one (concatenate).

        Args:
            exp (Experience): Another experience object.
        """
        for attr in [
            "observations",
            "actions",
            "rewards",
            "next_observations",
            "terminated",
            "truncated",
            "info",
            "agent_versions",
            "worker_ids",
            "episodes",
        ]:
            getattr(self, attr).extend(getattr(exp, attr))

    def copy(self) -> Self:
        """
        Return a deep copy of the experience.

        Returns:
            Experience: A copy of this experience.
        """
        return Experience(
            self.observations.copy(),
            self.actions.copy(),
            self.rewards.copy(),
            self.next_observations.copy(),
            self.terminated.copy(),
            self.truncated.copy(),
            self.info.copy(),
            self.agent_versions.copy(),
            self.worker_ids.copy(),
            self.episodes.copy(),
        )

    def get(self, columns: Optional[List[str]] = None) -> Tuple:
        """
        Get experience as a tuple of lists for specified columns.

        Args:
            columns (Optional[List[str]]): List of column names to return. If None, returns default columns.
        Returns:
            Tuple: Tuple of lists for each requested column.
        """
        data = {
            "observations": self.observations,
            "actions": self.actions,
            "rewards": self.rewards,
            "next_observations": self.next_observations,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "info": self.info,
            "agent_versions": self.agent_versions,
            "worker_ids": self.worker_ids,
            "episodes": self.episodes,
        }
        if columns is None:
            columns = [
                "observations",
                "actions",
                "rewards",
                "next_observations",
                "terminated",
                "truncated",
                "info",
            ]
        return tuple(data[col] for col in columns if col in data)

    def get_experience_batch(self, size: int = None) -> Self:
        """
        Get the last `size` steps as a new Experience object.

        Args:
            size (int, optional): Number of steps to include. If None, includes all.
        Returns:
            Experience: New experience object with the last `size` steps.
        """
        if size is None:
            size = len(self)
        return Experience(
            self.observations[-size:],
            self.actions[-size:],
            self.rewards[-size:],
            self.next_observations[-size:],
            self.terminated[-size:],
            self.truncated[-size:],
            self.info[-size:],
            self.agent_versions[-size:],
            self.worker_ids[-size:],
            self.episodes[-size:],
        )
