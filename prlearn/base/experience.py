import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar

T = TypeVar("T", bound="Experience")


EXPERIENCE_FIELDS = [
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
]


@dataclass
class Experience:
    """
    Container for storing and manipulating experience tuples in RL.
    Implemented as a dataclass for conciseness and serialization support.

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

    observations: List[Any] = field(default_factory=list)
    actions: List[Any] = field(default_factory=list)
    rewards: List[Any] = field(default_factory=list)
    next_observations: List[Any] = field(default_factory=list)
    terminated: List[bool] = field(default_factory=list)
    truncated: List[bool] = field(default_factory=list)
    info: List[Dict[str, Any]] = field(default_factory=list)
    agent_versions: List[int] = field(default_factory=list)
    worker_ids: List[int] = field(default_factory=list)
    episodes: List[int] = field(default_factory=list)

    def __post_init__(self):
        # Validate that all non-empty fields have the same length
        lengths = [
            len(getattr(self, field.name))
            for field in self.__dataclass_fields__.values()
        ]
        nonzero_lengths = [length for length in lengths if length > 0]
        if nonzero_lengths and len(set(nonzero_lengths)) > 1:
            raise ValueError(
                f"All fields in Experience must have the same length or be empty, got lengths: {lengths}"
            )

    def __len__(self) -> int:
        """
        Returns the number of experience steps.

        Returns:
            int: Number of steps in the experience buffer.
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
        Clear all stored experience from the buffer.
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

    def add_experience(self, exp: "Experience") -> None:
        """
        Concatenate another Experience object to this one.

        Args:
            exp (Experience): Another experience object to concatenate.
        """
        for attr in EXPERIENCE_FIELDS:
            getattr(self, attr).extend(getattr(exp, attr))

    def copy(self) -> "Experience":
        """
        Return a deep copy of the experience buffer.

        Returns:
            Experience: A copy of this experience buffer.
        """
        return Experience(*[getattr(self, attr).copy() for attr in EXPERIENCE_FIELDS])

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

    def get_experience_batch(self, size: Optional[int] = None) -> "Experience":
        """
        Get the last `size` steps as a new Experience object.

        Args:
            size (int, optional): Number of steps to include. If None, includes all.
        Returns:
            Experience: New experience object with the last `size` steps.
        """
        if size is None:
            size = len(self)
        elif not isinstance(size, int):
            raise TypeError("Parameter 'size' must be an integer or None.")
        elif size < 0:
            raise ValueError("Parameter 'size' must be non-negative.")
        elif size == 0:
            return Experience()
        return Experience(
            *[getattr(self, attr)[-size:] for attr in EXPERIENCE_FIELDS]
        )

    def pop_experience_batch(self, size: Optional[int] = None) -> "Experience":
        """
        Remove and return the last `size` steps as a new Experience object.

        Args:
            size (int, optional): Number of steps to include. If None or greater
                than the available steps, removes and returns all.
        Returns:
            Experience: New experience object with the removed steps.
        """
        current_len = len(self)
        if not current_len:
            return Experience()

        if size is None or size >= current_len:
            batch = self.copy()
            self.clear()
            return batch

        slice_idx = slice(-size, None)
        batch_data = []
        for attr in EXPERIENCE_FIELDS:
            src = getattr(self, attr)
            batch_data.append(src[slice_idx])
            del src[slice_idx]
        return Experience(*batch_data)

    def to_dict(self) -> dict:
        """
        Convert the Experience object to a dictionary.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> T:
        """
        Create an Experience object from a dictionary.
        """
        return cls(**data)

    def to_json(self) -> str:
        """
        Serialize the Experience object to a JSON string.
        """
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls: Type[T], s: str) -> T:
        """
        Deserialize a JSON string to an Experience object.
        """
        return cls.from_dict(json.loads(s))
