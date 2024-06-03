from enum import Enum
from typing import Any, Dict, List, NamedTuple, Optional

from prlearn.base.experience import Experience


class MessageType(Enum):
    TRAINER_AGENT = "trainer_agent"
    TRAINER_START = "trainer_start"
    TRAINER_DONE = "trainer_done"

    WORKER_EXPERIENCE = "worker_experience"
    WORKER_AGENT = "worker_agent"
    WORKER_START = "worker_start"
    WORKER_DONE = "worker_done"


class Mode(Enum):
    PARALLEL_COLLECTING = "parallel_collecting"
    PARALLEL_LEARNING = "parallel_learning"


class SyncMode(Enum):
    SYNCHRONOUS = "sync"
    ASYNCHRONOUS = "async"


class NewAgentData(NamedTuple):
    agent_version: int
    agent: Any


class SnapshotAgentData(NamedTuple):
    agent_version: int
    agent: Any
    n_steps: int
    n_total_steps: int
    n_episodes: int
    n_total_episodes: int
    rewards: Optional[List[Any]] = None
    stats: Optional[Dict[str, Any]] = None


class ExperienceData(NamedTuple):
    agent_version: int
    n_steps: int
    n_total_steps: int
    n_episodes: int
    n_total_episodes: int
    experience: Experience
    rewards: Optional[List[Any]] = None
    stats: Optional[Dict[str, Any]] = None


class WorkerMessage(NamedTuple):
    type: MessageType
    data: Optional[ExperienceData | SnapshotAgentData] = None


class TrainerMessage(NamedTuple):
    type: MessageType
    data: Optional[NewAgentData] = None
