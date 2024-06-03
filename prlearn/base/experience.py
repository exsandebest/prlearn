from typing import Any, Dict, List, Optional, Self, Tuple


class Experience:
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

    def __len__(self):
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
    ):
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

    def clear(self):
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_observations.clear()
        self.terminated.clear()
        self.truncated.clear()
        self.info.clear()
        self.agent_versions.clear()
        self.worker_ids.clear()
        self.episodes.clear()

    def add_experience(self, exp: Self):
        self.observations.extend(exp.observations)
        self.actions.extend(exp.actions)
        self.rewards.extend(exp.rewards)
        self.next_observations.extend(exp.next_observations)
        self.terminated.extend(exp.terminated)
        self.truncated.extend(exp.truncated)
        self.info.extend(exp.info)
        self.agent_versions.extend(exp.agent_versions)
        self.worker_ids.extend(exp.worker_ids)
        self.episodes.extend(exp.episodes)

    def copy(self) -> Self:
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

    def get_experience_batch(self, size: int) -> Self:
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
