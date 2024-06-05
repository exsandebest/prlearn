import time
from typing import Dict, List, Optional, Tuple, Union

from prlearn.common.dataclasses import Mode


class ProcessActionScheduler:
    """
    Class to manage the scheduling of various actions in parallel reinforcement learning.

    Args:
        config (Optional[List[Tuple[str, Union[int, float], str]]]): Configuration for scheduling actions.
            Each tuple should contain:
                - action (str): The name of the action (e.g., "train_agent").
                - interval (Union[int, float]): The interval for the action.
                - units (str): The units of the interval ("seconds", "steps", "episodes").
    Raises:
        ValueError: If any element of config is invalid.
    """

    def __init__(
            self,
            config: Optional[List[Tuple[str, int | float, str]]] = None,
            mode: Mode = Mode.PARALLEL_COLLECTING,
            n_workers: int = 1,
    ):
        self.possible_actions = [
            "train_agent",
            "worker_send_data",
            "finish",
            "combine_agents",
        ]

        self.config: Dict[str, Dict[str, Optional[Union[int, float]]]] = {
            action: {
                "seconds_interval": None,
                "steps_interval": None,
                "episodes_interval": None,
            }
            for action in self.possible_actions
        }

        self.state: Dict[str, Dict[str, Union[int, float]]] = {
            action: {"steps": 0, "episodes": 0, "seconds": time.time()}
            for action in self.possible_actions
        }

        self.input_config = config or []
        for item in self.input_config:
            if not isinstance(item, tuple) or len(item) != 3:
                raise ValueError(
                    f"Invalid config item '{item}'. Must be a tuple of (str, int|float, str)"
                )

            action, interval, units = item

            if action not in self.possible_actions:
                raise ValueError(
                    f"Invalid action '{action}'. Must be one of {self.possible_actions}"
                )
            if not isinstance(interval, (int, float)):
                raise ValueError(
                    f"Invalid interval '{interval}' for action '{action}'. Must be an int or float"
                )
            if units not in {"seconds", "steps", "episodes"}:
                raise ValueError(
                    f"Invalid units '{units}' for action '{action}'. Must be one of 'seconds', 'steps', 'episodes'"
                )

            self.config[action][f"{units}_interval"] = interval

        if mode == Mode.PARALLEL_COLLECTING:
            for key in self.config["train_agent"].keys():
                if self.config["train_agent"][key] is not None and self.config["worker_send_data"][key] is None:
                    self.config["worker_send_data"][key] = self.config["train_agent"][key] / n_workers

        if mode == Mode.PARALLEL_LEARNING:
            for key in self.config["combine_agents"].keys():
                if self.config["combine_agents"][key] is not None and self.config["worker_send_data"][key] is None:
                    self.config["worker_send_data"][key] = self.config["combine_agents"][key] / n_workers

            for key in self.config["train_agent"].keys():
                if self.config["train_agent"][key] is not None and self.config["worker_send_data"][key] is None:
                    self.config["worker_send_data"][key] = self.config["train_agent"][key] / n_workers

    def set_time(self, seconds: float = None, action: str = None):
        if seconds is None:
            seconds = time.time()
        if action is None:
            for action in self.possible_actions:
                self.state[action]["seconds"] = seconds
        else:
            self.state[action]["seconds"] = seconds

    def _update_state(
        self,
        action: str,
        n_steps: int = 0,
        n_episodes: int = 0,
        check_time: float = None,
    ):
        self.state[action]["steps"] = n_steps
        self.state[action]["episodes"] = n_episodes
        self.state[action]["seconds"] = check_time

    def _get_state(self, action):
        return self.state[action]

    def _get_diff(
        self,
        action: str,
        n_steps: int = 0,
        n_episodes: int = 0,
        check_time: float = None,
    ):
        return {
            "steps": n_steps - self.state[action]["steps"],
            "episodes": n_episodes - self.state[action]["episodes"],
            "seconds": check_time - self.state[action]["seconds"],
        }

    def _update_with_diff(
        self,
        action: str,
        n_steps: int = 0,
        n_episodes: int = 0,
        check_time: float = None,
    ):
        diff = self._get_diff(action, n_steps, n_episodes, check_time)
        self._update_state(action, n_steps, n_episodes, check_time)
        return diff

    def check(
        self,
        action: str,
        n_steps: int = 0,
        n_episodes: int = 0,
        check_time: float = None,
    ):
        """
        Check if the action should be executed based on the given intervals.

        Args:
            action (str): The action to check.
            n_steps (int): The number of steps taken.
            n_episodes (int): The number of episodes completed.
            check_time (Optional[float]): The current time in seconds.

        Returns:
            Optional[Dict[str, float]]: The differences if the action should be executed, otherwise None.

        Raises:
            ValueError: If action is not one of the possible actions.
        """

        if action not in self.possible_actions:
            raise ValueError(
                f"Invalid action '{action}'. Must be one of {self.possible_actions}"
            )

        if check_time is None:
            check_time = time.time()

        if (
            (
                self.config[action]["seconds_interval"] is not None
                and check_time - self.state[action]["seconds"]
                >= self.config[action]["seconds_interval"]
            )
            or (
                self.config[action]["steps_interval"] is not None
                and n_steps - self.state[action]["steps"]
                >= self.config[action]["steps_interval"]
            )
            or (
                self.config[action]["episodes_interval"] is not None
                and n_episodes - self.state[action]["episodes"]
                >= self.config[action]["episodes_interval"]
            )
        ):
            return self._update_with_diff(action, n_steps, n_episodes, check_time)
        return None

    def check_worker_send(
        self, n_steps: int = 0, n_episodes: int = 0, check_time: float = None
    ):
        return self.check("worker_send_data", n_steps, n_episodes, check_time)

    def check_agent_train(
        self, n_steps: int = 0, n_episodes: int = 0, check_time: float = None
    ):
        return self.check("train_agent", n_steps, n_episodes, check_time)

    def check_combine_agents(
        self, n_steps: int = 0, n_episodes: int = 0, check_time: float = None
    ):
        return self.check("combine_agents", n_steps, n_episodes, check_time)

    def check_train_finish(
        self, n_steps: int = 0, n_episodes: int = 0, check_time: float = None
    ):
        return self.check("finish", n_steps, n_episodes, check_time)
