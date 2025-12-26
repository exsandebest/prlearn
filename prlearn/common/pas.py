import time
from typing import Dict, List, Literal, Optional, Tuple, Union

from prlearn.common.dataclasses import Mode


PAS_ACTION_TRAIN_AGENT = "train_agent"
PAS_ACTION_WORKER_SEND = "worker_send_data"
PAS_ACTION_FINISH = "finish"
PAS_ACTION_COMBINE_AGENTS = "combine_agents"
PAS_ACTIONS = (
    PAS_ACTION_TRAIN_AGENT,
    PAS_ACTION_WORKER_SEND,
    PAS_ACTION_FINISH,
    PAS_ACTION_COMBINE_AGENTS,
)
PAS_UNITS = ("steps", "episodes", "seconds")
ActionName = Literal[
    "train_agent", "worker_send_data", "finish", "combine_agents"
]


def validate_schedule_config(
    config: Optional[List[Tuple[str, int | float, str]]],
) -> None:
    """
    Validate the schedule configuration for ProcessActionScheduler.

    Args:
        config (Optional[List[Tuple[str, int | float, str]]]): Schedule to validate.
    Raises:
        ValueError: If the schedule is not valid.
    """
    if config is None:
        return
    for idx, item in enumerate(config):
        if not isinstance(item, tuple) or len(item) != 3:
            raise ValueError(
                f"Schedule item at index {idx} must be a tuple of (str, int|float, str), got: {item}"
            )
        action, interval, units = item
        if not isinstance(action, str) or action not in PAS_ACTIONS:
            raise ValueError(
                f"Schedule item at index {idx}: action '{action}' is not valid. Allowed: {PAS_ACTIONS}"
            )
        if not isinstance(interval, (int, float)) or interval <= 0:
            raise ValueError(
                f"Schedule item at index {idx}: interval '{interval}' must be a positive number."
            )
        if not isinstance(units, str) or units not in PAS_UNITS:
            raise ValueError(
                f"Schedule item at index {idx}: units '{units}' is not valid. Allowed: {PAS_UNITS}"
            )


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
        config: Optional[List[Tuple[ActionName, int | float, str]]] = None,
        mode: Mode = Mode.PARALLEL_COLLECTING,
        n_workers: int = 1,
    ):
        """
        Initialize the ProcessActionScheduler.

        Args:
            config (Optional[List[Tuple[str, int | float, str]]]): Scheduling configuration.
            mode (Mode): Mode of operation.
            n_workers (int): Number of workers.
        """
        validate_schedule_config(config)
        self.possible_actions = list(PAS_ACTIONS)

        interval_template: Dict[str, Optional[Union[int, float]]] = {
            "seconds_interval": None,
            "steps_interval": None,
            "episodes_interval": None,
        }
        self.config: Dict[str, Dict[str, Optional[Union[int, float]]]] = {
            action: interval_template.copy() for action in self.possible_actions
        }

        now = time.time()
        self.state: Dict[str, Dict[str, Union[int, float]]] = {
            action: {"steps": 0, "episodes": 0, "seconds": now}
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
            if units not in PAS_UNITS:
                raise ValueError(
                    f"Invalid units '{units}' for action '{action}'. Must be one of {PAS_UNITS}"
                )
            self.config[action][f"{units}_interval"] = interval

        self.agent_combination = any(self.config["combine_agents"].values())
        self.agent_training = any(self.config["train_agent"].values())

        if mode == Mode.PARALLEL_COLLECTING:
            for key in self.config["train_agent"].keys():
                if (
                    self.config["train_agent"][key] is not None
                    and self.config["worker_send_data"][key] is None
                ):
                    self.config["worker_send_data"][key] = (
                        self.config["train_agent"][key] / n_workers
                    )

        if mode == Mode.PARALLEL_LEARNING:
            for key in self.config["combine_agents"].keys():
                if (
                    self.config["combine_agents"][key] is not None
                    and self.config["worker_send_data"][key] is None
                ):
                    self.config["worker_send_data"][key] = (
                        self.config["combine_agents"][key] / n_workers
                    )

        for key in self.config["finish"].keys():
            if (
                self.config["finish"][key] is not None
                and self.config["worker_send_data"][key] is None
            ):
                self.config["worker_send_data"][key] = self.config["finish"][key]

    def set_time(self, seconds: float = None, action: ActionName | None = None) -> None:
        """
        Set the current time for all or a specific action.

        Args:
            seconds (float, optional): The time to set. Defaults to current time.
            action (str, optional): The action to set time for. If None, sets for all actions.
        """
        if seconds is None:
            seconds = time.time()
        if action is None:
            for act in self.possible_actions:
                self.state[act]["seconds"] = seconds
        else:
            self.state[action]["seconds"] = seconds

    def _update_state(
        self,
        action: ActionName,
        n_steps: int = 0,
        n_episodes: int = 0,
        check_time: float = None,
    ) -> None:
        """
        Update the state for a specific action.

        Args:
            action (str): The action to update.
            n_steps (int): Number of steps.
            n_episodes (int): Number of episodes.
            check_time (float): Time value.
        """
        self.state[action]["steps"] = n_steps
        self.state[action]["episodes"] = n_episodes
        self.state[action]["seconds"] = check_time

    def _get_state(self, action: ActionName) -> Dict[str, Union[int, float]]:
        """
        Get the state for a specific action.

        Args:
            action (str): The action to get state for.
        Returns:
            Dict[str, Union[int, float]]: State dictionary.
        """
        return self.state[action]

    def _get_diff(
        self,
        action: ActionName,
        n_steps: int = 0,
        n_episodes: int = 0,
        check_time: float = None,
    ) -> Dict[str, float]:
        """
        Get the difference in steps, episodes, and seconds for an action.

        Args:
            action (str): The action to check.
            n_steps (int): Number of steps.
            n_episodes (int): Number of episodes.
            check_time (float): Time value.
        Returns:
            Dict[str, float]: Differences for steps, episodes, seconds.
        """
        return {
            "steps": n_steps - self.state[action]["steps"],
            "episodes": n_episodes - self.state[action]["episodes"],
            "seconds": check_time - self.state[action]["seconds"],
        }

    def _update_with_diff(
        self,
        action: ActionName,
        n_steps: int = 0,
        n_episodes: int = 0,
        check_time: float = None,
    ) -> Dict[str, float]:
        """
        Update state and return the difference for an action.

        Args:
            action (str): The action to update.
            n_steps (int): Number of steps.
            n_episodes (int): Number of episodes.
            check_time (float): Time value.
        Returns:
            Dict[str, float]: Differences for steps, episodes, seconds.
        """
        diff = self._get_diff(action, n_steps, n_episodes, check_time)
        self._update_state(action, n_steps, n_episodes, check_time)
        return diff

    def check(
        self,
        action: ActionName,
        n_steps: int = 0,
        n_episodes: int = 0,
        check_time: float = None,
    ) -> Optional[Dict[str, float]]:
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
        state = self.state[action]
        intervals = (
            ("seconds_interval", check_time - state["seconds"]),
            ("steps_interval", n_steps - state["steps"]),
            ("episodes_interval", n_episodes - state["episodes"]),
        )
        for interval_key, diff in intervals:
            interval_limit = self.config[action][interval_key]
            if interval_limit is not None and diff >= interval_limit:
                return self._update_with_diff(
                    action, n_steps, n_episodes, check_time
                )
        return None
