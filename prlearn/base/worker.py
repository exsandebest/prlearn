import os
from bisect import insort
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from copy import copy
from statistics import mean, median
from threading import Event, Lock
from typing import Any, Dict, List, Optional, Tuple, Union

from prlearn.base.agent import Agent
from prlearn.base.environment import Environment
from prlearn.base.experience import Experience
from prlearn.common.config import (
    BASE_DEBUG_WORKER_PRINT_STATS_EPISODES,
    BASE_LAST_X_EPISODES_STATS,
    BASE_QUEUE_RECEIVE_TIMEOUT,
)
from prlearn.common.dataclasses import (
    ExperienceData,
    MessageType,
    Mode,
    SnapshotAgentData,
    SyncMode,
    TrainerMessage,
    WorkerMessage,
)
from prlearn.common.pas import ProcessActionScheduler
from prlearn.utils.logger import get_logger
from prlearn.utils.message_utils import queue_receive, queue_send
from prlearn.utils.multiproc_lib import mp

logger = get_logger(__name__)


class Worker:
    def __init__(
        self,
        worker_id: int,
        env: Environment,
        agent: Agent,
        connection: Tuple[mp.Queue, mp.Queue],
        global_params: Dict[str, Any],
    ) -> None:
        """
        Initialize the Worker.

        Args:
            worker_id (int): Worker index.
            env (Environment): Environment object.
            agent (Agent): Agent object.
            connection (Tuple[mp.Queue, mp.Queue]): Queues for communication.
            global_params (Dict[str, Any]): Global parameters (mode, sync_mode, scheduler).
        """
        self.worker_id = worker_id
        self.env = env
        self.agent = agent
        self.conn_in, self.conn_out = connection
        self.pid = os.getpid()
        self.stats = self._initialize_stats()
        self.total_steps = 0
        self.total_episodes = 0
        self.agent_version = 0
        self.rewards: List[Union[int, float]] = []
        self._reward_sum: Union[int, float] = 0
        self._sorted_rewards: List[Union[int, float]] = []
        self._recent_rewards: deque = deque(
            maxlen=BASE_LAST_X_EPISODES_STATS
        )
        self._stats_rewards_count = 0
        self.global_params = global_params
        self.mode: Mode = global_params["mode"]
        self.experience = Experience()
        self.agent_store = None
        self.agent_store_version = 0
        self.store_agent_available = False
        self.stop_listener = False
        self.sync_mode: SyncMode = global_params["sync_mode"]
        self.scheduler: ProcessActionScheduler = global_params["scheduler"]
        self.agent_store_lock = Lock()
        self._agent_ready_event = Event()
        self.wait_new_agent: bool = self.sync_mode == SyncMode.SYNCHRONOUS and (
            (self.mode == Mode.PARALLEL_COLLECTING and self.scheduler.agent_training)
            or (
                self.mode == Mode.PARALLEL_LEARNING and self.scheduler.agent_combination
            )
        )

    def _initialize_stats(self) -> Dict[str, Optional[float]]:
        """
        Initialize the statistics dictionary.

        Returns:
            Dict[str, Optional[float]]: Initialized statistics dictionary.
        """
        keys = [
            "max_reward",
            "mean_reward",
            "median_reward",
            "min_reward",
            "max_x_episodes_reward",
            "mean_x_episodes_reward",
            "median_x_episodes_reward",
            "min_x_episodes_reward",
        ]
        return {k: None for k in keys}

    def _start(self) -> None:
        """
        Start the worker by sending a START message to the trainer and waiting for a response.
        """
        logger.debug(f"Worker {self.worker_id} ({self.pid}): Starting")
        trainer_message: TrainerMessage = queue_receive(self.conn_in)
        if trainer_message.type != MessageType.TRAINER_START:
            raise ValueError("Incorrect START message from Trainer to Worker")
        queue_send(self.conn_out, WorkerMessage(MessageType.WORKER_START))
        logger.debug(f"Worker {self.worker_id} ({self.pid}): Start complete")

    def _finish(self) -> None:
        """
        Finish the worker by sending a DONE message to the trainer and waiting for a response.
        """
        logger.debug(f"Worker {self.worker_id} ({self.pid}): Finishing")
        queue_send(
            self.conn_out, WorkerMessage(MessageType.WORKER_DONE, data=self.get())
        )
        while True:
            trainer_message: TrainerMessage = queue_receive(self.conn_in)
            if trainer_message and trainer_message.type == MessageType.TRAINER_DONE:
                break
            elif trainer_message:
                logger.warning(f"Unexpected message type: {trainer_message.type}")
        logger.debug(f"Worker {self.worker_id} ({self.pid}): Finish complete")

    def _update_stats(self) -> None:
        """
        Update the worker's statistics based on recent rewards.
        """
        if not self.rewards:
            return
        if len(self.rewards) != len(self._sorted_rewards):
            self._reward_sum = sum(self.rewards)
            self._sorted_rewards = sorted(self.rewards)
            self._recent_rewards = deque(
                self.rewards[-BASE_LAST_X_EPISODES_STATS :],
                maxlen=BASE_LAST_X_EPISODES_STATS,
            )
        if len(self.rewards) == self._stats_rewards_count:
            return
        self._stats_rewards_count = len(self.rewards)
        last_x_episodes = BASE_LAST_X_EPISODES_STATS
        recent_rewards = list(self._recent_rewards)
        total_rewards = len(self._sorted_rewards)
        mid = total_rewards // 2
        if total_rewards % 2:
            median_reward = self._sorted_rewards[mid]
        else:
            median_reward = (
                self._sorted_rewards[mid - 1] + self._sorted_rewards[mid]
            ) / 2
        self.stats.update(
            {
                "max_reward": self._sorted_rewards[-1],
                "mean_reward": self._reward_sum / total_rewards,
                "median_reward": median_reward,
                "min_reward": self._sorted_rewards[0],
                "max_x_episodes_reward": max(recent_rewards),
                "mean_x_episodes_reward": mean(recent_rewards),
                "median_x_episodes_reward": median(recent_rewards),
                "min_x_episodes_reward": min(recent_rewards),
            }
        )

    def _print_stats(self) -> None:
        """
        Print the current statistics of the worker.
        """
        logger.debug(f"Worker {self.worker_id} ({self.pid}): Stats: {self.stats}")

    def _get_new_agent(self) -> None:
        """
        Get a new agent from the agent store, optionally waiting if the agent is not yet available.
        """
        if self.store_agent_available:
            with self.agent_store_lock:
                if hasattr(self.agent, "set"):
                    self.agent.set(self.agent_store)
                else:
                    self.agent = self.agent_store
                self.agent_version = self.agent_store_version
                self.store_agent_available = False
                self._agent_ready_event.clear()
            logger.debug(
                f"Worker {self.worker_id}: New agent (version {self.agent_version}) acquired"
            )
            return
        if self.wait_new_agent:
            self._agent_ready_event.wait()
            self._get_new_agent()

    def _run_listener(self) -> None:
        """
        Run the listener to receive messages from the trainer.
        """
        while not self.stop_listener:
            trainer_message: TrainerMessage = queue_receive(
                self.conn_in, timeout=BASE_QUEUE_RECEIVE_TIMEOUT
            )
            if not trainer_message:
                continue
            if trainer_message.type == MessageType.TRAINER_AGENT:
                with self.agent_store_lock:
                    self.agent_store = trainer_message.data.agent
                    self.agent_store_version = trainer_message.data.agent_version
                    self.store_agent_available = True
                    self._agent_ready_event.set()
            else:
                logger.warning(
                    f"Worker {self.worker_id}: Unexpected message type: {trainer_message.type}"
                )

    def _parallel_learning_step(self) -> None:
        """
        Perform a parallel learning step by training the agent with a batch of experiences.
        """
        pas_diffs = self.scheduler.check_agent_train(
            self.total_steps, self.total_episodes
        )
        if pas_diffs:
            exp = self.experience.get_experience_batch(pas_diffs["steps"])
            self.agent.train(exp)
            self.agent_version += 1

    def _send_message(self) -> None:
        """
        Send a message from the worker to the trainer, including agent data or experience data.
        """
        pas_diffs = self.scheduler.check_worker_send(
            self.total_steps, self.total_episodes
        )
        if not pas_diffs:
            return
        self._update_stats()
        if self.mode == Mode.PARALLEL_LEARNING:
            message_type = MessageType.WORKER_AGENT
            data = SnapshotAgentData(
                agent_version=self.agent_version,
                agent=self.agent.get() if hasattr(self.agent, "get") else self.agent,
                n_steps=pas_diffs["steps"],
                n_total_steps=self.total_steps,
                n_episodes=pas_diffs["episodes"],
                n_total_episodes=self.total_episodes,
                rewards=self.rewards[-pas_diffs["episodes"] :],
                stats=self.stats,
            )
        else:
            message_type = MessageType.WORKER_EXPERIENCE
            data = ExperienceData(
                agent_version=self.agent_version,
                n_steps=pas_diffs["steps"],
                n_total_steps=self.total_steps,
                n_episodes=pas_diffs["episodes"],
                n_total_episodes=self.total_episodes,
                experience=self.experience.get_experience_batch(pas_diffs["steps"]),
                rewards=self.rewards[-pas_diffs["episodes"] :],
                stats=self.stats,
            )
        queue_send(self.conn_out, WorkerMessage(type=message_type, data=data))
        self._get_new_agent()

    def _run_simulation(self) -> None:
        """
        Run the simulation, performing actions and collecting experiences.
        """
        if hasattr(self.env, "before"):
            self.env.before()
        if hasattr(self.agent, "before"):
            self.agent.before()
        episode_reward = 0
        observation, info = self.env.reset()
        n_steps = 0
        self.scheduler.set_time()
        while True:
            action = self.agent.action((observation, info))
            next_observation, reward, terminated, truncated, info = self.env.step(
                action
            )
            self.experience.add_step(
                observation=copy(observation),
                action=action,
                reward=reward,
                next_observation=next_observation,
                terminated=terminated,
                truncated=truncated,
                info=info,
                agent_version=self.agent_version,
                worker_id=self.worker_id,
                episode=self.total_episodes + 1,
            )
            episode_reward += reward
            observation = next_observation
            n_steps += 1
            self.total_steps += 1
            if terminated or truncated:
                observation, info = self.env.reset()
                self.rewards.append(episode_reward)
                self._reward_sum += episode_reward
                insort(self._sorted_rewards, episode_reward)
                self._recent_rewards.append(episode_reward)
                self.total_episodes += 1
                if self.total_episodes % BASE_DEBUG_WORKER_PRINT_STATS_EPISODES == 0:
                    self._update_stats()
                    logger.debug(
                        f"{self.worker_id} ({self.pid}): Episode {self.total_episodes} done (rew: {episode_reward}) "
                        f"after {n_steps} steps (total: {self.total_steps}))"
                    )
                    self._print_stats()
                n_steps = 0
                episode_reward = 0
            if self.mode == Mode.PARALLEL_LEARNING:
                self._parallel_learning_step()
            self._send_message()
            if self.scheduler.check_train_finish(
                n_steps=self.total_steps,
                n_episodes=self.total_episodes,
            ):
                break
        if hasattr(self.env, "after"):
            self.env.after()
        if hasattr(self.agent, "after"):
            self.agent.after()

    def get(self) -> Dict[str, Any]:
        """
        Get the current state of the worker.

        Returns:
            Dict[str, Any]: State dictionary with agent, experience, rewards, etc.
        """
        return {
            "id": self.worker_id,
            "agent": self.agent,
            "agent_version": self.agent_version,
            "experience": self.experience,
            "rewards": self.rewards,
            "total_episodes": self.total_episodes,
            "total_steps": self.total_steps,
        }

    def run(self) -> "Worker":
        """
        Run the worker, starting the listener and the simulation.

        Returns:
            Worker: The instance of the Worker class.
        """
        self._start()
        executor = ThreadPoolExecutor(max_workers=1)
        executor.submit(self._run_listener)
        self._run_simulation()
        self.stop_listener = True
        executor.shutdown(cancel_futures=True)
        self._finish()
        return self
