from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

from prlearn.base.agent import Agent
from prlearn.base.agent_combiner import AgentCombiner
from prlearn.base.environment import Environment
from prlearn.base.experience import Experience
from prlearn.base.worker import Worker
from prlearn.collection.agent_combiners import FixedAgentCombiner, RandomAgentCombiner
from prlearn.common.config import BASE_QUEUE_RECEIVE_TIMEOUT
from prlearn.common.dataclasses import (
    ExperienceData,
    MessageType,
    Mode,
    NewAgentData,
    QueueConn,
    SnapshotAgentData,
    SyncMode,
    TrainerMessage,
    WorkerMessage,
)
from prlearn.common.pas import ProcessActionScheduler
from prlearn.utils.logger import get_logger
from prlearn.utils.message_utils import queue_receive, queue_send, try_queue_send
from prlearn.utils.multiproc_lib import mp

logger = get_logger(__name__)


class Trainer:
    """
    Trainer class to manage the training process of an agent in a distributed environment.

    The Trainer class is responsible for coordinating multiple workers that perform
    parallel data collection and training tasks. It handles the distribution of tasks,
    the collection of experience data from workers, and the aggregation of training
    results.
    """

    def __init__(
        self,
        agent: Agent | List[Agent],
        env: Environment,
        n_workers: int = 1,
        mode: str = "parallel_collecting",
        combiner: Optional[AgentCombiner] = None,
        schedule: Optional[List[Tuple[str, float, str]]] = None,
        sync_mode: str = "async",
    ):
        """
        Initialize the Trainer.

        Args:
            agent (Agent | List[Agent]): Agent object or list of Agents if mode is "parallel_collecting".
            env (Environment): Environment object or a list of Environment objects.
            n_workers (int): Number of workers.
            mode (str): Mode of operation ("parallel_collecting" or "parallel_learning").
            combiner (Optional[AgentCombiner]): AgentCombiner object, required if mode is "parallel_learning".
            schedule (Optional[List[Tuple[str, float, str]]]): Optional schedule for the ProcessActionScheduler.
            sync_mode (str): Synchronization mode ("sync" or "async").
        """
        if not isinstance(n_workers, int) or n_workers <= 0:
            raise ValueError("The 'n_workers' parameter must be a positive integer.")
        self.n_workers = n_workers

        try:
            self.mode = Mode(mode)
        except ValueError:
            raise ValueError(
                "The 'mode' parameter must be a string of one of the following values:"
                f"{[e.value for e in Mode]}."
            )

        try:
            self.sync_mode = SyncMode(sync_mode)
        except ValueError:
            raise ValueError(
                "The 'sync_mode' parameter must be a string of one of the following values:"
                f"{[e.value for e in SyncMode]}."
            )

        self.use_multiple_agents = isinstance(agent, list)

        if self.use_multiple_agents:
            if len(agent) != n_workers:
                raise ValueError(
                    "The length of the 'agent' list must be equal to 'num_workers'."
                )
            if self.mode != Mode.PARALLEL_LEARNING:
                raise ValueError(
                    f"Multiple agents can be provided only in '{Mode.PARALLEL_LEARNING.value}' mode."
                )
            self.agent = agent[0]
            self.start_agents = agent
        else:
            self.agent = agent

        if schedule and not all(
            isinstance(item, tuple) and len(item) == 3 for item in schedule
        ):
            raise ValueError(
                "The 'schedule' parameter must be a list of tuples with three elements each."
            )
        self.schedule_config = schedule

        if self.mode == Mode.PARALLEL_COLLECTING and n_workers == 1:
            logger.info(
                "Parameter n_workers is set to 1. Using parallel learning mode."
            )
            self.mode = Mode.PARALLEL_LEARNING

        if self.mode == Mode.PARALLEL_LEARNING and not combiner:
            logger.info("Parameter combiner is set to None.")
            if n_workers == 1:
                logger.info("Using FixedAgentCombiner(0).")
                combiner = FixedAgentCombiner(0)
            else:
                logger.info("Using RandomAgentCombiner(0).")
                combiner = RandomAgentCombiner(0)
        self.combiner = combiner

        self.env = env
        self.agent_version = 0
        self.experience_lock = Lock()
        self.experience = Experience()
        self.scheduler = ProcessActionScheduler(
            self.schedule_config, n_workers=self.n_workers, mode=self.mode
        )

        self.workers_processes = []
        self.workers_queues = []
        self.workers_steps = [0] * n_workers
        self.workers_episodes = [0] * n_workers
        self.workers_agent_versions = [0] * n_workers
        self.workers_agents = (
            self.start_agents if self.use_multiple_agents else [self.agent] * n_workers
        )
        self.workers_stats = [{}] * n_workers
        self.workers_rewards = [[] for _ in range(n_workers)]
        self.workers_finished = [False] * n_workers
        self.workers_done_accepted = [False] * n_workers
        self.workers_messages = [[] for _ in range(n_workers)]
        self.workers_results = [None for _ in range(n_workers)]

    def _update_worker_data(self, worker_index: int, data: Any) -> None:
        """
        Update the worker data with the provided information.

        Args:
            worker_index (int): Index of the worker.
            data (Any): Data received from the worker.
        """
        self.workers_episodes[worker_index] = data.n_total_episodes
        self.workers_steps[worker_index] = data.n_total_steps
        self.workers_agent_versions[worker_index] = data.agent_version
        self.workers_stats[worker_index] = data.stats
        self.workers_rewards[worker_index].extend(data.rewards)

    def _run_worker_handler(self, worker_index: int, queue: mp.Queue) -> None:
        """
        Handle messages from a worker.

        Args:
            worker_index (int): Index of the worker.
            queue (mp.Queue): Queue for receiving messages from the worker.
        """
        logger.debug(f"Starting worker handler for worker {worker_index}")
        while True:
            worker_message: WorkerMessage = queue_receive(queue)
            if worker_message and worker_message.type == MessageType.WORKER_START:
                logger.debug(f"Worker {worker_index} received START message")
                break
        while True:
            worker_message: WorkerMessage = queue_receive(
                queue, timeout=BASE_QUEUE_RECEIVE_TIMEOUT
            )
            if not worker_message:
                continue
            if worker_message.type == MessageType.WORKER_EXPERIENCE:
                data: ExperienceData = worker_message.data
                with self.experience_lock:
                    self.experience.add_experience(data.experience)
                self._update_worker_data(worker_index, data)
            elif worker_message.type == MessageType.WORKER_AGENT:
                data: SnapshotAgentData = worker_message.data
                self._update_worker_data(worker_index, data)
            elif worker_message.type == MessageType.WORKER_DONE:
                logger.debug(f"Worker {worker_index} received DONE message")
                self.workers_finished[worker_index] = True
                self.workers_results[worker_index] = worker_message.data
                break
        logger.debug(f"Worker handler for worker {worker_index} done")

    def _run_worker(
        self,
        idx: int,
        env: Environment,
        agent: Agent,
        connection: Tuple[mp.Queue, mp.Queue],
        global_params: Dict[str, Any],
    ) -> Worker:
        """
        Run a worker.

        Args:
            idx (int): Index of the worker.
            env (Environment): Environment object.
            agent (Agent): Agent object.
            connection (Tuple[mp.Queue, mp.Queue]): Queues for communication.
            global_params (Dict[str, Any]): Global parameters.
        Returns:
            int: Result of the worker's run method.
        """
        worker = Worker(idx, env, agent, connection, global_params)
        return worker.run()

    def _send_agent_to_workers(self, agent: Agent) -> None:
        """
        Send the updated agent to all active workers.

        Args:
            agent (Agent): The agent to be sent to workers.
        """
        trainer_message = TrainerMessage(
            type=MessageType.TRAINER_AGENT,
            data=NewAgentData(agent_version=self.agent_version, agent=agent),
        )
        for i in range(self.n_workers):
            if self.workers_processes[i].is_alive() and not self.workers_finished[i]:
                try_queue_send(
                    self.workers_queues[i].parent_to_child_queue, trainer_message
                )

    def _train_agent(self, total_steps: int, total_episodes: int) -> Optional[Agent]:
        """
        Train the agent based on the collected experience.

        Args:
            total_steps (int): Total number of steps taken by all workers.
            total_episodes (int): Total number of episodes completed by all workers.
        Returns:
            Optional[Agent]: The updated agent after training.
        """
        pas_diffs = self.scheduler.check_agent_train(total_steps, total_episodes)
        if pas_diffs:
            logger.debug(
                f"Training agent: Steps: {total_steps}, Episodes: {total_episodes}"
            )
            with self.experience_lock:
                exp_batch = self.experience.get_experience_batch()
                self.experience.clear()
            self.agent.train(exp_batch)
            self.agent_version += 1
            return self.agent.get() if hasattr(self.agent, "get") else self.agent

    def _combine_agents(self, total_steps: int, total_episodes: int) -> Optional[Agent]:
        """
        Combine agents from different workers into a single model.

        Args:
            total_steps (int): Total number of steps taken by all workers.
            total_episodes (int): Total number of episodes completed by all workers.
        Returns:
            Optional[Agent]: The combined agent after aggregation.
        """
        pas_diffs = self.scheduler.check_combine_agents(total_steps, total_episodes)
        if pas_diffs:
            new_agent = self.combiner.combine(
                self.workers_agents,
                self.agent.get() if hasattr(self.agent, "get") else self.agent,
                self.workers_stats,
            )
            self.agent_version = max(self.workers_agent_versions) + 1
            if hasattr(self.agent, "set"):
                self.agent = new_agent
            return new_agent

    def run(self) -> Tuple[Agent, Dict[str, Any]]:
        """
        Run the training process.

        This method initializes the worker processes, sends start messages to them,
        and coordinates the collection and processing of experience data. It also
        manages the synchronization and updating of agents among workers.

        The method performs the following steps:
        1. Initializes multiprocessing context and queues for worker communication.
        2. Starts worker processes and submits worker handlers to a ThreadPoolExecutor.
        3. Sends a start message to all workers.
        4. Enters a loop to monitor the progress of workers and manage agent updates.
        5. Waits for all workers to complete their tasks.
        6. Collects results from worker processes and returns the trained agent and worker results.

        Returns:
            Tuple[Agent, Dict[str, Any]]: The trained agent and a dictionary with worker and trainer results.
        """
        logger.info("Launching processes...")
        logger.debug("Creating queues")
        mp_context = mp.get_context("spawn")
        self.workers_queues = [
            QueueConn(mp_context.Queue(), mp_context.Queue())
            for _ in range(self.n_workers)
        ]
        logger.debug("Starting processes")
        for i in range(self.n_workers):
            process = mp_context.Process(
                target=self._run_worker,
                args=(
                    i,
                    self.env,
                    self.start_agents[i] if self.use_multiple_agents else self.agent,
                    (
                        self.workers_queues[i].parent_to_child_queue,
                        self.workers_queues[i].child_to_parent_queue,
                    ),
                    {
                        "mode": self.mode,
                        "sync_mode": self.sync_mode,
                        "scheduler": self.scheduler,
                    },
                ),
            )
            process.start()
            self.workers_processes.append(process)
        logger.debug("Starting workers handlers")
        executor = ThreadPoolExecutor(self.n_workers)
        futures = [
            executor.submit(
                self._run_worker_handler,
                i,
                self.workers_queues[i].child_to_parent_queue,
            )
            for i in range(self.n_workers)
        ]
        logger.debug("Sending START messages to all workers")
        for queue in self.workers_queues:
            queue_send(
                queue.parent_to_child_queue, TrainerMessage(MessageType.TRAINER_START)
            )
        self.scheduler.set_time()
        logger.info("Main process started")
        while True:
            current_total_steps = sum(self.workers_steps)
            current_total_episodes = sum(self.workers_episodes)
            agent_data = (
                self._train_agent(current_total_steps, current_total_episodes)
                if self.mode == Mode.PARALLEL_COLLECTING
                else self._combine_agents(current_total_steps, current_total_episodes)
            )
            if agent_data:
                self._send_agent_to_workers(agent_data)
            for i in range(self.n_workers):
                if self.workers_finished[i] and not self.workers_done_accepted[i]:
                    logger.debug(f"Sending DONE to Worker {i}")
                    queue_send(
                        self.workers_queues[i].parent_to_child_queue,
                        TrainerMessage(MessageType.TRAINER_DONE),
                    )
                    self.workers_done_accepted[i] = True
            if all(self.workers_finished) and all(self.workers_done_accepted):
                logger.debug("All workers are done")
                break
        logger.info("Main process finished")
        for future in futures:
            future.result()
        for process in self.workers_processes:
            process.join()
        results = {
            "workers": self.workers_results,
            "trainer": self,
        }
        result_agent = (
            self.agent if self.n_workers > 1 else self.workers_results[0]["agent"]
        )
        return result_agent, results
