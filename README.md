# PRLearn

PRLearn is a Python library designed for **P**arallel **R**einforcement **Learn**ing. 
It leverages multiprocessing to streamline the training of RL agents, making it easier and more efficient to experiment and develop new RL approaches.

## Key Features

- **Simple and Flexible**: Easy-to-use API built on Gymnasium, enabling seamless integration with existing environments.
- **No Dependencies**: No mandatory dependencies, but optional use of multiprocess for enhanced parallelism.
- **Parallel Data Collection and Training**: Efficiently collects and processes data using parallel execution, reducing training time.
- **Agent Combination**: Combines multiple agents to enhance learning outcomes through methods like averaging, boosting performance and stability.

## Installation

Install PRLearn using pip:

```sh
pip install prlearn
```

or 

```sh
pip install prlearn[multiprocess]
```

## Quick Start

Here's a brief example to get you started with PRLearn:

### Define Your Agent

```python
from prlearn import Agent, Experience
from typing import Any, Dict, Tuple


class MyAgent(Agent):
    
    def action(self, state: Tuple[Any, Dict[str, Any]]) -> Any:
        observation, info = state
        # Define action logic
        pass

    def train(self, experience: Experience):
        obs, actions, rewards, terminated, truncated, info = experience.get()
        # Define training logic
        pass

    # May define optional methods:
    # before(), after(), get(), set(agent)
```

### Using Parallel Training

```python
import gymnasium as gym
from best_agent import MyBestAgent
from prlearn import Trainer
from prlearn.collection.agent_combiners import FixedStatAgentCombiner

# Define your environment and agent
env = gym.make("LunarLander-v2")
agent = MyBestAgent()

# Create and configure the trainer
trainer = Trainer(
    agent=agent,
    env=env,
    n_workers=4,
    schedule=[
        ("finish", 1000, "episodes"),
        ("train_agent", 10, "episodes"),
    ],
    mode="parallel_learning",  # optional
    sync_mode="sync",  # optional
    combiner=FixedStatAgentCombiner("mean_reward"),  # optional
)

# Run the trainer
agent, result = trainer.run()
```

- **Environment**: We use the `LunarLander-v2` environment from Gymnasium.
- **Agent**: `BestAgent` is a custom agent class you should define.
- **Trainer Configuration**: The trainer is configured with 4 parallel workers, a schedule that specifies training completion after 1000 episodes and agent training every 10 episodes. 
- Optional parameters include the mode `parallel_learning`, synchronization mode `sync`, and a combiner `FixedStatAgentCombiner` that averages agent rewards.

### Using Custom Environment

```python
from prlearn import Environment
from typing import Any, Dict, Tuple


class MyEnv(Environment):
    
    def reset(self) -> Tuple[Any, Dict[str, Any]]:
        # Define reset logic
        observation = [[1, 2], [3, 4]]
        info = {"info": "description"}
        return observation, info

    def step(self, action: Any) -> Tuple[Any, Any, bool, bool, Dict[str, Any]]:
        # Define step logic
        observation = [[1, 2], [3, 4]]
        reward = 1
        terminated, truncated = False, False
        info = {"info": "description"}
        return observation, reward, terminated, truncated, info
```

**See more usage examples and scenarios at [docs/examples.md](https://github.com/exsandebest/prlearn/blob/master/docs/examples.md)**

## License

Licensed under the MIT License. See the [LICENSE](https://github.com/exsandebest/prlearn/blob/master/LICENSE) file for more information.
