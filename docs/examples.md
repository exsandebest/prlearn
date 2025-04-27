# PRLearn Usage Examples

This document provides practical usage scenarios and code examples for PRLearn. It covers basic and advanced parallel RL workflows, as well as how to define your own agents, environments, and combiners.

## Basic Scenarios

### Base Imports

```python
from prlearn import Trainer
from my_env import MyEnv
from my_agent import MyAgent

env = MyEnv()
agent = MyAgent()
```

### Simple Training (Single Process)

```python
trainer = Trainer(
    agent=agent,
    env=env,
    schedule=[
        ("finish", 1000, "episodes"),
        ("train_agent", 10, "episodes")
    ]
)

# Run the trainer
agent, result = trainer.run()
```

### Parallel Data Collection (No Parallel Training)

```python
trainer = Trainer(
    agent=agent,
    env=env,
    n_workers=4,
    schedule=[
        ("finish", 1000, "episodes")
    ]
)

# Run the trainer
agent, result = trainer.run()
```

### Parallel Data Collection with Periodic Training

```python
trainer = Trainer(
    agent=agent,
    env=env,
    n_workers=4,
    schedule=[
        ("finish", 1000, "episodes"),
        ("train_agent", 100, "episodes")
    ],
    sync_mode='sync'  # optional: synchronize workers
)

# Run the trainer
agent, result = trainer.run()
```

### Parallel Learning (Each Worker Trains Its Own Agent)

```python
from prlearn.collection.agent_combiners import FixedStatAgentCombiner

trainer = Trainer(
    agent=agent,
    env=env,
    n_workers=4,
    schedule=[
        ("finish", 1000, "episodes"),
        ("train_agent", 10, "episodes"),
        ("combine_agents", 100, "episodes")
    ],
    mode="parallel_learning",
    combiner=FixedStatAgentCombiner("mean_reward"),
    sync_mode='sync'  # optional
)

# Run the trainer
agent, result = trainer.run()
```

### Parallel Learning with Multiple Agents

```python
from prlearn.collection.agent_combiners import FixedStatAgentCombiner

trainer = Trainer(
    agent=[agent, agent, agent, agent],  # List of agents, one per worker
    env=env,
    n_workers=4,
    schedule=[
        ("finish", 1000, "episodes"),
        ("train_agent", 10, "episodes"),
        ("combine_agents", 100, "episodes")
    ],
    mode="parallel_learning",
    combiner=FixedStatAgentCombiner("max_reward"),
    sync_mode='sync'  # optional
)

# Run the trainer
agent, result = trainer.run()
```

---

## Defining Base Objects

### Custom Agent Example

```python
from prlearn import Agent, Experience
from typing import Any, Dict, Tuple
from my_model import Model

class MyAgent(Agent):
    def __init__(self):
        self.model = Model()

    def before(self):  # optional
        pass

    def action(self, state: Tuple[Any, Dict[str, Any]]) -> Any:
        observation, info = state
        # Implement your action selection logic here
        pass

    def train(self, experience: Experience):
        obs, actions, rewards, terminated, truncated, info = experience.get()
        # Implement your training logic here
        pass

    def after(self):  # optional
        pass

    def get(self) -> Any:  # optional
        return self.model.get_weights()

    def set(self, data: Any):  # optional
        self.model.set_weights(data)
```

### Custom Environment Example

```python
from prlearn import Environment
from typing import Any, Dict, Tuple

class MyEnv(Environment):
    def before(self):  # optional
        pass

    def reset(self) -> Tuple[Any, Dict[str, Any]]:
        # Reset environment state
        observation = [[1, 2], [3, 4]]
        info = {"info": "description"}
        return observation, info

    def step(self, action: Any) -> Tuple[Any, Any, bool, bool, Dict[str, Any]]:
        # Apply action and return new state, reward, done flags, and info
        observation = [[1, 2], [3, 4]]
        reward = 1
        terminated, truncated = False, False
        info = {"info": "description"}
        return observation, reward, terminated, truncated, info

    def after(self):  # optional
        pass
```

### Custom AgentCombiner Example

```python
import random
from prlearn import Agent, AgentCombiner
from typing import Any, Dict, List, Optional

class MyRandomAgentCombiner(AgentCombiner):
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)

    def combine(
        self,
        workers_agents: List[Agent],
        main_agent: Agent,
        workers_stats: Optional[List[Dict[str, Any]]] = None,
        main_agent_stats: Optional[Dict[str, Any]] = None
    ) -> Agent:
        # Example: randomly select one of the worker agents
        return random.choice(workers_agents)
```


## Example: Trainer Result Structure

After running the trainer, you receive the final agent and a result dictionary with detailed statistics:

```python
# ... Trainer initialization ...

agent, result = trainer.run()

print(result)

# Example output
{
    'workers': [
        {
            'id': 0,
            'agent': <MyAgent object>,
            'agent_version': 5,
            'experience': <Experience object>,
            'rewards': [98, 114],
            'total_episodes': 2,
            'total_steps': 100
        }
    ],
    'trainer': <Trainer object>
}
```