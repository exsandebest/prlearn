## PRLearn Usage

### Scenarios

#### Base imports

```python
from prlearn import Trainer
from my_env import MyEnv
from my_agent import MyAgent

env = MyEnv()
agent = MyAgent()
```

#### Simple learning
```python
trainer = Trainer(
    agent=agent,
    env=env,
    schedule=[
        ("finish", 1000, "episodes"),
        ("train_agent", 10, "episodes"),
    ]
)

# Run the trainer
agent, result = trainer.run()
```
#### Observations collecting
```python
trainer = Trainer(
    agent=agent,
    env=env,
    n_workers=4,
    schedule=[
        ("finish", 1000, "episodes"),
        ("worker_send_data", 1000, "episodes"),
    ]
)

# Run the trainer
agent, result = trainer.run()
```

#### Learning with parallel data collecting
```python
trainer = Trainer(
    agent=agent,
    env=env,
    n_workers=4,
    schedule=[
        ("finish", 1000, "episodes"),
        ("train_agent ", 100, "episodes"),
    ],
    sync_mode = 'sync' # optional
)

# Run the trainer
agent, result = trainer.run()
```

#### Parallel learning
```python
from prlearn.collection.agent_combiners import FixedStatAgentCombiner

trainer = Trainer(
    agent=agent,
    env=env,
    n_workers=4,
    schedule=[
        ("finish", 1000, "episodes"),
        ("train_agent ", 10, "episodes"),
        ("combine_agents ", 100, "episodes"),
    ],
    mode="parallel_learning",
    combiner=FixedStatAgentCombiner("mean_reward"),
    sync_mode = 'sync', # optional
)

# Run the trainer
agent, result = trainer.run()
```

#### Multiple agents parallel learning
```python
from prlearn.collection.agent_combiners import FixedStatAgentCombiner


trainer = Trainer(
    agent=[agent_1, agent_2, agent_3, agent_4],
    env=env,
    n_workers=4,
    schedule=[
        ("finish", 1000, "episodes"),
        ("train_agent ", 10, "episodes"),
        ("combine_agents ", 100, "episodes"),
    ],
    mode="parallel_learning",
    combiner=FixedStatAgentCombiner("max_reward"),
    sync_mode = 'sync', # optional
)

# Run the trainer
agent, result = trainer.run()
```

### Base objects defining

#### Agent
```python
from prlearn import Agent, Experience
from typing import Any, Dict, Tuple
from my_model import Model


class MyAgent(Agent):
    
    # example
    def __init__(self):
        self.model = Model()
    
    # optional
    def before(self):
        pass
    
    def action(self, state: Tuple[Any, Dict[str, Any]]) -> Any:
        observation, info = state
        # Define action logic
        pass

    def train(self, experience: Experience):
        obs, actions, rewards, terminated, truncated, info = experience.get()
        # Define training logic
        pass
    
    # optional
    def after(self):
        pass
   
    # optional
    def get(self) -> Any:
        return self.model.get_weights() 
    
    # optional
    def set(self, data: Any):
        self.model.set_weights(data)

```

#### Environment
```python
from prlearn import Environment
from typing import Any, Dict, Tuple


class MyEnv(Environment):
    
    # optional
    def before(self):
        pass
    
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

    # optional
    def after(self):
        pass
```

#### AgentCombiner
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
        main_agent_stats: Optional[Dict[str, Any]] = None,
    ) -> Agent:
        
        return random.choice(workers_agents)
```
