# PRLearn

PRLearn is a Python library for **P**arallel **R**einforcement **Learn**ing. It leverages multiprocessing to accelerate experience collection and agent training, making RL experimentation faster and more efficient.

## Key Features

- **Flexible architecture**: Easily extendable with custom agents, environments, and combiners.
- **Minimal dependencies**: Only Python 3.11+ and (optionally) multiprocess.
- **Parallel data collection and training**: Reduce training time via multiprocessing.
- **Agent combination**: Multiple strategies for aggregating agents (statistical, random, fixed, etc.).
- **Flexible scheduling**: Control training stages via ProcessActionScheduler.


## Installation

```sh
pip install prlearn
```
Or with multiprocess support:
```sh
pip install prlearn[multiprocess]
```

## Quick Start

### Define Your Agent

```python
from prlearn import Agent, Experience
from typing import Any, Dict, Tuple

class MyAgent(Agent):
    def action(self, state: Tuple[Any, Dict[str, Any]]) -> Any:
        observation, info = state
        # Action selection logic
        pass
    def train(self, experience: Experience):
        obs, actions, rewards, terminated, truncated, info = experience.get()
        # Training logic
        pass
```

### Use Trainer for Parallel Training

```python
import gymnasium as gym
from prlearn import Trainer
from prlearn.collection.agent_combiners import FixedStatAgentCombiner

env = gym.make("LunarLander-v2")
agent = MyAgent()

trainer = Trainer(
    agent=agent,
    env=env,
    n_workers=4,
    schedule=[
        ("finish", 1000, "episodes"),
        ("train_agent", 10, "episodes"),
    ],
    mode="parallel_learning",  # optional
    sync_mode="sync",          # optional
    combiner=FixedStatAgentCombiner("mean_reward"),  # optional
)

agent, result = trainer.run()
```

### Custom Environment

```python
from prlearn import Environment
from typing import Any, Dict, Tuple

class MyEnv(Environment):
    def reset(self) -> Tuple[Any, Dict[str, Any]]:
        # Reset logic
        return [[1, 2], [3, 4]], {"info": "description"}
    def step(self, action: Any) -> Tuple[Any, Any, bool, bool, Dict[str, Any]]:
        # Step logic
        return [[1, 2], [3, 4]], 1, False, False, {"info": "description"}
```

**See more usage examples in [docs/examples.md](docs/examples.md)**


## Extending

- **Custom agent**: Inherit from `Agent`, implement `action` and `train` methods.
- **Custom environment**: Inherit from `Environment`, implement `reset` and `step` methods.
- **Custom combiner**: Inherit from `AgentCombiner`, implement the `combine` method.


## Testing

To run tests:
```sh
pytest tests/
```

## License

MIT License. See [LICENSE](LICENSE).
