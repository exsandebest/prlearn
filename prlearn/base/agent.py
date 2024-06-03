from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

from prlearn.base.experience import Experience


class Agent(ABC):
    @abstractmethod
    def action(self, state: Tuple[Any, Dict[str, Any]]) -> Any:
        raise NotImplementedError("Not implemented agent method 'action'")

    @abstractmethod
    def train(self, experience: Experience):
        raise NotImplementedError("Not implemented agent method 'train'")
