from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple


class Environment(ABC):
    def before(self):
        pass

    @abstractmethod
    def reset(self) -> Tuple[Any, Dict[str, Any]]:
        raise NotImplementedError(
            "Method 'reset' of class Environment is not implemented"
        )

    @abstractmethod
    def step(self, action: Any) -> Tuple[Any, Any, bool, bool, Dict[str, Any]]:
        raise NotImplementedError(
            "Method 'step' of class Environment is not implemented"
        )

    def after(self):
        pass
