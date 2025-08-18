from abc import ABC, abstractmethod
from collections.abc import Coroutine
from typing import Any


class Handler(ABC):
    """Abstract base class for all handlers."""

    @abstractmethod
    def handle(self, query: str) -> dict[str, Any] | Coroutine[Any, Any, dict[str, Any]]:
        """
        Abstract method that must be implemented by subclasses.

        Raises:
            NotImplementedError: Indicates the method is not implemented.
        """
        raise NotImplementedError
