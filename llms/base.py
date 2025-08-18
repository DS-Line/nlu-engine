from abc import ABC, abstractmethod

from langchain.agents import BaseSingleActionAgent
from langchain_core.language_models import BaseLanguageModel


class BaseLLM(ABC):
    """
    Abstract base class for Large Language Model (LLM) implementations.
    """

    @abstractmethod
    def get_llm(self) -> BaseLanguageModel:
        """
        Retrieve the underlying LLM instance.

        Returns:
            Any: The concrete LLM object.
        """
        raise NotImplementedError

    @abstractmethod
    def completion(self, prompt: str) -> str:
        """
        Generate a completion for a given prompt.

        Args:
            prompt (str): Input prompt string.

        Returns:
            str: Generated text from the LLM.
        """
        raise NotImplementedError

    @abstractmethod
    def get_assistant(self) -> BaseSingleActionAgent:
        """
        Retrieve an LLM-based assistant instance.

        Returns:
            Any: Assistant object wrapping the LLM.
        """
        raise NotImplementedError


class LLM:
    """
    A wrapper class for a language model (LLM).

    This class stores an instance of a language model and can be
    extended with additional helper methods for processing
    inputs and generating outputs.
    """

    def __init__(self, llm: BaseLLM) -> None:
        """
        Initializes an instance of the LLM class.

        Args:
            llm: The language model to be used.

        Returns:
            None
        """
        self.llm: BaseLLM = llm

    def completion(self, prompt: str) -> str:
        """
        The function generates a completion for a given prompt using the LLM model.

        Args:
            prompt (str): The input prompt to be completed.

        Returns:
            str: The completed prompt.
        """
        return self.llm.completion(prompt)
