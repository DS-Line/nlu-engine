import os
from dataclasses import dataclass
from typing import Optional

from decouple import config
from langchain.agents.openai_assistant import OpenAIAssistantRunnable
from langchain_openai import ChatOpenAI

from llms.base import BaseLLM


@dataclass
class OpenAIConfig:
    """Configuration for OpenAI class."""

    openai_api_key: str | None = None
    name: str | None = None
    instruction: str | None = None
    model: str | None = None
    temperature: float | None = None


class OpenAI(BaseLLM):
    """OpenAI wrapper class for managing LLM instances and assistant interaction."""

    _instance: Optional["OpenAI"] = None
    _assistant_name: str = "Inteliome Assistant"
    _assistant_instruction: str = (
        "You are an intelligent agent to help users on retrieval and analysis of structured and unstructured data."
    )

    def __init__(self, cfg: OpenAIConfig | None = None, *, stream: bool = False) -> None:
        """
        Initializes a new instance of the OpenAI connection class.

        Args:
            cfg: OpenAIConfig object containing optional parameters.
            stream: Whether to stream responses.
        """
        cfg = cfg or OpenAIConfig()
        self._name = cfg.name
        self._instruction = cfg.instruction
        self._openai_api_key = cfg.openai_api_key or config("API_KEY")
        self._model = cfg.model or config("MODEL")
        self._temperature = cfg.temperature or config("TEMPERATURE", 0.1)
        self._stream = stream
        self._llm = self.create_llm()
        self._assistant: OpenAIAssistantRunnable | None = None

    def create_llm(self, model: str | None = None, temperature: float | None = None) -> ChatOpenAI:
        """
        Creates a ChatOpenAI instance.

        Args:
            model: Optional model name. Defaults to self._model.
            temperature: Optional temperature. Defaults to self._temperature.

        Returns:
            ChatOpenAI instance configured with the given model and temperature.
        """
        model = model if model is not None else self._model
        temperature = temperature if temperature is not None else self._temperature
        return ChatOpenAI(api_key=self._openai_api_key, model=model, temperature=temperature)

    def create_assistant(self) -> None:
        """
        Creates the OpenAI assistant instance with configured instructions and tools.
        """
        self._assistant = OpenAIAssistantRunnable.create_assistant(
            name=self._name or OpenAI._assistant_name,
            instructions=self._instruction or OpenAI._assistant_instruction,
            model=self._model,
            temperature=self._temperature,
            tools=[],
        )

    def get_llm(self, invoking_method: str | None = None) -> ChatOpenAI:
        """
        Retrieves a language model (LLM) instance from the connection pool.

        Args:
            invoking_method (str): The name of the method invoking the LLM.

        Returns:
            LLM: The language model instance.
        """
        if not self._llm:
            raise ValueError("LLM has not been created.")

        reasoning_model = config("REASONING_MODEL", default=os.getenv("MODEL"))
        mini_model = config("MINI_MODEL", default=os.getenv("MODEL"))

        invoking_method_mapping = {
            "SQL Query Generator": reasoning_model,
            "Result Explanation": mini_model,
            "Code Retry": reasoning_model,
            "Correct Plot": mini_model,
            "Pandas Code Generation": reasoning_model,
            "Pandas Plot Generation": reasoning_model,
            "Pandas DataFrame Manipulation": reasoning_model,
            "JSON Formatting": mini_model,
        }
        if invoking_method in invoking_method_mapping:
            self._llm = self.create_llm(model=invoking_method_mapping.get(invoking_method, self._model))
        elif not self._llm:
            raise ValueError("LLM has not been created.")

        return self._llm

    def get_assistant(self) -> OpenAIAssistantRunnable:
        """
        Retrieves the assistant instance.

        Returns:
            OpenAIAssistantRunnable instance.
        """
        if not self._assistant:
            raise ValueError("Assistant has not been created.")
        return self._assistant

    def completion(self, prompt: str, mode: str = "llm") -> str:
        """
        Generates a completion for a given prompt using the specified model and temperature.

        Args:
            prompt (str): The input prompt to be completed.
            mode: Either 'llm' to use LLM or 'assistant' to use assistant.

        Returns:
            str: The generated completion.

        Raises:
            Exception: If an error occurs during the completion process.
        """
        try:
            result = self._llm.invoke(prompt) if mode == "llm" else self._assistant.invoke(prompt)
            return result.content

        except Exception as e:
            raise Exception(f"LLM is unable to process your query: {e}") from e
