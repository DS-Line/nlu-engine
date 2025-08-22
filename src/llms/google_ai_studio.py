from decouple import config
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables.base import Runnable, RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI

from llms.base import BaseLLM


class GoogleAIStudio(BaseLLM, Runnable):
    """
    LangChain-compatible class for interacting with Google's Gemini models
    via the Google Generative AI API.
    This is a concrete LLM implementation compatible with LLMFactory.
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        temperature: float | None = None,
    ) -> None:
        """
        Initializes the Google Generative AI model with configuration
        from environment variables or provided arguments.

        Args:
            model: Model name to use (defaults to "gemini-pro" from env).
            api_key: Google API key (defaults to env variable API_KEY).
            temperature: Sampling temperature (defaults to 0.1 from env).
        """
        self._api_key = api_key or config("API_KEY")
        self._model = model or config("MODEL", default="gemini-2.5-flash")
        self._temperature = temperature if temperature is not None else float(config("TEMPERATURE", default=0.1))

        self.llm = ChatGoogleGenerativeAI(
            model=self._model,
            temperature=self._temperature,
            google_api_key=self._api_key,
        )

    def get_llm(self, invoking_method: str | None = None) -> ChatGoogleGenerativeAI:
        """
        Returns the underlying LLM instance.

        Args:
            invoking_method: Optional string to indicate the invoking method
                              (kept for API consistency, unused).

        Returns:
            The ChatGoogleGenerativeAI instance.
        """
        _invoking_method = invoking_method
        return self.llm

    def get_assistant(self) -> None:
        """
        Gemini via AI Studio does not directly support agents or assistants in this way.
        This method is not implemented.
        """
        raise NotImplementedError("Gemini via AI Studio does not support agents or assistants in this context.")

    def completion(self, prompt: str | list[BaseMessage], mode: str = "llm") -> str:
        """
        Generate a completion from a given prompt.

        Args:
            prompt (Union[str, List[BaseMessage]]): The prompt to send to the LLM.
                Can be either a plain string or a list of message objects.
            mode (str): The mode of operation. Only "llm" is currently supported.

        Returns:
            str: The generated completion content.

        Raises:
            NotImplementedError: If the mode is not "llm".
            RuntimeError: If the LLM invocation fails.
        """
        if mode != "llm":
            raise NotImplementedError("Only 'llm' mode is supported for GoogleAIStudio.")

        try:
            filtered_prompt = [HumanMessage(content=prompt)] if isinstance(prompt, str) else prompt
            return self.llm.invoke(filtered_prompt).content
        except Exception as err:
            raise RuntimeError(f"GoogleAIStudio LLM failed: {err}") from err

    def invoke(self, prompt: str | list[BaseMessage], _config: RunnableConfig | None = None, **_kwargs: object) -> str:
        """
        Invokes the LLM to generate a response.

        Args:
            prompt: The input prompt as a string or list of BaseMessage objects.
            _config: Optional RunnableConfig object for the invocation.
            **_kwargs: Additional keyword arguments (currently unused).

        Returns:
            Generated text from the LLM.
        """
        return self.completion(prompt)
