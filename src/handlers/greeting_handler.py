from typing import Any

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

from src.handlers.base import Handler
from src.llms.llm_factory import llm
from src.prompts.prompt import Prompt
from src.utils.execution_time_tracker import capture_execution_time


class GreetingHandler(Handler):
    """Handler to classify user's query as a greeting or not."""

    def __init__(self) -> None:
        """
        Initializes the GreetingHandler with the prompt chain.
        """
        self.llm_model_type = "lite"

        prompt = Prompt.get("GreetingPrompt")
        prompt_template = PromptTemplate(input_variables=["user_query"], template=prompt)

        if llm is not None:
            self.chain = prompt_template | llm.get_llm() | JsonOutputParser()

    @capture_execution_time
    async def handle(self, query: str) -> dict[str, Any]:
        """
        Classify the user's query as a greeting or not, returning the result.

        Args:
            query: User's query to classify.

        Returns:
            Dictionary containing the query, classification result, and messages.
        """
        return await self.chain.ainvoke({"user_query": query})
