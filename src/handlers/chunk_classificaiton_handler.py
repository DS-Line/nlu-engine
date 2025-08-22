from collections.abc import Coroutine
from typing import Any

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

from src.handlers.base import Handler
from src.llms.llm_factory import llm
from src.prompts.prompt import Prompt
from src.utils.execution_time_tracker import capture_execution_time


class ChunkClassificationHandler(Handler):
    """
    Handles the classification of text chunks using a predefined prompt.

    This class retrieves the 'ChunkClassificationPrompt' and uses it
    to classify input text into appropriate categories. It extends
    the base Handler class to provide chunk-specific classification logic.
    """

    def __init__(self) -> None:
        prompt = Prompt.get("ChunkClassificationPrompt")
        prompt_template = PromptTemplate(
            input_variables=["user_query", "extracted_metadata", "past_questions"], template=prompt
        )
        self.chain = prompt_template | llm.get_llm() | JsonOutputParser()

    @capture_execution_time
    async def handle(self, query: str, metadata: str, past_questions: list) -> Coroutine[Any, Any, dict[str, Any]]:
        """
        Classify the user's query as a greeting or not, and return a dictionary
        containing the user's query, the classification result, and a list of
        messages to be displayed to the user.

        :param query: User's query to classify
        :param metadata: Additional metadata that may influence classification.
        :param past_questions: List of past user queries or messages.
        """
        return await self.chain.ainvoke({
            "query": query,
            "extracted_metadata": metadata,
            "past_questions": past_questions,
        })
