from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from nlu_engine.handlers.base import Handler
from nlu_engine.llms.llm_factory import llm
from nlu_engine.prompts.prompt import Prompt
from nlu_engine.utils.execution_time_tracker import capture_execution_time
from nlu_engine.utils.logger import create_logger

logger = create_logger(level="DEBUG")


class QuestionCondenseHandler(Handler):
    """
    Handler responsible for condensing user queries based on conversation history.

    This class uses a language model (LLM) along with a prompt template to
    generate a concise version of the user's query. It is useful in multi-turn
    conversations where context from previous queries needs to be incorporated.

    Attributes:
        history (list[tuple]): Conversation history used to guide query condensation.
        llm_model_type (str): Type of language model to use; defaults to 'lite'.
        chain: Processing chain combining the prompt template, LLM, and output parser.
    """

    def __init__(self, history: list[tuple]) -> None:
        """
        Initialize the Question Condense Handler.

        :param history: Conversation history used in the query condensation
        """
        self.history = history
        self.llm_model_type = "lite"
        prompt = Prompt.get("QuestionCondensationPrompt")
        prompt_template = PromptTemplate(input_variables=["history", "user_query"], template=prompt)

        self.chain = prompt_template | llm.get_llm(invoking_method="Question Condenser") | StrOutputParser()

    def _format_history(self) -> str:
        """
        Convert the history list of tuples into a readable string for the LLM.
        """
        return "\n".join([f"SQL: {sql}\nUser Query: {uq}" for sql, uq in self.history])

    @capture_execution_time
    async def handle(self, query: str) -> str:
        """
        Condense the user's query based on the conversation history.

        :param query: The user's question
        :return: Condensed version of the user's question
        """
        try:
            history_str = self._format_history()
            response = await self.chain.ainvoke({"history": history_str, "user_query": query})
            logger.info(f"Condensed response: {response}")
            return response
        except Exception as e:
            raise Exception(f"Something went wrong while condensing the question: {e}") from e
