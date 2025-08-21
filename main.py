from src.exceptions import AgentIDNotFoundError, DatabaseConfigNotFoundError, UserQueryNotFoundError
from src.managers.memory_manager import MemoryManager
from src.orchestrator import Orchestrator
from src.utils.logger import create_logger
from src.utils.responses import BaseResponse, error_response, success_response

logger = create_logger(level="DEBUG")


class NLUEngine:
    """
    Natural Language Understanding (NLU) Engine.

    This class initializes and manages the NLU pipeline,
    including handling user queries, database configuration,
    and logging setup.
    """

    def __init__(self, **kwargs: object) -> None:
        logger.info("Denzing NLU Engine Initialized")
        self.kwargs = kwargs
        self.agent_id = kwargs.get("agent_id")
        self.user_id = kwargs.get("user_id")
        self._db_config = kwargs.get("db_config", {})
        self._tables = kwargs.get("tables", {})
        self._metadata = kwargs.get("metadata", [{}, {}, {}, {}])
        self._clarification_history = kwargs.get("clarification_history", [])
        self._ignore_history = kwargs.get("ignore_history", True)
        self._validate_inputs()

    def _validate_inputs(self) -> None:
        if self.agent_id is None:
            raise AgentIDNotFoundError("Agent ID cannot be None")
        if not self._db_config:
            raise DatabaseConfigNotFoundError("Database configuration not found")

    async def invoke(self, user_query: str) -> BaseResponse:
        """
        Invoke the NLU engine with the given user query.

        Args:
            user_query (str): The user query text to process.

        Returns:
            dict[str, Any]: The processed response from the query handler.

        Raises:
            UserQueryNotFoundError: If `user_query` is not provided.
        """
        if user_query is None:
            raise UserQueryNotFoundError("User Query Not Found")
        try:
            if user_query.strip().startswith("/memory."):
                return await MemoryManager.invoke(
                    user_query=user_query,
                    agent_id=self.agent_id,
                    user_id=self.user_id,
                )

            query_handler = Orchestrator(
                agent_id=self.agent_id,
                user_id=self.user_id,
                db_config=self._db_config,
                tables=self._tables,
                clarification_history=self._clarification_history,
                ignore_history=self._ignore_history,
            )

            result = await query_handler.invoke(user_query)
            logger.info(f"Result: {result}")
            return success_response(code="DZ_SUCCESS_200", message="NLU Engine run successful", data=result)

        except (AgentIDNotFoundError, DatabaseConfigNotFoundError, UserQueryNotFoundError) as e:
            logger.error(f"Error: {e}")
            return error_response(message="Validation Error", code="DZ_ERROR_400", errors=str(e))
        except Exception as e:
            logger.critical(f"Unhandled Exception: {e}")
            return error_response(message="Unhandled Exception", code="DZ_ERROR_500", errors=str(e))
