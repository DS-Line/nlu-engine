from typing import Any

from metadata.loader import get_metadata_from_directory
from src.exceptions import AgentIDNotFoundError, DatabaseConfigNotFoundError, UserQueryNotFoundError
from src.managers.memory_manager import MemoryManager
from src.managers.mongodb_manager import log_database_diagnostics, setup_mongo
from src.orchestrator import Orchestrator
from src.services.data_search_service import AsyncSearchEngine
from src.utils.logger import create_logger

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

    async def invoke(self, user_query: str) -> dict[str, Any]:
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

            asset_names_for_loader = [f"sem_{t.lower()}" for t in self._tables]
            metadata = list(get_metadata_from_directory(asset_names_for_loader))

            mongo_manager = await setup_mongo()
            await log_database_diagnostics(mongo_manager)

            async_search_engine = AsyncSearchEngine(
                mongo_manager=mongo_manager, tables_to_sync=list(self._tables.keys())
            )

            query_handler = Orchestrator(
                agent_id=self.agent_id,
                user_id=self.user_id,
                db_config=self._db_config,
                metadata=metadata,
                clarification_history=self._clarification_history,
                ignore_history=self._ignore_history,
                search_engine=async_search_engine,
            )

            result = await query_handler.invoke(user_query)
            logger.info(f"Result: {result}")
            return result

        except (AgentIDNotFoundError, DatabaseConfigNotFoundError, UserQueryNotFoundError) as e:
            logger.error(f"Error: {e}")
            return {"Error": str(e)}
        except Exception as e:
            logger.critical(f"Unhandled Exception: {e}")
            return {"Error": f"Unhandled Exception: {e}"}
