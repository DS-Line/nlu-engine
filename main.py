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
        self._db_config = kwargs.get("db_config", {})
        self._metadata = kwargs.get("metadata", [{}, {}, {}, {}])
        self._clarification_history = kwargs.get("clarification_history", [])
        self._ignore_history = kwargs.get("ignore_history", True)
        self._search_engine = kwargs.get("search_engine")
        self._validate_inputs()
        self.query_handler = Orchestrator(
            agent_id=self.agent_id,
            db_config=self._db_config,
            metadata=self._metadata,
            clarification_history=self._clarification_history,
            ignore_history=self._ignore_history,
            search_engine=self._search_engine,
        )

    def _validate_inputs(self) -> None:
        if self.agent_id is None:
            raise AgentIDNotFoundError("Agent ID cannot be None")
        if not self._db_config:
            raise DatabaseConfigNotFoundError("Database configuration not found")

    async def invoke(self, **kwargs: object) -> dict[str, Any]:
        """
        Invoke the NLU engine with the given user query.

        Args:
            **kwargs: Arbitrary keyword arguments.
                - user_query (str): The user query text to process.

        Returns:
            dict[str, Any]: The processed response from the query handler.

        Raises:
            UserQueryNotFoundError: If `user_query` is not provided.
        """
        user_query = kwargs.get("user_query")
        if user_query is None:
            raise UserQueryNotFoundError("User Query Not Found")
        return await self.query_handler.invoke(user_query)


async def nlu_engine(**kwargs: object) -> dict[str, Any] | None:
    """
    Entry point for running the NLU Engine.

    Initializes the engine with default configuration values
    (e.g., agent ID, database settings) and invokes the query
    processing pipeline.
    """
    agent_id = kwargs.get("agent_id")
    user_id = kwargs.get("user_id")
    user_query = kwargs.get("user_query")
    tables = kwargs.get("tables")
    db_config = kwargs.get("db_config")
    clarification_history = kwargs.get("clarification_history")
    ignore_history = kwargs.get("ignore_history")
    asset_names_for_loader = [f"sem_{t.lower()}" for t in tables]
    metadata = list(get_metadata_from_directory(asset_names_for_loader))

    try:
        mongo_manager = await setup_mongo()
    except Exception as e:
        logger.critical(f"Failed MongoDB setup: {e}")
        return {"Error": f"Failed MongoDB setup: {e}"}

    await log_database_diagnostics(mongo_manager)

    async_search_engine = AsyncSearchEngine(mongo_manager=mongo_manager, tables_to_sync=list(tables.keys()))

    arguments = {
        "agent_id": agent_id,
        "metadata": metadata,
        "db_config": db_config,
        "clarification_history": clarification_history,
        "ignore_history": ignore_history,
        "search_engine": async_search_engine,
    }
    try:
        if user_query.strip().startswith("/memory."):
            result = await MemoryManager.invoke(user_query=user_query, agent_id=agent_id, user_id=user_id)
        else:
            engine = NLUEngine(**arguments)
            result = await engine.invoke(user_query=user_query)
        logger.info(f"Result: {result}")
        return result
    except (AgentIDNotFoundError, DatabaseConfigNotFoundError, UserQueryNotFoundError) as e:
        logger.error(f"Error: {e}")
        return {"Error": e}
