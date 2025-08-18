import asyncio
import time
from typing import Any

from metadata.loader import get_metadata_from_directory
from src.exceptions import AgentIDNotFoundError, DatabaseConfigNotFoundError, UserQueryNotFoundError
from src.managers.memory_manager import MemoryManager
from src.orchestrator import Orchestrator
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
        self._validate_inputs()
        self.query_handler = Orchestrator(
            agent_id=self.agent_id,
            db_config=self._db_config,
            metadata=self._metadata,
            clarification_history=self._clarification_history,
            ignore_history=self._ignore_history,
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


async def main() -> None:
    """
    Entry point for running the NLU Engine.

    Initializes the engine with default configuration values
    (e.g., agent ID, database settings) and invokes the query
    processing pipeline.
    """
    agent_id = "1839af40-70c3-45bf-af7c-d8b08bab3c15"
    user_id = "cfecc156-785a-4e9f-94ac-47261693b3b0"
    final_metadata = get_metadata_from_directory([
        "sem_dwh_d_games",
        "sem_dwh_d_players",
        "sem_dwh_d_teams",
        "sem_dwh_f_player_boxscore",
    ])
    metadata = list(final_metadata)
    db_config = {"host": "localhost", "port": 5432}
    clarification_history = [
        (
            "SELECT player_name, points, assists FROM sem_dwh_f_player_boxscore WHERE player_name='Stephen Curry';",
            "What were Stephen Curry's stats this season?",
        ),
        (
            "SELECT player_name, points, assists FROM sem_dwh_f_player_boxscore WHERE player_name='Kevin Durant';",
            "What were Kevin Durant's stats this season?",
        ),
    ]
    user_query = "Compare his performance to Durant."
    ignore_history = False

    arguments = {
        "agent_id": agent_id,
        "metadata": metadata,
        "db_config": db_config,
        "clarification_history": clarification_history,
        "ignore_history": ignore_history,
    }

    try:
        if user_query.strip().startswith("/memory."):
            result = await MemoryManager.invoke(user_query=user_query, agent_id=agent_id, user_id=user_id)
            logger.info(f"Memory Command Result: {result}")
        else:
            engine = NLUEngine(**arguments)
            logger.info(f"User Query: {user_query}")
            result = await engine.invoke(user_query=user_query)
            logger.info(f"Result: {result}")
    except (AgentIDNotFoundError, DatabaseConfigNotFoundError, UserQueryNotFoundError) as e:
        logger.error(f"Error: {e!s}")


if __name__ == "__main__":
    tic = time.time()
    asyncio.run(main())
    toc = time.time() - tic
    logger.info(f"Run Time: {toc}")
