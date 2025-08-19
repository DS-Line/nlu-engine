import asyncio
import time
from typing import Any

from decouple import config

from metadata.loader import get_metadata_from_directory
from src.exceptions import AgentIDNotFoundError, DatabaseConfigNotFoundError, UserQueryNotFoundError
from src.managers.memory_manager import MemoryManager
from src.managers.mongodb_manager import AsyncMongoDBManager
from src.orchestrator import Orchestrator
from src.services.data_search_service import AsyncSearchEngine
from src.utils.logger import create_logger

logger = create_logger(level="DEBUG")
MONGO_URI = config("KV_STORE_CONNECTION_URI")
MONGO_DB = config("KV_STORE_NAME")


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


async def setup_mongo() -> AsyncMongoDBManager:
    """Initialize and connect to MongoDB."""
    mongo_manager = AsyncMongoDBManager(uri=MONGO_URI, database=MONGO_DB)
    await mongo_manager.connect()
    if not await mongo_manager.health_check():
        raise RuntimeError("MongoDB connection is not healthy")
    return mongo_manager


async def log_database_diagnostics(mongo_manager: AsyncMongoDBManager) -> None:
    """Log collections and sample documents for diagnostics."""
    logger.info("--- STARTING ASYNC DATABASE DIAGNOSTICS ---")
    try:
        db = mongo_manager.db
        all_collections = await db.list_collection_names()
        logger.info(f"Collections in '{MONGO_DB}': {all_collections}")

        expected_collection = "DWH_D_PLAYERS_attributes"
        if expected_collection in all_collections:
            player_collection = db[expected_collection]
            doc_count = await player_collection.count_documents({})
            logger.info(f"'{expected_collection}' contains {doc_count} documents.")
            if doc_count > 0:
                sample_doc = await player_collection.find_one({})
                logger.info(sample_doc)
        else:
            logger.error(f"Expected collection '{expected_collection}' not found.")
    except Exception as e:
        logger.error(f"Database diagnostics error: {e}")
    logger.info("--- ENDING ASYNC DATABASE DIAGNOSTICS ---")


def prepare_nlu_engine_arguments(**kwargs: object) -> dict:
    """Return the arguments dictionary for NLUEngine initialization."""
    return {
        "agent_id": kwargs.get("agent_id"),
        "metadata": kwargs.get("metadata"),
        "db_config": kwargs.get("db_config"),
        "clarification_history": kwargs.get("clarification_history"),
        "ignore_history": kwargs.get("ignore_history"),
        "search_engine": kwargs.get("async_search_engine"),
    }


async def main() -> None:
    """
    Entry point for running the NLU Engine.

    Initializes the engine with default configuration values
    (e.g., agent ID, database settings) and invokes the query
    processing pipeline.
    """
    agent_id = "1839af40-70c3-45bf-af7c-d8b08bab3c15"
    user_id = "cfecc156-785a-4e9f-94ac-47261693b3b0"
    tables_to_sync = {
        "DWH_D_GAMES": ["gamecode", "season", "season_name", "game_time", "arena_name"],
        "DWH_D_PLAYERS": ["full_name", "playercode", "player_slug", "games_played_flag"],
        "DWH_D_TEAMS": ["full_name", "abbreviation", "nickname", "city", "state"],
        "DWH_F_PLAYER_BOXSCORE": ["position", "jerseynum"],
        "DWH_F_PLAYER_TRACKING": ["position"],
        "DWH_F_TEAM_BOX_SCORE": [],
    }

    asset_names_for_loader = [f"sem_{t.lower()}" for t in tables_to_sync]
    metadata = list(get_metadata_from_directory(asset_names_for_loader))
    db_config = {"host": "localhost", "port": 5432}

    try:
        mongo_manager = await setup_mongo()
    except Exception as e:
        logger.critical(f"Failed MongoDB setup: {e}")
        return

    await log_database_diagnostics(mongo_manager)

    async_search_engine = AsyncSearchEngine(mongo_manager=mongo_manager, tables_to_sync=list(tables_to_sync.keys()))

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
    user_query = "whats the scores of kobe bryant this season"
    arguments = prepare_nlu_engine_arguments(
        agent_id=agent_id,
        metadata=metadata,
        db_config=db_config,
        clarification_history=clarification_history,
        ignore_history=True,
        async_search_engine=async_search_engine,
    )

    try:
        if user_query.strip().startswith("/memory."):
            result = await MemoryManager.invoke(user_query=user_query, agent_id=agent_id, user_id=user_id)
        else:
            engine = NLUEngine(**arguments)
            result = await engine.invoke(user_query=user_query)
        logger.info(f"Result: {result}")
    except (AgentIDNotFoundError, DatabaseConfigNotFoundError, UserQueryNotFoundError) as e:
        logger.error(f"Error: {e}")


if __name__ == "__main__":
    tic = time.time()
    asyncio.run(main())
    toc = time.time() - tic
    logger.info(f"Run Time: {toc}")
