import datetime

from decouple import config
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection
from pymongo.errors import ConnectionFailure

from src.utils.logger import create_logger

logger = create_logger(level="DEBUG")
MONGO_URI = config("KV_STORE_CONNECTION_URI")
MONGO_DB = config("KV_STORE_NAME")


class AsyncMongoDBManager:
    """Asynchronous MongoDB manager using Motor."""

    def __init__(self, uri: str, database: str) -> None:
        """Initialize with connection URI and database name."""
        self.uri = uri
        self.database_name = database
        self.client = AsyncIOMotorClient(uri)
        self.db = self.client[self.database_name]

    async def connect(self) -> None:
        """Establish and verify the connection."""
        try:
            await self.client.admin.command("ismaster")
            logger.info("Successfully connected to async MongoDB")
        except Exception as e:
            logger.error(f"Async MongoDB connection failed: {e}")
            raise

    def disconnect(self) -> None:
        """Closes the MongoDB connection."""
        if self.client:
            self.client.close()
            logger.info("Async MongoDB connection closed.")

    def get_collection(self, collection_name: str) -> AsyncIOMotorCollection:
        """Return a collection instance by name."""
        return self.db[collection_name]

    async def update_refresh_timestamp(self, collection_name: str = "system_metadata") -> None:
        """Update the last refresh timestamp in the given collection."""
        collection = self.get_collection(collection_name)
        today = datetime.date.today().isoformat()
        await collection.update_one(
            {"key": "last_refresh_date"},
            {"$set": {"value": today, "updated_at": datetime.datetime.now(datetime.timezone.utc)}},
            upsert=True,
        )
        logger.info(f"Updated async refresh timestamp to {today}")

    async def get_refresh_timestamp(self, collection_name: str = "system_metadata") -> str:
        """Retrieve the last refresh timestamp from the given collection."""
        collection = self.get_collection(collection_name)
        metadata = await collection.find_one({"key": "last_refresh_date"})
        return metadata.get("value") if metadata else None

    async def health_check(self) -> bool:
        """Check if the MongoDB connection is healthy."""
        if not self.client:
            return False
        try:
            await self.client.admin.command("ping")
            logger.info("Async MongoDB connection is healthy.")
            return True
        except ConnectionFailure as e:
            logger.error(f"Async MongoDB health check failed: {e}")
            return False


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
