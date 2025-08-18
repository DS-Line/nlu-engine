import datetime
import logging
from collections.abc import Collection

from pymongo import MongoClient
from pymongo.database import Database

logger = logging.getLogger(__name__)


class MongoDBManager:
    """Manages the MongoDB connection and provides utility methods for collections."""

    def __init__(self, uri: str, database: str) -> None:
        self.uri = uri
        self.database = database
        self.client = None
        self.db = None

    def connect(self) -> Database:
        """Establishes and returns a connection to MongoDB."""
        try:
            self.client = MongoClient(self.uri)
            self.db = self.client[self.database]
            self.client.admin.command("ismaster")  # Check connection
            logger.info("Connected to MongoDB")
            return self.db
        except Exception as e:
            logger.error(f"MongoDB connection failed: {e}")
            raise

    def get_collection(self, collection_name: str) -> Collection:
        """Returns a MongoDB collection instance."""
        return self.db[collection_name]

    def update_refresh_timestamp(self, collection_name: str = "system_metadata") -> None:
        """Updates the last refresh date in a metadata collection."""
        collection = self.get_collection(collection_name)
        today = datetime.date.today().isoformat()
        collection.update_one(
            {"key": "last_refresh_date"},
            {"$set": {"value": today, "updated_at": datetime.datetime.now(datetime.timezone.utc)}},
            upsert=True,
        )
        logger.info(f"Updated refresh timestamp to {today}")

    def get_refresh_timestamp(self, collection_name: str = "system_metadata") -> str:
        """Retrieves the last refresh date from a metadata collection."""
        collection = self.get_collection(collection_name)
        metadata = collection.find_one({"key": "last_refresh_date"})
        return metadata.get("value") if metadata else None
