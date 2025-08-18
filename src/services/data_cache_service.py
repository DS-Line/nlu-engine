import concurrent
import datetime
import json
import logging
import re
from collections.abc import Collection
from typing import Any

from pymongo import TEXT
from sqlalchemy import text

from src.managers.mongodb_manager import MongoDBManager

logger = logging.getLogger(__name__)


class DataCache:
    """
    Handles fetching data from Snowflake and storing it in MongoDB.
    This class now focuses purely on the data processing logic.
    """

    def __init__(
        self, snowflake_conn: object, mongo_manager: MongoDBManager, tables_to_sync: dict[str, list[str]]
    ) -> None:
        self.sf_conn = snowflake_conn
        self.mongo_manager = mongo_manager
        self.tables_to_sync = tables_to_sync
        self.attribute_threshold = 50
        self.max_workers = 4

    def connect_mongodb(self) -> None:
        """Initializes the MongoDB connection."""
        self.mongo_manager.connect()
        logger.info("MongoDB connection initialized")

    @staticmethod
    def _flatten_values_for_search(values: list[Any]) -> str:
        """Flattens a list of values into a single string for text indexing."""
        processed_values = [
            json.dumps(val, ensure_ascii=False) if isinstance(val, (list, dict)) else str(val).strip() for val in values
        ]
        return " ".join(processed_values)[:50000]

    @staticmethod
    def _validate_identifier(identifier: str) -> str:
        """Allow only letters, numbers, and underscores for table/column names."""
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", identifier):
            raise ValueError(f"Invalid SQL identifier: {identifier}")
        return identifier

    def _get_column_values(self, table_name: str, column_name: str) -> list[str]:
        """Fetch distinct, non-null values for a given column from a Snowflake table."""

        try:
            # Validate table and column names
            table_name = self._validate_identifier(table_name)
            column_name = self._validate_identifier(column_name)

            # Build the query safely
            query = text("SELECT DISTINCT :column_name FROM :table_name WHERE :column_name IS NOT NULL LIMIT 5000")

            # Execute
            with self.sf_conn.cursor() as cursor:
                cursor.execute(query, {"column_name": column_name, "table_name": table_name})
                return [row[0] for row in cursor if row[0] is not None]

        except Exception as e:
            logger.debug(f"Error fetching values for column {column_name} in table {table_name}: {e}")
            return []

    @staticmethod
    def _create_text_index(collection: Collection, index_fields: list[tuple[str, int]]) -> None:
        """Drops any existing text index and creates a new one."""
        try:
            existing_indexes = collection.index_information()
            for index_name in existing_indexes:
                if "text" in str(existing_indexes[index_name]):
                    collection.drop_index(index_name)
            if index_fields:
                collection.create_index(index_fields, name="text_search_idx", background=True)
                logger.debug(f"Created text index for {collection.name}")
        except Exception as e:
            logger.warning(f"Could not create text index for {collection.name}: {e}")

    def process_table_single_document(self, table_name: str, selected_attributes: list[str]) -> None:
        """Processes a single table and stores all its attributes in one MongoDB document."""
        collection_name = f"{table_name}_attributes"
        collection = self.mongo_manager.get_collection(collection_name)
        collection.delete_many({})
        table_doc = {"table_name": table_name, "attributes": {}, "search_text": {}}

        for column in selected_attributes:
            values = self._get_column_values(table_name, column)
            if values:
                table_doc["attributes"][column] = {"name": column, "sample_values": values}
                table_doc["search_text"][column] = self._flatten_values_for_search(values)

        if table_doc["attributes"]:
            collection.insert_one(table_doc)
            index_fields = [(f"search_text.{col}", TEXT) for col in table_doc["attributes"]]
            self._create_text_index(collection, index_fields)
            logger.info(
                f"Stored {len(table_doc['attributes'])} attributes for table {table_name} in a single document."
            )
        else:
            logger.warning(f"No valid attributes processed for table {table_name}")

    def process_table_multiple_documents(self, table_name: str, selected_attributes: list[str]) -> None:
        """Processes a single table, storing each attribute in its own MongoDB document."""
        collection_name = f"{table_name}_attributes"
        collection = self.mongo_manager.get_collection(collection_name)
        collection.delete_many({})

        documents_to_insert = []
        for column in selected_attributes:
            values = self._get_column_values(table_name, column)
            if values:
                search_text = self._flatten_values_for_search(values)
                documents_to_insert.append({
                    "table_name": table_name,
                    "attribute_name": column,
                    "sample_values": values,
                    "search_text": search_text,
                })

        if documents_to_insert:
            collection.insert_many(documents_to_insert)
            self._create_text_index(collection, [("search_text", TEXT)])
            logger.info(f"Stored {len(documents_to_insert)} attributes for table {table_name} in multiple documents.")

    def process_table(self, table_name: str, attributes: list[str]) -> None:
        """Main method to determine processing strategy for a table."""
        if not attributes:
            logger.warning(f"No attributes to process for table {table_name}")
            return
        try:
            if len(attributes) <= self.attribute_threshold:
                self.process_table_single_document(table_name, attributes)
            else:
                self.process_table_multiple_documents(table_name, attributes)
        except Exception as e:
            logger.error(f"Error processing table {table_name}: {e}")
            raise

    def should_refresh_data(self) -> bool:
        """
        Determines if a full refresh is needed.
        A refresh is needed if:
        1. The last refresh date is not today.
        2. Any required collection is empty or non-existent.
        """
        last_refresh_date = self.mongo_manager.get_refresh_timestamp()
        today = datetime.date.today().isoformat()

        if last_refresh_date != today:
            logger.info("Last refresh date is not today. Refreshing data.")
            return True

        for table in self.tables_to_sync:
            collection_name = f"{table}_attributes"
            collection = self.mongo_manager.get_collection(collection_name)

            if collection.count_documents({}) == 0:
                logger.info(f"Collection {collection_name} is empty. Refreshing data.")
                return True

        logger.info("Data is up-to-date. Skipping refresh.")
        return False

    def refresh_tables(self) -> None:
        """Coordinates the concurrent refreshing of all tables if a refresh is needed."""
        try:
            if self.mongo_manager.db is None:
                self.connect_mongodb()

            if not self.should_refresh_data():
                return

            logger.info("Data refresh triggered for tables: " + ", ".join(self.tables_to_sync.keys()))
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(self.process_table, table, attributes): table
                    for table, attributes in self.tables_to_sync.items()
                }
                for future in concurrent.futures.as_completed(futures):
                    table = futures[future]
                    try:
                        future.result()
                        logger.info(f"Successfully processed table: {table}")
                    except Exception as e:
                        logger.error(f"Error processing table {table}: {e}")

            self.mongo_manager.update_refresh_timestamp()
            logger.info("Data refresh complete.")
        except Exception as e:
            logger.error(f"An error occurred during table processing: {e}")
            raise
