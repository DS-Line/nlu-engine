import asyncio
import datetime
import json
import logging
import re
from typing import Any

import nltk
from nltk.corpus import wordnet

from nlu_engine.utils.logger import create_logger

try:
    wordnet.synsets("example")
except (LookupError, OSError):
    nltk.download("wordnet", quiet=True)

from itertools import starmap

from motor.motor_asyncio import AsyncIOMotorCollection
from pymongo import TEXT
from sqlalchemy import text

from nlu_engine.managers.mongodb_manager import AsyncMongoDBManager

logger = create_logger("DEBUG")


class AsyncDataCache:
    """
    Handles fetching data from Snowflake and storing it in MongoDB.
    """

    def __init__(
        self, snowflake_conn: object, mongo_manager: AsyncMongoDBManager, tables_to_sync: dict[str, list[str]]
    ) -> None:
        """
        Initializes the async data cache.

        Args:
            snowflake_conn: An Snowflake connection object.
            mongo_manager: An instance of AsyncMongoDBManager.
            tables_to_sync: A dictionary of table names to lists of their attributes.
        """
        self.sf_conn = snowflake_conn
        self.mongo_manager = mongo_manager
        self.tables_to_sync = tables_to_sync
        self.attribute_threshold = 50

    @staticmethod
    def _validate_identifier(identifier: str) -> str:
        """Allow only letters, numbers, and underscores for table/column names."""
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", identifier):
            raise ValueError(f"Invalid SQL identifier: {identifier}")
        return identifier

    @staticmethod
    def log_task_results(results: list[str], table_names: list[str], logger: logging.Logger) -> None:
        """
        Logs the outcome of processing tasks for a set of tables.

        For each result-table pair, logs an error message if the result is an Exception,
        or an info message if the task was successful.
        """
        for task_result, table_name in zip(results, table_names, strict=False):
            if isinstance(task_result, Exception):
                logger.error(f"Error processing table {table_name}: {task_result}")
            else:
                logger.info(f"Successfully processed table: {table_name}")

    async def refresh_tables(self) -> None:
        """
        Coordinates the concurrent refreshing of all tables if a refresh is needed.
        This is the main entry point for the caching process.
        """
        try:
            await self.mongo_manager.connect()
            if not await self.should_refresh_data():
                return
            logger.info("Data refresh triggered for tables: " + ", ".join(self.tables_to_sync.keys()))
            process_tasks = list(starmap(self.process_table, self.tables_to_sync.items()))
            results = await asyncio.gather(*process_tasks, return_exceptions=True)

            self.log_task_results(results, self.tables_to_sync.keys(), logger)

            for task_result, table_name in zip(results, self.tables_to_sync.keys(), strict=False):
                if isinstance(task_result, Exception):
                    logger.error(f"Error processing table {table_name}: {task_result}")
                else:
                    logger.info(f"Successfully processed table: {table_name}")

            await self.mongo_manager.update_refresh_timestamp()
            logger.info("Data refresh complete.")

        except Exception as e:
            logger.error(f"An error occurred during the table refresh process: {e}")

    async def should_refresh_data(self) -> bool:
        """
        Determines if a full data refresh is needed by checking
        the timestamp and for the existence of data in collections.
        """
        last_refresh_date = await self.mongo_manager.get_refresh_timestamp()
        today = datetime.date.today().isoformat()

        if last_refresh_date != today:
            logger.info("Last refresh date is not today. Refreshing data.")
            return True

        check_tasks = []
        for table in self.tables_to_sync:
            collection = self.mongo_manager.get_collection(f"{table}_attributes")
            check_tasks.append(collection.count_documents({}))

        counts = await asyncio.gather(*check_tasks)

        for count, table in zip(counts, self.tables_to_sync.keys(), strict=False):
            if count == 0:
                logger.info(f"Collection {table}_attributes is empty. Refreshing data.")
                return True

        logger.info("Data is up-to-date. Skipping refresh.")
        return False

    async def process_table(self, table_name: str, attributes: list[str]) -> None:
        """Main method to determine the processing strategy for a table."""
        if not attributes:
            logger.warning(f"No attributes to process for table {table_name}")
            return

        if len(attributes) <= self.attribute_threshold:
            await self.process_table_single_document(table_name, attributes)
        else:
            await self.process_table_multiple_documents(table_name, attributes)

    async def process_table_single_document(self, table_name: str, selected_attributes: list[str]) -> None:
        """Processes a table and stores all its attributes in one MongoDB document."""
        collection_name = f"{table_name}_attributes"
        collection = self.mongo_manager.get_collection(collection_name)
        await collection.delete_many({})

        table_doc = {"table_name": table_name, "attributes": {}, "search_text": {}}

        for column in selected_attributes:
            values = await self._get_column_values(table_name, column)
            if values:
                table_doc["attributes"][column] = {"name": column, "sample_values": values}
                table_doc["search_text"][column] = self._flatten_values_for_search(values)

        if table_doc["attributes"]:
            await collection.insert_one(table_doc)
            index_fields = [(f"search_text.{col}", TEXT) for col in table_doc["attributes"]]
            await self._create_text_index(collection, index_fields)
            logger.info(f"Stored {len(table_doc['attributes'])} attributes for {table_name} in a single document.")
        else:
            logger.warning(f"No valid attributes processed for table {table_name}")

    async def process_table_multiple_documents(self, table_name: str, selected_attributes: list[str]) -> None:
        """Processes a table, storing each attribute in its own MongoDB document."""
        collection_name = f"{table_name}_attributes"
        collection = self.mongo_manager.get_collection(collection_name)
        await collection.delete_many({})

        column_value_tasks = {col: self._get_column_values(table_name, col) for col in selected_attributes}
        results = await asyncio.gather(*column_value_tasks.values())
        column_values_map = dict(zip(column_value_tasks.keys(), results, strict=False))

        documents_to_insert = []
        for column, values in column_values_map.items():
            if values:
                search_text = self._flatten_values_for_search(values)
                documents_to_insert.append({
                    "table_name": table_name,
                    "attribute_name": column,
                    "sample_values": values,
                    "search_text": search_text,
                })

        if documents_to_insert:
            await collection.insert_many(documents_to_insert)
            await self._create_text_index(collection, [("search_text", TEXT)])
            logger.info(f"Stored {len(documents_to_insert)} attributes for {table_name} in multiple documents.")

    async def _get_column_values(self, table_name: str, column_name: str) -> list[Any]:
        """Asynchronously fetches distinct, non-null values for a column from Snowflake."""
        try:
            table_name = self._validate_identifier(table_name)
            column_name = self._validate_identifier(column_name)

            query = text("SELECT DISTINCT :column_name FROM :table_name WHERE :column_name IS NOT NULL LIMIT 5000")

            with self.sf_conn.cursor() as cursor:
                cursor.execute(query, {"column_name": column_name, "table_name": table_name})
                return [row[0] for row in cursor if row[0] is not None]

        except Exception as e:
            logger.debug(f"Error fetching values for {table_name}.{column_name}: {e}")
            return []

    @staticmethod
    async def _create_text_index(collection: AsyncIOMotorCollection, index_fields: list[tuple[str, int]]) -> None:
        """Drops any existing text index and creates a new one."""
        try:
            existing_indexes = await collection.index_information()
            for index_name, index_info in existing_indexes.items():
                if any("text" in str(key) for key in index_info.get("key", [])):
                    await collection.drop_index(index_name)
                    break
            if index_fields:
                await collection.create_index(index_fields, name="text_search_idx", background=True)
                logger.debug(f"Created text index for {collection.name}")
        except Exception as e:
            logger.warning(f"Could not create text index for {collection.name}: {e}")

    @staticmethod
    def _flatten_values_for_search(values: list[Any]) -> str:
        """Flattens a list of values into a single string for text indexing. (CPU-bound)"""
        processed_values = [
            json.dumps(val, ensure_ascii=False) if isinstance(val, (list, dict)) else str(val).strip() for val in values
        ]
        return " ".join(processed_values)[:50000]
