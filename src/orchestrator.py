import asyncio
import re
from collections import defaultdict
from typing import Any

import nltk
from nltk.corpus import stopwords

from metadata.loader import get_metadata_from_directory
from src.exceptions import AgentIDNotFoundError, DatabaseConfigNotFoundError
from src.extractors.date_type_extractor import DateTypeHandler
from src.handlers.chunk_classificaiton_handler import ChunkClassificationHandler
from src.handlers.greeting_handler import GreetingHandler
from src.handlers.question_condensation_handler import QuestionCondenseHandler
from src.managers.memory_manager import match_memory
from src.managers.mongodb_manager import log_database_diagnostics, setup_mongo
from src.processors.query_processor import QueryProcessor
from src.processors.text_preprocessor import TextPreprocessor
from src.services.data_search_service import AsyncSearchEngine
from src.services.metadata_service import MetadataService
from src.utils.logger import create_logger
from src.utils.metadata_helper import filter_value_mappings_by_query, normalize_and_filter_terms, transform_raw_metadata

logger = create_logger(__name__)


class Orchestrator:
    """
    Handles a user query by processing it through a series of NLP and LLM-based
    steps to extract meaningful metadata and classify query chunks.
    """

    def __init__(self, agent_id: str, user_id: str, **kwargs: object) -> None:
        """
        Initializes the Orchestrator with essential configurations.

        Args:
            agent_id (str): The unique identifier for the agent.
            user_id (str): The unique identifier for the user.
            **kwargs: A dictionary of additional keyword arguments.
                - db_config (Dict[str, Any]): Database configuration dictionary.
                - clarification_history (List[Tuple[str, str]], optional): A history of previous SQL queries and user queries.
                - metadata (Any, optional): Metadata object for the database schema.
                - ignore_history (bool, optional): If True, the existing clarification history will be ignored.
        """
        self.agent_id = agent_id
        self.user_id = user_id
        self._database_config = kwargs.get("db_config")
        self._metadata: list[dict] = []
        self._tables = kwargs.get("tables")
        self.clarification_history = kwargs.get("clarification_history")
        self._ignore_history = kwargs.get("ignore_history")
        self.search_engine: AsyncSearchEngine | None = None
        self.state: dict[str, Any] = {}
        self._validate_inputs()

    def _validate_inputs(self) -> None:
        """
        Validates that required inputs like agent_id and database config are provided.
        """
        if self.agent_id is None:
            raise AgentIDNotFoundError("Agent ID cannot be None")
        if not self._database_config:
            raise DatabaseConfigNotFoundError("Database configuration not found")
        # Download NLTK data if not present, a robust approach.
        try:
            stopwords.words("english")
            nltk.data.find("tokenizers/punkt")
        except (LookupError, OSError):
            nltk.download("stopwords", quiet=True)
            nltk.download("punkt", quiet=True)

    async def _initialize_assets_and_search(self) -> None:
        """
        Prepares asset names, loads metadata, sets up MongoDB, and initializes
        the async search engine.
        """
        asset_names_for_loader = [f"sem_{t.lower()}" for t in self._tables]
        self._metadata = list(get_metadata_from_directory(asset_names_for_loader))

        mongo_manager = await setup_mongo()
        await log_database_diagnostics(mongo_manager, self._tables)

        self.search_engine = AsyncSearchEngine(mongo_manager=mongo_manager, tables_to_sync=list(self._tables.keys()))

    def _get_previous_queries(self) -> tuple[list[str], list[str]]:
        """
        Extracts previous SQL and user queries from the clarification history.
        """
        previous_sql_queries = [q for q, _ in self.clarification_history if q]
        previous_user_queries = [u for _, u in self.clarification_history if u]
        return previous_sql_queries, previous_user_queries

    @staticmethod
    def _extract_where_filters_from_sql(previous_sql_queries: list[str]) -> dict[str, str]:
        """
        Parses SQL queries to extract WHERE clause filters (value-to-column mappings).
        """
        value_to_column = {}
        pattern = re.compile(r"(?:UPPER|LOWER)?\(?([\w\.]+)\)?\s*(=|ILIKE|LIKE)\s*'([^']+)'", re.IGNORECASE)
        for query in previous_sql_queries:
            where_match = re.search(r"\bWHERE\b(.*)", query, re.IGNORECASE | re.DOTALL)
            if where_match:
                where_clause = where_match.group(1)
                conditions = re.split(r"\s+(?:AND|OR)\s+", where_clause, flags=re.IGNORECASE)
                for condition in conditions:
                    match = pattern.search(condition.strip())
                    if match:
                        column, _, value = match.groups()
                        column = column.split(".")[-1].upper()
                        value = value.strip("%").lower()
                        value_to_column[value] = column
        return value_to_column

    @staticmethod
    async def _classify_chunks_concurrently(
        query_chunks: list[str], extracted_metadata: str, previous_user_queries: list[str]
    ) -> tuple[set[str], set[str]]:
        """
        Runs chunk classification for all query chunks concurrently.

        The method gathers all results, which are expected to be dictionaries
        containing 'cols' and 'vals' keys, and aggregates them into sets.
        """
        chunk_classifier = ChunkClassificationHandler()
        classification_tasks = [
            chunk_classifier.handle(chunk, extracted_metadata, previous_user_queries) for chunk in query_chunks
        ]

        results = await asyncio.gather(*classification_tasks)

        all_columns = set()
        all_values = set()

        for result in results:
            # Assuming the dictionary keys are 'cols' and 'vals'.
            all_columns.update(result.get("COLUMNS", []))
            all_values.update(result.get("VALUES", []))

        return all_columns, all_values

    @staticmethod
    def _start_initial_tasks(query: str) -> asyncio.Task:
        """Helper to start the greeting handler task."""
        greeting_handler = GreetingHandler()
        return asyncio.create_task(greeting_handler.handle(query))

    async def _condense_question(self, query: str) -> str:
        """Helper to condense the user query based on history."""
        if not self._ignore_history:
            question_condensation_handler = QuestionCondenseHandler(self.clarification_history)
            return await question_condensation_handler.handle(query)
        return query

    def _process_metadata_and_query(self, query: str) -> tuple[dict, set, list]:
        """Helper to perform synchronous metadata and query processing."""
        metrics, attributes, columns, _functions = transform_raw_metadata(self._metadata)
        extracted_metadata, used_keywords_from_query = MetadataService(metrics, attributes, columns).invoke(query)
        query_processor = QueryProcessor(max_words=10)
        query_chunks = query_processor.split(query)
        return extracted_metadata, used_keywords_from_query, query_chunks

    @staticmethod
    async def _handle_greetings(greeting_task: asyncio.Task) -> bool:
        """Helper to check and handle a greeting response."""
        greeting_result = await greeting_task
        return isinstance(greeting_result, dict) and greeting_result.get("is_greeting")

    @staticmethod
    def _resolve_tokens(query: str, resolved_phrases: set[str]) -> list[str]:
        """Helper to resolve tokens and identify unresolved ones."""
        single_resolved_words = set()
        for phrase in resolved_phrases:
            single_resolved_words.update(phrase.split())

        date_handler = DateTypeHandler()
        date_tokens = date_handler.handle(query)
        single_resolved_words.update(date_tokens)

        text_preprocessor = TextPreprocessor()
        query_tokens = text_preprocessor.process(query)

        return [token for token in query_tokens if token not in single_resolved_words]

    async def _resolve_unresolved_tokens_async(self, unresolved_tokens: list[str]) -> dict[str, Any]:
        """
        Asynchronously resolves a list of tokens by running all database searches
        concurrently and formats the output into the desired detailed structure.
        """
        if not unresolved_tokens:
            return {
                "resolution_details": [],
                "keys_resolved_values": [],
                "keys_unresolved_values": [],
            }

        logger.debug(f"Attempting to resolve tokens concurrently: {unresolved_tokens}")

        search_tasks = [self.search_engine.perform_combined_search_async(token) for token in unresolved_tokens]
        suggestion_tasks = [self.search_engine.perform_spelling_suggestions_async(token) for token in unresolved_tokens]

        all_search_results = await asyncio.gather(*search_tasks)
        all_suggestion_results = await asyncio.gather(*suggestion_tasks)

        resolution_details = []
        keys_resolved = []
        keys_still_unresolved = []

        for token, separated_results, spelling_suggestions in zip(
            unresolved_tokens, all_search_results, all_suggestion_results, strict=False
        ):
            exact_matches = separated_results.get("exact", [])

            if exact_matches:
                keys_resolved.append(token)
                mappings = defaultdict(set)
                for match in exact_matches:
                    table, attribute = match.get("table"), match.get("attribute")
                    if table and attribute:
                        mappings[table].add(attribute)

                final_mappings = {table: sorted(attrs) for table, attrs in mappings.items()}
                resolution_details.append({
                    "token": token,
                    "resolved": "Yes",
                    "key_value_mappings": final_mappings,
                    "spelling_suggestion": [],
                })
            else:
                keys_still_unresolved.append(token)
                fuzzy_matches = separated_results.get("fuzzy", [])
                suggestions = set(spelling_suggestions or [])
                for f_match in fuzzy_matches:
                    suggestions.update(f_match.get("matched_values", []))

                resolution_details.append({
                    "token": token,
                    "resolved": "No",
                    "key_value_mappings": {},
                    "spelling_suggestion": sorted(suggestions),
                })

        logger.info(f"Resolved values: {keys_resolved}, Unresolved values: {keys_still_unresolved}")
        return {
            "resolution_details": resolution_details,
            "keys_resolved_values": keys_resolved,
            "keys_unresolved_values": keys_still_unresolved,
        }

    async def _classify_and_process(
        self, query_chunks: list[str], extracted_metadata: dict[str, Any]
    ) -> tuple[set, set]:
        """Helper to handle concurrent chunk classification and return results."""
        _previous_sql_queries, previous_user_queries = self._get_previous_queries()

        classification_task = asyncio.create_task(
            self._classify_chunks_concurrently(query_chunks, extracted_metadata, previous_user_queries)
        )

        return await classification_task

    def _assemble_final_state(self, **kwargs: object) -> None:
        """Helper to assemble the final state dictionary."""
        previous_sql_queries, previous_user_queries = self._get_previous_queries()
        value_to_column_map = self._extract_where_filters_from_sql(previous_sql_queries)

        filtered_mappings = filter_value_mappings_by_query(value_to_column_map, kwargs.get("query"))
        self.state["user_query"] = kwargs.get("query")
        self.state["condensed_query"] = kwargs.get("condensed_query")
        self.state["matched_memory"] = kwargs.get("matched_memory")
        self.state["key_resolution"] = kwargs.get("resolved_keys_info")
        self.state["value_mappings"] = filtered_mappings
        self.state["extracted_metadata"] = kwargs.get("extracted_metadata")
        self.state["unresolved_tokens"] = kwargs.get("unresolved_tokens")
        all_columns = kwargs.get("all_columns")
        all_values = kwargs.get("all_values")
        self.state["all_columns"] = normalize_and_filter_terms(list(all_columns), self.state["value_mappings"])
        self.state["all_values"] = normalize_and_filter_terms(list(all_values), self.state["value_mappings"])
        self.state["previous_user_queries"] = previous_user_queries

    async def invoke(self, query: str) -> dict[str, Any]:
        """
        The main handler method to process a user's query.
        """
        await self._initialize_assets_and_search()
        greeting_task = self._start_initial_tasks(query)
        condensed_query = await self._condense_question(query)

        extracted_metadata, used_keywords_from_query, query_chunks = self._process_metadata_and_query(query)

        query_processor = QueryProcessor(max_words=10)
        query_chunks = query_processor.split(query)

        resolved_phrases = set(used_keywords_from_query)
        single_resolved_words = set()
        for phrase in resolved_phrases:
            single_resolved_words.update(phrase.split())

        if await self._handle_greetings(greeting_task):
            return {"is_greeting": True, "response": await greeting_task}

        matched_memory = await match_memory(query, self.agent_id, self.user_id)

        unresolved_tokens = list(self._resolve_tokens(query, used_keywords_from_query) - matched_memory.keys())

        token_resolution_task = asyncio.create_task(self._resolve_unresolved_tokens_async(unresolved_tokens))

        resolved_keys_info = await asyncio.gather(token_resolution_task)

        all_columns, all_values = await self._classify_and_process(query_chunks, extracted_metadata)

        self._assemble_final_state(
            query=query,
            condensed_query=condensed_query,
            extracted_metadata=extracted_metadata,
            all_columns=all_columns,
            all_values=all_values,
            resolved_keys_info=resolved_keys_info,
            matched_memory=matched_memory,
            unresolved_tokens=unresolved_tokens,
        )

        return self.state
