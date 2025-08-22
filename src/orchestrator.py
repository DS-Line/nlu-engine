import asyncio
import re
from collections import defaultdict
from typing import Any

import nltk
from decouple import config
from nltk.corpus import stopwords

from src.exceptions import AgentIDNotFoundError, DatabaseConfigNotFoundError
from src.extractors.date_type_extractor import DateTypeHandler
from src.handlers.chunk_classificaiton_handler import ChunkClassificationHandler
from src.handlers.greeting_handler import GreetingHandler
from src.handlers.question_condensation_handler import QuestionCondenseHandler
from src.managers.memory_manager import match_memory
from src.managers.mongodb_manager import log_database_diagnostics, setup_mongo
from src.metadata.loader import get_metadata_from_directory
from src.processors.query_processor import QueryProcessor
from src.processors.text_preprocessor import TextPreprocessor
from src.services.data_search_service import AsyncSearchEngine
from src.services.metadata_service import MetadataService
from src.utils.logger import create_logger
from src.utils.metadata_helper import filter_value_mappings_by_query, transform_raw_metadata
from src.utils.responses import BaseResponse, success_response

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
        self._metadata = kwargs.get("metadata", [{}, {}, {}, {}])
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
        if self._database_config is None:
            raise DatabaseConfigNotFoundError("Database configuration not found")
        try:
            stopwords.words("english")
            nltk.data.find("tokenizers/punkt")
        except (LookupError, OSError):
            nltk.download("stopwords", quiet=True)
            nltk.download("punkt", quiet=True)

    async def _setup_mongodb(self) -> None:
        """
        Checks MongoDB connection and initializes the async search engine.
        """
        mongo_manager = await setup_mongo()
        await log_database_diagnostics(mongo_manager, self._tables)

        self.search_engine = AsyncSearchEngine(mongo_manager=mongo_manager, tables_to_sync=list(self._tables.keys()))

    def _load_metadata(self) -> None:
        """
        Loads metadata for all tables and stores it in self._metadata.
        """
        asset_names_for_loader = [f"sem_{t.lower()}" for t in self._tables]
        self._metadata = list(get_metadata_from_directory(asset_names_for_loader))

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
            if not where_match:
                continue
            where_clause = where_match.group(1)
            conditions = re.split(r"\s+(?:AND|OR)\s+", where_clause, flags=re.IGNORECASE)
            for condition in conditions:
                match = pattern.search(condition.strip())
                if not match:
                    continue
                column, _, value = match.groups()
                value_to_column[value.strip("%").lower()] = column.split(".")[-1].upper()
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

        all_columns = {col for result in results for col in result.get("COLUMNS", [])}
        all_values = {val for result in results for val in result.get("VALUES", [])}

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

    def _process_metadata_and_query(self, query: str) -> tuple[dict, set, list, dict]:
        """Helper to perform synchronous metadata and query processing."""
        metrics, attributes, columns, _functions = transform_raw_metadata(self._metadata)
        extracted_metadata, used_keywords_from_query = MetadataService(metrics, attributes, columns).invoke(query)
        query_processor = QueryProcessor(max_words=10)
        query_chunks = query_processor.split(query)
        return extracted_metadata, used_keywords_from_query, query_chunks, columns

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

        search_results, suggestion_results = await self._run_searches(unresolved_tokens)

        resolution_details: list[dict[str, Any]] = []
        keys_resolved: list[str] = []
        keys_unresolved: list[str] = []

        for token, separated_results, spelling_suggestions in zip(
            unresolved_tokens, search_results, suggestion_results, strict=False
        ):
            self._process_token_resolution(
                token=token,
                separated_results=separated_results,
                spelling_suggestions=spelling_suggestions,
                keys_resolved=keys_resolved,
                keys_unresolved=keys_unresolved,
                resolution_details=resolution_details,
            )

        simplified_dict = self._build_simplified_dict(resolution_details)
        logger.info(f"Resolved values: {keys_resolved}, Unresolved values: {keys_unresolved}")
        return simplified_dict

    async def _run_searches(self, tokens: list[str]) -> tuple[list[Any], list[Any]]:
        """Runs both combined search and spelling suggestion tasks concurrently."""
        search_tasks = [self.search_engine.perform_combined_search_async(token) for token in tokens]
        suggestion_tasks = [self.search_engine.perform_spelling_suggestions_async(token) for token in tokens]
        return await asyncio.gather(*search_tasks), await asyncio.gather(*suggestion_tasks)

    def _process_token_resolution(self, **kwargs: object) -> None:
        """Processes a single token and updates resolution lists/details."""
        token: str = kwargs["token"]
        separated_results: dict[str, Any] = kwargs["separated_results"]
        spelling_suggestions: list[str] | None = kwargs.get("spelling_suggestions")
        keys_resolved: list[str] = kwargs["keys_resolved"]
        keys_unresolved: list[str] = kwargs["keys_unresolved"]
        resolution_details: list[dict[str, Any]] = kwargs["resolution_details"]
        exact_matches = separated_results.get("exact", [])
        if exact_matches:
            keys_resolved.append(token)
            resolution_details.append({
                "token": token,
                "resolved": "Yes",
                "key_value_mappings": self._build_exact_mapping(exact_matches),
                "spelling_suggestion": [],
            })
            return

        keys_unresolved.append(token)
        resolution_details.append({
            "token": token,
            "resolved": "No",
            "key_value_mappings": {},
            "spelling_suggestion": self._build_fuzzy_suggestions(separated_results, spelling_suggestions),
        })

    @staticmethod
    def _build_exact_mapping(exact_matches: list[dict[str, Any]]) -> dict[str, list[str]]:
        """Builds table-to-attributes mapping from exact matches."""
        mappings: dict[str, set[str]] = defaultdict(set)
        for match in exact_matches:
            table, attribute = match.get("table"), match.get("attribute")
            if table and attribute:
                mappings[table].add(attribute)
        return {table: sorted(attrs) for table, attrs in mappings.items()}

    @staticmethod
    def _build_fuzzy_suggestions(
        separated_results: dict[str, Any], spelling_suggestions: list[str] | None
    ) -> list[str]:
        """Combines fuzzy match values and spelling suggestions into a sorted list."""
        suggestions = set(spelling_suggestions or [])
        for f_match in separated_results.get("fuzzy", []):
            suggestions.update(f_match.get("matched_values", []))
        return sorted(suggestions)

    @staticmethod
    def _build_simplified_dict(resolution_details: list[dict[str, Any]]) -> dict[str, list[str]]:
        """Builds a simplified token-to-columns mapping for resolved tokens."""
        return {
            item["token"]: [
                f"{table}.{field}" for table, fields in item["key_value_mappings"].items() for field in fields
            ]
            for item in resolution_details
            if item["resolved"] == "Yes"
        }

    async def _classify_and_process(self, query_chunks: list[str], extracted_metadata: str) -> tuple[set, set]:
        """Helper to handle concurrent chunk classification and return results."""
        _previous_sql_queries, previous_user_queries = self._get_previous_queries()

        classification_task = asyncio.create_task(
            self._classify_chunks_concurrently(query_chunks, extracted_metadata, previous_user_queries)
        )

        return await classification_task

    def _assemble_final_state(self, **kwargs: object) -> None:
        """Helper to assemble the final state dictionary."""
        previous_sql_queries, _previous_user_queries = self._get_previous_queries()
        value_to_column_map = self._extract_where_filters_from_sql(previous_sql_queries)

        filtered_mappings = filter_value_mappings_by_query(value_to_column_map, kwargs.get("query"))
        self.state["query"] = kwargs.get("query")
        self.state["memory_mappings"] = kwargs.get("memory_mappings")
        self.state["attribute_mappings"] = kwargs.get("resolved_keys_info")
        self.state["context_from_conditions"] = filtered_mappings
        self.state["metadata"] = kwargs.get("metadata")
        self.state["final_columns"] = kwargs.get("final_columns")

    async def invoke(self, query: str) -> dict[str, Any] | BaseResponse:
        """
        The main handler method to process a user's query.
        """
        greeting_task = asyncio.create_task(GreetingHandler().handle(query))
        setup_mongo_task = asyncio.create_task(self._setup_mongodb())
        condense_task = asyncio.create_task(self._condense_question(query))

        if config("LOAD_METADATA", default="True") == "True":
            self._load_metadata()
        extracted_metadata, used_keywords_from_query, _query_chunks, final_columns = self._process_metadata_and_query(
            query
        )

        resolved_phrases = set(used_keywords_from_query)
        single_resolved_words = set()
        for phrase in resolved_phrases:
            single_resolved_words.update(phrase.split())

        _setup_mongo, query, greeting_result = await asyncio.gather(setup_mongo_task, condense_task, greeting_task)
        if isinstance(greeting_result, dict) and greeting_result.get("is_greeting"):
            return success_response(message="Greeting Handled", data=greeting_result.get("response"))
        (memory_mappings,) = await asyncio.gather(asyncio.create_task(match_memory(query, self.agent_id, self.user_id)))

        unresolved_tokens = list(self._resolve_tokens(query, used_keywords_from_query) - memory_mappings.keys())
        (resolved_keys_info,) = await asyncio.gather(
            asyncio.create_task(self._resolve_unresolved_tokens_async(unresolved_tokens))
        )
        # TODO: Use _classify_and_process  to handle query_chunks and extracted_metadata with asyncio tasks.
        #       Filter _all_columns and _all_values using normalize_and_filter_terms against value mappings.

        self._assemble_final_state(
            query=query,
            metadata=extracted_metadata,
            resolved_keys_info=resolved_keys_info,
            memory_mappings=memory_mappings,
            final_columns=final_columns,
        )

        return self.state
