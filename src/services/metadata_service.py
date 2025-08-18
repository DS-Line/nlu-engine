import logging
import re
from operator import itemgetter
from typing import Any

import yaml
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

logger = logging.getLogger(__name__)


def remove_keys(item_data_items: dict[str, Any]) -> None:
    """
    Removes 'primary_key' and 'foreign_key' from the dictionary if their values are explicitly False.
    """
    if item_data_items.get("primary_key") is False:
        del item_data_items["primary_key"]
    if item_data_items.get("foreign_key") is False:
        del item_data_items["foreign_key"]


class NoAliasDumper(yaml.SafeDumper):
    """Custom YAML dumper that disables aliasing."""

    @staticmethod
    def ignore_aliases(_data: object) -> bool:
        """Disable YAML aliases for all data."""
        return True


class MetadataService:
    """
    An optimized service to load, index, and query metadata.
    It replaces repetitive searching with fast, pre-computed indices.
    """

    def __init__(self, metrics: dict, attributes: dict, columns: dict) -> None:
        """
        Initializes the service by loading, validating, and indexing all metadata.
        This one-time setup makes subsequent queries very fast.
        """
        self._master_index: dict[str, dict] = {}
        self._keyword_to_ids: dict[str, set[str]] = {}
        self._dependency_graph: dict[str, set[str]] = {}
        self._lemmatizer = WordNetLemmatizer()
        self._stop_words = set(stopwords.words("english"))

        self._load_and_index_all(metrics, attributes, columns)
        self._build_keyword_and_word_indices()
        self._build_dependency_graph()

    def _process_text(self, text: str, *, as_set: bool = False) -> any:
        """
        Processes a text string by tokenizing, removing stop words, and lemmatizing.
        Returns a string or a set of unique lemmas based on the as_set flag.
        """
        text = text.replace("-", " ")
        tokens = word_tokenize(text.lower())
        filtered_tokens = [token for token in tokens if token.isalpha() and token not in self._stop_words]
        lemmas = [self._lemmatizer.lemmatize(token) for token in filtered_tokens]
        return set(lemmas) if as_set else " ".join(lemmas)

    def _load_and_index_all(self, metrics: dict, attributes: dict, columns: dict) -> None:
        """
        Loads all data from different sources into a unified master index.
        """
        all_metadata = {"metric": metrics, "attribute": attributes, "column": columns}

        for item_type, data in all_metadata.items():
            if not data:
                continue
            for item_id, values in data.items():
                self._master_index[item_id] = {**values, "_type": item_type}

    def _build_keyword_and_word_indices(self) -> None:
        """
        Builds the inverted keyword index and the set of valid words using processed text.
        """
        for item_id, values in self._master_index.items():
            all_names = [values.get("name", ""), *values.get("synonym", []), *values.get("synonyms", [])]
            for name in all_names:
                if not name:
                    continue
                processed_term = self._process_text(name)
                if processed_term not in self._keyword_to_ids:
                    self._keyword_to_ids[processed_term] = set()
                self._keyword_to_ids[processed_term].add(item_id)

    def _build_dependency_graph(self) -> None:
        """
        Builds a directed graph of dependencies from 'calculation', 'include', and 'filters'.
        An edge from A -> B means A depends on B.
        """
        bracket_pattern = re.compile(r"\[([^\]]+)\]")
        raw_col_pattern = re.compile(r"\b([A-Z_]{2,})\b")
        table_col_pattern = re.compile(r"\b[A-Z_]+\.([A-Z_]+)\b")

        for item_id, data in self._master_index.items():
            dependencies = set(data.get("include", []))
            strings_to_check = [data.get("calculation", ""), *data.get("filters", []), *data.get("filter", [])]

            for s in strings_to_check:
                if not s:
                    continue
                dependencies.update(bracket_pattern.findall(s))
                dependencies.update(raw_col_pattern.findall(s))
                dependencies.update(table_col_pattern.findall(s))

            self._dependency_graph[item_id] = {dep for dep in dependencies if dep in self._master_index}

    def _find_seed_matches(self, query: str) -> tuple[set[str], set[str]]:
        """
        Finds seed matches by checking if all lemmas from a metadata keyword
        are present in the set of lemmas from the user query.

        Returns a tuple containing the set of matched IDs and the set of keywords used for matching.
        """
        seed_ids = set()
        used_keywords = set()
        query_lemmas = self._process_text(query, as_set=True)

        potential_matches = []

        for keyword, ids in self._keyword_to_ids.items():
            keyword_lemmas = set(keyword.split())

            if keyword_lemmas.issubset(query_lemmas):
                potential_matches.append((len(keyword_lemmas), keyword, ids))

        potential_matches.sort(key=itemgetter(0), reverse=True)

        if potential_matches:
            best_len = potential_matches[0][0]
            for length, keyword, ids in potential_matches:
                if length == best_len:
                    seed_ids.update(ids)
                    used_keywords.add(keyword)

        return seed_ids, used_keywords

    def _resolve_all_dependencies(self, seed_ids: set[str]) -> set[str]:
        """
        Performs a graph traversal (BFS) to find all items required by the initial seeds.
        """
        queue = list(seed_ids)
        visited = set(seed_ids)

        head = 0
        while head < len(queue):
            current_id = queue[head]
            head += 1

            for dependency_id in self._dependency_graph.get(current_id, set()):
                if dependency_id not in visited:
                    visited.add(dependency_id)
                    queue.append(dependency_id)

        return visited

    def _aggregate_and_format_sources(self, all_required_ids: set[str]) -> list[dict]:
        """
        Collects all unique source definitions and formats them for the final output.
        """
        all_source_definitions = set()
        for item_id in all_required_ids:
            if item_id not in self._master_index:
                continue

            item_data = self._master_index[item_id]
            item_type = item_data.get("_type")

            if item_type == "column" and "table_source" in item_data:
                for src in item_data["table_source"]:
                    if isinstance(src, dict):
                        all_source_definitions.add(yaml.dump(src, Dumper=NoAliasDumper))

        source_list = [yaml.safe_load(s) for s in all_source_definitions]
        sorted_sources = sorted(source_list, key=lambda x: x.get("table", ""))

        formatted_sources = []
        for src in sorted_sources:
            ordered_src = {"table": src.get("table")}
            if src.get("joins"):
                ordered_src["joins"] = src["joins"]
            formatted_sources.append(ordered_src)

        return formatted_sources

    def _categorize_required_items(self, all_required_ids: set[str]) -> tuple:
        """
        Sorts the required metadata items into metrics, attributes, and columns,
        and simplifies the table source for each column.
        """

        matched_metrics, matched_attributes, matched_columns = {}, {}, {}

        category_map = {
            "metric": matched_metrics,
            "attribute": matched_attributes,
            "column": matched_columns,
        }

        for item_id in all_required_ids:
            if item_id not in self._master_index:
                continue

            item_data = self._master_index[item_id].copy()
            item_type: str | None = item_data.pop("_type", None)

            if item_type == "column" and "table_source" in item_data:
                primary_source = None
                full_source_list = item_data.get("table_source", [])

                for src in full_source_list:
                    if isinstance(src, dict) and not src.get("joins"):
                        primary_source = src.get("table")
                        break

                if not primary_source and full_source_list:
                    primary_source = full_source_list[0].get("table")

                item_data["table_source"] = [primary_source] if primary_source else []

            remove_keys(item_data)

            if item_type is not None and item_type in category_map:
                category_map[item_type][item_id] = item_data

        return matched_metrics, matched_attributes, matched_columns

    def _format_final_output(self, all_required_ids: set[str]) -> str:
        """
        Aggregates and formats the final metadata into a clean YAML string.
        """
        formatted_sources = self._aggregate_and_format_sources(all_required_ids)
        metrics, attributes, columns = self._categorize_required_items(all_required_ids)

        final_metadata = {
            "sources": formatted_sources or "No Source Found",
            "metrics": metrics or "No Metrics Matched",
            "attributes": attributes or "No Attributes Matched",
            "columns": columns or "No Columns Matched",
        }

        return yaml.dump(
            final_metadata,
            default_flow_style=False,
            sort_keys=False,
            Dumper=NoAliasDumper,
        )

    def process_query(self, user_query: str) -> str:
        """
        Processes a user query to find all relevant metadata.
        """
        seed_ids, used_keywords = self._find_seed_matches(user_query)
        all_required_ids = self._resolve_all_dependencies(seed_ids)

        if not all_required_ids:
            return "No metadata found for the given query.", used_keywords

        formatted_output = self._format_final_output(all_required_ids)

        return formatted_output, used_keywords

    def invoke(self, query: str) -> tuple[Any, Any] | None:
        """
        Process the input query and return extracted metadata and used keywords.

        Args:
            query (str): The input query string.

        Returns:
            Optional[Tuple[Any, Any]]: A tuple containing extracted metadata and used keywords,
            or None if processing fails.
        """
        try:
            extracted_metadata, used_keywords = self.process_query(query)
            logger.info(f"Extracted metadata: {extracted_metadata}")
            return extracted_metadata, used_keywords
        except Exception as e:
            logger.info("Skipping MetadataService Creation")
            logger.error(f"Error extracting metadata: {e!s}")
            raise
        return None, None
