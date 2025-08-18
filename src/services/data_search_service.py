import concurrent
import difflib
import logging
import re
from typing import Any

from pymongo.collection import Collection

from src.managers.mongodb_manager import MongoDBManager

logger = logging.getLogger(__name__)
FUZZY_MATCH_THRESHOLD = 0.85
VALUE_SIMILARITY_THRESHOLD = 0.8
VALUE_MATCH_THRESHOLD = 0.8


class SearchEngine:
    """
    Handles all search operations (fuzzy, partial, combined) against the MongoDB collections.
    This class now centralizes search logic, making it reusable.
    """

    def __init__(
        self, mongo_manager: MongoDBManager, tables_to_sync: dict[str, list[str]], max_workers: int = 4
    ) -> None:
        self.mongo_manager = mongo_manager
        self.tables_to_sync = tables_to_sync
        self.max_workers = max_workers
        self.metric_threshold = 0.8

    def _get_collection_info(self, table_name: str) -> tuple[Collection, bool]:
        """Returns the collection and a flag indicating if it's a single-document collection."""
        collection = self.mongo_manager.get_collection(f"{table_name}_attributes")
        is_single_doc = collection.count_documents({}) == 1
        return collection, is_single_doc

    @staticmethod
    def _search_single_doc_partial(doc: dict[str, Any], search_term: str, table_name: str) -> list[dict[str, Any]]:
        """Helper for partial search on a single-document collection."""
        results = []
        search_term_lower = search_term.lower()
        if doc:
            for attribute, text in doc.get("search_text", {}).items():
                if search_term_lower in text.lower():
                    results.append({"table": table_name, "attribute": attribute, "match_type": "partial_match"})
        return results

    @staticmethod
    def _search_multi_doc_partial(collection: Collection, search_term: str, table_name: str) -> list[dict[str, Any]]:
        """Helper for partial search on a multiple-document collection."""
        regex_pattern = re.escape(search_term)
        cursor = collection.find({"search_text": {"$regex": regex_pattern, "$options": "i"}})
        return [
            {"table": table_name, "attribute": doc["attribute_name"], "match_type": "partial_match"} for doc in cursor
        ]

    def perform_partial_search(self, table_name: str, search_term: str) -> list[dict[str, Any]]:
        """Performs a partial search for a given search term within a table's attributes."""
        collection, is_single_doc = self._get_collection_info(table_name)
        if collection.count_documents({}) == 0:
            return []
        if is_single_doc:
            doc = collection.find_one({})
            return self._search_single_doc_partial(doc, search_term, table_name)
        return self._search_multi_doc_partial(collection, search_term, table_name)

    def _search_single_doc_fuzzy(
        self, collection: Collection, doc: dict[str, Any], search_term: str, table_name: str
    ) -> list[dict[str, Any]]:
        """Helper for fuzzy search on a single-document collection."""
        results = []
        try:
            query = {"$text": {"$search": f'"{search_term}"'}}
            projection = {"score": {"$meta": "textScore"}}
            text_search_doc = collection.find_one(query, projection)
            if text_search_doc and "search_text" in doc:
                results.extend([
                    {
                        "table": table_name,
                        "attribute": column,
                        "match_type": "fuzzy_match",
                        "score": text_search_doc.get("score", 1.0),
                    }
                    for column in doc["search_text"]
                    if search_term.lower() in doc["search_text"][column].lower()
                ])

        except Exception as e:
            logger.debug(f"Text search failed for single doc collection {collection.name}: {e}")
            results = self._search_single_doc_fuzzy_fallback(doc, search_term, table_name)
        return results

    @staticmethod
    def _search_single_doc_fuzzy_fallback(
        doc: dict[str, Any], search_term: str, table_name: str
    ) -> list[dict[str, Any]]:
        """Fallback for fuzzy search on single doc, using difflib."""
        results = []
        if doc and "attributes" in doc:
            for attribute in doc["attributes"]:
                sample_values = doc["attributes"][attribute].get("sample_values", [])
                for val in sample_values:
                    fuzzy_match_score = difflib.SequenceMatcher(None, search_term.lower(), str(val).lower()).ratio()
                    if fuzzy_match_score >= FUZZY_MATCH_THRESHOLD:
                        results.append({
                            "table": table_name,
                            "attribute": attribute,
                            "match_type": "fuzzy_match",
                            "score": round(fuzzy_match_score, 2),
                            "matched_values": val,
                        })
        return results

    def _search_multi_doc_fuzzy(
        self, collection: Collection, search_term: str, table_name: str
    ) -> list[dict[str, Any]]:
        """Helper for fuzzy search on a multiple-document collection."""
        results = []
        try:
            search_phrases = [f'"{word}"' for word in search_term.split()]
            query = {"$text": {"$search": " ".join(search_phrases)}}
            projection = {"score": {"$meta": "textScore"}}
            cursor = collection.find(query, projection).sort([("score", {"$meta": "textScore"})])
            results.extend([
                {
                    "table": table_name,
                    "attribute": doc["attribute_name"],
                    "match_type": "fuzzy_match",
                    "score": doc.get("score", 1.0),
                }
                for doc in cursor
                if "attribute_name" in doc
            ])

        except Exception as e:
            logger.debug(f"Text search failed for multi-doc collection {collection.name}: {e}")
            results = self.perform_partial_search(table_name, search_term)
        return results

    def perform_fuzzy_search(self, table_name: str, search_term: str) -> list[dict[str, Any]]:
        """Performs a fuzzy search on a given table's attributes."""
        collection, is_single_doc = self._get_collection_info(table_name)
        if collection.count_documents({}) == 0:
            return []

        results = []
        if is_single_doc:
            doc = collection.find_one({})
            results = self._search_single_doc_fuzzy(collection, doc, search_term, table_name)
        else:
            results = self._search_multi_doc_fuzzy(collection, search_term, table_name)

        results.sort(key=lambda x: x.get("score", 0), reverse=True)
        return results

    @staticmethod
    def _process_single_attribute_match(**kwargs: object) -> None:
        """Helper to check for different match types for a single attribute."""

        table = kwargs.get("table")
        attribute = kwargs.get("attribute")
        sample_values = kwargs.get("sample_values", [])
        search_term_lower = kwargs.get("search_term_lower", "")
        results_list = kwargs.get("results_list", [])
        search_text = kwargs.get("search_text")

        exact_match = search_term_lower in search_text
        value_similarity = any(
            difflib.SequenceMatcher(None, search_term_lower, str(val).lower()).ratio() >= VALUE_SIMILARITY_THRESHOLD
            for val in sample_values
        )
        fuzzy_match = exact_match or value_similarity

        if exact_match or fuzzy_match:
            matched_vals = [
                val
                for val in sample_values
                if (search_term_lower in str(val).lower())
                or (difflib.SequenceMatcher(None, search_term_lower, str(val).lower()).ratio() >= VALUE_MATCH_THRESHOLD)
            ]
            if matched_vals:
                results_list.append({
                    "table": table.upper(),
                    "attribute": attribute.lower(),
                    "exact_match": exact_match,
                    "fuzzy_match": fuzzy_match,
                    "matched_values": matched_vals,
                })

    @staticmethod
    def _get_unique_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Filters out duplicate results based on table and attribute name."""
        seen = set()
        unique_results = []
        for result in results:
            key = (result["table"], result["attribute"])
            if key not in seen:
                seen.add(key)
                unique_results.append(result)
        return unique_results

    def perform_combined_search(self, search_term: str) -> list[dict[str, Any]]:
        """Executes a series of searches and consolidates the results."""
        all_results = []
        search_term_lower = search_term.lower()

        for table_name in self.tables_to_sync:
            try:
                collection, is_single_doc = self._get_collection_info(table_name)
                if collection.count_documents({}) == 0:
                    continue

                if is_single_doc:
                    self._process_single_doc_collection(collection, table_name, search_term_lower, all_results)
                else:
                    self._process_multi_doc_collection(collection, table_name, search_term_lower, all_results)
            except Exception as e:
                logger.error(f"Error during combined search in table {table_name}: {e}")

        return self._get_unique_results(all_results)

    def _process_single_doc_collection(
        self,
        collection: Collection,
        table_name: str,
        search_term_lower: str,
        all_results: list,
    ) -> None:
        doc = collection.find_one({})
        if not doc:
            return

        for attribute, data in doc.get("attributes", {}).items():
            self._process_single_attribute_match(
                table=table_name,
                attribute=attribute,
                search_text=data.get("search_text", ""),
                sample_values=data.get("sample_values", []),
                search_term_lower=search_term_lower,
                results_list=all_results,
            )

    def _process_multi_doc_collection(
        self,
        collection: Collection,
        table_name: str,
        search_term_lower: str,
        all_results: list,
    ) -> None:
        for doc in collection.find():
            attribute = doc.get("attribute_name")
            if not attribute:
                continue

            self._process_single_attribute_match(
                table=table_name,
                attribute=attribute,
                search_text=doc.get("search_text", ""),
                sample_values=doc.get("sample_values", []),
                search_term_lower=search_term_lower,
                results_list=all_results,
            )

    def perform_batch_combined_search(self, search_terms: list[str]) -> dict[str, Any]:
        """
        Performs a batch combined search for a list of search terms.
        Uses a thread pool to run searches for each term concurrently.
        """
        if not search_terms:
            return {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_term = {executor.submit(self.perform_combined_search, term): term for term in search_terms}
            batch_results = {}
            for future in concurrent.futures.as_completed(future_to_term):
                term = future_to_term[future]
                try:
                    results = future.result()
                    batch_results[term] = results
                except Exception as e:
                    logger.error(f"Error during batch search for term '{term}': {e}")
                    batch_results[term] = []
        return batch_results
