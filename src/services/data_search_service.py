import asyncio
import difflib
import logging
import re
from typing import Any

try:
    from nltk.corpus import wordnet
    from nltk.stem import WordNetLemmatizer
except ImportError:
    wordnet = None
    WordNetLemmatizer = None

from motor.motor_asyncio import AsyncIOMotorCollection

from src.managers.mongodb_manager import AsyncMongoDBManager

logger = logging.getLogger(__name__)
SINGLE_CHAR_CHECK_LENGTH = 3
FUZZY_THRESHOLD = 0.8
SHORT_WORD_LENGTH = 4
MEDIUM_WORD_LENGTH = 8
SHORT_WORD_MAX_EDIT = 1
MEDIUM_WORD_MAX_EDIT = 2
LONG_WORD_MAX_EDIT = 3


class AsyncSearchEngine:
    """
    Handles all ASYNCHRONOUS search operations (combined, spelling) against MongoDB collections.
    This class is designed to work with an async event loop.
    """

    def __init__(self, mongo_manager: AsyncMongoDBManager, tables_to_sync: list[str]) -> None:
        self.mongo_manager = mongo_manager
        self.tables_to_sync = tables_to_sync
        self.lemmatizer = WordNetLemmatizer() if WordNetLemmatizer else None

    async def perform_combined_search_async(self, search_term: str) -> dict[str, list[dict[str, Any]]]:
        """
        Asynchronously executes a search for a single term across all configured tables
        and returns separated exact and fuzzy results.
        """
        exact_results = []
        fuzzy_results = []
        search_term_lower = search_term.lower()

        table_search_tasks = [
            self._search_single_table_async(table_name, search_term_lower) for table_name in self.tables_to_sync
        ]

        all_table_results = await asyncio.gather(*table_search_tasks)

        for table_result in all_table_results:
            exact_results.extend(table_result["exact"])
            fuzzy_results.extend(table_result["fuzzy"])

        return {"exact": self._get_unique_results(exact_results), "fuzzy": self._get_unique_results(fuzzy_results)}

    async def perform_spelling_suggestions_async(self, search_term: str) -> list[str]:
        """
        Asynchronously searches the database for values that are likely spelling
        corrections for the given search term.
        """
        suggestions = set()
        search_term_lower = search_term.lower()
        search_term_is_proper = self._is_proper_noun_lemma(search_term)

        suggestion_tasks = [
            self._get_suggestions_from_table_async(
                table_name, search_term_lower, search_term_is_proper=search_term_is_proper
            )
            for table_name in self.tables_to_sync
        ]

        suggestion_sets = await asyncio.gather(*suggestion_tasks)

        for s_set in suggestion_sets:
            suggestions.update(s_set)

        filtered_suggestions = self._apply_spacing_preference(search_term_lower, suggestions)
        sorted_suggestions = sorted(
            filtered_suggestions,
            key=lambda x: (
                self._levenshtein_distance(search_term_lower, x.lower()),
                -difflib.SequenceMatcher(None, search_term_lower, x.lower()).ratio(),
            ),
        )
        return sorted_suggestions[:10]

    async def _search_single_table_async(self, table_name: str, search_term_lower: str) -> dict[str, list]:
        """Helper to run the combined search on a single table."""
        exact_results = []
        fuzzy_results = []
        try:
            collection, is_single_doc = await self._get_collection_info(table_name)

            if is_single_doc:
                doc = await collection.find_one({})
                if doc:
                    for attribute, data in doc.get("attributes", {}).items():
                        self._process_separate_attribute_matches(
                            table=table_name,
                            attribute=attribute,
                            search_text=data.get("search_text", ""),
                            sample_values=data.get("sample_values", []),
                            search_term_lower=search_term_lower,
                            exact_results=exact_results,
                            fuzzy_results=fuzzy_results,
                        )
            else:
                async for doc in collection.find():
                    attribute = doc.get("attribute_name")
                    if not attribute:
                        continue
                    self._process_separate_attribute_matches(
                        table=table_name,
                        attribute=attribute,
                        search_text=doc.get("search_text", ""),
                        sample_values=doc.get("sample_values", []),
                        search_term_lower=search_term_lower,
                        exact_results=exact_results,
                        fuzzy_results=fuzzy_results,
                    )
        except Exception as e:
            logger.error(f"Error during async search in table {table_name}: {e}")

        return {"exact": exact_results, "fuzzy": fuzzy_results}

    def _add_valid_suggestions(
        self,
        suggestions: set[str],
        search_term_lower: str,
        values: list[Any],
        *,
        search_term_is_proper: bool,
    ) -> None:
        """Add valid spelling suggestions from a list of values."""
        for val in values:
            if not isinstance(val, str) or val.isdigit():
                continue
            if self._is_spelling_match(search_term_lower, val, search_term_is_proper=search_term_is_proper):
                suggestions.add(val)

    async def _get_suggestions_from_table_async(
        self, table_name: str, search_term_lower: str, *, search_term_is_proper: bool
    ) -> set[str]:
        """Helper to get spelling suggestions from a single table."""
        suggestions = set()
        try:
            collection, is_single_doc = await self._get_collection_info(table_name)

            if is_single_doc:
                doc = await collection.find_one({})
                if doc:
                    for attribute_values in doc.get("attributes", {}).values():
                        self._add_valid_suggestions(
                            suggestions,
                            search_term_lower,
                            attribute_values.get("sample_values", []),
                            search_term_is_proper=search_term_is_proper,
                        )
            else:
                async for doc in collection.find():
                    self._add_valid_suggestions(
                        suggestions,
                        search_term_lower,
                        doc.get("sample_values", []),
                        search_term_is_proper=search_term_is_proper,
                    )
        except Exception as e:
            logger.error(f"Error getting async spelling suggestions from table {table_name}: {e}")
        return suggestions

    async def _get_collection_info(self, table_name: str) -> tuple[AsyncIOMotorCollection, bool]:
        """Asynchronously returns the collection and a flag indicating if it's a single-document collection."""
        collection = self.mongo_manager.get_collection(f"{table_name}_attributes")
        count = await collection.count_documents({})
        return collection, count == 1

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

    @staticmethod
    def _process_separate_attribute_matches(**kwargs: object) -> None:
        """Helper to find exact and fuzzy matches and append them to separate result lists."""
        table: str = kwargs.get("table")
        attribute: str = kwargs.get("attribute")
        search_text: str = kwargs.get("search_text")
        sample_values: list[Any] = kwargs.get("sample_values", [])
        search_term_lower: str = kwargs.get("search_term_lower", "")
        exact_results: list[dict] = kwargs.get("exact_results", [])
        fuzzy_results: list[dict] = kwargs.get("fuzzy_results", [])

        exact_matched_values = [v for v in sample_values if search_term_lower in str(v).lower()]
        is_exact_attribute_name_match = search_term_lower in search_text.lower()

        if is_exact_attribute_name_match or exact_matched_values:
            vals_to_report = exact_matched_values or sample_values
            if vals_to_report:
                exact_results.append({
                    "table": table.upper(),
                    "attribute": attribute.lower(),
                    "matched_values": vals_to_report,
                })

        non_exact_values = [v for v in sample_values if search_term_lower not in str(v).lower()]
        fuzzy_matched_values = [
            v
            for v in non_exact_values
            if difflib.SequenceMatcher(None, search_term_lower, str(v).lower()).ratio() >= FUZZY_THRESHOLD
        ]

        if fuzzy_matched_values:
            fuzzy_results.append({
                "table": table.upper(),
                "attribute": attribute.lower(),
                "matched_values": fuzzy_matched_values,
            })

    def _is_proper_noun_lemma(self, word: str) -> bool:
        """Determines if a word is likely a proper noun."""
        if not word or not isinstance(word, str) or not wordnet or not self.lemmatizer:
            return False

        word_lower = word.lower()

        def has_synsets(word_form: str, pos: str | None = None) -> bool:
            return bool(wordnet.synsets(word_form, pos=pos))

        lemma = self.lemmatizer.lemmatize(word_lower, pos="n")
        if (lemma == word_lower or (word_lower.endswith("s") and lemma == word_lower[:-1])) and has_synsets(
            lemma, pos=wordnet.NOUN
        ):
            return False

        if not has_synsets(word_lower):
            return True

        return word[0].isupper()

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculates the Levenshtein distance between two strings."""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]

    def _is_spelling_match(self, search_term_lower: str, candidate: str, *, search_term_is_proper: bool) -> bool:
        """Checks if a candidate string is a likely spelling match for a search term."""
        candidate_lower = candidate.lower()
        if not candidate_lower or candidate_lower == search_term_lower:
            return False

        candidate_is_proper = self._is_proper_noun_lemma(candidate)
        if search_term_is_proper != candidate_is_proper:
            return False

        search_normalized = self._normalize_for_grouping(search_term_lower)
        candidate_normalized = self._normalize_for_grouping(candidate_lower)

        if search_normalized == candidate_normalized:
            return True
        if abs(len(candidate_lower) - len(search_term_lower)) > max(2, len(search_term_lower) // 4):
            return False

        if " " in search_term_lower or " " in candidate_lower:
            return self._is_multiword_match(search_term_lower, candidate_lower)
        return self._is_single_word_match(search_normalized, candidate_normalized)

    def _is_single_word_match(self, search_word: str, candidate_word: str) -> bool:
        if len(search_word) <= SINGLE_CHAR_CHECK_LENGTH and search_word[0] != candidate_word[0]:
            return False
        edit_dist = self._levenshtein_distance(search_word, candidate_word)

        if len(search_word) <= SHORT_WORD_LENGTH:
            max_edit_dist = SHORT_WORD_MAX_EDIT
        elif len(search_word) <= MEDIUM_WORD_LENGTH:
            max_edit_dist = MEDIUM_WORD_MAX_EDIT
        else:
            max_edit_dist = LONG_WORD_MAX_EDIT

        return edit_dist <= max_edit_dist

    def _is_multiword_match(self, search_phrase: str, candidate_phrase: str) -> bool:
        search_words = search_phrase.split()
        candidate_words = candidate_phrase.split()
        if len(search_words) != len(candidate_words):
            return False
        return all(
            sw == cw or self._is_single_word_match(sw, cw)
            for sw, cw in zip(search_words, candidate_words, strict=False)
        )

    @staticmethod
    def _normalize_for_grouping(text: str) -> str:
        return re.sub(r"[^a-zA-Z0-9]", "", text.lower())

    def _apply_spacing_preference(self, search_term_lower: str, suggestions: set[str]) -> set[str]:
        if not suggestions:
            return set()

        normalized_groups = {}
        for suggestion in suggestions:
            normalized = self._normalize_for_grouping(suggestion.lower())
            if normalized not in normalized_groups:
                normalized_groups[normalized] = []
            normalized_groups[normalized].append(suggestion)

        filtered_suggestions = set()
        for group in normalized_groups.values():
            if len(group) == 1:
                filtered_suggestions.update(group)
            else:
                preferred = self._select_preferred_spacing(search_term_lower, group)
                filtered_suggestions.update(preferred)
        return filtered_suggestions

    @staticmethod
    def _select_preferred_spacing(search_term_lower: str, candidates: list) -> list:
        search_has_spaces = " " in search_term_lower
        spaced_candidates = [c for c in candidates if " " in c]
        no_space_candidates = [c for c in candidates if " " not in c]
        return (
            (spaced_candidates or no_space_candidates)
            if search_has_spaces
            else (no_space_candidates or spaced_candidates)
        )
