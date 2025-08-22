import re
from typing import ClassVar

from nlu_engine.handlers.base import Handler


class RatioTypeHandler(Handler):
    """
    A flexible and robust handler that splits strings representing ratios
    into their constituent parts.
    """

    _RATIO_SEPARATORS: ClassVar[set[str]] = {"to", "and", "vs"}
    _IGNORED_KEYWORDS: ClassVar[set[str]] = {"ratio", "proportion"}

    def __init__(self, separators: set[str] | None = None, ignored_keywords: set[str] | None = None) -> None:
        """
        Initializes the handler. Allows custom separators and keywords.

        Args:
            separators (Optional[Set[str]]): A custom set of word separators.
            ignored_keywords (Optional[Set[str]]): A custom set of words to ignore.
        """
        # Use custom sets or fall back to the class defaults
        active_separators = separators or self._RATIO_SEPARATORS
        active_ignored = ignored_keywords or self._IGNORED_KEYWORDS

        # --- Compile Regex Patterns ---
        # Pattern to find and remove keywords like "ratio of" or "proportion"
        ignore_pattern = r"\b(?:" + "|".join(active_ignored) + r")(?:\s+of)?\b"
        self._ignore_regex = re.compile(ignore_pattern, re.IGNORECASE)

        # Pattern to split by words OR by a colon, handling surrounding punctuation
        separator_pattern = r"[\s,;.]*(?:\b(?:" + "|".join(active_separators) + r")\b|:)" + r"[\s,;.]*"
        self._separator_regex = re.compile(separator_pattern, re.IGNORECASE)

    def _parse_single_query(self, query: str) -> list[str]:
        """Parses a single query string to extract ratio parts."""
        cleaned_query = self._ignore_regex.sub("", query)
        parts = self._separator_regex.split(cleaned_query)
        return [part.strip() for part in parts if part.strip()]

    def handle(self, queries: list[str]) -> dict[str, list[list[str]]]:
        """
        Processes a list of ratio strings, preserving the original structure.

        Args:
            queries (List[str]): A list of strings to process.

        Returns:
            Dict[str, List[List[str]]]: A dictionary where "extracted_columns"
            contains a list of lists, with each inner list corresponding to an
            input query.
        """
        if not isinstance(queries, list):
            return {"extracted_columns": []}

        # Use a more Pythonic list comprehension for a clean transformation
        extracted_groups = [self._parse_single_query(query) for query in queries]

        return {"extracted_columns": extracted_groups}
