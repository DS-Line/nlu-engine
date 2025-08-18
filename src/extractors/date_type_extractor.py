import re

import dateparser

from src.handlers.base import Handler


class DateTypeHandler(Handler):
    """
    A stateless handler that identifies date-related terms from a query string.

    This version is compatible with older versions of dateparser (< 1.0.0)
    and parses one token at a time. It uses regex to correctly tokenize the
    query, stripping punctuation.
    """

    def __init__(self) -> None:
        """Initializes the DateTypeHandler"""

        self._token_pattern = re.compile(r"\d+[-/]\d+[-/]\d+|\w+")

    def handle(self, query: str) -> list[str]:
        """
        Parses a query string to find and return tokens that represent dates.

        Args:
            query (str): The raw input string to parse.

        Returns:
            A list of single-word, date-related terms.
        """
        if not isinstance(query, str) or not query.strip():
            return []

        tokens = self._token_pattern.findall(query.lower())
        return [token for token in tokens if len(token) > 1 and dateparser.parse(token) is not None]
