import re
from collections.abc import Iterable, Mapping
from typing import Any


def extract_where_filters(queries: Iterable[str]) -> dict[str, str]:
    """
    Extracts value-to-column mappings from WHERE clauses in SQL queries.

    Args:
        queries: An iterable of SQL query strings.

    Returns:
        A dictionary mapping normalized values (lowercased, without %)
        to uppercase column names (without table prefixes).
    """
    value_to_column: dict[str, str] = {}

    pattern = re.compile(r"(?:UPPER|LOWER)?\(?([\w\.]+)\)?\s*(=|ILIKE|LIKE)\s*'([^']+)'", re.IGNORECASE)

    for query in queries:
        where_match = re.search(r"\bWHERE\b(.*)", query, re.IGNORECASE | re.DOTALL)
        if not where_match:
            continue

        where_clause = where_match.group(1)
        conditions = re.split(r"\s+(?:AND|OR)\s+", where_clause, flags=re.IGNORECASE)

        for condition in conditions:
            match = pattern.search(condition.strip())
            if match:
                column, _, value = match.groups()
                column = column.split(".")[-1]  # Remove table prefix if present
                value = value.strip("%").lower()
                value_to_column[value] = column.upper()

    return value_to_column


def _get_clarification_history(config: Mapping[str, Any] | object) -> list[tuple[str, str]] | None:
    """
    Safely extracts clarification history from a config object or mapping.

    Args:
        config: Either a mapping (like dict) or an object with a 'configurable' attribute.

    Returns:
        A list of (question, sql_query) tuples, or None if not available.
    """
    if hasattr(config, "configurable"):
        return getattr(config.configurable, "clarification_history", None)
    if isinstance(config, Mapping):
        return config.get("configurable", {}).get("clarification_history")
    return None


def extract_questions_from_history(config: Mapping[str, Any] | object) -> list[str]:
    """
    Extracts all questions from the clarification history.

    Args:
        config: Mapping or object containing clarification history.

    Returns:
        A list of all questions.
    """
    history = _get_clarification_history(config)
    return [q for q, _ in history if q] if history else []


def extract_sql_queries_from_history(config: Mapping[str, Any] | object) -> list[str]:
    """
    Extracts all SQL queries from the clarification history.

    Args:
        config: Mapping or object containing clarification history.

    Returns:
        A list of all SQL queries.
    """
    history = _get_clarification_history(config)
    return [sql for _, sql in history if sql] if history else []
