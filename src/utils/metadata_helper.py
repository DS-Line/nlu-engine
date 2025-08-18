import re
from typing import Any

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

EXPECTED_METADATA_LENGTH = 4


def transform_raw_metadata(raw_data: list[dict]) -> tuple[dict, dict, Any, Any]:
    """
    Transforms the raw 4-element metadata list into the expected
    4-element tuple (metrics, attributes, columns, functions).
    """
    if not isinstance(raw_data, list) or len(raw_data) < EXPECTED_METADATA_LENGTH:
        raise ValueError(f"Invalid raw metadata format. Expected a list of {EXPECTED_METADATA_LENGTH} items.")

    mixed_definitions = {**raw_data[0], **raw_data[1]}
    final_columns = raw_data[2]
    final_functions = raw_data[3]
    final_metrics, final_attributes = {}, {}

    aggregate_functions = ("COUNT(", "AVG(", "SUM(", "MIN(", "MAX(")

    if mixed_definitions:
        for key, definition in mixed_definitions.items():
            calculation = definition.get("calculation", "")
            if (
                calculation and calculation.upper().startswith(aggregate_functions)
            ) or "GROUP BY" in calculation.upper():
                final_metrics[key] = definition
            else:
                final_attributes[key] = definition

    return final_metrics, final_attributes, final_columns, final_functions


def lemmatize_and_normalize(text: str) -> str:
    """
    Normalizes and lemmatizes a string.

    :param text: The input string.
    :return: The normalized and lemmatized string, or an empty string if the input is None.
    """
    if not text:
        return ""

    text = text.lower().strip()
    words = re.split(r"[_\-\s]+", text)
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]

    return " ".join(lemmatized_words)


def filter_value_mappings_by_query(value_to_column: dict, user_query: str) -> list[dict]:
    """
    Filters the value-to-column mappings based on the presence of values in the normalized user query.
    Returns a list containing a single filtered dictionary for compatibility with downstream processing.
    """
    if not value_to_column:
        return []

    text = re.sub(r"[^a-zA-Z0-9]", " ", user_query.lower())
    normalized_query = " ".join(lemmatizer.lemmatize(w) for w in text.split())

    filtered = {key: col for key, col in value_to_column.items() if lemmatize_and_normalize(key) in normalized_query}

    return [filtered] if filtered else []


def extract_keys_from_mappings(value_mappings: list[dict]) -> set[str]:
    """
    Recursively extracts normalized keys from nested value-to-column mappings.

    Args:
        value_mappings (Iterable[dict]): List or iterable of dictionaries (possibly nested).

    Returns:
        Set[str]: A set of normalized keys.
    """
    keys = set()

    def recurse(mapping: dict) -> None:
        """
        Recursively extract keys from nested value-to-column mappings.
        """
        if not isinstance(mapping, dict):
            return
        for key, value in mapping.items():
            keys.add(lemmatize_and_normalize(key))
            if isinstance(value, dict):
                recurse(value)

    for item in value_mappings:
        recurse(item)

    return keys


def normalize_and_filter_terms(terms: list[str], value_mappings: list[dict]) -> list[str]:
    """Normalize and filter out terms that already exist in value_mappings."""
    normalized_mappings_keys = extract_keys_from_mappings(value_mappings)

    return [term for term in terms if lemmatize_and_normalize(term) not in normalized_mappings_keys]
