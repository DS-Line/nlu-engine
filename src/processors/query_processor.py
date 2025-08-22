import re
from collections.abc import Callable

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class QueryProcessor:
    """
    Splits a text query into chunks, respecting a word count limit while
    preserving natural language constructs.

    The splitting strategy is hierarchical:
    1.  First, it attempts to group whole sentences.
    2.  If a sentence exceeds the word limit, it's split by natural
        delimiters (e.g., commas, conjunctions).
    3.  If a part between delimiters is still too long, it's split by a
        hard word count as a final fallback.
    """

    def __init__(self, max_words: int = 10, language: str = "english") -> None:
        """
        Initializes the QuerySplitter.

        Args:
            max_words (int): The maximum number of meaningful (non-stop)
                             words allowed in each chunk.
            language (str): The language for stopwords.
        """
        if max_words <= 0:
            raise ValueError("max_words must be a positive integer.")

        self.max_words = max_words
        self._download_nltk_data()
        self.stop_words: set[str] = set(stopwords.words(language))

        # Regex to split text by sentences, keeping the delimiter.
        self.sentence_pattern = re.compile(r"(?<=[.!?])\s+")

        # Regex to split phrases by natural delimiters, keeping the delimiter.
        self.delimiter_pattern = re.compile(r"(\s*,\s*|\s+and\s+|\s+or\s+|\s+but\s+|;|\s*:\s*|\s*-\s*)", re.IGNORECASE)

    @staticmethod
    def _download_nltk_data() -> None:
        """
        Downloads required NLTK data if not already present.
        This is more robust than assuming the user has it.
        """
        try:
            stopwords.words("english")
            nltk.data.find("tokenizers/punkt")
        except (LookupError, OSError):
            nltk.download("stopwords", quiet=True)
            nltk.download("punkt", quiet=True)

    def _get_meaningful_word_count(self, text: str) -> int:
        """
        Calculates the count of non-stop, alphanumeric words in a text.
        """
        if not text:
            return 0
        tokens = word_tokenize(text.lower())
        return sum(1 for word in tokens if word.isalnum() and word not in self.stop_words)

    def _split_by_word_count(self, text: str) -> list[str]:
        """
        Fallback splitter: Splits text into chunks based on a hard word count.
        This is the lowest level of splitting.
        """
        words = text.split()
        if not words:
            return []

        chunks: list[str] = []
        current_chunk: list[str] = []
        current_word_count = 0

        for word in words:
            is_meaningful = word.lower() not in self.stop_words and word.isalnum()
            current_chunk, current_word_count, chunks = self._add_word_to_chunks(
                word, current_chunk, current_word_count, chunks, is_meaningful=is_meaningful
            )

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _add_word_to_chunks(
        self, word: str, current_chunk: list[str], current_word_count: int, chunks: list[str], *, is_meaningful: bool
    ) -> tuple[list[str], int, list[str]]:
        """
        Handles adding a word to the current chunk, starting a new chunk if the max word count is reached.
        """
        if is_meaningful and current_word_count >= self.max_words:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_word_count = 1
        else:
            current_chunk.append(word)
            if is_meaningful:
                current_word_count += 1

        return current_chunk, current_word_count, chunks

    def _build_chunks_from_parts(self, parts: list[str], recursive_splitter: Callable[[str], list[str]]) -> list[str]:
        """
        A generalized helper to assemble chunks from a list of text parts.
        This function contains the core chunking logic, preventing code duplication.

        Args:
            parts: The pieces of text to assemble.
            recursive_splitter: The function to call if a single part is too large.

        Returns:
            A list of assembled chunks.
        """
        chunks: list[str] = []
        current_chunk_parts: list[str] = []
        current_word_count = 0

        for part in parts:
            if not part or not part.strip():
                continue

            part_word_count = self._get_meaningful_word_count(part)

            if current_chunk_parts and current_word_count + part_word_count > self.max_words:
                chunks.append("".join(current_chunk_parts).strip())
                current_chunk_parts = []
                current_word_count = 0

            if part_word_count > self.max_words:
                sub_chunks = recursive_splitter(part)
                chunks.extend(sub_chunks)
            else:
                current_chunk_parts.append(part)
                current_word_count += part_word_count

        if current_chunk_parts:
            chunks.append("".join(current_chunk_parts).strip())

        return chunks

    def _split_by_natural_breaks(self, text: str) -> list[str]:
        """
        Splits a single sentence or phrase by natural delimiters.
        """
        parts = self.delimiter_pattern.split(text)
        return self._build_chunks_from_parts(parts, self._split_by_word_count)

    def split(self, query: str) -> list[str]:
        """
        Splits the main query into chunks using the hierarchical strategy.

        Args:
            query: The input text to be split.

        Returns:
            A list of query chunks.
        """
        if not isinstance(query, str) or not query.strip():
            return []

        if self._get_meaningful_word_count(query) <= self.max_words:
            return [query.strip()]

        sentences = self.sentence_pattern.split(query)
        return self._build_chunks_from_parts(sentences, self._split_by_natural_breaks)
