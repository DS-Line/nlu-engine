import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag


def get_wordnet_pos(treebank_tag: str) -> str:
    """
    Convert a Penn Treebank part-of-speech tag to a WordNet POS tag.

    Args:
        treebank_tag (str): A POS tag from Penn Treebank (e.g., 'NN', 'VB', 'JJ').

    Returns:
        str: Corresponding WordNet POS tag (ADJ, VERB, NOUN, ADV). Defaults to NOUN.
    """
    if treebank_tag.startswith("J"):
        return nltk.corpus.wordnet.ADJ
    if treebank_tag.startswith("V"):
        return nltk.corpus.wordnet.VERB
    if treebank_tag.startswith("N"):
        return nltk.corpus.wordnet.NOUN
    if treebank_tag.startswith("R"):
        return nltk.corpus.wordnet.ADV
    return nltk.corpus.wordnet.NOUN  # Default to noun


class TextPreprocessor:
    """
    Cleans, tokenizes, and lemmatizes a query string.
    """

    def __init__(self, language: str = "english") -> None:
        self._token_pattern = re.compile(r"\d+[-/]\d+[-/]\d+|\w+")
        self._stopwords: set[str] = set(stopwords.words(language))
        self._lemmatizer = WordNetLemmatizer()

    def process(self, query: str) -> list[str]:
        """
        Tokenize, POS-tag, and lemmatize the input query string, removing stopwords.

        Args:
            query (str): The input text to process.

        Returns:
            list[str]: A list of lemmatized tokens with stopwords removed.
        """
        if not isinstance(query, str) or not query.strip():
            return []

        lower_query = query.lower()
        tokens = self._token_pattern.findall(lower_query)
        tagged_tokens = pos_tag(tokens)

        return [
            self._lemmatizer.lemmatize(word, get_wordnet_pos(tag))
            for word, tag in tagged_tokens
            if self._lemmatizer.lemmatize(word, get_wordnet_pos(tag)) not in self._stopwords
        ]
