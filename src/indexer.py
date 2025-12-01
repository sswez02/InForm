from __future__ import annotations

from collections import Counter
from math import log
from typing import Dict, List, Tuple

import numpy as np

from .text_utils import tokenise
from .models import Passage
from .retriever import Retriever


class TfIdfIndex:
    """
    A mini search engine over your Passage objects

    If you give it passages, it builds math-y vectors
    If you give it a query, it returns the most relevant passages
    """

    def __init__(self) -> None:
        self.passages: List[Passage] = []
        self.vocab: Dict[str, int] = {}  # token -> index map
        self.idf: np.ndarray | None = None  # vector of IDF weight per token
        self.doc_token_counts: List[Counter[str]] = (
            []
        )  # token counts grouped by passage
        self.passage_vectors: np.ndarray | None = None  # matrix of TF-IDF vectors

    def add_passages(self, passages: List[Passage]) -> None:
        self.passages.extend(passages)

        # Store Passage
        for p in passages:
            tokens = tokenise(p.text)
            counts = Counter(tokens)
            self.doc_token_counts.append(counts)
            # Build our vocab
            for token in counts.keys():
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)

    def build(self) -> None:
        """
        Compute IDF and TF-IDF vectors for all passages

        TF (term frequency) - bigger if the word shows up a lot in a passage
        IDF (inverse document frequency) - bigger if the word appears in few passages overall (rare word)

        Common boring words (“the”, “and”, “of”)
            high TF, but low IDF = low TF-IDF weight.

        Meaningful specific words (“creatine”, “hypertrophy”)
            good TF and high IDF = high TF-IDF weight.
        """
        n_passages = len(self.doc_token_counts)  # number of passages
        vocab_size = len(self.vocab)  # number of unique tokens
        if n_passages == 0 or vocab_size == 0:
            raise ValueError(
                "No documents or empty vocabulary, .add_passages(passages) first"
            )

        df = np.zeros(vocab_size)
        for counts in self.doc_token_counts:  # loop through each passage
            for token in counts.keys():
                word = self.vocab[token]
                df[
                    word
                ] += 1  # df corresponds to in how many documents each token in our vocab appears

        # Weighting using df to find word frequency (how common or rare across passages)
        self.idf = np.log((1.0 + n_passages) / (1.0 + df)) + 1.0

        # TF IDF matrix
        self.passage_vectors = np.zeros((n_passages, vocab_size))
        for i, counts in enumerate(self.doc_token_counts):  # loop through each passage
            length = sum(counts.values())  # total counted words
            if length == 0:
                continue
            for token, count in counts.items():
                j = self.vocab[token]
                tf = count / length  # term frequency
                self.passage_vectors[i, j] = tf * self.idf[j]  # vector: tf x idf

        # L2 normalise doc vectors for cosine similarity (dot product)
        norms = np.linalg.norm(self.passage_vectors, axis=1, keepdims=True) + 1e-8
        self.passage_vectors = self.passage_vectors / norms

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Passage, float]]:
        """
        Return top_k (passage, score) pairs for the query search
        """
        if self.passage_vectors is None or self.idf is None:
            raise ValueError("No vectors build, call .build() first")

        tokens = tokenise(query)
        if not tokens:
            return []

        vocab_size = len(self.vocab)
        query_vector = np.zeros(vocab_size)  # TF-IDF vector for the query
        query_counts = Counter(tokens)
        query_length = sum(query_counts.values())
        if query_length == 0:
            return []

        for token, count in query_counts.items():
            if token not in self.vocab:  # searchword not in vocab
                continue
            j = self.vocab[token]
            tf = count / query_length
            query_vector[j] = tf * self.idf[j]

        # Normalise query vector
        query_vector = query_vector / (np.linalg.norm(query_vector) + 1e-8)
        scores = (
            self.passage_vectors @ query_vector
        )  # single similarity score per passage using matrix multiplication

        # Get indices of top scores
        top_index = np.argsort(-scores)[:top_k]
        results: List[Tuple[Passage, float]] = []  # (passage, score) pairs
        for i in top_index:
            score = float(scores[i])
            if score <= 0:
                continue
            results.append((self.passages[i], score))

        return results
