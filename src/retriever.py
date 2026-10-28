import re
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import TOP_K_SECTIONS, RETRIEVAL_THRESHOLD


@dataclass
class RetrievedSection:
    name: str
    text: str
    score: float


class TfidfRetriever:
    """Fast lexical retriever for a small closed-domain manual."""

    def __init__(self, sections: Dict[str, str]):
        if not sections:
            raise ValueError('The manual is empty.')
        self.names = list(sections)
        self.texts = [sections[n] for n in self.names]
        self.vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(self.texts)

    def top_k(self, question: str, k: int = TOP_K_SECTIONS) -> List[RetrievedSection]:
        question = question.strip()
        if not question:
            return []
        scores = cosine_similarity(self.vectorizer.transform([question]), self.matrix)[0]
        indices = np.argsort(scores)[::-1][:k]
        return [RetrievedSection(self.names[i], self.texts[i], float(scores[i])) for i in indices]

    def is_relevant(self, score: float) -> bool:
        return score >= RETRIEVAL_THRESHOLD
