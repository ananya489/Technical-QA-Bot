from __future__ import annotations

from typing import Dict, Optional

from config import TOP_K_SECTIONS, RETRIEVAL_THRESHOLD
from src.qa_model import ExtractiveQAModel
from src.retriever import TfidfRetriever


ABSTAIN = 'I couldn\'t find an answer to this question in the provided university documentation.'


class TechnicalQABot:
    def __init__(self, manual: Dict[str, str], reader: Optional[ExtractiveQAModel] = None, use_model: bool = True):
        self.retriever = TfidfRetriever(manual)
        self.reader = reader or ExtractiveQAModel()
        if use_model:
            self.reader.load()

    def ask(self, question: str) -> dict:
        if not question.strip():
            return {'answer': 'Please enter a question.', 'section': None, 'retrieval_score': 0.0, 'confidence': None, 'context': ''}

        retrieved = self.retriever.top_k(question, TOP_K_SECTIONS)
        relevant = [r for r in retrieved if self.retriever.is_relevant(r.score)]
        if not relevant:
            return {'answer': ABSTAIN, 'section': None, 'retrieval_score': retrieved[0].score if retrieved else 0.0, 'confidence': None, 'context': ''}

        best = None
        for item in relevant:
            if self.reader.available:
                answer, confidence = self.reader.predict(question, item.text)
            else:
                answer, confidence = self.reader.fallback_predict(question, item.text)
            if answer and (best is None or confidence > best['confidence']):
                best = {'answer': answer, 'section': item.name, 'retrieval_score': item.score, 'confidence': confidence, 'context': item.text}

        if best is None:
            return {'answer': ABSTAIN, 'section': None, 'retrieval_score': relevant[0].score, 'confidence': None, 'context': relevant[0].text}

        # For the fallback, require meaningful overlap. For RoBERTa, its no-answer score handles abstention.
        if not self.reader.available and best['confidence'] < 0.25:
            best['answer'] = ABSTAIN
            best['section'] = None
        elif self.reader.available and best['confidence'] <= self.reader.threshold:
            best['answer'] = ABSTAIN
            best['section'] = None
        return best
