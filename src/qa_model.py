from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from config import MODEL_DIR, MODEL_NAME, MAX_LENGTH, DOC_STRIDE, MAX_ANSWER_LENGTH, QA_THRESHOLD


class ExtractiveQAModel:
    """RoBERTa extractive QA reader with a lightweight local fallback.

    The Hugging Face model is loaded lazily. This keeps the app importable and
    testable even when the model is not available offline.
    """

    def __init__(self, model_dir: Path = MODEL_DIR, model_name: str = MODEL_NAME):
        self.model_dir = Path(model_dir)
        self.model_name = model_name
        self._tokenizer = None
        self._model = None
        self.device = 'cpu'
        self.threshold = self._load_threshold()

    @property
    def available(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        try:
            import torch
            from transformers import AutoModelForQuestionAnswering, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError('Install requirements.txt before using the RoBERTa reader.') from exc

        source = str(self.model_dir) if self.model_dir.exists() else self.model_name
        self._tokenizer = AutoTokenizer.from_pretrained(source)
        self._model = AutoModelForQuestionAnswering.from_pretrained(source)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._model.to(self.device).eval()

    def _load_threshold(self) -> float:
        path = self.model_dir / 'threshold.json'
        if path.exists():
            try:
                return float(json.loads(path.read_text())['threshold'])
            except Exception:
                pass
        return QA_THRESHOLD

    def predict(self, question: str, context: str) -> Tuple[str, float]:
        if self._model is None:
            raise RuntimeError('RoBERTa model is not loaded. Call load() first.')

        import torch

        encoded = self._tokenizer(
            question.strip(), context,
            max_length=MAX_LENGTH,
            truncation='only_second',
            stride=DOC_STRIDE,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding=True,
            return_tensors='pt',
        )
        offsets = encoded.pop('offset_mapping').numpy()
        encoded.pop('overflow_to_sample_mapping')
        sequence_ids = [encoded.sequence_ids(i) for i in range(encoded['input_ids'].shape[0])]

        with torch.no_grad():
            outputs = self._model(**{k: v.to(self.device) for k, v in encoded.items()})
        starts = outputs.start_logits.detach().cpu().numpy()
        ends = outputs.end_logits.detach().cpu().numpy()

        best_text, best_score, null_score = '', -np.inf, -np.inf
        for i, (start, end) in enumerate(zip(starts, ends)):
            ids = sequence_ids[i]
            if ids:
                cls_indices = [j for j, sid in enumerate(ids) if sid == 0]
                if cls_indices:
                    null_score = max(null_score, float(start[cls_indices[0]] + end[cls_indices[0]]))
            start_candidates = np.argsort(start)[-20:]
            end_candidates = np.argsort(end)[-20:]
            for a in start_candidates:
                for b in end_candidates:
                    if a >= len(ids) or b >= len(ids) or ids[a] != 1 or ids[b] != 1:
                        continue
                    if b < a or b - a + 1 > MAX_ANSWER_LENGTH:
                        continue
                    score = float(start[a] + end[b])
                    if score > best_score:
                        best_score = score
                        begin, finish = offsets[i][a]
                        _, finish = offsets[i][b]
                        if finish > begin:
                            best_text = context[begin:finish]
        return best_text.strip(), float(best_score - null_score)

    def fallback_predict(self, question: str, context: str) -> Tuple[str, float]:
        """Deterministic extractive fallback used only when no model is available."""
        question_words = set(re.findall(r'[a-zA-Z0-9]+', question.lower()))
        candidates = []
        for sentence in re.split(r'(?<=[.!?])\s+', context):
            words = set(re.findall(r'[a-zA-Z0-9]+', sentence.lower()))
            overlap = len(question_words & words) / max(1, len(question_words))
            if overlap:
                candidates.append((overlap, sentence.strip()))
        if not candidates:
            return '', 0.0
        score, sentence = max(candidates)
        return sentence, float(score)
