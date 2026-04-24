'''
- Document indexing with preprocessing
- Query processing and scoring
- Index persistence and loading
- Parameter tuning (k1, b values)
'''
import json
import math
import re
from collections import Counter
from typing import Dict, List, Optional


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents: List[Dict] = []
        self._doc_lengths: Dict[str, int] = {}       # fast O(1) length lookup
        self._total_doc_length: int = 0
        self.avg_doc_length: float = 0.0
        self.inverted_index: Dict[str, List[Dict]] = {}

    # --- preprocessing ---

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        return text.split()

    # --- indexing ---

    def add_document(self, doc_id: str, text: str, metadata: Dict) -> None:
        tokens = self._tokenize(text)
        doc_length = len(tokens)

        self.documents.append({'id': doc_id, 'text': text, 'metadata': metadata})
        self._doc_lengths[doc_id] = doc_length
        self._total_doc_length += doc_length
        self.avg_doc_length = self._total_doc_length / len(self.documents)

        for token, freq in Counter(tokens).items():
            if token not in self.inverted_index:
                self.inverted_index[token] = []
            self.inverted_index[token].append({'doc_id': doc_id, 'term_freq': freq})

    # --- scoring ---

    def score(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        query_tokens = self._tokenize(query)
        n = len(self.documents)
        scores: Dict[str, float] = {}

        for token in query_tokens:
            if token not in self.inverted_index:
                continue
            postings = self.inverted_index[token]
            df = len(postings)
            idf = math.log((n - df + 0.5) / (df + 0.5) + 1)

            for posting in postings:
                doc_id = posting['doc_id']
                tf = posting['term_freq']
                dl = self._doc_lengths[doc_id]
                tf_norm = (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1 - self.b + self.b * dl / self.avg_doc_length)
                )
                scores[doc_id] = scores.get(doc_id, 0.0) + idf * tf_norm

        results = sorted(
            [
                {**doc, 'score': scores[doc['id']]}
                for doc in self.documents
                if doc['id'] in scores
            ],
            key=lambda x: x['score'],
            reverse=True,
        )
        return results[:top_k] if top_k is not None else results

    # --- persistence ---

    def save(self, file_path: str) -> None:
        with open(file_path, 'w') as f:
            json.dump(
                {
                    'k1': self.k1,
                    'b': self.b,
                    'documents': self.documents,
                    'doc_lengths': self._doc_lengths,
                    'total_doc_length': self._total_doc_length,
                    'avg_doc_length': self.avg_doc_length,
                    'inverted_index': self.inverted_index,
                },
                f,
            )

    @classmethod
    def load(cls, file_path: str) -> "BM25Index":
        with open(file_path, 'r') as f:
            data = json.load(f)
        index = cls(k1=data['k1'], b=data['b'])
        index.documents = data['documents']
        index._doc_lengths = data['doc_lengths']
        index._total_doc_length = data['total_doc_length']
        index.avg_doc_length = data['avg_doc_length']
        index.inverted_index = data['inverted_index']
        return index
