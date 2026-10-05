from __future__ import annotations

"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import os, sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass
from functools import lru_cache

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K, RRF_K)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words."""
    from underthesea import word_tokenize

    return word_tokenize(text, format="text").replace("_", " ")



class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        from rank_bm25 import BM25Okapi

        self.documents = list(chunks)
        self.corpus_tokens = [
            segment_vietnamese(chunk["text"]).lower().split()
            for chunk in self.documents
        ]
        self.bm25 = (BM25Okapi(self.corpus_tokens)
                     if any(self.corpus_tokens) else None)

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        if self.bm25 is None or top_k <= 0 or not query.strip():
            return []
        tokens = segment_vietnamese(query).lower().split()
        scores = self.bm25.get_scores(tokens)
        indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [SearchResult(
            text=self.documents[i]["text"], score=float(scores[i]),
            metadata=dict(self.documents[i].get("metadata", {})), method="bm25",
        ) for i in indices if scores[i] > 0][:top_k]



@lru_cache(maxsize=1)
def _load_encoder(model_name):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


class DenseSearch:
    def __init__(self):
        from qdrant_client import QdrantClient
        try:
            self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=2)
            self.client.get_collections()
        except Exception:
            self.client = QdrantClient(":memory:")
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            self._encoder = _load_encoder(EMBEDDING_MODEL)
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant."""
        from qdrant_client.models import Distance, VectorParams, PointStruct

        if self.client.collection_exists(collection):
            self.client.delete_collection(collection)
        self.client.create_collection(
            collection,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        if not chunks:
            return
        vectors = self._get_encoder().encode(
            [chunk["text"] for chunk in chunks], show_progress_bar=True,
        )
        points = [PointStruct(
            id=i, vector=vector.tolist(),
            payload={**chunk.get("metadata", {}), "text": chunk["text"]},
        ) for i, (chunk, vector) in enumerate(zip(chunks, vectors))]
        for start in range(0, len(points), 64):
            self.client.upsert(collection, points=points[start:start + 64], wait=True)

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors."""
        if top_k <= 0 or not query.strip():
            return []
        if not self.client.collection_exists(collection):
            return []
        query_vector = self._get_encoder().encode(query).tolist()
        response = self.client.query_points(
            collection, query=query_vector, limit=top_k,
        )
        results = []
        for point in response.points:
            payload = dict(point.payload or {})
            text = payload.pop("text", "")
            results.append(SearchResult(
                text=text, score=float(point.score), metadata=payload, method="dense",
            ))
        return results



def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = RRF_K,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank + 1)."""
    if k < 0:
        raise ValueError("k must be non-negative")
    if top_k <= 0:
        return []
    scores = {}
    documents = {}
    for results in results_list:
        seen = set()
        for rank, result in enumerate(results):
            if result.text in seen:
                continue
            seen.add(result.text)
            documents.setdefault(result.text, result)
            scores[result.text] = scores.get(result.text, 0.0) + 1.0 / (k + rank + 1)
    ranked = sorted(scores, key=scores.get, reverse=True)[:top_k]
    return [SearchResult(
        text=text, score=scores[text], metadata=dict(documents[text].metadata),
        method="hybrid",
    ) for text in ranked]



class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print(f"Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
