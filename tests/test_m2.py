"""Tests for Module 2: Hybrid Search."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.m2_search import segment_vietnamese, BM25Search, reciprocal_rank_fusion, SearchResult

CHUNKS = [
    {"text": "Nhân viên được nghỉ phép năm 12 ngày.", "metadata": {"source": "policy"}},
    {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "metadata": {"source": "it"}},
    {"text": "Thời gian thử việc là 60 ngày.", "metadata": {"source": "hr"}},
]

def test_segment_returns_string():
    assert isinstance(segment_vietnamese("nghỉ phép năm"), str)

def test_bm25_search():
    bm25 = BM25Search()
    bm25.index(CHUNKS)
    results = bm25.search("nghỉ phép", top_k=2)
    assert len(results) > 0 and results[0].method == "bm25"

def test_bm25_relevant_first():
    bm25 = BM25Search()
    bm25.index(CHUNKS)
    results = bm25.search("nghỉ phép năm", top_k=2)
    if results:
        assert "nghỉ" in results[0].text.lower() or "12" in results[0].text

def test_rrf_merges():
    a = [SearchResult("doc1", 0.9, {}, "bm25"), SearchResult("doc2", 0.8, {}, "bm25")]
    b = [SearchResult("doc2", 0.95, {}, "dense"), SearchResult("doc3", 0.85, {}, "dense")]
    merged = reciprocal_rank_fusion([a, b], top_k=3)
    assert len(merged) > 0 and "doc2" in [r.text for r in merged]

def test_rrf_method():
    a = [SearchResult("d1", 0.9, {}, "bm25")]
    b = [SearchResult("d1", 0.8, {}, "dense")]
    merged = reciprocal_rank_fusion([a, b], top_k=1)
    if merged:
        assert merged[0].method == "hybrid"


def test_segment_normalizes_compound_words():
    assert "_" not in segment_vietnamese("Nhân viên nghỉ phép năm")


def test_bm25_empty_and_unrelated_queries():
    bm25 = BM25Search()
    bm25.index([])
    assert bm25.search("nghỉ phép") == []
    bm25.index(CHUNKS)
    assert bm25.search("zzzzzz") == []
    assert bm25.search("nghỉ phép", top_k=0) == []


def test_rrf_exact_scores_and_ranking():
    import pytest

    a = [SearchResult("doc1", 999, {}, "bm25"),
         SearchResult("doc2", 1, {"source": "policy"}, "bm25")]
    b = [SearchResult("doc2", 0.01, {}, "dense")]
    merged = reciprocal_rank_fusion([a, b], k=60, top_k=2)
    assert [result.text for result in merged] == ["doc2", "doc1"]
    assert merged[0].score == pytest.approx(1 / 62 + 1 / 61)
    assert merged[0].metadata == {"source": "policy"}
    assert a[1].method == "bm25"


def test_dense_qdrant_round_trip(monkeypatch):
    import numpy as np
    from qdrant_client import QdrantClient
    from src.m2_search import DenseSearch, EMBEDDING_DIM

    class Encoder:
        def encode(self, texts, **kwargs):
            def vector(text):
                result = np.zeros(EMBEDDING_DIM)
                result[0 if "nghỉ" in text else 1] = 1
                return result
            return (vector(texts) if isinstance(texts, str)
                    else np.array([vector(text) for text in texts]))

    client = QdrantClient(":memory:")
    monkeypatch.setattr("qdrant_client.QdrantClient", lambda **kwargs: client)
    dense = DenseSearch()
    dense._encoder = Encoder()
    try:
        dense.index(CHUNKS, collection="module2_test")
        results = dense.search("nghỉ phép", top_k=1, collection="module2_test")
        assert len(results) == 1
        assert results[0].text == CHUNKS[0]["text"]
        assert results[0].metadata == {"source": "policy"}
        assert results[0].method == "dense"
        dense.index([], collection="module2_test")
        assert dense.search("nghỉ phép", collection="module2_test") == []
    finally:
        client.close()
