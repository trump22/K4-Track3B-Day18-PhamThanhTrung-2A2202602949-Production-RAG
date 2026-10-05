"""Tests for Module 3: Reranking."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.m3_rerank import CrossEncoderReranker, benchmark_reranker, RerankResult

Q = "Nhân viên được nghỉ phép bao nhiêu ngày?"
DOCS = [
    {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
    {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
    {"text": "VPN dùng WireGuard AES-256.", "score": 0.6, "metadata": {}},
]

def test_rerank_returns():
    r = CrossEncoderReranker().rerank(Q, DOCS, top_k=2)
    assert len(r) > 0 and len(r) <= 2

def test_rerank_type():
    assert all(isinstance(x, RerankResult) for x in CrossEncoderReranker().rerank(Q, DOCS))

def test_rerank_sorted():
    r = CrossEncoderReranker().rerank(Q, DOCS)
    if len(r) >= 2:
        assert r[0].rerank_score >= r[1].rerank_score

def test_rerank_relevant_first():
    r = CrossEncoderReranker().rerank(Q, DOCS)
    if r:
        assert "nghỉ" in r[0].text.lower() or "12" in r[0].text

def test_benchmark_stats():
    stats = benchmark_reranker(CrossEncoderReranker(), Q, DOCS, n_runs=2)
    assert "avg_ms" in stats and "min_ms" in stats and "max_ms" in stats


def test_rerank_empty_skips_loading(monkeypatch):
    reranker = CrossEncoderReranker()
    def unexpected_load():
        raise AssertionError("Model should not load without candidates")
    monkeypatch.setattr(reranker, "_load_model", unexpected_load)
    assert reranker.rerank(Q, []) == []
    assert reranker.rerank(Q, DOCS, top_k=0) == []


def test_rerank_preserves_scores_metadata_and_ranks():
    class Model:
        def predict(self, pairs):
            assert pairs == [(Q, doc["text"]) for doc in DOCS]
            return [0.3, 0.9, 0.6]

    reranker = CrossEncoderReranker()
    reranker._model = Model()
    results = reranker.rerank(Q, DOCS, top_k=2)
    assert [result.text for result in results] == [DOCS[1]["text"], DOCS[2]["text"]]
    assert [result.rank for result in results] == [0, 1]
    assert [result.original_score for result in results] == [0.7, 0.6]
    assert results[0].metadata == DOCS[1]["metadata"]
    assert results[0].metadata is not DOCS[1]["metadata"]


def test_rerank_single_scalar_score():
    class Model:
        def predict(self, pairs):
            return 0.75

    reranker = CrossEncoderReranker()
    reranker._model = Model()
    results = reranker.rerank(Q, [DOCS[0]])
    assert len(results) == 1
    assert results[0].rerank_score == 0.75
    assert results[0].rank == 0
