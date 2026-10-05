import json
from types import SimpleNamespace

from src.m4_eval import EvalResult, save_report
from src.pipeline import run_query


def test_pipeline_returns_parent_context(monkeypatch):
    monkeypatch.setattr("src.llm.has_gemini_key", lambda: False)
    class Search:
        parent_contexts = {"source:parent_0": "Full parent with current policy and exceptions"}
        def search(self, query):
            return [SimpleNamespace(text="small child", score=1,
                                    metadata={"parent_id": "source:parent_0"})]
    class Reranker:
        def rerank(self, query, documents, top_k):
            return [SimpleNamespace(text=d["text"], metadata=d["metadata"])
                    for d in documents]
    answer, contexts = run_query("question", Search(), Reranker())
    assert contexts == ["Full parent with current policy and exceptions"]
    assert answer == contexts[0]


def test_report_saves_per_question_scores(tmp_path):
    row = EvalResult("q", "a", ["context"], "ground truth", 1, 0.5, 0.6, 0.7)
    path = tmp_path / "report.json"
    save_report({"faithfulness": 1, "status": "ok", "per_question": [row]}, [], str(path))
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["num_questions"] == 1
    assert report["per_question"][0]["answer"] == "a"
    assert report["per_question"][0]["contexts"] == ["context"]
    assert report["per_question"][0]["context_recall"] == 0.7
    assert report["aggregate"]["status"] == "ok"



def test_baseline_cache_changes_with_corpus_or_query():
    from naive_baseline import _cache_signature
    docs = [{"text": "policy", "metadata": {"source": "policy.md"}}]
    questions = [{"question": "q", "ground_truth": "g"}]
    signature = _cache_signature(docs, questions)
    assert signature == _cache_signature(docs, questions)
    assert signature != _cache_signature([{"text": "new policy", "metadata": docs[0]["metadata"]}], questions)
    assert signature != _cache_signature(docs, [{"question": "new q", "ground_truth": "g"}])
