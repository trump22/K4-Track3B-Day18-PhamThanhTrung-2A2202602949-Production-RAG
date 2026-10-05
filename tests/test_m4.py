"""Tests for Module 4: Evaluation."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.m4_eval import load_test_set, evaluate_ragas, failure_analysis, EvalResult

def test_load_test_set():
    ts = load_test_set()
    assert len(ts) > 0 and "question" in ts[0] and "ground_truth" in ts[0]

def test_evaluate_returns_metrics():
    r = evaluate_ragas(["q"], ["a"], [["c"]], ["gt"])
    for k in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        assert k in r and isinstance(r[k], (int, float))

def test_failure_analysis_returns():
    results = [EvalResult("Q1", "A1", ["C1"], "GT1", 0.5, 0.6, 0.4, 0.3)]
    f = failure_analysis(results, bottom_n=1)
    assert len(f) == 1

def test_failure_has_diagnosis():
    results = [EvalResult("Q1", "A1", ["C1"], "GT1", 0.5, 0.6, 0.4, 0.3)]
    f = failure_analysis(results, bottom_n=1)
    if f:
        assert "diagnosis" in f[0] and "suggested_fix" in f[0]


import pytest
from src.m4_eval import _evaluation_dependencies as real_dependencies


@pytest.fixture(autouse=True)
def mocked_ragas(monkeypatch):
    """Unit tests are deterministic and never spend API quota."""
    from src import m4_eval
    import ragas

    def fake_evaluate(dataset, **kwargs):
        import asyncio
        assert asyncio.get_running_loop().is_running()
        assert kwargs["llm"] == "gemini-test-llm"
        assert kwargs["embeddings"] == "local-test-embeddings"
        assert len(kwargs["metrics"]) == 4
        rows = dataset.to_pandas()
        for i, metric in enumerate(m4_eval.METRICS):
            rows[metric] = [0.2 + 0.1 * i + 0.1 * j for j in range(len(rows))]
        class Result:
            def to_pandas(self):
                return rows
        return Result()

    monkeypatch.setattr(m4_eval, "_evaluation_dependencies", lambda: (
        "gemini-test-llm", "local-test-embeddings", list(m4_eval.METRICS),
    ))
    monkeypatch.setattr(ragas, "evaluate", fake_evaluate)


def test_evaluation_aggregates_and_preserves_rows():
    result = evaluate_ragas(["q1", "q2"], ["a1", "a2"], [["c1"], ["c2"]], ["g1", "g2"])
    assert result["status"] == "ok"
    assert result["faithfulness"] == pytest.approx(0.25)
    assert result["context_recall"] == pytest.approx(0.55)
    assert len(result["per_question"]) == 2
    row = result["per_question"][1]
    assert isinstance(row, EvalResult)
    assert (row.question, row.answer, row.contexts, row.ground_truth) == ("q2", "a2", ["c2"], "g2")


def test_evaluation_error_is_explicit(monkeypatch, capsys):
    from src import m4_eval
    def unavailable():
        raise RuntimeError("secret-key-should-not-be-printed")
    monkeypatch.setattr(m4_eval, "_evaluation_dependencies", unavailable)
    result = evaluate_ragas(["q"], ["a"], [["c"]], ["g"])
    assert result["status"] == "error"
    assert result["per_question"] == []
    assert "secret-key" not in str(result)
    assert "secret-key" not in capsys.readouterr().out


def test_evaluation_input_validation():
    assert evaluate_ragas([], [], [], [])["status"] == "empty"
    with pytest.raises(ValueError, match="same length"):
        evaluate_ragas(["q"], [], [], [])


def test_failure_analysis_orders_by_average_and_selects_worst():
    good = EvalResult("good", "a", ["c"], "g", 0.9, 0.9, 0.9, 0.8)
    bad = EvalResult("bad", "a", ["c"], "g", 0.2, 0.5, 0.3, 0.4)
    result = failure_analysis([good, bad], bottom_n=1)
    assert result[0]["question"] == "bad"
    assert result[0]["worst_metric"] == "faithfulness"
    assert result[0]["score"] == pytest.approx(0.35)
    assert failure_analysis([bad], bottom_n=0) == []


@pytest.mark.parametrize("worst", ["faithfulness", "answer_relevancy", "context_precision", "context_recall"])
def test_each_diagnostic_branch(worst):
    from src.m4_eval import METRICS
    scores = {metric: 0.9 for metric in METRICS}
    scores[worst] = 0.1
    result = failure_analysis([EvalResult("q", "a", ["c"], "g", **scores)])[0]
    assert result["worst_metric"] == worst
    assert result["diagnosis"] and result["suggested_fix"]



def test_local_embedding_adapter_handles_numpy2(monkeypatch):
    import config
    import numpy as np
    import sentence_transformers
    from src import m4_eval, llm

    class Encoder:
        def __init__(self, model_name):
            assert model_name == config.EMBEDDING_MODEL
        def encode(self, texts, normalize_embeddings):
            assert normalize_embeddings
            return np.ones((len(texts), config.EMBEDDING_DIM))

    monkeypatch.setattr(llm, "has_gemini_key", lambda: True)
    monkeypatch.setattr(config, "GEMINI_API_KEY", "test-key")
    monkeypatch.setattr("src.m2_search._load_encoder", Encoder)
    chat, embeddings, metrics = real_dependencies()
    assert str(chat.langchain_llm.openai_api_base) == config.GEMINI_BASE_URL
    assert chat.langchain_llm.model_name == config.GEMINI_MODEL
    assert len(embeddings.embed_query("test")) == 1024
    assert len(metrics) == 4



def test_gemini_ragas_uses_single_candidates(monkeypatch):
    import asyncio
    import config
    import numpy as np
    from src import llm
    from langchain_core.outputs import Generation, LLMResult
    from langchain_core.prompt_values import StringPromptValue

    class Encoder:
        def __init__(self, model_name): pass
        def encode(self, texts, **kwargs): return np.ones((len(texts), 1024))
    monkeypatch.setattr(llm, "has_gemini_key", lambda: True)
    monkeypatch.setattr(config, "GEMINI_API_KEY", "test-key")
    monkeypatch.setattr("src.m2_search._load_encoder", Encoder)
    wrapper, _, _ = real_dependencies()
    real_chat = wrapper.langchain_llm
    calls = []
    class Model:
        def generate_prompt(self, prompts, n=1, **kwargs):
            assert len(prompts) == 1 and n == 1
            calls.append(n)
            return LLMResult(generations=[[Generation(text="sample")]])
        async def agenerate_prompt(self, **kwargs):
            return self.generate_prompt(**kwargs)
    wrapper.langchain_llm = Model()
    prompt = StringPromptValue(text="question")
    try:
        assert len(wrapper.generate_text(prompt, n=3).generations[0]) == 3
        assert len(asyncio.run(wrapper.agenerate_text(prompt, n=3)).generations[0]) == 3
        assert calls == [1] * 6
    finally:
        real_chat.http_client.close()
        asyncio.run(real_chat.http_async_client.aclose())
