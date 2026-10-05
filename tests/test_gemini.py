"""Verify Gemini routing without contacting a paid API."""
import json
from types import SimpleNamespace

import httpx
import pytest

from src import llm


def test_gemini_client_sends_key_model_and_google_endpoint(monkeypatch):
    from openai import OpenAI
    import config
    from src.pipeline import run_query

    monkeypatch.setattr(llm, "GEMINI_API_KEY", "test-gemini-key")
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json={
            "id": "test", "object": "chat.completion", "created": 0,
            "model": config.GEMINI_MODEL,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "12 days"}}],
        })

    transport_client = httpx.Client(transport=httpx.MockTransport(respond))
    def client_factory(**kwargs):
        kwargs.pop("http_client").close()
        return OpenAI(**kwargs, http_client=transport_client)
    monkeypatch.setattr("openai.OpenAI", client_factory)

    class Search:
        def search(self, query):
            return [SimpleNamespace(text="12 days", score=1, metadata={})]
    class Reranker:
        def rerank(self, query, docs, top_k):
            return [SimpleNamespace(text=docs[0]["text"])]

    try:
        answer, contexts = run_query("Annual leave?", Search(), Reranker())
        assert answer == "12 days"
        assert contexts == ["12 days"]
        assert len(requests) == 1
        request = requests[0]
        assert str(request.url) == llm.GEMINI_BASE_URL + "chat/completions"
        assert request.headers["authorization"] == "Bearer test-gemini-key"
        assert json.loads(request.content)["model"] == config.GEMINI_MODEL
    finally:
        transport_client.close()


@pytest.mark.parametrize("key", ["", "your_gemini_api_key", "sk-..."])
def test_missing_or_placeholder_key_is_rejected(monkeypatch, key):
    monkeypatch.setattr(llm, "GEMINI_API_KEY", key)
    assert not llm.has_gemini_key()
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        llm.create_gemini_client()



def test_quota_response_adjusts_pacing(monkeypatch):
    monkeypatch.setattr(llm, "_request_interval", 4.1)
    monkeypatch.setattr(llm, "_next_request", 100.0)
    monkeypatch.setattr(llm.time, "monotonic", lambda: 100.0)
    response = httpx.Response(429, json=[{"error": {"details": [
        {"violations": [{"quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier", "quotaValue": "5"}]},
        {"retryDelay": "10s"},
    ]}}])
    llm._update_quota(response)
    assert llm._request_interval == pytest.approx(12.2)
    assert llm._reserve_request_slot() == pytest.approx(10.0)
    assert llm._next_request == pytest.approx(110.0)
    monkeypatch.setattr(llm.time, "monotonic", lambda: 110.0)
    assert llm._reserve_request_slot() == 0
    assert llm._next_request == pytest.approx(122.2)
