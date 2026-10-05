"""Tests for Module 5: Enrichment Pipeline."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.m5_enrichment import (
    summarize_chunk, generate_hypothesis_questions,
    contextual_prepend, extract_metadata, enrich_chunks, EnrichedChunk,
)

SAMPLE = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm."
CHUNKS = [
    {"text": SAMPLE, "metadata": {"source": "policy.md"}},
    {"text": "Mật khẩu phải thay đổi mỗi 90 ngày.", "metadata": {"source": "it.md"}},
]


def test_summarize_returns_string():
    result = summarize_chunk(SAMPLE)
    assert isinstance(result, str)


def test_summarize_shorter_than_original():
    result = summarize_chunk(SAMPLE)
    if result:  # May be empty if no API key
        assert len(result) <= len(SAMPLE) * 2  # Summary should not be much longer


def test_hyqa_returns_list():
    result = generate_hypothesis_questions(SAMPLE, n_questions=2)
    assert isinstance(result, list)


def test_hyqa_generates_questions():
    result = generate_hypothesis_questions(SAMPLE, n_questions=2)
    if result:
        assert len(result) >= 1
        assert any("?" in q or "bao" in q.lower() or "mấy" in q.lower() for q in result)


def test_contextual_prepend_returns_string():
    result = contextual_prepend(SAMPLE, "Sổ tay nhân viên")
    assert isinstance(result, str)
    assert len(result) >= len(SAMPLE)  # Should be at least as long as original


def test_contextual_contains_original():
    result = contextual_prepend(SAMPLE, "Sổ tay nhân viên")
    assert SAMPLE in result  # Original text must be preserved


def test_extract_metadata_returns_dict():
    result = extract_metadata(SAMPLE)
    assert isinstance(result, dict)


def test_enrich_chunks_returns_list():
    result = enrich_chunks(CHUNKS, methods=["contextual"])
    assert isinstance(result, list)


def test_enrich_chunks_type():
    result = enrich_chunks(CHUNKS, methods=["contextual"])
    if result:
        assert all(isinstance(c, EnrichedChunk) for c in result)


def test_enrich_preserves_original():
    result = enrich_chunks(CHUNKS, methods=["contextual"])
    if result:
        assert result[0].original_text == SAMPLE


import json
from types import SimpleNamespace
import pytest
from src import m5_enrichment as m5


@pytest.fixture(autouse=True)
def no_live_api(monkeypatch, tmp_path):
    """Use local fallback by default; tests never spend real API quota."""
    monkeypatch.setattr(m5, "has_gemini_key", lambda: False)
    monkeypatch.setattr(m5, "ENRICHMENT_CACHE_PATH", tmp_path / "cache.json")


def fake_client(monkeypatch, content=None, error=None):
    calls = []
    class Client:
        def with_options(self, **kwargs):
            assert kwargs == {"timeout": 30, "max_retries": 3}
            return self
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def create(self, **kwargs):
            calls.append(kwargs)
            if error:
                raise error
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content=content))])
    client = Client()
    client.chat = SimpleNamespace(completions=client)
    monkeypatch.setattr(m5, "has_gemini_key", lambda: True)
    monkeypatch.setattr(m5, "create_gemini_client", lambda: client)
    return calls


def valid_response():
    return {
        "summary": "Nghỉ phép năm 12 ngày.",
        "questions": ["Nhân viên được nghỉ bao nhiêu ngày?"],
        "context": "Đoạn văn nói về chính sách nghỉ phép năm.",
        "metadata": {"topic": "nghỉ phép", "entities": [], "category": "hr", "language": "vi"},
    }


def test_combined_one_call_per_chunk_and_indexed_questions(monkeypatch):
    response = valid_response()
    response["metadata"].update(source="fake-source", parent_id="fake-parent")
    calls = fake_client(monkeypatch, json.dumps(response))
    chunk = {"text": SAMPLE, "metadata": {"source": "policy.md", "parent_id": "parent_0"}}
    result = enrich_chunks([chunk])[0]
    assert len(calls) == 1
    assert calls[0]["model"] == m5.GEMINI_MODEL
    assert calls[0]["response_format"] == {"type": "json_object"}
    assert result.summary == response["summary"]
    assert response["context"] in result.enriched_text
    assert response["questions"][0] in result.enriched_text
    assert SAMPLE in result.enriched_text
    assert result.auto_metadata["source"] == "policy.md"
    assert result.auto_metadata["parent_id"] == "parent_0"
    assert chunk["metadata"] == {"source": "policy.md", "parent_id": "parent_0"}


@pytest.mark.parametrize("content", ["invalid json", "[]", '{"summary": null}',
    json.dumps({**valid_response(), "questions": "wrong type"}),
    json.dumps({**valid_response(), "metadata": None})])
def test_malformed_output_falls_back(monkeypatch, content):
    calls = fake_client(monkeypatch, content)
    result = m5._enrich_single_call(SAMPLE, "policy.md")
    assert len(calls) == 1
    assert result["summary"] and result["questions"]
    assert "policy.md" in result["context"]
    assert isinstance(result["metadata"], dict)


def test_api_exception_falls_back_without_exposing_secrets(monkeypatch, capsys):
    calls = fake_client(monkeypatch, error=RuntimeError("private-api-key"))
    result = m5._enrich_single_call(SAMPLE, "policy.md")
    assert len(calls) == 1 and result["summary"]
    assert "private-api-key" not in capsys.readouterr().out


def test_fallback_without_key_never_creates_client(monkeypatch):
    def unexpected_client():
        raise AssertionError("Must not call API without key")
    monkeypatch.setattr(m5, "create_gemini_client", unexpected_client)
    result = enrich_chunks(CHUNKS)
    assert all(chunk.summary and chunk.hypothesis_questions for chunk in result)
    assert "policy.md" in result[0].enriched_text
    assert result[0].auto_metadata["source"] == "policy.md"


def test_empty_inputs_and_invalid_methods():
    assert enrich_chunks([]) == []
    assert summarize_chunk("") == ""
    assert generate_hypothesis_questions(SAMPLE, 0) == []
    assert generate_hypothesis_questions("", 2) == []
    with pytest.raises(ValueError, match="Unknown"):
        enrich_chunks(CHUNKS, methods=["typo"])


def test_fenced_json_is_accepted(monkeypatch):
    fake_client(monkeypatch, "```json\n" + json.dumps(valid_response()) + "\n```")
    assert m5._enrich_single_call(SAMPLE, "")["metadata"]["category"] == "hr"
