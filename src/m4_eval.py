from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
import math
import asyncio
from copy import deepcopy
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass, asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float | None
    answer_relevancy: float | None
    context_precision: float | None
    context_recall: float | None


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Evaluate with Gemini and local embeddings; report unavailable scores explicitly."""
    if not (len(questions) == len(answers) == len(contexts) == len(ground_truths)):
        raise ValueError("All evaluation inputs must have the same length")
    empty = {metric: 0.0 for metric in METRICS}
    empty["per_question"] = []
    if not questions:
        return {**empty, "status": "empty"}
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.run_config import RunConfig

        llm, embeddings, metrics = _evaluation_dependencies()
        dataset = Dataset.from_dict({
            "question": questions, "answer": answers,
            "contexts": contexts, "ground_truth": ground_truths,
        })
        async def run_evaluation():
            # RAGAS 0.1 creates asyncio.as_completed before its own asyncio.run.
            # Python 3.14 requires a running loop; RAGAS supports nested loops.
            return evaluate(
                dataset, metrics=metrics, llm=llm, embeddings=embeddings,
                run_config=RunConfig(timeout=600, max_retries=2, max_workers=2),
                raise_exceptions=True,
            ).to_pandas()

        import nest_asyncio
        try:
            loop = asyncio.get_running_loop()
            own_loop = False
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            own_loop = True
        nest_asyncio.apply(loop)
        try:
            df = loop.run_until_complete(run_evaluation())
        finally:
            if own_loop:
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                loop.close()
                asyncio.set_event_loop(None)
        per_question = []
        for _, row in df.iterrows():
            scores = {metric: float(row[metric]) if math.isfinite(float(row[metric]))
                      else None for metric in METRICS}
            per_question.append(EvalResult(
                question=row["question"], answer=row["answer"],
                contexts=list(row["contexts"]), ground_truth=row["ground_truth"],
                **scores,
            ))
        if len(per_question) != len(questions):
            raise ValueError("RAGAS result count does not match question count")
        defined = {metric: [getattr(item, metric) for item in per_question
                            if getattr(item, metric) is not None] for metric in METRICS}
        if any(not values for values in defined.values()):
            raise ValueError("RAGAS returned no defined scores for a metric")
        return {
            **{metric: sum(values) / len(values) for metric, values in defined.items()},
            "valid_counts": {metric: len(values) for metric, values in defined.items()},
            "per_question": per_question, "status": "ok",
        }
    except Exception as exc:
        # Numeric zeros preserve callers' contract; status/error distinguish failure
        # from genuinely measured zero scores. Never print credentials from errors.
        error = f"RAGAS evaluation unavailable ({type(exc).__name__})"
        print(f"  ⚠️  {error}. Check Gemini key, quota and dependencies.")
        return {**empty, "status": "error", "error": error}


METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


def _evaluation_dependencies():
    from src.llm import has_gemini_key, gemini_http_client, gemini_async_http_client
    if not has_gemini_key():
        raise ValueError("Set GEMINI_API_KEY in .env before evaluating")
    from config import GEMINI_API_KEY, GEMINI_BASE_URL, GEMINI_MODEL, EMBEDDING_MODEL
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper
    from langchain_core.outputs import LLMResult
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_core.embeddings import Embeddings
    from src.m2_search import _load_encoder
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

    llm = ChatOpenAI(
        model=GEMINI_MODEL, api_key=GEMINI_API_KEY,
        base_url=GEMINI_BASE_URL, temperature=0, request_timeout=30, max_retries=3,
        http_client=gemini_http_client(), http_async_client=gemini_async_http_client(),
    )
    class LocalEmbeddings(Embeddings):
        # RAGAS 0.1's HuggingfaceEmbeddings uses bool(empty NumPy array),
        # which fails on NumPy 2. Wrap the encoder directly instead.
        def __init__(self):
            self.encoder = _load_encoder(EMBEDDING_MODEL)

        def embed_documents(self, texts):
            return self.encoder.encode(texts, normalize_embeddings=True).tolist()

        def embed_query(self, text):
            return self.embed_documents([text])[0]

    class GeminiRagasLLM(LangchainLLMWrapper):
        def set_run_config(self, run_config):
            super().set_run_config(run_config)
            # RAGAS overwrites the HTTP timeout with the whole metric timeout.
            # Keep individual Gemini calls bounded so SDK retries can run.
            self.langchain_llm.request_timeout = 30

        # Gemini Flash Lite does not support n>1 candidates. Preserve RAGAS'
        # default three relevance samples using three separate n=1 requests.
        def generate_text(self, prompt, n=1, temperature=None, stop=None, callbacks=None):
            temperature = self.get_temperature(n) if temperature is None else temperature
            runs = [super(GeminiRagasLLM, self).generate_text(
                prompt, n=1, temperature=temperature, stop=stop, callbacks=callbacks,
            ) for _ in range(n)]
            return LLMResult(generations=[[run.generations[0][0] for run in runs]])

        async def agenerate_text(self, prompt, n=1, temperature=None, stop=None, callbacks=None):
            temperature = self.get_temperature(n) if temperature is None else temperature
            runs = []
            for _ in range(n):
                runs.append(await super().agenerate_text(
                    prompt, n=1, temperature=temperature, stop=stop, callbacks=callbacks,
                ))
            return LLMResult(generations=[[run.generations[0][0] for run in runs]])

    llm = GeminiRagasLLM(llm)
    embeddings = LangchainEmbeddingsWrapper(LocalEmbeddings())
    metrics = deepcopy([faithfulness, answer_relevancy, context_precision, context_recall])
    return llm, embeddings, metrics


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    if bottom_n <= 0:
        return []
    diagnostic_tree = {
        "faithfulness": (
            "LLM tự bịa câu trả lời ngoài tài liệu",
            "Thắt chặt system prompt, giảm temperature về 0",
        ),
        "context_recall": (
            "Hệ thống tìm kiếm bỏ sót đoạn văn đúng",
            "Cải thiện bước cắt đoạn hoặc bổ sung từ khóa BM25",
        ),
        "context_precision": (
            "Đoạn văn không liên quan bị xếp lên đầu",
            "Bổ sung Cross-Encoder reranking hoặc lọc theo metadata",
        ),
        "answer_relevancy": (
            "Câu trả lời bị lệch trọng tâm câu hỏi",
            "Viết lại prompt hướng dẫn mô hình trả lời trực tiếp hơn",
        ),
    }
    failures = []
    for result in eval_results:
        scores = {metric: getattr(result, metric) for metric in METRICS
                  if getattr(result, metric) is not None}
        if not scores:
            continue
        worst_metric = min(scores, key=scores.get)
        diagnosis, suggested_fix = diagnostic_tree[worst_metric]
        failures.append({
            "question": result.question, "worst_metric": worst_metric,
            "score": sum(scores.values()) / len(scores),
            "diagnosis": diagnosis, "suggested_fix": suggested_fix,
        })
    return sorted(failures, key=lambda item: item["score"])[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "reports/ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    from config import GEMINI_MODEL, EMBEDDING_MODEL
    from importlib.metadata import version
    report = {
        "run_info": {"python": sys.version.split()[0], "provider": "Gemini",
                     "llm_model": GEMINI_MODEL, "embedding_model": EMBEDDING_MODEL,
                     "ragas_version": version("ragas")},
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
        "per_question": [asdict(item) for item in results.get("per_question", [])],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
