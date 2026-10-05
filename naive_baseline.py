"""
Basic RAG Baseline — Chạy TRƯỚC để có scores so sánh.
=====================================================
Basic = paragraph chunking + dense-only search (không hybrid, không rerank, không enrichment).
Đây là RAG đã học ở buổi trước — hôm nay sẽ cải thiện từng bước.
"""

import sys, os, time
import json
import hashlib
from pathlib import Path
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.m1_chunking import load_documents, chunk_basic
from src.m2_search import DenseSearch
from src.m4_eval import load_test_set, evaluate_ragas, save_report
from config import NAIVE_COLLECTION, GEMINI_MODEL, EMBEDDING_MODEL


def main():
    print("=" * 60)
    print("BASIC RAG BASELINE")
    print("(paragraph chunking + dense-only, no rerank, no enrichment)")
    print("=" * 60)

    docs = load_documents()
    chunks = []
    for doc in docs:
        for c in chunk_basic(doc["text"], metadata=doc["metadata"]):
            chunks.append({"text": c.text, "metadata": c.metadata})
    print(f"  {len(chunks)} basic paragraph chunks")

    # Rebuild the collection even when generation is cached: a previous
    # interrupted run may have deleted its points before the upsert.
    search = DenseSearch()
    search.index(chunks, collection=NAIVE_COLLECTION)

    test_set = load_test_set()
    signature = _cache_signature(docs, test_set)
    cache_path = Path("reports/naive_answers_cache.json")
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("signature") == signature and len(cached.get("rows", [])) == len(test_set):
            print("  Reusing saved baseline answers/context for unchanged model and corpus.", flush=True)
            _evaluate_rows(cached["rows"])
            return

    questions, answers, all_contexts, ground_truths = [], [], [], []

    from config import GEMINI_MODEL
    from src.llm import has_gemini_key, create_gemini_client
    llm_client = None
    if has_gemini_key():
        llm_client = create_gemini_client()

    for i, item in enumerate(test_set):
        results = search.search(item["question"], top_k=3, collection=NAIVE_COLLECTION)
        contexts = [r.text for r in results]

        if llm_client and contexts:
            try:
                context_str = "\n\n".join(contexts)
                resp = llm_client.chat.completions.create(model=GEMINI_MODEL, messages=[
                    {"role": "system", "content": "Trả lời CHỈ dựa trên context. Nếu không có → nói 'Không tìm thấy.'"},
                    {"role": "user", "content": f"Context:\n{context_str}\n\nCâu hỏi: {item['question']}"},
                ])
                answer = resp.choices[0].message.content
            except Exception:
                answer = contexts[0]
        else:
            answer = contexts[0] if contexts else "Không tìm thấy."

        answers.append(answer)
        questions.append(item["question"])
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])
        with open("reports/naive_answers.json", "w", encoding="utf-8") as checkpoint:
            json.dump([{"question": q, "answer": a, "contexts": c, "ground_truth": g}
                       for q, a, c, g in zip(questions, answers, all_contexts, ground_truths)],
                      checkpoint, ensure_ascii=False, indent=2)
        print(f"  [{i+1}/{len(test_set)}] {item['question'][:50]}...", flush=True)

    rows = [{"question": q, "answer": a, "contexts": c, "ground_truth": g}
            for q, a, c, g in zip(questions, answers, all_contexts, ground_truths)]
    cache_path.write_text(json.dumps({"signature": signature, "rows": rows},
                                    ensure_ascii=False, indent=2), encoding="utf-8")
    _evaluate_rows(rows)


def _cache_signature(docs, test_set):
    return hashlib.sha256(json.dumps({
        "model": GEMINI_MODEL, "embedding_model": EMBEDDING_MODEL,
        "documents": docs, "test_set": test_set, "chunk_size": 500, "top_k": 3,
        "prompt": "Trả lời CHỈ dựa trên context. Nếu không có → nói 'Không tìm thấy.'",
    }, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def _evaluate_rows(rows):
    results = evaluate_ragas(
        [r["question"] for r in rows], [r["answer"] for r in rows],
        [r["contexts"] for r in rows], [r["ground_truth"] for r in rows],
    )
    print("\nBASIC BASELINE SCORES")
    for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        print(f"  {metric}: {results.get(metric, 0):.4f}")
    save_report(results, [], path="reports/naive_baseline_report.json")
    if results.get("status") != "ok":
        print("\nEvaluation incomplete; zero placeholders are not measured scores.")


if __name__ == "__main__":
    start = time.time()
    main()
    print(f"Total: {time.time() - start:.1f}s")
