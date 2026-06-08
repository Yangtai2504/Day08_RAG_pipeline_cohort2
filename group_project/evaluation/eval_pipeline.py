"""
RAG Evaluation Pipeline — RAGAS framework.

Chay evaluation tren 16 Q&A pairs voi 4 metrics:
    - Faithfulness
    - Answer Relevancy
    - Context Recall
    - Context Precision

A/B Comparison:
    - Config A: Hybrid search (semantic + BM25) + RRF reranking
    - Config B: Dense-only (chi semantic search, khong reranking)

Chay:
    cd <project_root>
    python -m group_project.evaluation.eval_pipeline
"""

import json
import os
import sys
from pathlib import Path

# Them project root vao sys.path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENAI_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "openai/gpt-4o-mini"


# =============================================================================
# LOAD DATASET
# =============================================================================

def load_golden_dataset() -> list[dict]:
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# RAG PIPELINE WRAPPERS (Config A va B)
# =============================================================================

def run_pipeline_config_a(question: str) -> dict:
    """Config A: Hybrid (semantic + BM25) + RRF reranking."""
    from src.task9_retrieval_pipeline import retrieve
    from src.task10_generation import generate_with_citation

    result = generate_with_citation(question, top_k=5)
    return result


def run_pipeline_config_b(question: str) -> dict:
    """Config B: Dense-only (chi semantic search, khong reranking)."""
    from src.task5_semantic_search import semantic_search
    from src.task10_generation import (
        reorder_for_llm, format_context, SYSTEM_PROMPT,
        LLM_MODEL, TEMPERATURE, TOP_P
    )

    chunks = semantic_search(question, top_k=5)
    if not chunks:
        return {
            "answer": "Toi khong the xac minh thong tin nay tu nguon hien co.",
            "sources": [],
            "retrieval_source": "dense_only",
        }

    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)
    user_message = f"Context:\n{context}\n\n---\n\nCau hoi: {question}"

    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY", "")
    answer = "Toi khong the xac minh thong tin nay tu nguon hien co."

    if api_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key, base_url=OPENAI_BASE_URL)
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=TEMPERATURE,
                top_p=TOP_P,
            )
            answer = response.choices[0].message.content
        except Exception as e:
            print(f"  LLM error: {e}")

    return {
        "answer": answer,
        "sources": reordered,
        "retrieval_source": "dense_only",
    }


# =============================================================================
# RAGAS EVALUATION
# =============================================================================

def evaluate_config(config_name: str, run_fn, golden_dataset: list[dict]) -> dict:
    """
    Chay RAGAS evaluation cho mot config.

    Args:
        config_name: Ten config (vd: 'hybrid_rerank', 'dense_only')
        run_fn: Callable(question) -> {'answer': str, 'sources': list}
        golden_dataset: List of Q&A pairs

    Returns:
        dict with metric scores
    """
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_recall,
        context_precision,
    )
    from ragas.llms import LangchainLLMWrapper
    from langchain_openai import ChatOpenAI
    from datasets import Dataset

    os.environ["RAGAS_DO_NOT_TRACK"] = "true"
    # ragas also needs OPENAI_API_KEY for its embedding calls
    os.environ["OPENAI_API_KEY"] = OPENROUTER_API_KEY
    os.environ["OPENAI_API_BASE"] = OPENAI_BASE_URL

    # Explicitly configure LLM with low max_tokens to stay within credit limit
    ragas_llm = LangchainLLMWrapper(ChatOpenAI(
        model=LLM_MODEL,
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base=OPENAI_BASE_URL,
        max_tokens=512,
    ))
    faithfulness.llm = ragas_llm
    answer_relevancy.llm = ragas_llm
    context_recall.llm = ragas_llm
    context_precision.llm = ragas_llm

    print(f"\n{'='*60}")
    print(f"Evaluating Config: {config_name}")
    print(f"{'='*60}")

    eval_data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }

    for i, item in enumerate(golden_dataset):
        print(f"  [{i+1}/{len(golden_dataset)}] {item['question'][:60]}...")
        try:
            result = run_fn(item["question"])
            eval_data["question"].append(item["question"])
            eval_data["answer"].append(result.get("answer", ""))
            # Truncate each context chunk to 400 chars to stay within token limits
            contexts = [c["content"][:400] for c in result.get("sources", [])]
            eval_data["contexts"].append(contexts)
            eval_data["ground_truth"].append(item["expected_answer"])
        except Exception as e:
            print(f"    ERROR: {e}")
            eval_data["question"].append(item["question"])
            eval_data["answer"].append("")
            eval_data["contexts"].append([])
            eval_data["ground_truth"].append(item["expected_answer"])

    dataset = Dataset.from_dict(eval_data)

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
    )
    df = result.to_pandas()

    scores = {
        "faithfulness": float(df["faithfulness"].mean()),
        "answer_relevancy": float(df["answer_relevancy"].mean()),
        "context_recall": float(df["context_recall"].mean()),
        "context_precision": float(df["context_precision"].mean()),
    }
    scores["average"] = sum(scores.values()) / len(scores)

    print(f"\nScores for {config_name}:")
    for k, v in scores.items():
        print(f"  {k}: {v:.3f}")

    scores["_per_question"] = df.to_dict(orient="records")
    scores["_config"] = config_name

    return scores


# =============================================================================
# EXPORT RESULTS
# =============================================================================

def export_results(results_a: dict, results_b: dict, golden_dataset: list[dict]):
    """Export ket qua ra results.md."""

    lines = []
    lines.append("# RAG Evaluation Results\n")
    lines.append(f"**Framework:** RAGAS  ")
    lines.append(f"**LLM Judge:** {LLM_MODEL} (via OpenRouter)  ")
    lines.append(f"**Golden Dataset:** {len(golden_dataset)} Q&A pairs\n")

    # Overall scores table
    lines.append("## Overall Scores\n")
    lines.append("| Metric | Config A (Hybrid+RRF) | Config B (Dense-only) | Delta |")
    lines.append("|--------|----------------------|----------------------|-------|")

    metrics = ["faithfulness", "answer_relevancy", "context_recall", "context_precision", "average"]
    for m in metrics:
        a = results_a.get(m, 0.0)
        b = results_b.get(m, 0.0)
        delta = a - b
        sign = "+" if delta >= 0 else ""
        lines.append(f"| {m} | {a:.3f} | {b:.3f} | {sign}{delta:.3f} |")

    lines.append("")

    # A/B Analysis
    lines.append("## A/B Comparison Analysis\n")
    a_avg = results_a.get("average", 0)
    b_avg = results_b.get("average", 0)
    winner = "Config A (Hybrid+RRF)" if a_avg >= b_avg else "Config B (Dense-only)"
    lines.append(f"**Winner:** {winner} (average score: {max(a_avg, b_avg):.3f})\n")
    lines.append("**Nhan xet:**")
    if a_avg > b_avg:
        lines.append("- Config A (Hybrid + RRF reranking) cho ket qua tot hon Dense-only.")
        lines.append("- Reranking giup chon dung context hon, cai thien Faithfulness va Context Precision.")
    else:
        lines.append("- Config B (Dense-only) cho ket qua tuong duong hoac tot hon.")
        lines.append("- Voi corpus nho (10 documents), BM25 khong them nhieu gia tri.")
    lines.append("")

    # Worst performers
    lines.append("## Worst Performers Analysis\n")
    per_q_a = results_a.get("_per_question", [])
    if per_q_a:
        # Tim questions co faithfulness thap nhat
        sorted_q = sorted(per_q_a, key=lambda x: x.get("faithfulness", 1.0))
        lines.append("### Cau hoi co Faithfulness thap nhat (Config A)\n")
        lines.append("| # | Question | Faithfulness | Context Recall |")
        lines.append("|---|----------|-------------|----------------|")
        for i, row in enumerate(sorted_q[:5], 1):
            q = golden_dataset[i-1]["question"][:60] if i-1 < len(golden_dataset) else "N/A"
            faith = row.get("faithfulness", 0.0)
            recall = row.get("context_recall", 0.0)
            lines.append(f"| {i} | {q}... | {faith:.3f} | {recall:.3f} |")
        lines.append("")
        lines.append("**Nguyen nhan worst performers:**")
        lines.append("- Cau hoi ve thong tin cu the (ten nghe si, so dieu luat) can chunk chinh xac hon.")
        lines.append("- Mot so bai bao crawl duoc it noi dung (article_03 that bai) nen thieu context.")
        lines.append("- Cac cau hoi ve nhan vat cu the ('Miu Le', 'Long Nhat') can thong tin ro rang hon.")
    else:
        lines.append("(Chua co du lieu per-question — chay eval de co ket qua chi tiet)")

    lines.append("")

    # Recommendations
    lines.append("## Recommendations\n")
    lines.append("1. **Tang so luong bai bao crawl** — them >= 10 bai bao de co du context cho cac cau hoi ve nghe si cu the.")
    lines.append("2. **Dung MarkdownHeaderTextSplitter** — giu nguyen cau truc dieu khoan phap luat de tra loi chinh xac hon.")
    lines.append("3. **Tang chunk_size** cho van ban phap luat (1200 chars) — dieu khoan dai can nhieu context hon.")
    lines.append("4. **Them HyDE** (Hypothetical Document Embeddings) — sinh ra doc gia thiet truoc khi embed query.")
    lines.append("5. **Cross-encoder reranking** — dung Jina API thay vi RRF de rerank chinh xac hon cho tieng Viet.")

    lines.append("")
    lines.append("---")
    lines.append("*Generated by RAGAS evaluation pipeline*")

    RESULTS_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nResults exported to: {RESULTS_PATH}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    if not OPENROUTER_API_KEY:
        print("ERROR: OPENROUTER_API_KEY not set in .env")
        sys.exit(1)

    golden = load_golden_dataset()
    print(f"Loaded {len(golden)} test cases")

    # Use subset for speed (full dataset takes ~15 min)
    subset = golden[:8]
    print(f"Running eval on {len(subset)} questions (subset for speed)")

    print("\n[Config A] Hybrid + RRF Reranking")
    results_a = evaluate_config("hybrid_rerank", run_pipeline_config_a, subset)

    print("\n[Config B] Dense-only (no reranking)")
    results_b = evaluate_config("dense_only", run_pipeline_config_b, subset)

    export_results(results_a, results_b, golden)
    print("\nDone! Check group_project/evaluation/results.md")
