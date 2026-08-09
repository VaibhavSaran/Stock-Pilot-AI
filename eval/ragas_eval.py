"""
StockPilot AI — RAGAS Evaluation Script

Evaluates the News RAG pipeline using 4 core RAGAS metrics:
  - Faithfulness:      Does the answer stick to retrieved context?
  - Answer Relevancy:  Does the answer address the question?
  - Context Precision: Are retrieved docs actually relevant?
  - Context Recall:    Did retrieval find everything needed?

Also reports two ranking-metric proxies (MRR, NDCG@6) based on ticker-match,
since RAGAS doesn't compute ranking metrics and no hand-labeled per-document
relevance judgments exist. See ranking_metrics_note in the saved JSON.

Usage (inside the API container):
    docker exec stockpilot_api python eval/ragas_eval.py

Requirements:
    pip install --no-cache-dir ragas==0.1.9 datasets==2.19.0
    (heavy, eval-only deps — install on demand, not part of the built image)

Note: Uses Claude Sonnet 4.6 as the judge LLM via LangChain Anthropic.
      Each question makes ~5-10 LLM calls, so 15 questions = ~75-150 API calls.
"""

import argparse
import os
import sys
import json
import logging
import math
from datetime import datetime

# ── Path setup ────────────────────────────────────────────────────────────────
# Add /app to path so we can import StockPilot modules
sys.path.insert(0, "/app")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Test dataset ──────────────────────────────────────────────────────────────
# 15 questions grounded in actual data collected March-June 2026
# ground_truth = the correct answer based on known facts in the data
# expected_ticker = the tracked ticker the question is actually about, or
# None for broad/index questions not about one tracked ticker.

TEST_DATASET = [
    {
        "question": "What was NVIDIA's revenue in Q1 fiscal year 2027?",
        "ground_truth": "NVIDIA reported record revenue of $81.6 billion for Q1 fiscal year 2027, up 85% year-over-year and up 20% from the previous quarter.",
        "expected_ticker": "NVDA",
    },
    {
        "question": "What major restructuring did Meta announce in 2026?",
        "ground_truth": "Meta announced a major restructuring in 2026, cutting approximately 8,000 jobs while reassigning 7,000 employees to AI-focused teams as part of its push into artificial intelligence.",
        "expected_ticker": "META",
    },
    {
        "question": "Did NVIDIA announce any stock buyback program recently?",
        "ground_truth": "Yes, NVIDIA authorized an additional $80 billion share repurchase program alongside its Q1 fiscal 2027 earnings, its second $80 billion buyback authorization in three quarters.",
        "expected_ticker": "NVDA",
    },
    {
        "question": "What is the latest news about Apple stock performance?",
        "ground_truth": "Apple stock reached an all-time high near $300-$305 in May-June 2026, with analysts from Bank of America raising the price target to $380 and Morgan Stanley reiterating an overweight rating.",
        "expected_ticker": "AAPL",
    },
    {
        "question": "What happened to Meta stock after they announced paid subscriptions?",
        "ground_truth": "Meta shares jumped after the company announced premium subscription plans under the Meta One banner, offering plans at $7.99 and $19.99 per month for its apps and AI chatbot.",
        "expected_ticker": "META",
    },
    {
        "question": "What concerns were raised about NVIDIA stock despite strong earnings?",
        "ground_truth": "Despite record earnings, NVIDIA stock dipped after hours as investors worried about the sustainability of its extraordinary growth momentum and rising competition from AMD, Cerebras, Amazon, and Google.",
        "expected_ticker": "NVDA",
    },
    {
        "question": "What is the latest news on Tesla stock?",
        "ground_truth": "Tesla reported its strongest quarter in years with Q1 2026 EPS of $0.41 beating the $0.36 consensus. Tesla also holds a SpaceX stake with roughly $890 million in related revenue since 2023.",
        "expected_ticker": "TSLA",
    },
    {
        "question": "How did Amazon stock perform and what is the analyst outlook?",
        "ground_truth": "Wall Street analysts believe Amazon stock can rally 40% to $370, with the company well-positioned due to its AWS cloud business and AI investments. Analysts highlighted its strong long-term cash generation.",
        "expected_ticker": "AMZN",
    },
    {
        "question": "What major IPO was expected in 2026 related to Elon Musk?",
        "ground_truth": "SpaceX filed for what could become the largest IPO in history, seeking to raise up to $75-80 billion on public markets. The filing revealed Musk would maintain majority voting control through a dual-class share structure.",
        "expected_ticker": "TSLA",
    },
    {
        "question": "What happened to the Nasdaq on June 5 2026?",
        "ground_truth": "The Nasdaq suffered its worst day of 2026, falling 4.18% as AI stocks tumbled following a stronger-than-expected jobs report that erased hopes for near-term Federal Reserve rate cuts. The S&P 500 fell 2.64%.",
        "expected_ticker": None,
    },
    {
        "question": "What is the latest news about Microsoft and AI?",
        "ground_truth": "Microsoft is reshaping its leadership for the AI era under CEO Satya Nadella, pushing flatter teams and AI-focused executives. The company launched GridSFM to cut grid congestion losses and faces competition from Claude and Gemini.",
        "expected_ticker": "MSFT",
    },
    {
        "question": "What did Meta do with its Forum app?",
        "ground_truth": "Meta launched Forum, a standalone app for Facebook Groups combined with AI assistants, directly competing with Reddit. Reddit shares fell 6% following the announcement.",
        "expected_ticker": "META",
    },
    {
        "question": "What is the latest on Alphabet Google stock?",
        "ground_truth": "Alphabet stock received a raised price target from Piper Sandler in June 2026. Jim Cramer highlighted Alphabet's expensive new capital needs for AI infrastructure, while the stock was seen as a top pick-and-shovel AI play.",
        "expected_ticker": "GOOGL",
    },
    {
        "question": "What concerns did JPMorgan raise about AI spending?",
        "ground_truth": "JPMorgan sounded the alarm on the runaway cost of AI in June 2026, warning that AI infrastructure spending had reached levels difficult to comprehend, comparing the figures to infinity in terms of scale.",
        "expected_ticker": None,
    },
    {
        "question": "What was the market reaction to NVIDIA earnings in May 2026?",
        "ground_truth": "Despite NVIDIA reporting record Q1 revenue of $81.6 billion, the stock received a tepid reaction and dipped after hours. Investors focused on whether the extraordinary growth pace could be sustained amid rising competition.",
        "expected_ticker": "NVDA",
    },
]


# ── Run the RAG pipeline for each question ────────────────────────────────────

def run_rag_for_question(question: str) -> dict:
    """
    Run a question through the StockPilot News RAG pipeline via a single
    run_query() call, so the answer and the contexts used for RAGAS come
    from the exact same graph execution (ticker extraction, filtered
    retrieval, doc grading, possible web-search fallback all included).

    Returns:
        dict with keys:
            question : str
            answer   : str
            contexts : list[str]   — plain text, for RAGAS
            docs     : list[dict]  — raw docs with metadata intact
                                      (used for the ticker-match ranking
                                      proxy below)
            route    : str | None
    """
    from agents.graph import run_query

    result = run_query(question)
    route  = result.get("route")
    answer = result.get("final_answer", "") or ""

    if route == "stock_data_rag":
        # No retrieved_docs/web_search_results exist on this path —
        # RAGAS's context-based metrics aren't meaningful here.
        logger.info(
            "    note: route=stock_data_rag — no retrieved context, "
            "context-based RAGAS metrics not meaningful for this question"
        )
        docs = []
    else:
        docs = result.get("retrieved_docs", []) + result.get("web_search_results", [])

    contexts = [d.get("document", "") for d in docs]

    return {
        "question": question,
        "answer":   answer,
        "contexts": contexts,
        "docs":     docs,
        "route":    route,
    }


# ── Ranking metrics (ticker-match proxy) ──────────────────────────────────────
# RAGAS has no ranking metrics, and there are no hand-labeled per-document
# relevance judgments available for this dataset. As an explicitly-documented
# proxy, we treat a retrieved doc as "relevant" if its metadata ticker
# matches the question's expected_ticker. This is NOT ground-truth relevance
# — it's a cheap signal that at least confirms the retriever surfaced docs
# about the right company. Questions with expected_ticker=None are skipped
# (not scored as 0), since there is no ticker to match against.

def _reciprocal_rank(docs: list[dict], expected_ticker: str) -> float:
    for rank, doc in enumerate(docs, 1):
        if doc.get("metadata", {}).get("ticker") == expected_ticker:
            return 1.0 / rank
    return 0.0


def _ndcg_at_k(docs: list[dict], expected_ticker: str, k: int = 6) -> float:
    top_k = docs[:k]
    relevances = [
        1.0 if doc.get("metadata", {}).get("ticker") == expected_ticker else 0.0
        for doc in top_k
    ]

    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))

    ideal_relevances = sorted(relevances, reverse=True)
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_relevances))

    return dcg / idcg if idcg > 0 else 0.0


# ── RAGAS evaluation ──────────────────────────────────────────────────────────

def run_ragas_evaluation():
    """
    Run RAGAS evaluation on all test questions and print a report.
    """
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    from langchain_anthropic import ChatAnthropic
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from config.config import LLMConfig

    logger.info("=" * 60)
    logger.info("StockPilot AI — RAGAS Evaluation")
    logger.info(f"Questions: {len(TEST_DATASET)}")
    logger.info(f"Started:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    judge_llm = LangchainLLMWrapper(ChatAnthropic(
        model=LLMConfig.CLAUDE_MODEL,
        api_key=LLMConfig.ANTHROPIC_API_KEY,
        max_tokens=2048,
    ))

    judge_embeddings = LangchainEmbeddingsWrapper(GoogleGenerativeAIEmbeddings(
        model=LLMConfig.GEMINI_EMBEDDING_MODEL,
        google_api_key=LLMConfig.GEMINI_API_KEY,
    ))

    logger.info("\nRunning RAG pipeline for each question...")
    results = []

    for i, item in enumerate(TEST_DATASET, 1):
        question        = item["question"]
        ground_truth    = item["ground_truth"]
        expected_ticker = item["expected_ticker"]

        logger.info(f"  [{i}/{len(TEST_DATASET)}] {question[:70]}...")

        try:
            rag_output = run_rag_for_question(question)
            results.append({
                "question":        question,
                "answer":          rag_output["answer"],
                "contexts":        rag_output["contexts"],
                "docs":            rag_output["docs"],
                "route":           rag_output["route"],
                "ground_truth":    ground_truth,
                "expected_ticker": expected_ticker,
            })
            logger.info(f"    ✓ Answer: {rag_output['answer'][:80]}...")
            logger.info(f"    ✓ Contexts retrieved: {len(rag_output['contexts'])}")

        except Exception as exc:
            logger.error(f"    ✗ Failed: {exc}")
            results.append({
                "question":        question,
                "answer":          "",
                "contexts":        [],
                "docs":            [],
                "route":           None,
                "ground_truth":    ground_truth,
                "expected_ticker": expected_ticker,
            })

    logger.info(f"\nBuilding RAGAS dataset from {len(results)} results...")

    dataset = Dataset.from_dict({
        "question":    [r["question"]    for r in results],
        "answer":      [r["answer"]      for r in results],
        "contexts":    [r["contexts"]    for r in results],
        "ground_truth":[r["ground_truth"] for r in results],
    })

    logger.info("\nRunning RAGAS metrics (this may take 5-10 minutes)...")
    logger.info("Metrics: faithfulness, answer_relevancy, context_precision, context_recall")

    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]

    for metric in metrics:
        metric.llm       = judge_llm
        if hasattr(metric, "embeddings"):
            metric.embeddings = judge_embeddings

    score = evaluate(
        dataset,
        metrics=metrics,
    )

    logger.info("\n" + "=" * 60)
    logger.info("RAGAS EVALUATION RESULTS")
    logger.info("=" * 60)

    df = score.to_pandas()
    metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    score_dict = {}
    for m in metric_names:
        if m in df.columns:
            score_dict[m] = df[m].dropna().mean()
        else:
            score_dict[m] = float("nan")

    # Ranking metrics (ticker-match proxy) — computed separately from RAGAS
    mrr_scores, ndcg_scores, skipped_ranking = [], [], 0
    for r in results:
        expected_ticker = r["expected_ticker"]
        if expected_ticker is None:
            skipped_ranking += 1
            logger.info(f"  [ranking] '{r['question'][:50]}...' skipped: no ticker proxy")
            continue
        docs = r["docs"]
        mrr_scores.append(_reciprocal_rank(docs, expected_ticker))
        ndcg_scores.append(_ndcg_at_k(docs, expected_ticker, k=6))

    mean_mrr  = sum(mrr_scores) / len(mrr_scores) if mrr_scores else float("nan")
    mean_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else float("nan")
    score_dict["mrr"] = mean_mrr
    score_dict["ndcg_at_6"] = mean_ndcg

    metric_labels = {
        "faithfulness":       "Faithfulness       (answer grounded in context)",
        "answer_relevancy":   "Answer Relevancy   (answer addresses question)",
        "context_precision":  "Context Precision  (retrieved docs are relevant)",
        "context_recall":     "Context Recall     (context covers the answer)",
    }

    overall = 0.0
    count   = 0

    for key, label in metric_labels.items():
        value = score_dict.get(key, 0.0)
        bar   = "█" * int(value * 20) + "░" * (20 - int(value * 20))
        logger.info(f"  {label}")
        logger.info(f"  [{bar}] {value:.3f}")
        logger.info("")
        overall += value
        count   += 1

    avg = overall / count if count > 0 else 0.0
    logger.info(f"  Overall Average Score: {avg:.3f}")
    logger.info("=" * 60)

    logger.info("\nRANKING METRICS (ticker-match proxy, not hand-labeled ground truth)")
    logger.info(f"  Scorable questions: {len(mrr_scores)} | Skipped (no ticker proxy): {skipped_ranking}")
    for key, label in (("mrr", "MRR        (rank of first ticker-matched doc)"),
                        ("ndcg_at_6", "NDCG@6     (ranking quality of top 6 docs)")):
        value = score_dict.get(key, 0.0)
        if value != value:  # NaN check
            logger.info(f"  {label}")
            logger.info("  [no scorable questions]")
            logger.info("")
            continue
        bar = "█" * int(value * 20) + "░" * (20 - int(value * 20))
        logger.info(f"  {label}")
        logger.info(f"  [{bar}] {value:.3f}")
        logger.info("")
    logger.info("=" * 60)

    logger.info("\nINTERPRETATION:")
    logger.info("  0.8 - 1.0 : Excellent")
    logger.info("  0.6 - 0.8 : Good")
    logger.info("  0.4 - 0.6 : Needs improvement")
    logger.info("  0.0 - 0.4 : Poor — significant issues")
    logger.info("")

    if score_dict.get("faithfulness", 0) < 0.6:
        logger.info("  ⚠ Low Faithfulness — Claude may be hallucinating beyond the docs")
    if score_dict.get("answer_relevancy", 0) < 0.6:
        logger.info("  ⚠ Low Answer Relevancy — answers not addressing questions well")
    if score_dict.get("context_precision", 0) < 0.6:
        logger.info("  ⚠ Low Context Precision — hybrid search retrieving irrelevant docs")
    if score_dict.get("context_recall", 0) < 0.6:
        logger.info("  ⚠ Low Context Recall — missing relevant docs in retrieval")

    output = {
        "timestamp":     datetime.now().isoformat(),
        "num_questions": len(TEST_DATASET),
        "scores":        {k: round(v, 4) for k, v in score_dict.items()},
        "average":       round(avg, 4),
        "ranking_metrics_note": (
            "mrr and ndcg_at_6 use each retrieved doc's metadata ticker "
            "compared against the question's expected_ticker as a proxy "
            "relevance signal, not hand-labeled ground truth. Questions "
            "with expected_ticker=None are excluded from these two scores."
        ),
        "ranking_metrics_scorable_questions": len(mrr_scores),
        "ranking_metrics_skipped_questions":  skipped_ranking,
        "per_question": score.to_pandas().to_dict(orient="records"),
    }

    output_path = "/tmp/ragas_results.json"

    def _json_default(obj):
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=_json_default)

    logger.info(f"\nDetailed results saved to: {output_path}")
    logger.info("To view: docker exec stockpilot_api cat /tmp/ragas_results.json")
    logger.info("=" * 60)

    return output


# ── Helper: suggest new questions for future dataset expansion ────────────────
# Opt-in only, not run automatically. Prints candidate headlines/snippets
# grouped by ticker for human review — does NOT auto-generate or insert
# anything into TEST_DATASET.

def suggest_new_questions(months_back: int = 2) -> None:
    """
    Connects directly to MongoDB (same connection pattern as
    airflow/dags/dag_s3_archive.py's archive_mongo_news, including
    authSource=admin) and prints real article headlines/snippets grouped by
    ticker for the most recent `months_back` months.

    For human review only — turn interesting candidates into properly
    grounded new TEST_DATASET entries by hand.
    """
    from collections import defaultdict
    from dateutil.relativedelta import relativedelta
    from pymongo import MongoClient

    from config.config import MongoConfig

    cutoff = datetime.now() - relativedelta(months=months_back)

    client     = MongoClient(MongoConfig.URL, serverSelectionTimeoutMS=5000)
    collection = client[MongoConfig.DB]["articles"]

    articles = list(collection.find(
        {"published_at": {"$gte": cutoff}},
        {
            "ticker":       1,
            "headline":     1,
            "summary":      1,
            "published_at": 1,
            "_id":          0,
        }
    ).sort("published_at", -1))

    logger.info("=" * 60)
    logger.info(f"Candidate articles for new questions (last {months_back} months)")
    logger.info(f"Total articles found: {len(articles)}")
    logger.info("=" * 60)

    by_ticker: dict[str, list[dict]] = defaultdict(list)
    for article in articles:
        by_ticker[article.get("ticker") or "UNTICKERED"].append(article)

    for ticker in sorted(by_ticker):
        group = by_ticker[ticker]
        logger.info(f"\n--- {ticker} ({len(group)} articles) ---")
        for article in group:
            published = article.get("published_at")
            published_str = published.isoformat() if published else "unknown date"
            headline = article.get("headline", "")
            summary  = article.get("summary", "")[:200]
            logger.info(f"  [{published_str}] {headline}")
            if summary:
                logger.info(f"    {summary}")

    logger.info("\n" + "=" * 60)
    logger.info("Review the above and manually add well-grounded questions/")
    logger.info("ground_truth to TEST_DATASET — nothing was auto-inserted.")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StockPilot AI RAGAS evaluation")
    parser.add_argument(
        "--suggest",
        action="store_true",
        help="Print candidate article headlines/snippets for future TEST_DATASET "
             "expansion (human review only) instead of running the RAGAS evaluation.",
    )
    parser.add_argument(
        "--months-back",
        type=int,
        default=2,
        help="Used with --suggest: how many months back to pull candidate articles from.",
    )
    args = parser.parse_args()

    if args.suggest:
        suggest_new_questions(months_back=args.months_back)
    else:
        run_ragas_evaluation()
