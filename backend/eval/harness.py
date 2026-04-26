# harness.py
"""Main evaluation harness for the CS department chatbot.

Usage:
    cd backend
    python eval/harness.py
    python eval/harness.py --questions adv_001 adv_002   # run specific questions
    python eval/harness.py --category financial_aid       # run one category
    python eval/harness.py --no-judge                     # skip LLM judge (faster)

Outputs (written to backend/eval/results/):
    eval_results_{run_id}.jsonl   — one JSON record per question (full detail)
    eval_answers_{run_id}.csv     — question_id, question, answer, passed
    eval_scores_{run_id}.csv      — question_id, scores, judge reasoning
    eval_summary_{run_id}.json    — aggregate metrics and category breakdown
"""

import argparse
import csv
import json
import os
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import yaml

try:
    import textstat
    _TEXTSTAT_AVAILABLE = True
except ImportError:
    _TEXTSTAT_AVAILABLE = False

# Allow imports from backend/ when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval import generate_grounded_answer, build_context
from eval.judge import judge_question, judge_total

GROUND_TRUTH_PATH = Path(__file__).parent / "ground_truth.yaml"
RESULTS_DIR = Path(__file__).parent / "results"

DEPARTMENT_ID = "cs"

# Passing thresholds (from EVALUATION_PLAN.md)
RETRIEVAL_SCORE_THRESHOLD = 0.7
JUDGE_TOTAL_THRESHOLD = 0.7


# ── Ground truth loader ───────────────────────────────────────────────────────

def load_ground_truth(
    question_ids: list[str] | None = None,
    category: str | None = None,
) -> list[dict]:
    with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
        records = yaml.safe_load(f)

    if question_ids:
        records = [r for r in records if r["question_id"] in question_ids]
    if category:
        records = [r for r in records if r["category"] == category]

    return records


# ── Deterministic metrics ─────────────────────────────────────────────────────

def compute_retrieval_hit(chunks: list[dict], expected_ids: list[str]) -> bool:
    """True if at least one expected chunk_id appears in the top-5 results."""
    if not expected_ids:
        return False
    retrieved_ids = {c.get("chunk_id") for c in chunks}
    return bool(retrieved_ids.intersection(expected_ids))


def compute_top1_match(chunks: list[dict], expected_ids: list[str]) -> bool:
    """True if the rank-1 chunk is one of the expected chunk IDs."""
    if not expected_ids or not chunks:
        return False
    return chunks[0].get("chunk_id") in expected_ids


def compute_source_type_correct(chunks: list[dict], expected_source_type: str) -> bool:
    """True if the dominant source type in top-3 matches expected.

    'either' and 'redirect' are always considered correct — no preference enforced.
    """
    if expected_source_type in ("either", "redirect"):
        return True
    if not chunks:
        return False
    top3 = chunks[:3]
    sources = [c.get("content_source", "") for c in top3]
    dominant = max(set(sources), key=sources.count) if sources else ""
    return dominant == expected_source_type


def compute_banner_redirect_triggered(answer: str) -> bool:
    """True if the Banner URL appears anywhere in the system answer."""
    return "banner-public.nmsu.edu" in answer.lower()


def compute_citation_format_valid(chunks: list[dict], answer: str) -> bool:
    """Basic heuristic: catalog chunks should produce a catalog citation,
    web chunks should produce a URL citation.

    Returns True if at least one citation of the expected type is present.
    Falls back to True when no chunks are available (cannot evaluate).
    """
    if not chunks:
        return True

    has_catalog = any(c.get("content_source") == "catalog" for c in chunks)
    has_web = any(c.get("content_source") == "web" for c in chunks)

    answer_lower = answer.lower()

    if has_catalog:
        # Expect something like "catalog" + a year pattern
        has_catalog_cite = (
            "catalog" in answer_lower and
            any(str(y) in answer for y in range(2020, 2030))
        )
        if not has_catalog_cite:
            return False

    if has_web:
        # Expect at least one URL
        has_url = "http" in answer_lower
        if not has_url:
            return False

    return True


def compute_retrieval_score(
    retrieval_hit: bool,
    top1_match: bool,
    source_type_correct: bool,
) -> float:
    """Weighted retrieval score: hit×0.4 + top1×0.3 + source_correct×0.3."""
    return (
        (0.4 if retrieval_hit else 0.0) +
        (0.3 if top1_match else 0.0) +
        (0.3 if source_type_correct else 0.0)
    )


# ── Deterministic content metrics ─────────────────────────────────────────────

_FILLER_PATTERNS = re.compile(
    r"\b(great question|certainly[!,]?|of course[!,]?|i'?d be happy to help"
    r"|as an ai|i'?m here to help|absolutely[!,]?|sure[!,]?)\b",
    re.IGNORECASE,
)

_COURSE_CODE_RE = re.compile(r'\b([A-Z][A-Z\s]{0,5}\s+\d{3,4}[A-Z]?)\b')

_URL_RE = re.compile(r'https?://[^\s)\]>,"\']+')

_OPENER_RE = re.compile(
    r"^\s*(great|certainly|of course|i'?d be happy|as an ai|i'?m here|absolutely|sure)[!,\s]",
    re.IGNORECASE,
)


def compute_key_fact_coverage(answer: str, key_facts: list[str]) -> float:
    """Fraction of key facts whose significant terms appear in the answer.

    A key fact is considered 'covered' if at least half of its non-stopword
    tokens appear anywhere in the answer text (case-insensitive).
    Returns 0.0 if there are no key facts.
    """
    if not key_facts:
        return 0.0

    _STOPWORDS = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been",
        "being", "of", "in", "to", "for", "and", "or", "that", "this",
        "it", "at", "by", "with", "as", "on", "from", "not", "no",
    }

    answer_lower = answer.lower()
    covered = 0

    for fact in key_facts:
        tokens = [t for t in re.findall(r"[a-z0-9]+", fact.lower()) if t not in _STOPWORDS]
        if not tokens:
            covered += 1
            continue
        hits = sum(1 for t in tokens if t in answer_lower)
        if hits >= max(1, len(tokens) // 2):
            covered += 1

    return round(covered / len(key_facts), 3)


def compute_filler_detected(answer: str) -> bool:
    """True if the answer contains a known filler phrase."""
    return bool(_FILLER_PATTERNS.search(answer))


def compute_leads_with_answer(answer: str) -> bool:
    """True if the answer does NOT start with a filler opener.

    Checks whether the first non-blank line begins with a disqualifying
    opener phrase (e.g., "Great question!", "Certainly!").
    """
    return not bool(_OPENER_RE.match(answer.strip()))


def compute_course_entity_valid(answer: str, key_facts: list[str]) -> bool | None:
    """Check that course codes in key facts are not misquoted in the answer.

    Returns True if no key-fact course codes appear in the answer with wrong
    formatting. Returns None if key facts contain no course codes (not applicable).
    """
    fact_text = " ".join(key_facts)
    expected_codes = {
        re.sub(r'\s+', ' ', m.group(1)).strip()
        for m in _COURSE_CODE_RE.finditer(fact_text)
    }

    if not expected_codes:
        return None

    answer_codes = {
        re.sub(r'\s+', ' ', m.group(1)).strip()
        for m in _COURSE_CODE_RE.finditer(answer)
    }

    # All expected codes should appear verbatim in the answer when cited
    mismatched = expected_codes - answer_codes
    # Only flag as invalid if codes appear in the answer but differ from expected
    if not answer_codes:
        return True  # answer didn't mention course codes at all — not a mismatch
    return len(mismatched) == 0


def compute_url_hallucination(answer: str, chunks: list[dict]) -> bool:
    """True if all URLs in the answer can be traced to a retrieved chunk source or
    a known redirect URL (Banner, apply.nmsu.edu, computerscience.nmsu.edu).

    Returns True (no hallucination) when no URLs appear in the answer.
    """
    _KNOWN_URLS = {
        "banner-public.nmsu.edu",
        "apply.nmsu.edu",
        "computerscience.nmsu.edu",
        "nmsu.edu",
    }

    answer_urls = _URL_RE.findall(answer)
    if not answer_urls:
        return True

    chunk_sources = {(c.get("source") or "").lower() for c in chunks}

    for url in answer_urls:
        url_lower = url.lower()
        # Accept if domain matches a known redirect domain
        if any(known in url_lower for known in _KNOWN_URLS):
            continue
        # Accept if the URL (or its prefix) matches a chunk source
        if any(url_lower.startswith(src) or src.startswith(url_lower)
               for src in chunk_sources if src):
            continue
        return False  # URL found that cannot be traced

    return True


def compute_response_length(answer: str) -> int:
    """Word count of the answer."""
    return len(answer.split())


def compute_readability(answer: str) -> float | None:
    """Flesch Reading Ease score (0–100; higher = easier).

    Returns None if textstat is not installed.
    Scores: 90-100 very easy, 60-70 standard, below 30 very difficult.
    """
    if not _TEXTSTAT_AVAILABLE:
        return None
    return round(textstat.flesch_reading_ease(answer), 1)


# ── Per-question evaluation ───────────────────────────────────────────────────

def evaluate_question(record: dict, use_judge: bool) -> dict:
    """Run the full pipeline for one ground truth record and return a result dict."""
    question_id = record["question_id"]
    question = record["question"]
    expected_ids = record.get("expected_chunk_ids") or []
    expected_source_type = record.get("expected_source_type", "either")
    banner_redirect_expected = record.get("banner_redirect_expected", False)
    key_facts = record.get("key_facts", [])

    result = {
        "question_id":    question_id,
        "category":       record.get("category", ""),
        "question":       question,
        "key_facts":      key_facts,
        "expected_chunk_ids":    expected_ids,
        "expected_source_type":  expected_source_type,
        "banner_redirect_expected": banner_redirect_expected,
        # filled below
        "system_answer":  None,
        "system_sources": None,
        "retrieved_chunks": None,
        # deterministic retrieval metrics
        "retrieval_hit":  None,
        "top1_match":     None,
        "source_type_correct": None,
        "banner_redirect_triggered": None,
        "citation_format_valid": None,
        "retrieval_score": None,
        # deterministic content metrics
        "key_fact_coverage": None,
        "filler_detected": None,
        "leads_with_answer": None,
        "course_entity_valid": None,
        "url_hallucination_clean": None,
        "response_length_words": None,
        "readability_flesch": None,
        # LLM judge scores
        "faithfulness":   None,
        "completeness":   None,
        "source_preference": None,
        "citation_quality": None,
        "hallucination":  None,
        "response_quality": None,
        "judge_total":    None,
        "judge_reasoning": None,
        "judge_tokens_used": None,
        "judge_error":    None,
        "passed":         None,
        "latency_ms":     None,
        "error":          None,
    }

    # ── Pipeline call ─────────────────────────────────────────────────────────
    t0 = time.monotonic()
    try:
        pipeline_result = generate_grounded_answer(question, DEPARTMENT_ID)
    except Exception as e:
        result["error"] = str(e)
        result["latency_ms"] = int((time.monotonic() - t0) * 1000)
        return result

    result["latency_ms"] = int((time.monotonic() - t0) * 1000)

    answer = pipeline_result.get("answer", "")
    chunks = pipeline_result.get("chunks", [])
    sources = pipeline_result.get("sources", [])

    result["system_answer"] = answer
    result["system_sources"] = sources
    result["retrieved_chunks"] = [
        {
            "chunk_id":       c.get("chunk_id"),
            "rank":           c.get("rank"),
            "hybrid_score":   c.get("hybrid_score"),
            "metadata_boost": c.get("metadata_boost"),
            "final_score":    c.get("final_score"),
            "chunk_type":     c.get("chunk_type"),
            "content_source": c.get("content_source"),
            "heading":        c.get("heading"),
            "source":         c.get("source"),
        }
        for c in chunks
    ]

    # ── Deterministic metrics ─────────────────────────────────────────────────
    retrieval_hit = compute_retrieval_hit(chunks, expected_ids)
    top1_match = compute_top1_match(chunks, expected_ids)
    source_type_correct = compute_source_type_correct(chunks, expected_source_type)
    banner_triggered = compute_banner_redirect_triggered(answer)
    citation_valid = compute_citation_format_valid(chunks, answer)
    ret_score = compute_retrieval_score(retrieval_hit, top1_match, source_type_correct)

    result["retrieval_hit"] = retrieval_hit
    result["top1_match"] = top1_match
    result["source_type_correct"] = source_type_correct
    result["banner_redirect_triggered"] = banner_triggered if banner_redirect_expected else None
    result["citation_format_valid"] = citation_valid
    result["retrieval_score"] = ret_score

    # ── Deterministic content metrics ─────────────────────────────────────────
    result["key_fact_coverage"]     = compute_key_fact_coverage(answer, key_facts)
    result["filler_detected"]       = compute_filler_detected(answer)
    result["leads_with_answer"]     = compute_leads_with_answer(answer)
    result["course_entity_valid"]   = compute_course_entity_valid(answer, key_facts)
    result["url_hallucination_clean"] = compute_url_hallucination(answer, chunks)
    result["response_length_words"] = compute_response_length(answer)
    result["readability_flesch"]    = compute_readability(answer)

    # ── LLM judge ─────────────────────────────────────────────────────────────
    if use_judge:
        context_str = build_context(chunks)
        judge_scores = judge_question(
            question=question,
            key_facts=key_facts,
            context=context_str,
            answer=answer,
            banner_redirect_expected=banner_redirect_expected,
        )
        jt = judge_total(judge_scores)

        result["faithfulness"]      = judge_scores["faithfulness"]
        result["completeness"]      = judge_scores["completeness"]
        result["source_preference"] = judge_scores["source_preference"]
        result["citation_quality"]  = judge_scores["citation_quality"]
        result["hallucination"]     = judge_scores["hallucination"]
        result["response_quality"]  = judge_scores["response_quality"]
        result["judge_total"]       = jt
        result["judge_reasoning"]   = judge_scores["reasoning"]
        result["judge_tokens_used"] = judge_scores["judge_tokens_used"]
        result["judge_error"]       = judge_scores["judge_error"]

        if jt is not None:
            if banner_redirect_expected:
                # Redirect questions have no retrievable chunk — pass on
                # redirect trigger + judge quality, not retrieval score.
                result["passed"] = (banner_triggered and
                                    jt >= JUDGE_TOTAL_THRESHOLD)
            elif not expected_ids:
                # Unanswerable questions have no expected chunks — pass on
                # judge quality alone (system must say "not available").
                result["passed"] = jt >= JUDGE_TOTAL_THRESHOLD
            else:
                result["passed"] = (ret_score >= RETRIEVAL_SCORE_THRESHOLD and
                                    jt >= JUDGE_TOTAL_THRESHOLD)
    else:
        # Without judge, pass based on retrieval score only
        # (redirect/unanswerable questions fall back to banner trigger check)
        if banner_redirect_expected:
            result["passed"] = banner_triggered
        elif not expected_ids:
            result["passed"] = True  # cannot evaluate without judge
        else:
            result["passed"] = ret_score >= RETRIEVAL_SCORE_THRESHOLD

    return result


# ── Summary builder ───────────────────────────────────────────────────────────

def build_summary(results: list[dict], run_id: str, run_timestamp: str) -> dict:
    total = len(results)
    passed = sum(1 for r in results if r.get("passed") is True)
    errors = [r["question_id"] for r in results if r.get("error")]
    failed = [r["question_id"] for r in results if r.get("passed") is False]

    latencies = [r["latency_ms"] for r in results if r.get("latency_ms") is not None]
    avg_latency = int(sum(latencies) / len(latencies)) if latencies else None
    p90_latency = int(sorted(latencies)[int(len(latencies) * 0.9)]) if latencies else None

    # Per-category breakdown
    categories: dict[str, list] = {}
    for r in results:
        cat = r.get("category", "unknown")
        categories.setdefault(cat, []).append(r)

    category_breakdown = {}
    for cat, cat_results in sorted(categories.items()):
        cat_passed = sum(1 for r in cat_results if r.get("passed") is True)
        ret_scores = [r["retrieval_score"] for r in cat_results
                      if r.get("retrieval_score") is not None]
        judge_totals = [r["judge_total"] for r in cat_results
                        if r.get("judge_total") is not None]
        category_breakdown[cat] = {
            "count":              len(cat_results),
            "passed":             cat_passed,
            "pass_rate":          round(cat_passed / len(cat_results), 3) if cat_results else None,
            "avg_retrieval_score": round(sum(ret_scores) / len(ret_scores), 3) if ret_scores else None,
            "avg_judge_total":    round(sum(judge_totals) / len(judge_totals), 3) if judge_totals else None,
        }

    # Low retrieval questions
    low_retrieval = [
        r["question_id"] for r in results
        if r.get("retrieval_score") is not None and r["retrieval_score"] < RETRIEVAL_SCORE_THRESHOLD
    ]

    return {
        "run_id":            run_id,
        "run_timestamp":     run_timestamp,
        "total_questions":   total,
        "total_passed":      passed,
        "pass_rate":         round(passed / total, 3) if total else None,
        "avg_latency_ms":    avg_latency,
        "p90_latency_ms":    p90_latency,
        "category_breakdown": category_breakdown,
        "failed_questions":  failed,
        "low_retrieval_ids": low_retrieval,
        "error_questions":   errors,
    }


# ── Stdout summary table ──────────────────────────────────────────────────────

def print_summary(summary: dict) -> None:
    print(f"\n{'='*60}")
    print(f"Run ID:   {summary['run_id']}")
    print(f"Time:     {summary['run_timestamp']}")
    print(f"Total:    {summary['total_questions']} questions")
    print(f"Passed:   {summary['total_passed']}  ({summary['pass_rate']:.1%})")
    print(f"Latency:  avg {summary['avg_latency_ms']}ms  p90 {summary['p90_latency_ms']}ms")

    print(f"\n{'Category':<30} {'Count':>5} {'Passed':>6} {'Pass%':>6} {'AvgRet':>7} {'AvgJudge':>9}")
    print("-" * 68)
    for cat, stats in summary["category_breakdown"].items():
        pass_pct = f"{stats['pass_rate']:.0%}" if stats["pass_rate"] is not None else "—"
        avg_ret = f"{stats['avg_retrieval_score']:.2f}" if stats["avg_retrieval_score"] is not None else "—"
        avg_j = f"{stats['avg_judge_total']:.2f}" if stats["avg_judge_total"] is not None else "—"
        print(f"{cat:<30} {stats['count']:>5} {stats['passed']:>6} {pass_pct:>6} {avg_ret:>7} {avg_j:>9}")

    if summary["failed_questions"]:
        print(f"\nFailed ({len(summary['failed_questions'])}):")
        for qid in summary["failed_questions"][:10]:
            print(f"  {qid}")
        if len(summary["failed_questions"]) > 10:
            print(f"  ... and {len(summary['failed_questions']) - 10} more")

    if summary["error_questions"]:
        print(f"\nErrors ({len(summary['error_questions'])}):")
        for qid in summary["error_questions"]:
            print(f"  {qid}")

    print(f"{'='*60}\n")


# ── CSV export ────────────────────────────────────────────────────────────────

_CSV_ANSWERS_FIELDS = [
    "question_id", "category", "question", "passed", "system_answer",
]

_CSV_SCORES_FIELDS = [
    "question_id", "category", "passed",
    "retrieval_score", "judge_total",
    "retrieval_hit", "top1_match", "source_type_correct",
    "faithfulness", "completeness", "hallucination", "response_quality",
    "latency_ms", "judge_reasoning",
]


def _write_csv(results: list, answers_path: Path, scores_path: Path) -> None:
    with open(answers_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_ANSWERS_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    with open(scores_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_SCORES_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CS chatbot evaluation harness")
    parser.add_argument(
        "--questions", nargs="+", metavar="ID",
        help="Run only these question IDs (e.g. adv_001 fin_002)"
    )
    parser.add_argument(
        "--category", metavar="CAT",
        help="Run only questions in this category (e.g. financial_aid)"
    )
    parser.add_argument(
        "--no-judge", action="store_true",
        help="Skip LLM-as-judge (faster, deterministic metrics only)"
    )
    args = parser.parse_args()

    use_judge = not args.no_judge

    # Load ground truth
    records = load_ground_truth(
        question_ids=args.questions,
        category=args.category,
    )
    if not records:
        print("No questions matched the filter. Check --questions or --category.")
        sys.exit(1)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    run_timestamp = datetime.now(timezone.utc).isoformat()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path  = RESULTS_DIR / f"eval_results_{run_id}.jsonl"
    answers_path  = RESULTS_DIR / f"eval_answers_{run_id}.csv"
    scores_path   = RESULTS_DIR / f"eval_scores_{run_id}.csv"
    summary_path  = RESULTS_DIR / f"eval_summary_{run_id}.json"

    print(f"Run ID: {run_id}")
    print(f"Questions: {len(records)}  |  Judge: {'yes' if use_judge else 'no'}")
    print(f"Results → {results_path}\n")

    results = []

    with open(results_path, "w", encoding="utf-8") as out:
        for i, record in enumerate(records, start=1):
            qid = record["question_id"]
            print(f"[{i}/{len(records)}] {qid} ...", end=" ", flush=True)

            result = evaluate_question(record, use_judge=use_judge)
            results.append(result)

            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            out.flush()

            status = "✓" if result.get("passed") else ("✗" if result.get("passed") is False else "?")
            latency = result.get("latency_ms", "?")
            ret = result.get("retrieval_score")
            jt = result.get("judge_total")
            ret_str = f"{ret:.2f}" if ret is not None else "—"
            jt_str = f"{jt:.2f}" if jt is not None else "—"
            print(f"{status}  {latency}ms  ret={ret_str}  judge={jt_str}")

    summary = build_summary(results, run_id, run_timestamp)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    _write_csv(results, answers_path, scores_path)

    print_summary(summary)
    print(f"Summary → {summary_path}")
    print(f"Answers → {answers_path}")
    print(f"Scores  → {scores_path}")


if __name__ == "__main__":
    main()
