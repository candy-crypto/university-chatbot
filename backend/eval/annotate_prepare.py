"""annotate_prepare.py
====================
Prepare a human-review CSV from the most recent eval run.

Includes every question in the eval set (sorted by category then question_id),
excluding questions marked unanswerable (listed in EXCLUDE_IDS).

Merges eval_scores and eval_answers, adds blank human_pass / human_notes
columns, and writes backend/eval/results/human_review_{run_id}.csv.

Usage:
    cd backend
    python eval/annotate_prepare.py                        # uses latest run
    python eval/annotate_prepare.py --run 20260426_034716  # specific run
"""

import argparse
import csv
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"

# Questions excluded from human annotation (known unanswerable from sources)
EXCLUDE_IDS = set()

# judge_total below this (but still passing) = borderline
BORDERLINE_THRESHOLD = 0.85


def latest_run_id() -> str:
    scores_files = sorted(RESULTS_DIR.glob("eval_scores_*.csv"), reverse=True)
    if not scores_files:
        sys.exit("No eval_scores_*.csv files found in results/")
    name = scores_files[0].stem          # e.g. eval_scores_20260426_034716_091def
    parts = name.split("_", maxsplit=2)  # ['eval', 'scores', '20260426_034716_091def']
    return parts[2]


def load_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser(description="Prepare human annotation review sheet.")
    parser.add_argument("--run", metavar="RUN_ID", help="Specific run ID to use.")
    args = parser.parse_args()

    run_id = args.run or latest_run_id()
    scores_path  = RESULTS_DIR / f"eval_scores_{run_id}.csv"
    answers_path = RESULTS_DIR / f"eval_answers_{run_id}.csv"

    if not scores_path.exists():
        sys.exit(f"Scores file not found: {scores_path}")
    if not answers_path.exists():
        sys.exit(f"Answers file not found: {answers_path}")

    scores  = {r["question_id"]: r for r in load_csv(scores_path)}
    answers = {r["question_id"]: r for r in load_csv(answers_path)}

    sample = sorted(
        [qid for qid in scores if qid not in EXCLUDE_IDS],
        key=lambda q: (scores[q].get("category", ""), q),
    )

    failures   = sum(1 for q in sample if scores[q]["passed"].strip().lower() != "true")
    borderline = sum(1 for q in sample
                     if scores[q]["passed"].strip().lower() == "true"
                     and float(scores[q].get("judge_total", 0)) < BORDERLINE_THRESHOLD)
    clear      = len(sample) - failures - borderline

    print(f"Run       : {run_id}")
    print(f"Failures  : {failures}")
    print(f"Borderline: {borderline}")
    print(f"Clear     : {clear}")
    print(f"Total     : {len(sample)}")

    out_path = RESULTS_DIR / f"human_review_{run_id}.csv"
    fieldnames = [
        "question_id",
        "category",
        "judge_pass",       # original judge verdict
        "judge_total",
        "human_pass",       # ← fill in: TRUE / FALSE
        "human_notes",      # ← fill in: optional free text
        "question",
        "system_answer",
        "judge_reasoning",
    ]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for qid in sample:
            s = scores.get(qid, {})
            a = answers.get(qid, {})
            writer.writerow({
                "question_id":     qid,
                "category":        s.get("category", ""),
                "judge_pass":      s.get("passed", ""),
                "judge_total":     s.get("judge_total", ""),
                "human_pass":      "",
                "human_notes":     "",
                "question":        a.get("question", ""),
                "system_answer":   a.get("system_answer", ""),
                "judge_reasoning": s.get("judge_reasoning", ""),
            })

    print(f"\nReview sheet written to:\n  {out_path}")
    print("\nInstructions:")
    print("  1. Open human_review_{run_id}.csv in Excel or Numbers.")
    print("  2. For each row, read the question, system_answer, and judge_reasoning.")
    print("  3. Enter TRUE or FALSE in the human_pass column.")
    print("  4. Optionally add notes in human_notes (e.g. 'missing key fact X').")
    print("  5. Save as CSV and run annotate_analyze.py to compute agreement metrics.")


if __name__ == "__main__":
    main()
