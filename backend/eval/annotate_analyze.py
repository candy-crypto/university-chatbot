"""annotate_analyze.py
=====================
Compute inter-rater agreement and error analysis from a completed human
review CSV produced by annotate_prepare.py.

Metrics computed:
  - Cohen's Kappa  (judge vs human, overall)
  - TPR / TNR      (true positive / true negative rates, treating human as gold)
  - Category breakdown: judge_pass, human_pass, agreement counts
  - Disagreement table: rows where judge and human differ

Usage:
    cd backend
    python eval/annotate_analyze.py                              # latest review file
    python eval/annotate_analyze.py --file results/human_review_20260426_034716_091def.csv
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"


def latest_review_file() -> Path:
    files = sorted(RESULTS_DIR.glob("human_review_*.csv"), reverse=True)
    if not files:
        sys.exit("No human_review_*.csv files found in results/")
    return files[0]


def load_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_bool(value: str) -> bool | None:
    v = value.strip().upper()
    if v in {"TRUE", "YES", "1", "T", "Y"}:
        return True
    if v in {"FALSE", "NO", "0", "F", "N"}:
        return False
    return None


def cohen_kappa(rows: list[dict]) -> float:
    """Compute Cohen's Kappa between judge_pass and human_pass."""
    tp = tn = fp = fn = 0
    for r in rows:
        j = parse_bool(r["judge_pass"])
        h = parse_bool(r["human_pass"])
        if j is None or h is None:
            continue
        if j and h:
            tp += 1
        elif not j and not h:
            tn += 1
        elif j and not h:
            fp += 1   # judge said pass, human said fail
        else:
            fn += 1   # judge said fail, human said pass

    n = tp + tn + fp + fn
    if n == 0:
        return float("nan")

    p_o = (tp + tn) / n           # observed agreement

    p_judge_pos  = (tp + fp) / n
    p_judge_neg  = (tn + fn) / n
    p_human_pos  = (tp + fn) / n
    p_human_neg  = (tn + fp) / n
    p_e = p_judge_pos * p_human_pos + p_judge_neg * p_human_neg  # chance agreement

    if p_e == 1.0:
        return float("nan")
    return (p_o - p_e) / (1 - p_e)


def main():
    parser = argparse.ArgumentParser(description="Analyze human annotation results.")
    parser.add_argument("--file", metavar="PATH", help="Path to human_review CSV.")
    args = parser.parse_args()

    path = Path(args.file) if args.file else latest_review_file()
    if not path.exists():
        sys.exit(f"File not found: {path}")

    rows = load_csv(path)

    # Filter to rows with human_pass filled in
    annotated = [r for r in rows if parse_bool(r.get("human_pass", "")) is not None]
    skipped   = len(rows) - len(annotated)

    if not annotated:
        print("No rows with human_pass filled in yet. Fill in the CSV and re-run.")
        return

    print(f"{'='*60}")
    print(f"Human annotation analysis")
    print(f"File    : {path.name}")
    print(f"Rows    : {len(annotated)} annotated  ({skipped} skipped / blank)")
    print(f"{'='*60}\n")

    # ── Overall confusion matrix ───────────────────────────────────���──────────
    tp = tn = fp = fn = 0
    for r in annotated:
        j = parse_bool(r["judge_pass"])
        h = parse_bool(r["human_pass"])
        if j and h:     tp += 1
        elif not j and not h: tn += 1
        elif j and not h:     fp += 1
        else:                 fn += 1

    n = tp + tn + fp + fn
    tpr = tp / (tp + fn) if (tp + fn) else float("nan")   # recall / sensitivity
    tnr = tn / (tn + fp) if (tn + fp) else float("nan")   # specificity
    kappa = cohen_kappa(annotated)

    print("Overall agreement (judge vs human)")
    print(f"  Cohen's Kappa : {kappa:.3f}  ", end="")
    if   kappa >= 0.8:  print("(almost perfect)")
    elif kappa >= 0.6:  print("(substantial)")
    elif kappa >= 0.4:  print("(moderate)")
    elif kappa >= 0.2:  print("(fair)")
    else:               print("(slight or poor)")

    print(f"\n  Confusion matrix (judge rows × human cols):")
    print(f"                 Human PASS   Human FAIL")
    print(f"  Judge PASS     {tp:5d}        {fp:5d}")
    print(f"  Judge FAIL     {fn:5d}        {tn:5d}")
    print(f"\n  TPR (judge sensitivity) : {tpr:.3f}  — of answers human marked PASS, judge caught {tpr:.0%}")
    print(f"  TNR (judge specificity) : {tnr:.3f}  — of answers human marked FAIL, judge caught {tnr:.0%}")

    # ── Category breakdown ────────────────────────────────────────────────────
    by_cat = defaultdict(lambda: {"tp": 0, "tn": 0, "fp": 0, "fn": 0})
    for r in annotated:
        cat = r.get("category", "unknown")
        j = parse_bool(r["judge_pass"])
        h = parse_bool(r["human_pass"])
        if j and h:         by_cat[cat]["tp"] += 1
        elif not j and not h: by_cat[cat]["tn"] += 1
        elif j and not h:   by_cat[cat]["fp"] += 1
        else:               by_cat[cat]["fn"] += 1

    print(f"\n{'─'*60}")
    print("Category breakdown")
    print(f"  {'Category':<28} {'n':>3}  {'Judge%':>7}  {'Human%':>7}  {'Agree':>6}")
    print(f"  {'─'*28}  {'─'*3}  {'─'*7}  {'─'*7}  {'─'*6}")
    for cat, c in sorted(by_cat.items()):
        n_cat      = c["tp"] + c["tn"] + c["fp"] + c["fn"]
        judge_pass = (c["tp"] + c["fp"]) / n_cat
        human_pass = (c["tp"] + c["fn"]) / n_cat
        agree      = (c["tp"] + c["tn"]) / n_cat
        print(f"  {cat:<28} {n_cat:>3}  {judge_pass:>6.0%}   {human_pass:>6.0%}   {agree:>5.0%}")

    # ── Disagreements ─────────────────────────────────────────────────────────
    disagreements = [
        r for r in annotated
        if parse_bool(r["judge_pass"]) != parse_bool(r["human_pass"])
    ]

    if disagreements:
        print(f"\n{'─'*60}")
        print(f"Disagreements ({len(disagreements)} rows)")
        for r in disagreements:
            j = "PASS" if parse_bool(r["judge_pass"]) else "FAIL"
            h = "PASS" if parse_bool(r["human_pass"]) else "FAIL"
            print(f"\n  {r['question_id']:12s}  judge={j}  human={h}  [{r['category']}]")
            print(f"  Q: {r['question'][:80]}")
            if r.get("human_notes", "").strip():
                print(f"  Note: {r['human_notes'].strip()}")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
