# compare_retrievals.py
"""Compare two chunk_id_results.json files to measure retrieval improvement.

Usage (from the backend/ directory):
    python eval/compare_retrievals.py
    python eval/compare_retrievals.py --baseline eval/chunk_id_results_baseline.json
                                      --new      eval/chunk_id_results.json

Output: per-question diff showing rank changes, new arrivals, and dropouts,
        plus a summary table of improvement / regression / unchanged counts.
"""

import argparse
import json
from pathlib import Path


def load(path: str) -> dict:
    """Load a chunk_id_results.json and index by question_id."""
    with open(path) as f:
        data = json.load(f)
    return {q["question_id"]: q for q in data}


def top_ids(question: dict, n: int = 5) -> list[str]:
    return [entry["chunk_id"] for entry in question.get("top5", [])[:n]]


def rank_of(chunk_id: str, question: dict) -> int | None:
    for entry in question.get("top5", []):
        if entry["chunk_id"] == chunk_id:
            return entry["rank"]
    return None


def compare(baseline: dict, new: dict) -> None:
    improved = []
    regressed = []
    unchanged = []
    only_baseline = []
    only_new = []

    all_ids = sorted(set(baseline) | set(new))

    for qid in all_ids:
        if qid not in baseline:
            only_new.append(qid)
            continue
        if qid not in new:
            only_baseline.append(qid)
            continue

        b_ids = set(top_ids(baseline[qid]))
        n_ids = set(top_ids(new[qid]))
        b_question = baseline[qid]
        n_question = new[qid]

        arrived   = n_ids - b_ids   # in new top5 but not baseline
        dropped   = b_ids - n_ids   # in baseline top5 but not new
        stayed    = b_ids & n_ids   # in both

        # Check for rank improvements among chunks that stayed
        rank_changes = []
        for cid in stayed:
            b_rank = rank_of(cid, b_question)
            n_rank = rank_of(cid, n_question)
            if b_rank is not None and n_rank is not None and n_rank != b_rank:
                rank_changes.append((cid, b_rank, n_rank))

        has_change = bool(arrived or dropped or rank_changes)

        # Determine if this is net improvement or regression
        # Simple heuristic: more chunks in new top5 that weren't in baseline = improvement
        # Fewer = regression; equal = rank-only change
        if not has_change:
            unchanged.append(qid)
            continue

        net = len(arrived) - len(dropped)
        if net > 0:
            improved.append(qid)
        elif net < 0:
            regressed.append(qid)
        else:
            # Same count in/out — check if rank 1 improved
            b1 = top_ids(b_question, 1)
            n1 = top_ids(n_question, 1)
            if n1 and b1 and n1[0] != b1[0]:
                # Rank 1 changed — call it a change, neutral
                unchanged.append(qid)
            else:
                unchanged.append(qid)

        # Print detail for changed questions
        question_text = b_question.get("question", qid)
        print(f"\n{'='*70}")
        print(f"{qid}: {question_text}")
        print(f"{'='*70}")

        # Rank 1 comparison
        b_top = b_question["top5"][0] if b_question.get("top5") else None
        n_top = n_question["top5"][0] if n_question.get("top5") else None
        if b_top and n_top:
            b1_head = b_top.get("heading", "")[:50]
            n1_head = n_top.get("heading", "")[:50]
            marker = "  " if b_top["chunk_id"] == n_top["chunk_id"] else "* "
            print(f"  Rank 1  before: [{b_top['final_score']:.3f}] {b1_head}")
            print(f"  Rank 1  after:  [{n_top['final_score']:.3f}] {n1_head}  {marker if marker.strip() else ''}")

        if arrived:
            print(f"  NEW in top5 ({len(arrived)}):")
            for cid in arrived:
                entry = next((e for e in n_question["top5"] if e["chunk_id"] == cid), {})
                print(f"    rank {entry.get('rank','?')} [{entry.get('final_score',0):.3f}]"
                      f" {entry.get('heading','')[:55]}")
        if dropped:
            print(f"  DROPPED from top5 ({len(dropped)}):")
            for cid in dropped:
                entry = next((e for e in b_question["top5"] if e["chunk_id"] == cid), {})
                print(f"    was rank {entry.get('rank','?')} [{entry.get('final_score',0):.3f}]"
                      f" {entry.get('heading','')[:55]}")
        if rank_changes:
            print(f"  RANK CHANGES:")
            for cid, br, nr in sorted(rank_changes, key=lambda x: x[2]):
                entry = next((e for e in n_question["top5"] if e["chunk_id"] == cid), {})
                arrow = "↑" if nr < br else "↓"
                print(f"    {arrow} rank {br}→{nr} [{entry.get('final_score',0):.3f}]"
                      f" {entry.get('heading','')[:50]}")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Improved  (more relevant chunks in top5): {len(improved):3d}")
    print(f"  Regressed (fewer relevant chunks in top5): {len(regressed):3d}")
    print(f"  Unchanged:                                 {len(unchanged):3d}")
    if only_new:
        print(f"  Questions only in new file:              {len(only_new):3d}  {only_new}")
    if only_baseline:
        print(f"  Questions only in baseline:              {len(only_baseline):3d}  {only_baseline}")
    print(f"  Total questions compared: {len(all_ids) - len(only_new) - len(only_baseline)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two chunk_id_results files.")
    parser.add_argument(
        "--baseline",
        default="eval/chunk_id_results_baseline.json",
        help="Path to baseline results (default: eval/chunk_id_results_baseline.json)",
    )
    parser.add_argument(
        "--new",
        default="eval/chunk_id_results.json",
        help="Path to new results (default: eval/chunk_id_results.json)",
    )
    args = parser.parse_args()

    baseline = load(args.baseline)
    new = load(args.new)
    compare(baseline, new)


if __name__ == "__main__":
    main()
