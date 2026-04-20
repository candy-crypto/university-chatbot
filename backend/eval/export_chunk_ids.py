# export_chunk_ids.py
"""Query Weaviate for each ground truth question and export the top-5 chunk IDs.

Candy runs this script on her machine (where Weaviate is running) to produce
chunk_id_results.json.  Barbara then uses that file to fill in
expected_chunk_ids in ground_truth.yaml.

Usage (from the backend/ directory):
    python eval/export_chunk_ids.py
    python eval/export_chunk_ids.py --output eval/chunk_id_results.json

Output format:
    [
      {
        "question_id": "adv_001",
        "question": "What is CAASS?",
        "top5": [
          {
            "rank": 1,
            "chunk_id": "https://advising.nmsu.edu/#chunk-0",
            "heading": "...",
            "content_source": "web",
            "chunk_type": "advising",
            "hybrid_score": 0.843,
            "metadata_boost": 0.08,
            "final_score": 0.923,
            "text_preview": "first 120 chars of chunk text..."
          },
          ...
        ]
      },
      ...
    ]
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval import search_chunks

GROUND_TRUTH_PATH = Path(__file__).parent / "ground_truth.yaml"
DEFAULT_OUTPUT     = Path(__file__).parent / "chunk_id_results.json"
DEPARTMENT_ID      = "cs"


def main():
    parser = argparse.ArgumentParser(description="Export top-5 chunk IDs for all ground truth questions")
    parser.add_argument(
        "--output", "-o",
        default=str(DEFAULT_OUTPUT),
        help="Path to write the JSON output (default: eval/chunk_id_results.json)"
    )
    parser.add_argument(
        "--questions", nargs="+", metavar="ID",
        help="Only run these question IDs (default: all)"
    )
    args = parser.parse_args()

    with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
        records = yaml.safe_load(f)

    if args.questions:
        records = [r for r in records if r["question_id"] in args.questions]

    print(f"Running {len(records)} queries against Weaviate (department: {DEPARTMENT_ID})\n")

    results = []

    for i, record in enumerate(records, start=1):
        qid      = record["question_id"]
        question = record["question"]
        print(f"[{i}/{len(records)}] {qid}: {question[:70]}...")

        try:
            chunks = search_chunks(question, DEPARTMENT_ID)
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "question_id": qid,
                "question":    question,
                "error":       str(e),
                "top5":        [],
            })
            continue

        top5 = []
        for c in chunks:
            top5.append({
                "rank":           c.get("rank"),
                "chunk_id":       c.get("chunk_id"),
                "heading":        c.get("heading", ""),
                "content_source": c.get("content_source", ""),
                "chunk_type":     c.get("chunk_type", ""),
                "hybrid_score":   round(c.get("hybrid_score") or 0.0, 4),
                "metadata_boost": round(c.get("metadata_boost") or 0.0, 4),
                "final_score":    round(c.get("final_score") or 0.0, 4),
                "text_preview":   (c.get("text") or "")[:120],
            })
            print(f"  {c.get('rank')}. [{c.get('content_source')}] {c.get('chunk_id')}")

        results.append({
            "question_id": qid,
            "question":    question,
            "top5":        top5,
        })

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Results written to {output_path}")
    print("Share this file with Barbara to fill in expected_chunk_ids in ground_truth.yaml.")


if __name__ == "__main__":
    main()
