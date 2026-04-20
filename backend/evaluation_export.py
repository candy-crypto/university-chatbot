import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPORT_DIR = Path(__file__).resolve().parent / "evaluation"
EXPORT_PATH = EXPORT_DIR / "retrieval_eval.csv"


CSV_COLUMNS = [
    "timestamp_utc",
    "department_id",
    "question",
    "answer",
    "sources",
    "prompt_context",
    "chunk_count",
    "chunk_rank",
    "chunk_id",
    "chunk_type",
    "content_source",
    "heading",
    "source",
    "course_code",
    "referenced_courses",
    "catalog_year",
    "catalog_page",
    "catalog_page_end",
    "level",
    "degree_type",
    "degree_full_title",
    "concentration",
    "policy_topic",
    "lab",
    "score",
    "hybrid_score",
    "metadata_boost",
    "final_score",
    "text",
]


def _as_text_list(value: Any) -> str:
    if isinstance(value, list):
        return " | ".join(str(item) for item in value)
    if value is None:
        return ""
    return str(value)


def append_chat_evaluation_row(question: str, department_id: str, result: dict[str, Any]) -> str:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).isoformat()
    sources = _as_text_list(result.get("sources", []))
    prompt_context = result.get("prompt_context", "")
    answer = result.get("answer", "")
    chunks = result.get("chunks", []) or []
    chunk_count = len(chunks)

    rows: list[dict[str, Any]] = []

    for chunk in chunks:
        rows.append({
            "timestamp_utc": timestamp,
            "department_id": department_id,
            "question": question,
            "answer": answer,
            "sources": sources,
            "prompt_context": prompt_context,
            "chunk_count": chunk_count,
            "chunk_rank": chunk.get("rank", ""),
            "chunk_id": chunk.get("chunk_id", ""),
            "chunk_type": chunk.get("chunk_type", ""),
            "content_source": chunk.get("content_source", ""),
            "heading": chunk.get("heading", ""),
            "source": chunk.get("source", ""),
            "course_code": chunk.get("course_code", ""),
            "referenced_courses": _as_text_list(chunk.get("referenced_courses", [])),
            "catalog_year": chunk.get("catalog_year", ""),
            "catalog_page": chunk.get("catalog_page", ""),
            "catalog_page_end": chunk.get("catalog_page_end", ""),
            "level": chunk.get("level", ""),
            "degree_type": chunk.get("degree_type", ""),
            "degree_full_title": chunk.get("degree_full_title", ""),
            "concentration": chunk.get("concentration", ""),
            "policy_topic": chunk.get("policy_topic", ""),
            "lab": chunk.get("lab", ""),
            "score": chunk.get("score", ""),
            "hybrid_score": chunk.get("hybrid_score", ""),
            "metadata_boost": chunk.get("metadata_boost", ""),
            "final_score": chunk.get("final_score", ""),
            "text": chunk.get("text", ""),
        })

    if not rows:
        rows.append({
            "timestamp_utc": timestamp,
            "department_id": department_id,
            "question": question,
            "answer": answer,
            "sources": sources,
            "prompt_context": prompt_context,
            "chunk_count": 0,
            "chunk_rank": "",
            "chunk_id": "",
            "chunk_type": "",
            "content_source": "",
            "heading": "",
            "source": "",
            "course_code": "",
            "referenced_courses": "",
            "catalog_year": "",
            "catalog_page": "",
            "catalog_page_end": "",
            "level": "",
            "degree_type": "",
            "degree_full_title": "",
            "concentration": "",
            "policy_topic": "",
            "lab": "",
            "score": "",
            "hybrid_score": "",
            "metadata_boost": "",
            "final_score": "",
            "text": "",
        })

    write_header = not EXPORT_PATH.exists()
    with EXPORT_PATH.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

    return str(EXPORT_PATH)
