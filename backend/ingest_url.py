# ingest_url.py
"""Ingest a single URL into Weaviate without re-running the full crawl.

Crawls the given URL, chunks it, embeds it with OpenAI, and upserts the
chunks to Weaviate.  Existing chunks for that URL are replaced; everything
else in the collection is untouched.

Usage (from the backend/ directory):
    python ingest_url.py https://apply.nmsu.edu/apply/
    python ingest_url.py https://apply.nmsu.edu/apply/ --department cs
    python ingest_url.py https://apply.nmsu.edu/apply/ --dry-run
"""

import argparse
import sys
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
from weaviate.classes.data import DataObject
from weaviate.classes.query import Filter

from ingest import (
    load_department_config,
    extract_page_data,
    chunk_page,
    embed_batch,
    extract_course_code,
    extract_referenced_courses,
)
from weaviate_client import get_weaviate_client, ensure_collection, get_collection

load_dotenv()

CONFIG_PATH   = "configs/departments/cs.yaml"
DEPARTMENT_ID = "cs"


def ingest_single_url(url: str, department_id: str, dry_run: bool = False) -> int:
    """Crawl one URL, chunk, embed, and upsert to Weaviate.

    Returns the number of chunks inserted.
    """
    config = load_department_config(CONFIG_PATH)

    page_types        = config.get("page_types", {})
    page_levels       = config.get("page_levels", {})
    page_degree_types = config.get("page_degree_types", {})
    campus            = config.get("campus", "")

    print(f"Crawling: {url}")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(user_agent="UniversityChatbotBot/1.0")
        page    = context.new_page()
        try:
            data = extract_page_data(page, url, config)
        finally:
            context.close()
            browser.close()

    if len(data.get("text", "")) < 100:
        print("ERROR: Page returned less than 100 characters of text. Aborting.")
        sys.exit(1)

    chunk_type  = page_types.get(url, "general")
    level       = page_levels.get(url, "")
    degree_type = page_degree_types.get(url, "")

    print(f"  chunk_type={chunk_type}  level={level or '(both)'}  degree_type={degree_type or '(all)'}")

    chunks = chunk_page(data["sections"], chunk_type)
    if not chunks:
        text = data.get("text", "").strip()
        if text:
            chunks = [{"heading": data.get("h1", ""), "text": text[:2000]}]

    if not chunks:
        print("ERROR: No chunks produced. Aborting.")
        sys.exit(1)

    print(f"  Produced {len(chunks)} chunk(s)")

    if dry_run:
        for i, c in enumerate(chunks):
            print(f"\n── Chunk {i} (heading: {c['heading']!r}) ({'%d chars' % len(c['text'])}) ──")
            print(c["text"][:300] + ("..." if len(c["text"]) > 300 else ""))
        print("\n[dry-run] No changes written to Weaviate.")
        return 0

    print(f"  Embedding {len(chunks)} chunk(s) with OpenAI...")
    embeddings = embed_batch([c["text"] for c in chunks])

    crawl_version = time.strftime("%Y%m%d_%H%M%S")

    objects = []
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        chunk_text = chunk["text"]
        chunk_heading = chunk.get("heading") or data.get("h1", "")
        objects.append(DataObject(
            uuid=str(uuid.uuid4()),
            properties={
                "chunk_id":            f"{url}#chunk-{i}",
                "chunk_type":          chunk_type,
                "department_id":       department_id.strip().lower(),
                "campus":              campus,
                "text":                chunk_text,
                "heading":             chunk_heading,
                "source":              url,
                "level":               level,
                "degree_type":         degree_type,
                "course_code":         extract_course_code(data.get("h1", ""), data.get("text", "")),
                "referenced_courses":  extract_referenced_courses(data.get("text", "")),
                "content_source":      "web",
                "catalog_year":        "",
                "catalog_page":        0,
                "catalog_page_end":    0,
                "degree_full_title":   "",
                "concentration":       "",
                "credits":             "",
                "has_prerequisites":   False,
                "policy_topic":        "",
                "lab":                 "",
                "research":            False,
                "crawl_version":       crawl_version,
            },
            vector=emb,
        ))

    client = get_weaviate_client()
    try:
        ensure_collection(client)
        collection = get_collection(client)

        # Delete any existing chunks for this URL before inserting new ones.
        deleted = collection.data.delete_many(
            where=(
                Filter.by_property("source").equal(url)
                & Filter.by_property("content_source").equal("web")
            )
        )
        if hasattr(deleted, "successful") and deleted.successful:
            print(f"  Removed {deleted.successful} existing chunk(s) for this URL")

        collection.data.insert_many(objects)
        print(f"  Inserted {len(objects)} chunk(s) into Weaviate")

    finally:
        client.close()

    return len(objects)


def main():
    parser = argparse.ArgumentParser(description="Ingest a single URL into Weaviate")
    parser.add_argument("url", help="The URL to crawl and ingest")
    parser.add_argument(
        "--department", "-d", default=DEPARTMENT_ID,
        help=f"Department ID (default: {DEPARTMENT_ID})"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Crawl and chunk but do not write to Weaviate"
    )
    args = parser.parse_args()

    n = ingest_single_url(args.url, args.department, dry_run=args.dry_run)
    if not args.dry_run:
        print(f"\nDone. {n} chunk(s) added to Weaviate.")


if __name__ == "__main__":
    main()
