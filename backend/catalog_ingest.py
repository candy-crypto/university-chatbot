# This file imports external chunkers, maps their chunks to your current stack,
# embeds with OpenAI and write to DepartmentChunk.
# Uses the OpenAI + Weaviate stack
# Catalog flow isolated from the web cralwer

import os
import re
import sys
import uuid
import time
from datetime import datetime, timezone

from dotenv import load_dotenv
from openai import OpenAI
from weaviate.classes.query import Filter
from weaviate.classes.data import DataObject

from weaviate_client import get_weaviate_client, ensure_collection, get_collection
from db import init_db, log_crawl_run

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
CATALOG_PDF_PATH = os.getenv("CATALOG_PDF_PATH", "25-26_New_Mexico_State_University_-_Las_Cruces.pdf")
CATALOG_YEAR = os.getenv("CATALOG_YEAR", "2025-2026")
CATALOG_DEPARTMENT_ID = os.getenv("CATALOG_DEPARTMENT_ID", "cs")
EXTERNAL_SCRIPTS_DIR = os.getenv("CATALOG_SCRIPTS_DIR", ".")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")

openai_client = OpenAI(api_key=OPENAI_API_KEY)

sys.path.insert(0, EXTERNAL_SCRIPTS_DIR)

from nmsu_catalog_chunker import run_pipeline as run_catalog_pipeline  # noqa: E402


EMBED_BATCH_MAX_INPUTS = int(os.getenv("EMBED_BATCH_MAX_INPUTS", "64"))
EMBED_BATCH_MAX_EST_TOKENS = int(os.getenv("EMBED_BATCH_MAX_EST_TOKENS", "240000"))


def estimate_tokens(text: str) -> int:
    # Conservative approximation for batching embeddings requests.
    return max(1, (len(text or "") + 3) // 4)


def embed_batch(texts):
    batches = []
    current_batch = []
    current_tokens = 0

    for text in texts:
        estimated_tokens = estimate_tokens(text)

        if current_batch and (
            len(current_batch) >= EMBED_BATCH_MAX_INPUTS
            or current_tokens + estimated_tokens > EMBED_BATCH_MAX_EST_TOKENS
        ):
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0

        current_batch.append(text)
        current_tokens += estimated_tokens

    if current_batch:
        batches.append(current_batch)

    embeddings = []
    for batch in batches:
        response = openai_client.embeddings.create(
            model=OPENAI_EMBED_MODEL,
            input=batch,
        )
        embeddings.extend(item.embedding for item in response.data)

    return embeddings


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_document_id(chunk) -> str:
    if getattr(chunk, "course_code", ""):
        code = re.sub(r"[^A-Za-z0-9]+", "-", chunk.course_code.strip()).strip("-").lower()
        return f"catalog::{CATALOG_YEAR}::course::{code}"

    if getattr(chunk, "degree_full_title", ""):
        title = re.sub(r"[^A-Za-z0-9]+", "-", chunk.degree_full_title.strip()).strip("-").lower()
        return f"catalog::{CATALOG_YEAR}::degree::{title}"

    return f"catalog::{CATALOG_YEAR}::{chunk.chunk_type}::p{chunk.catalog_page}"


def make_heading(chunk) -> str:
    """Derive a human-readable heading based on chunk type."""
    if getattr(chunk, "course_code", "") and getattr(chunk, "course_title", ""):
        return f"{chunk.course_code} — {chunk.course_title}"
    if getattr(chunk, "degree_full_title", ""):
        return chunk.degree_full_title
    if getattr(chunk, "policy_topic", ""):
        return chunk.policy_topic
    if getattr(chunk, "lab_name", ""):
        return chunk.lab_name
    return getattr(chunk, "chunk_type", "catalog")


def delete_catalog_chunks(collection, department_id: str, catalog_year: str):
    collection.data.delete_many(
        where=(
            Filter.by_property("department_id").equal(department_id)
            & Filter.by_property("content_source").equal("catalog")
            & Filter.by_property("catalog_year").equal(catalog_year)
        )
    )


def make_level(chunk) -> str:
    # Consolidate degree_level and course_number_level into a single level field.
    # course_number_level takes priority for course_description chunks.
    course_level = getattr(chunk, "course_number_level", "")
    if course_level:
        return "graduate" if course_level == "graduate" else "undergraduate"
    degree_level = getattr(chunk, "degree_level", "")
    if degree_level in ("ms", "phd"):
        return "graduate"
    return degree_level or ""


def make_source(chunk) -> str:
    page = getattr(chunk, "catalog_page", 0)
    year = getattr(chunk, "catalog_year", CATALOG_YEAR)
    return f"NMSU Academic Catalog {year}, p.{page}"


def map_catalog_chunk(chunk, crawl_version: str):
    text = normalize_text(chunk.text)

    return {
        # Core fields
        "chunk_id":            getattr(chunk, "chunk_id", str(uuid.uuid4())),
        "chunk_type":          getattr(chunk, "chunk_type", ""),
        "department_id":       CATALOG_DEPARTMENT_ID,
        "campus":              "las_cruces",
        "text":                text,
        "heading":             make_heading(chunk),
        "source":              make_source(chunk),
        "level":               make_level(chunk),
        "degree_type":         getattr(chunk, "degree_type", ""),
        "course_code":         getattr(chunk, "course_code", ""),
        "referenced_courses":  getattr(chunk, "referenced_courses", []),
        "content_source":      "catalog",

        # Catalog-specific fields
        "catalog_year":        getattr(chunk, "catalog_year", CATALOG_YEAR),
        "catalog_page":        getattr(chunk, "catalog_page", 0),
        "catalog_page_end":    getattr(chunk, "catalog_page_end", 0),
        "degree_full_title":   getattr(chunk, "degree_full_title", ""),
        "concentration":       getattr(chunk, "concentration", ""),
        "credits":             getattr(chunk, "credits", ""),
        "has_prerequisites":   getattr(chunk, "has_prerequisites", False),
        "policy_topic":        getattr(chunk, "policy_topic", ""),
        "lab":                 getattr(chunk, "lab_name", ""),
        "research":            getattr(chunk, "is_research_related", False),

        # Web-specific fields — empty for catalog
        "crawl_version":       "",
    }


def upsert_catalog_chunks(catalog_chunks, crawl_version: str):
    client = get_weaviate_client()
    try:
        ensure_collection(client)
        collection = get_collection(client)

        delete_catalog_chunks(collection, CATALOG_DEPARTMENT_ID, CATALOG_YEAR)

        mapped = [map_catalog_chunk(chunk, crawl_version) for chunk in catalog_chunks if normalize_text(chunk.text)]
        embeddings = embed_batch([item["text"] for item in mapped])

        objects = []
        for item, emb in zip(mapped, embeddings):
            objects.append(
                DataObject(
                    uuid=str(uuid.uuid5(uuid.NAMESPACE_URL, item["chunk_id"])),
                    properties=item,
                    vector=emb,
                )
            )

        if objects:
            collection.data.insert_many(objects)

        return len(objects)

    finally:
        client.close()


def main():
    init_db()

    crawl_version = time.strftime("%Y%m%d_%H%M%S")

    chunks = run_catalog_pipeline(
        pdf_path=CATALOG_PDF_PATH,
        dry_run=False,
        include_courses=True,
    )

    indexed = upsert_catalog_chunks(chunks, crawl_version)

    log_crawl_run(
        department_id=CATALOG_DEPARTMENT_ID,
        crawl_version=crawl_version,
        pages_scraped=0,
        chunks_indexed=indexed,
    )

    print(f"Parsed {len(chunks)} catalog chunks")
    print(f"Indexed {indexed} catalog chunks")
    print(f"Crawl version: {crawl_version}")


if __name__ == "__main__":
    main()
