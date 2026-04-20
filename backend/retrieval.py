# retrieval.py

"""Hybrid retrieval utilities for the university chatbot backend.

This module handles query embedding, Weaviate native hybrid search
(BM25 + vector via RRF), optional metadata boosting, and prompt
construction for grounded answer generation.
"""

import os
import re
from typing import List, Dict, Any

from dotenv import load_dotenv
from openai import OpenAI
from weaviate.classes.query import MetadataQuery, Filter

from weaviate_client import get_weaviate_client, ensure_collection, get_collection

load_dotenv()

# Function to turn off/on the OpenAI calls
def env_flag(name: str, default: str = "false") -> bool:
    """Parse common truthy environment variable values."""
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-5-mini")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
MOCK_OPENAI = env_flag("MOCK_OPENAI") 

# Top-K results returned to the caller.
TOP_K = int(os.getenv("TOP_K", "5"))

# Hybrid alpha: 0.0 = pure BM25, 1.0 = pure vector, 0.75 = mostly semantic.
HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", "0.75"))

# Code to turn on/off OpenAI calls
if not MOCK_OPENAI and not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")

openai_client = None if MOCK_OPENAI else OpenAI(api_key=OPENAI_API_KEY)

# Properties fetched from Weaviate for every retrieved chunk.
RETURN_PROPERTIES = [
    # ── Core fields ───────────────────────────────────────────────────────────
    "chunk_id",
    "chunk_type",
    "department_id",
    "campus",
    "text",
    "heading",
    "source",
    "level",
    "degree_type",
    "course_code",
    "referenced_courses",
    "content_source",

    # ── Catalog-specific fields ───────────────────────────────────────────────
    "catalog_year",
    "catalog_page",
    "catalog_page_end",
    "degree_full_title",
    "concentration",
    "credits",
    "has_prerequisites",
    "policy_topic",
    "lab",
    "research",

    # ── Web-specific fields ───────────────────────────────────────────────────
    "crawl_version",
]


def embed_text(text: str) -> List[float]:
    """Embed a query string using the OpenAI embedding model."""
    # Added code to turn on/off OpenAI calls
    if MOCK_OPENAI:
        raise RuntimeError("embed_text() should not be called when MOCK_OPENAI is enabled")

    response = openai_client.embeddings.create(
        model=OPENAI_EMBED_MODEL,
        input=text
    )
    return response.data[0].embedding


def tokenize(text: str) -> List[str]:
    """Normalize text and extract lowercase term tokens."""
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def metadata_boost(query: str, chunk: Dict[str, Any]) -> float:
    """Add a small score boost when query terms appear in chunk metadata fields."""
    query_tokens = set(tokenize(query))
    boost = 0.0

    if query_tokens.intersection(tokenize(chunk.get("heading", ""))):
        boost += 0.08

    # Course code and referenced courses — important for course-specific queries.
    course_tokens = tokenize(
        f"{chunk.get('course_code', '')} "
        f"{' '.join(chunk.get('referenced_courses') or [])}"
    )
    if query_tokens.intersection(course_tokens):
        boost += 0.10

    # Degree metadata — important for program and degree queries.
    degree_tokens = tokenize(
        f"{chunk.get('degree_full_title', '')} "
        f"{chunk.get('level', '')} "
        f"{chunk.get('degree_type', '')} "
        f"{chunk.get('concentration', '')}"
    )
    if query_tokens.intersection(degree_tokens):
        boost += 0.08

    if query_tokens.intersection(tokenize(chunk.get("policy_topic", ""))):
        boost += 0.06

    if query_tokens.intersection(tokenize(chunk.get("lab", ""))):
        boost += 0.05

    # Slight preference for catalog content, which tends to be more authoritative.
    if chunk.get("content_source") == "catalog":
        boost += 0.02

    return boost


def parse_weaviate_objects(objects: Any, score_attr: str | None = None) -> List[Dict[str, Any]]:
    """Convert Weaviate object results into plain dictionaries."""
    results = []
    for rank, obj in enumerate(objects, start=1):
        props = obj.properties or {}
        result = {key: props.get(key) for key in RETURN_PROPERTIES}
        result["referenced_courses"] = result.get("referenced_courses") or []
        result["rank"] = rank
        if obj.metadata is not None and score_attr:
            result[score_attr] = getattr(obj.metadata, score_attr, None)
        else:
            result[score_attr] = None if score_attr else None
        results.append(result)
    return results


def search_chunks(query: str, department_id: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    """
    Hybrid retrieval using Weaviate's native hybrid search.

    Weaviate fuses BM25 keyword scores and vector similarity scores using
    Reciprocal Rank Fusion (RRF) internally, then returns results ranked by
    the combined score. A metadata boost is applied as a final re-ranking step.
    """
    query_vector = embed_text(query)

    client = get_weaviate_client()
    try:
        ensure_collection(client)
        collection = get_collection(client)

        response = collection.query.hybrid(
            query=query,
            vector=query_vector,
            filters=Filter.by_property("department_id").equal(department_id),
            limit=top_k,
            alpha=HYBRID_ALPHA,
            return_metadata=MetadataQuery(score=True),
            return_properties=RETURN_PROPERTIES,
        )

        results = parse_weaviate_objects(response.objects, "score")

        # Re-rank results by adding a metadata boost to the hybrid score.
        for item in results:
            boost = metadata_boost(query, item)
            item["metadata_boost"] = boost
            item["hybrid_score"] = item.get("score") or 0.0
            item["final_score"] = item["hybrid_score"] + boost

        results.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)
        return results

    finally:
        client.close()


def build_context(chunks: List[Dict[str, Any]]) -> str:
    """Format retrieved chunks into a grounded context string for the prompt."""
    parts = []

    for i, chunk in enumerate(chunks, start=1):
        if chunk.get("content_source") == "catalog":
            parts.append(
                f"[Source {i}]\n"
                f"Content Source: catalog\n"
                f"Chunk Type: {chunk.get('chunk_type', '')}\n"
                f"Heading: {chunk.get('heading', '')}\n"
                f"Degree: {chunk.get('degree_full_title', '')}\n"
                f"Course: {chunk.get('course_code', '')}\n"
                f"Catalog Year: {chunk.get('catalog_year', '')}\n"
                f"Catalog Pages: {chunk.get('catalog_page', '')}-{chunk.get('catalog_page_end', '')}\n"
                f"Referenced Courses: {', '.join(chunk.get('referenced_courses') or [])}\n"
                f"Text: {chunk.get('text', '')}\n"
            )
        else:
            parts.append(
                f"[Source {i}]\n"
                f"Content Source: web\n"
                f"Chunk Type: {chunk.get('chunk_type', '')}\n"
                f"Heading: {chunk.get('heading', '')}\n"
                f"Source: {chunk.get('source', '')}\n"
                f"Course: {chunk.get('course_code', '')}\n"
                f"Text: {chunk.get('text', '')}\n"
            )

    return "\n---\n".join(parts)


def generate_grounded_answer(question: str, department_id: str) -> Dict[str, Any]:
    """
    RAG flow:
      1. Embed question with OpenAI.
      2. Retrieve top-K chunks from Weaviate via hybrid search.
      3. Ask OpenAI to answer using only the retrieved context.
    """
    normalized_department_id = (department_id or "").strip().lower()

    # Added code to use mock code instead of OpenAI calls
    if MOCK_OPENAI:
        mock_source = "mock://openai-disabled"
        mock_chunk = {
            "chunk_id": "mock-chunk-1",
            "chunk_type": "mock",
            "department_id": normalized_department_id,
            "campus": "",
            "text": (
                "Mock mode is enabled. No OpenAI embedding or response request was sent. "
                "This response is intended for local UI and API testing only."
            ),
            "heading": "Mock Mode Response",
            "source": mock_source,
            "level": "",
            "degree_type": "",
            "course_code": "",
            "referenced_courses": [],
            "content_source": "mock",
            "catalog_year": "",
            "catalog_page": "",
            "catalog_page_end": "",
            "degree_full_title": "",
            "concentration": "",
            "credits": "",
            "has_prerequisites": False,
            "policy_topic": "",
            "lab": "",
            "research": "",
            "crawl_version": "",
            "rank": 1,
            "score": None,
            "metadata_boost": 0.0,
            "hybrid_score": 0.0,
            "final_score": 0.0,
        }
        return {
            "answer": (
                f"Mock mode is enabled for department '{normalized_department_id or 'unknown'}'. "
                f"No OpenAI tokens were used. Test question received: {question}"
            ),
            "sources": [mock_source],
            "chunks": [mock_chunk],
            "prompt_context": build_context([mock_chunk]),
        }

    chunks = search_chunks(question, normalized_department_id)
    context = build_context(chunks)

    system_prompt = """
You are a helpful university department assistant.

Answer the user using ONLY the provided context.
The context may come from the department website and/or the academic catalog.
If the answer is not in the context, say:
"I could not find that information in the provided department sources."
Do not make up facts.
Be concise and factual.
At the end, list the most relevant source URLs and/or catalog page citations if available.
"""

    user_prompt = f"""
Department: {department_id}

Question:
{question}

Retrieved context:
{context}
"""

    response = openai_client.responses.create(
        model=OPENAI_CHAT_MODEL,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    answer = response.output_text

    # Build a deduplicated list of source citations for the response.
    sources = []
    for chunk in chunks:
        if chunk.get("content_source") == "catalog":
            citation = (
                f"NMSU Academic Catalog {chunk.get('catalog_year', '')}, "
                f"pp. {chunk.get('catalog_page', '')}-{chunk.get('catalog_page_end', '')}"
            )
            if citation not in sources:
                sources.append(citation)
        else:
            src = chunk.get("source")
            if src and src not in sources:
                sources.append(src)

    return {
        "answer": answer,
        "sources": sources,
        "chunks": chunks,
        "prompt_context": context,
    }
