# retrieval.py

"""Hybrid retrieval utilities for the university chatbot backend.

This module handles query embedding, semantic search, keyword ranking,
score fusion, and prompt construction for grounded answer generation.
"""

import os
import math
import re
from collections import Counter
from typing import List, Dict, Any

from dotenv import load_dotenv
from openai import OpenAI
from weaviate.classes.query import MetadataQuery, Filter

from weaviate_client import get_weaviate_client, ensure_collection, get_collection

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-5-mini")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

# Default retrieval limits and scoring weights.
TOP_K = int(os.getenv("TOP_K", "5"))
SEMANTIC_TOP_K = int(os.getenv("SEMANTIC_TOP_K", "20"))
KEYWORD_TOP_K = int(os.getenv("KEYWORD_TOP_K", "20"))
KEYWORD_CANDIDATE_LIMIT = int(os.getenv("KEYWORD_CANDIDATE_LIMIT", "200"))
RRF_K = int(os.getenv("RRF_K", "60"))
WEIGHT_BM25 = float(os.getenv("WEIGHT_BM25", "0.35"))
WEIGHT_SEMANTIC = float(os.getenv("WEIGHT_SEMANTIC", "0.45"))
WEIGHT_RRF = float(os.getenv("WEIGHT_RRF", "0.20"))

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")

openai_client = OpenAI(api_key=OPENAI_API_KEY)

RETURN_PROPERTIES = [
    "department_id",
    "document_id",
    "url",
    "source",
    "title",
    "section",
    "timestamp",
    "tags",
    "course_number",
    "course_title",
    "text",
    "chunk_id",
    "crawl_version",
]


def embed_text(text: str) -> List[float]:
    """Embed a query or document string using the OpenAI embedding model."""
    response = openai_client.embeddings.create(
        model=OPENAI_EMBED_MODEL,
        input=text
    )
    return response.data[0].embedding


def tokenize(text: str) -> List[str]:
    """Normalize text and extract lowercase term tokens for keyword matching."""
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def lexical_document(chunk: Dict[str, Any]) -> List[str]:
    """Build a combined token list from chunk metadata and document text."""
    fields = [
        chunk.get("title", ""),
        chunk.get("section", ""),
        chunk.get("source", ""),
        chunk.get("course_number", ""),
        chunk.get("course_title", ""),
        " ".join(chunk.get("tags") or []),
        chunk.get("text", ""),
    ]
    return tokenize(" ".join(fields))


def bm25_score_documents(query: str, chunks: List[Dict[str, Any]], k1: float = 1.5, b: float = 0.75) -> Dict[str, float]:
    """Compute BM25 similarity scores for a list of chunks against the query."""
    tokenized_query = tokenize(query)
    if not tokenized_query or not chunks:
        return {}

    # Tokenize all documents and prepare length statistics.
    tokenized_docs = [lexical_document(chunk) for chunk in chunks]
    doc_lengths = [len(doc) for doc in tokenized_docs]
    avg_doc_len = sum(doc_lengths) / max(len(doc_lengths), 1)

    # Document frequency per token is used in the inverse document frequency term.
    doc_freq = Counter()
    for doc in tokenized_docs:
        for token in set(doc):
            doc_freq[token] += 1

    total_docs = len(tokenized_docs)
    scores: Dict[str, float] = {}

    # Score each chunk relative to the query using BM25.
    for chunk, doc_tokens, doc_len in zip(chunks, tokenized_docs, doc_lengths):
        term_freq = Counter(doc_tokens)
        score = 0.0
        for token in tokenized_query:
            if token not in term_freq:
                continue
            numerator = total_docs - doc_freq[token] + 0.5
            denominator = doc_freq[token] + 0.5
            idf = math.log(1 + (numerator / denominator))
            freq = term_freq[token]
            denom = freq + k1 * (1 - b + b * (doc_len / max(avg_doc_len, 1)))
            score += idf * ((freq * (k1 + 1)) / max(denom, 1e-9))

        scores[chunk.get("chunk_id", "")] = score

    return scores


def normalize_scores(raw_scores: Dict[str, float]) -> Dict[str, float]:
    """Scale raw scores to the [0.0, 1.0] range for fair combination."""
    if not raw_scores:
        return {}

    values = list(raw_scores.values())
    low = min(values)
    high = max(values)
    if math.isclose(low, high):
        return {key: 1.0 for key in raw_scores}

    return {
        key: (value - low) / (high - low)
        for key, value in raw_scores.items()
    }


def metadata_boost(query: str, chunk: Dict[str, Any]) -> float:
    """Add a small score boost when query terms appear in chunk metadata."""
    query_tokens = set(tokenize(query))
    boost = 0.0

    if query_tokens.intersection(tokenize(chunk.get("title", ""))):
        boost += 0.08
    if query_tokens.intersection(tokenize(chunk.get("section", ""))):
        boost += 0.05
    if query_tokens.intersection(tokenize(" ".join(chunk.get("tags") or []))):
        boost += 0.03

    course_tokens = tokenize(f"{chunk.get('course_number', '')} {chunk.get('course_title', '')}")
    if query_tokens.intersection(course_tokens):
        boost += 0.10

    return boost


def reciprocal_rank_fusion(bm25_rank: int | None, semantic_rank: int | None, rrf_k: int = RRF_K) -> float:
    """Combine BM25 and semantic ranks using Reciprocal Rank Fusion (RRF)."""
    score = 0.0
    if bm25_rank is not None:
        score += 1.0 / (rrf_k + bm25_rank)
    if semantic_rank is not None:
        score += 1.0 / (rrf_k + semantic_rank)
    return score


def parse_weaviate_objects(objects: Any, score_attr: str | None = None) -> List[Dict[str, Any]]:
    """Convert Weaviate object results into plain dictionaries for ranking."""
    results = []
    for rank, obj in enumerate(objects, start=1):
        props = obj.properties or {}
        result = {key: props.get(key) for key in RETURN_PROPERTIES}
        result["tags"] = result.get("tags") or []
        result["rank"] = rank
        if obj.metadata is not None and score_attr:
            result[score_attr] = getattr(obj.metadata, score_attr, None)
        else:
            result[score_attr] = None if score_attr else None
        results.append(result)
    return results


def search_chunks(query: str, department_id: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    """
    Hybrid retrieval pipeline for one department.

    1. Semantic ranking with OpenAI embeddings and Weaviate near-vector search.
    2. Local keyword ranking with BM25 over candidate documents.
    3. Score fusion using normalized semantic scores, BM25 scores, RRF, and metadata boosts.
    """
    query_vector = embed_text(query)

    client = get_weaviate_client()
    try:
        ensure_collection(client)
        collection = get_collection(client)

        semantic_response = collection.query.near_vector(
            near_vector=query_vector,
            filters=Filter.by_property("department_id").equal(department_id),
            limit=max(top_k, SEMANTIC_TOP_K),
            return_metadata=MetadataQuery(distance=True),
            return_properties=RETURN_PROPERTIES,
        )

        semantic_results = parse_weaviate_objects(semantic_response.objects, "distance")

        # Convert Weaviate distance metadata into a semantic score.
        # Lower distance means higher semantic similarity.
        semantic_score_map = {
            item["chunk_id"]: 1.0 / (1.0 + float(item["distance"]))
            for item in semantic_results
            if item.get("chunk_id")
        }

        # Preserve the semantic ranking position for later RRF fusion.
        semantic_rank_map = {
            item["chunk_id"]: item["rank"]
            for item in semantic_results
            if item.get("chunk_id")
        }

        # Start with the semantic candidate pool and optionally expand to a broader keyword candidate set.
        keyword_candidates = semantic_results
        try:
            keyword_response = collection.query.fetch_objects(
                filters=Filter.by_property("department_id").equal(department_id),
                limit=KEYWORD_CANDIDATE_LIMIT,
                return_properties=RETURN_PROPERTIES,
            )
            fetched_candidates = parse_weaviate_objects(keyword_response.objects)
            if fetched_candidates:
                keyword_candidates = fetched_candidates
        except Exception:
            # Fall back to the semantic pool if fetch_objects is unavailable or schema is mid-migration.
            keyword_candidates = semantic_results

        bm25_raw_scores = bm25_score_documents(query, keyword_candidates)
        normalized_bm25 = normalize_scores(bm25_raw_scores)
        normalized_semantic = normalize_scores(semantic_score_map)

        # Build a ranked list of keyword-based candidates.
        keyword_ranked = sorted(
            normalized_bm25.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:max(top_k, KEYWORD_TOP_K)]
        keyword_rank_map = {
            chunk_id: rank
            for rank, (chunk_id, _) in enumerate(keyword_ranked, start=1)
        }

        merged_results = []
        semantic_chunk_map = {
            item["chunk_id"]: item
            for item in semantic_results
            if item.get("chunk_id")
        }
        keyword_chunk_map = {
            item["chunk_id"]: item
            for item in keyword_candidates
            if item.get("chunk_id")
        }

        # Combine semantic and keyword candidates into one merged result set.
        for chunk_id in set(keyword_rank_map) | set(semantic_chunk_map):
            item = semantic_chunk_map.get(chunk_id)
            if item is None:
                item = keyword_chunk_map.get(chunk_id)
            if item is None:
                continue

            chunk_id = item.get("chunk_id")
            if not chunk_id:
                continue

            bm25_score = normalized_bm25.get(chunk_id, 0.0)
            semantic_score = normalized_semantic.get(chunk_id, 0.0)
            rrf_score = reciprocal_rank_fusion(
                keyword_rank_map.get(chunk_id),
                semantic_rank_map.get(chunk_id),
            )
            boost = metadata_boost(query, item)
            final_score = (
                WEIGHT_BM25 * bm25_score
                + WEIGHT_SEMANTIC * semantic_score
                + WEIGHT_RRF * rrf_score
                + boost
            )

            item["bm25_score"] = bm25_score
            item["semantic_score"] = semantic_score
            item["rrf_score"] = rrf_score
            item["metadata_boost"] = boost
            item["final_score"] = final_score
            merged_results.append(item)

        merged_results.sort(key=lambda item: item.get("final_score", 0.0), reverse=True)
        return merged_results[:top_k]
    finally:
        client.close()


def build_context(chunks: List[Dict[str, Any]]) -> str:
    """Format retrieved chunks into a single grounded context string for the prompt."""
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(
            f"[Source {i}]\n"
            f"Document ID: {chunk.get('document_id', '')}\n"
            f"Title: {chunk.get('title', '')}\n"
            f"Source: {chunk.get('source', '')}\n"
            f"Section: {chunk.get('section', '')}\n"
            f"Timestamp: {chunk.get('timestamp', '')}\n"
            f"Tags: {', '.join(chunk.get('tags') or [])}\n"
            f"Course: {chunk.get('course_number', '')} {chunk.get('course_title', '')}\n"
            f"URL: {chunk.get('url', '')}\n"
            f"Text: {chunk.get('text', '')}\n"
        )
    return "\n---\n".join(parts)


def generate_grounded_answer(question: str, department_id: str) -> Dict[str, Any]:
    """
    RAG flow:
      1. embed question
      2. retrieve chunks from Weaviate
      3. ask OpenAI to answer only from retrieved context
    """
    # Normalize department identifiers to ensure consistent filtering in Weaviate.
    normalized_department_id = (department_id or "").strip().lower()
    chunks = search_chunks(question, normalized_department_id)
    context = build_context(chunks)

    system_prompt = """
You are a helpful university department assistant.

Answer the user using ONLY the provided context.
If the answer is not in the context, say:
"I could not find that information on the department website."
Do not make up facts.
Be concise and factual.
At the end, list the most relevant source URLs if available.
"""

    user_prompt = f"""
Department: {department_id}
Normalized department: {normalized_department_id}

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

    # Collect the distinct source URLs referenced by the retrieved chunks.
    sources = []
    for chunk in chunks:
        url = chunk.get("url")
        if url and url not in sources:
            sources.append(url)

    return {
        "answer": answer,
        "sources": sources,
        "chunks": chunks
    }
