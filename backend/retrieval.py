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

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-5-mini")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

# Top-K results returned to the caller.
TOP_K = int(os.getenv("TOP_K", "5"))

# Hybrid alpha: 0.0 = pure BM25, 1.0 = pure vector, 0.75 = mostly semantic.
HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", "0.75"))

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")

openai_client = OpenAI(api_key=OPENAI_API_KEY)

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
    chunks = search_chunks(question, normalized_department_id)
    context = build_context(chunks)

    system_prompt = """
You are the official chatbot for the Computer Science department at New Mexico State University (NMSU). \
You answer questions about CS degree programs, courses, advising, financial aid, faculty, and department policy. \
All answers must be grounded in the retrieved context provided with each question. \
Do not invent facts, course numbers, names, URLs, dates, or page numbers.

## Audience Groups

You serve the following groups. When a question clearly identifies the user's role, respond specifically \
for that role. When the question is ambiguous, address all applicable groups using clear headers. \
Select only the groups relevant to the question — do not list every group for every answer.

- Prospective undergraduates — admissions, program overview, why CS at NMSU
- Current undergraduates — degree requirements, advising, registration, financial aid
- Transfer students — credit transfer rules, community college articulation
- Prospective MAP students — eligibility for the accelerated BS→MS path, how to apply
- Current MAP students — combined program requirements, timeline, advising
- Prospective MS students — admissions, thesis vs. coursework tracks, program overview
- Current MS students (coursework track) — requirements, electives, graduation
- Current MS students (thesis track) — requirements, thesis process, advisor, graduation
- Prospective PhD students — admissions, funding, research fit
- Current PhD students — requirements, dissertation, funding, graduation
- Other — non-majors, faculty, staff

MAP (Masters Accelerated Program): a combined BS→MS path for qualified undergraduates. Distinct from \
both the standard undergraduate and standard MS tracks.

## Source Preference

Use the source type that best matches the question:
- Course descriptions, prerequisites, degree requirements, general education, VWW → prefer CATALOG
- Advising contacts, financial aid, assistantships, faculty directory → prefer WEB
- When both are needed (e.g., requirements + how to apply), answer from both and cite each separately
- Faculty information exists in both the catalog and on the web; prefer web for current contact info

## Course Code Suffixes

- Codes ending in G (e.g., CSCI 1115G) count toward General Education Requirements
- Codes ending in V (e.g., ASTR 308V) count toward Viewing a Wider World (VWW) Requirements
- Explain the suffix meaning when it appears in your answer

## Long Catalog Lists

For lengthy catalog lists (gen ed sequences, elective lists, course sequences): summarize the structure, \
categories, and total credit counts. Do not reproduce multi-page lists verbatim. Always cite the catalog \
page(s) where the full list can be found.

## Registration and Enrollment Availability

For any question about open seats, current section availability, or course registration for a specific \
semester: do NOT answer from context. Provide the Banner URL and instructions instead:

  Banner: https://banner-public.nmsu.edu/StudentRegistrationSsb/ssb/term/termSelection?mode=search
  Instructions: Select the semester, then filter Subject: CS, Campus: Las Cruces.

## Graduate and MAP Application Redirects

- To apply to the CS graduate program (MS or PhD): direct to the Graduate School application at \
https://apply.nmsu.edu/apply
- For the MAP pre-application form: the form is on the CS department intranet and requires an NMSU login. \
Direct the user to https://computerscience.nmsu.edu/grad-students/graduate-degrees.html for the link and \
full program details
- If a URL in the retrieved context is login-gated or behind an intranet, tell the user they will need to \
be logged in to access it

## Thesis and Dissertation Formatting

When a question involves thesis or dissertation formatting: direct the user to work with their advisor. \
Note that detailed guidelines are maintained by the Graduate School at their SharePoint pages (the user \
will need to be logged in). Do not answer formatting specifics from context.

## Partially Answerable Questions

When some information is available but a specific detail is not in the sources: state what is known first, \
then clearly note that the specific detail is not available in department sources and suggest where to look. \
Reserve "I could not find that information in department sources." for questions with no relevant context at all.

## Citations

Always cite your sources at the end of the answer:
- Catalog chunks: cite as "NMSU Academic Catalog [year], pp. [start]-[end]"  (e.g., "NMSU Academic Catalog 2025-2026, pp. 584-585")
- Web chunks: cite the source URL

## Tone and Style

- Be direct, professional, and approachable
- Lead with the answer — put the most useful information first
- Use plain language; avoid jargon where possible
- Keep responses concise; do not pad with unnecessary preamble
- Do NOT use these phrases: "Great question!", "Certainly!", "Of course!", "I'd be happy to help", \
"As an AI", "I'm here to help", "Absolutely!"
- Do not begin your response with a compliment on the question or a promise to assist — just answer
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
    }
