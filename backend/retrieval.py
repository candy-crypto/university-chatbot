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
from db import lookup_course_by_code, lookup_course_by_title, lookup_courses_by_suffix

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
    # Exclude bare "cs" — it is a department abbreviation used on some outdated
    # web pages (e.g. "CS 171G") but is not a valid Banner course subject code.
    # Matching on it causes any page mentioning a CS-prefix course to get a
    # spurious boost on queries like "What CS courses will be offered?".
    course_tokens = tokenize(
        f"{chunk.get('course_code', '')} "
        f"{' '.join(chunk.get('referenced_courses') or [])}"
    )
    matching_course_tokens = query_tokens.intersection(course_tokens) - {"cs"}
    if matching_course_tokens:
        boost += 0.10

    # Extra boost when the chunk's own course code is an exact match —
    # all tokens of the code appear in the query. Differentiates e.g.
    # "MATH 1521H" from "MATH 1531H" which both match on "math" alone.
    chunk_code_tokens = set(tokenize(chunk.get("course_code", ""))) - {"cs"}
    if chunk_code_tokens and chunk_code_tokens.issubset(query_tokens):
        boost += 0.15

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

    # Glossary chunks are preferred for definitional queries.
    if chunk.get("chunk_type") == "glossary":
        boost += 0.06

    # Course schedule chunks are authoritative for offering/scheduling queries.
    _schedule_terms = {"offered", "offer", "offering", "schedule", "rotation", "semester", "fall", "spring", "opportunity"}
    if chunk.get("chunk_type") == "course_schedule" and query_tokens.intersection(_schedule_terms):
        boost += 0.15

    # Minor requirement chunks are preferred when the query is about a minor.
    if chunk.get("chunk_type") == "minor_requirement" and "minor" in query_tokens:
        boost += 0.15

    # Degree requirement chunks are authoritative for "what is required" questions.
    # Prefer them over study_plan chunks, which list courses as a roadmap rather
    # than as a definitive requirement.
    _requirement_terms = {"required", "requirement", "requirements", "must", "need", "needs"}
    if chunk.get("chunk_type") in ("degree_core_requirement", "degree_requirement") and query_tokens.intersection(_requirement_terms):
        boost += 0.10

    # Course description chunks are authoritative for "what courses cover topic X" questions.
    _course_topic_terms = {"address", "addresses", "cover", "covers", "covering",
                           "about", "related", "focus", "focuses", "teach", "teaches",
                           "include", "includes", "involve", "involves", "topics"}
    if (chunk.get("chunk_type") == "course_description"
            and "courses" in query_tokens
            and query_tokens.intersection(_course_topic_terms)):
        boost += 0.15

    # Policy chunks are preferred for VWW, Gen Ed, and explicit policy queries.
    _policy_query_terms = {
        "vww", "viewing", "wider", "world",           # VWW-specific
        "policy", "policies", "rule", "rules",         # explicit policy questions
        "procedure", "procedures", "process",          # process questions
        "deadline", "deadlines", "hold", "holds",      # administrative process
    }
    if chunk.get("chunk_type") in ("policy", "grad_program_info") and query_tokens.intersection(_policy_query_terms):
        boost += 0.12

    # Level match — boost chunks whose level matches the audience implied by the query,
    # and penalize chunks whose level contradicts it.
    _undergrad_terms = {"undergraduate", "undergrad", "bachelor", "bachelors",
                        "freshman", "freshmen", "sophomore", "junior", "senior"}
    _graduate_terms  = {"graduate", "grad", "master", "masters", "phd",
                        "doctoral", "doctorate", "thesis", "dissertation"}
    chunk_level = chunk.get("level", "")
    if chunk_level == "undergraduate" and query_tokens.intersection(_undergrad_terms):
        boost += 0.08
    elif chunk_level == "graduate" and query_tokens.intersection(_graduate_terms):
        boost += 0.08
    elif chunk_level == "graduate" and query_tokens.intersection(_undergrad_terms):
        boost -= 0.20   # penalize graduate chunks for explicit undergrad queries
    elif chunk_level == "undergraduate" and query_tokens.intersection(_graduate_terms):
        boost -= 0.20   # penalize undergrad chunks for explicit graduate queries

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


# ── Availability redirect pre-retrieval shortcut ─────────────────────────────

_BANNER_URL = (
    "https://banner-public.nmsu.edu/StudentRegistrationSsb/ssb/term/"
    "termSelection?mode=search"
)

_AVAILABILITY_SIGNALS = {
    "offered", "offering", "available", "availability",
    "open", "seats", "sections", "register", "registration", "enroll", "enrollment",
}
_SEMESTER_SIGNALS = {
    "this fall", "this spring", "this summer", "next fall", "next spring",
    "next summer", "this semester", "next semester", "current semester",
    "fall semester", "spring semester", "summer session",
}


def _is_availability_question(query: str) -> bool:
    """Return True if the query is about current-semester course availability
    or open seats — something only Banner can answer in real time."""
    q = query.lower()
    has_availability = bool(set(re.findall(r"[a-z]+", q)).intersection(_AVAILABILITY_SIGNALS))
    has_semester = any(sig in q for sig in _SEMESTER_SIGNALS)
    return has_availability and has_semester


_AVAILABILITY_ANSWER = (
    "Course availability changes in real time as students register and drop, "
    "so it cannot be answered from department sources. "
    "To see which courses are currently offered, use NMSU's course search:\n\n"
    f"  {_BANNER_URL}\n\n"
    "Select the semester, then filter by Subject and Campus: Las Cruces."
)


def try_availability_redirect(query: str) -> Dict[str, Any] | None:
    """Return a Banner redirect answer for availability questions, skipping retrieval."""
    if not _is_availability_question(query):
        return None
    return {
        "answer":         _AVAILABILITY_ANSWER,
        "sources":        [_BANNER_URL],
        "chunks":         [],
        "prompt_context": "",
    }


# ── Course lookup pre-retrieval shortcut ──────────────────────────────────────

_COURSE_CODE_RE = re.compile(r'\b([A-Z]{1,5}\s+\d{3,4}[A-Z]?)\b')

_GEN_ED_TERMS  = {"gen ed", "general education"}
_VWW_TERMS     = {"vww", "viewing a wider world", "wider world"}
_CREDIT_TERMS  = {"how many credits", "credit hours", "how many credit", "worth"}

# Words to strip from the query before doing a course title FTS search.
# Keeps only the content words that describe the course topic.
_TITLE_STRIP_WORDS = {
    "can", "i", "take", "the", "course", "about", "courses", "a", "an",
    "credits", "credit", "toward", "my", "vww", "viewing", "wider", "world",
    "general", "education", "gen", "ed", "requirements", "requirement",
    "is", "does", "do", "will", "contribute", "to", "get", "satisfy",
    "satisfies", "fulfills", "fulfill", "count", "counts", "called", "named",
    "that", "which", "this", "for", "toward", "how", "many", "hours",
    "worth", "into",
}

_SUFFIX_LABELS = {
    "G": "General Education (G suffix)",
    "V": "Viewing a Wider World (V suffix)",
    "":  "neither General Education nor Viewing a Wider World",
}

# Maps lowercase department name keywords to Banner course prefix codes.
_DEPT_KEYWORDS = {
    "math": "MATH", "mathematics": "MATH",
    "physics": "PHYS",
    "chemistry": "CHEM",
    "biology": "BIOL",
    "history": "HIST",
    "english": "ENGL",
    "philosophy": "PHIL",
    "sociology": "SOCI",
    "psychology": "PSYC",
    "economics": "ECON",
    "political": "POLS",
    "anthropology": "ANTH",
    "geography": "GEOG",
    "astronomy": "ASTR",
    "communication": "COMM",
    "theatre": "THEA", "theater": "THEA",
    "music": "MUSC",
    "art": "ARTH",
    "honors": "HNRS",
    "computer": "CSCI",
}


def _detect_question_type(query: str) -> str:
    """Return 'gen_ed', 'vww', 'credits', 'list_gen_ed', 'list_vww', or ''."""
    q = query.lower()
    is_list = ("what " in q or "which " in q) and "courses" in q
    for term in _GEN_ED_TERMS:
        if term in q:
            return "list_gen_ed" if is_list else "gen_ed"
    for term in _VWW_TERMS:
        if term in q:
            return "list_vww" if is_list else "vww"
    for term in _CREDIT_TERMS:
        if term in q:
            return "credits"
    return ""


def _extract_dept_prefix(query: str) -> str:
    """Extract a Banner department code from the query.
    Tries uppercase code pattern first (e.g. 'MATH'), then keyword mapping."""
    # Explicit uppercase code in query (e.g. "MATH courses")
    match = re.search(r'\b([A-Z]{2,5})\b', query)
    if match:
        return match.group(1)
    # Keyword mapping
    q = query.lower()
    for keyword, prefix in _DEPT_KEYWORDS.items():
        if keyword in q:
            return prefix
    return ""


def _suffix_answer(course: dict, question_type: str) -> str:
    code    = course["course_code"]
    title   = course["course_title"]
    suffix  = course["suffix"]
    credits = course["credits"]
    year    = course["catalog_year"]
    label   = _SUFFIX_LABELS.get(suffix, "neither General Education nor Viewing a Wider World")

    if question_type == "credits":
        return (
            f"{code} — {title} carries {credits} credit(s) "
            f"(NMSU Academic Catalog {year})."
        )

    if question_type == "gen_ed":
        if suffix == "G":
            return (
                f"Yes. {code} — {title} has a 'G' suffix, which means it counts toward "
                f"General Education requirements (NMSU Academic Catalog {year})."
            )
        elif suffix == "V":
            return (
                f"No. {code} — {title} has a 'V' suffix, not 'G'. It counts toward "
                f"Viewing a Wider World (VWW) requirements, not General Education "
                f"(NMSU Academic Catalog {year})."
            )
        else:
            return (
                f"No. {code} — {title} does not carry a 'G' suffix, so it does not count "
                f"toward General Education requirements (NMSU Academic Catalog {year})."
            )

    if question_type == "vww":
        if suffix == "V":
            return (
                f"Yes. {code} — {title} has a 'V' suffix, which means it counts toward "
                f"Viewing a Wider World (VWW) requirements (NMSU Academic Catalog {year})."
            )
        elif suffix == "G":
            return (
                f"No. {code} — {title} has a 'G' suffix, not 'V'. It counts toward "
                f"General Education requirements, not VWW "
                f"(NMSU Academic Catalog {year})."
            )
        else:
            return (
                f"No. {code} — {title} does not carry a 'V' suffix, so it does not count "
                f"toward Viewing a Wider World requirements (NMSU Academic Catalog {year})."
            )

    return ""


def _format_course_list(courses: list, suffix: str, dept_prefix: str) -> str:
    """Format a list of courses with a given suffix into a readable answer."""
    label_map = {"G": "General Education (G)", "V": "Viewing a Wider World (V)"}
    label = label_map.get(suffix, suffix)
    dept_phrase = f" in {dept_prefix}" if dept_prefix else ""

    if not courses:
        return (
            f"No courses{dept_phrase} with a {label} suffix were found "
            f"in the course lookup table."
        )

    lines = [f"The following{dept_phrase} courses carry a {label} suffix:\n"]
    for c in courses:
        credit_str = f" ({c['credits']} cr.)" if c.get("credits") else ""
        lines.append(f"- {c['course_code']} — {c['course_title']}{credit_str}")

    year = courses[0].get("catalog_year", "")
    if year:
        lines.append(f"\nSource: NMSU Academic Catalog {year}.")
    return "\n".join(lines)


def try_course_lookup(query: str) -> Dict[str, Any] | None:
    """
    Pre-retrieval shortcut for suffix, credit, and list questions.

    1. Detect question type (gen_ed / vww / credits / list_gen_ed / list_vww).
    2. For list types: return all courses with the matching suffix, filtered by dept.
    3. For single-course types: find the course by code or title FTS.
    4. If found, return a direct answer without hitting Weaviate.
    5. Return None to fall through to full RAG.
    """
    question_type = _detect_question_type(query)
    if not question_type:
        return None

    # List queries: return all courses with the matching suffix for a department.
    if question_type in ("list_gen_ed", "list_vww"):
        suffix = "G" if question_type == "list_gen_ed" else "V"
        dept_prefix = _extract_dept_prefix(query)
        courses = lookup_courses_by_suffix(suffix, dept_prefix)
        answer = _format_course_list(courses, suffix, dept_prefix)
        source = f"NMSU Academic Catalog (course lookup table)"
        return {
            "answer":         answer,
            "sources":        [source],
            "chunks":         [],
            "prompt_context": "",
        }

    course = None

    # Try course code extraction first (most reliable)
    codes = _COURSE_CODE_RE.findall(query.upper())
    for code in codes:
        course = lookup_course_by_code(code)
        if course:
            break

    # Fall back to full-text title search using only topic content words.
    # Strip question/domain vocabulary so FTS matches course titles, not
    # question words like "credits", "vww", "requirements".
    if not course:
        content_words = " ".join(
            w for w in re.findall(r"[a-z]+", query.lower())
            if w not in _TITLE_STRIP_WORDS
        )
        if content_words:
            results = lookup_course_by_title(content_words)
            if results:
                # Use top result if it scores clearly above the rest,
                # or if there is only one result.
                top = results[0]
                if len(results) == 1 or (
                    len(results) > 1 and results[0].get("rank", 0) > 2 * results[1].get("rank", 0)
                ):
                    course = top

    if not course:
        return None

    answer = _suffix_answer(course, question_type)
    if not answer:
        return None

    source = f"NMSU Academic Catalog {course['catalog_year']} (course lookup table)"
    return {
        "answer":         answer,
        "sources":        [source],
        "chunks":         [],
        "prompt_context": "",
    }


def generate_grounded_answer(question: str, department_id: str) -> Dict[str, Any]:
    """
    RAG flow:
      0. Pre-retrieval course lookup for suffix/credit questions (no Weaviate call).
      1. Embed question with OpenAI.
      2. Retrieve top-K chunks from Weaviate via hybrid search.
      3. Ask OpenAI to answer using only the retrieved context.
    """
    normalized_department_id = (department_id or "").strip().lower()

    # Short-circuit for current-semester availability questions — Banner only.
    if not MOCK_OPENAI:
        availability_result = try_availability_redirect(question)
        if availability_result:
            return availability_result

    # Short-circuit for suffix (Gen Ed / VWW) and credit questions.
    if not MOCK_OPENAI:
        lookup_result = try_course_lookup(question)
        if lookup_result:
            return lookup_result

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

## Chunk Types and How to Read Them

The retrieved context includes a `Chunk Type` field. Use it to interpret completeness:
- `degree_core_requirement` — shared requirements for ALL concentrations of a degree family \
(gen ed, core CS courses, math, science). One chunk covers all concentrations. When this chunk \
is present, its requirements apply to every concentration unless explicitly overridden.
- `concentration_requirement` — ONLY the requirements unique to one concentration. It is \
incomplete on its own; combine it with the `degree_core_requirement` chunk to get the full picture.
- `degree_requirement` — full requirements for standalone degrees (BA, non-concentrated MS/PhD). \
Self-contained.
- `study_plan` — a suggested semester-by-semester roadmap. It lists courses but is not the \
authoritative requirement source; a course appearing in a study plan does not mean it is required.

## Source Preference

Use the source type that best matches the question:
- Course descriptions, prerequisites, degree requirements, general education, VWW → prefer CATALOG
- Advising contacts, financial aid, assistantships, faculty directory → prefer WEB
- When both are needed (e.g., requirements + how to apply), answer from both and cite each separately
- Faculty information exists in both the catalog and on the web; prefer web for current contact info

## Course Numbering and Level

At NMSU, course level is determined by the number:
- 3-digit numbers below 500 (e.g., C E 490, E E 465) — undergraduate
- 3-digit numbers 500 and above (e.g., CSCI 551) — graduate
- 4-digit numbers below 5000 (e.g., CSCI 4440, CSCI 1720) — undergraduate
- 4-digit numbers 5000 and above (e.g., CSCI 5405, CSCI 5750) — graduate

When a user asks specifically about undergraduate or graduate courses, apply this rule to \
include or exclude courses accordingly — even if the retrieved chunk does not explicitly state \
the level. Do not present a graduate-level course as an answer to an undergraduate question, \
or vice versa.

## Course Code Suffixes

- Codes ending in G (e.g., CSCI 1115G) count toward General Education Requirements
- Codes ending in V (e.g., ASTR 308V) count toward Viewing a Wider World (VWW) Requirements
- Both General Education and VWW are undergraduate degree requirements only — they do not apply to graduate students
- Explain the suffix meaning when it appears in your answer

## Long Catalog Lists

For lengthy catalog lists (gen ed sequences, elective lists, course sequences): summarize the structure, \
categories, and total credit counts. Do not reproduce multi-page lists verbatim. Always cite the catalog \
page(s) where the full list can be found.

## Registration and Enrollment Availability

For any question about open seats, current section availability, or course registration for a specific \
semester: do NOT answer from context. This information changes in real time as students register and drop, \
so it is not available here. Direct the user to NMSU's course search instead:

  https://banner-public.nmsu.edu/StudentRegistrationSsb/ssb/term/termSelection?mode=search
  Instructions: Select the semester, then filter Subject: CS, Campus: Las Cruces.

## Summer Session Offerings

The three-year course rotation covers only Fall and Spring semesters. Department sources do not list which \
courses are offered during summer sessions. For any question about summer course availability — whether \
asking about a specific course or about typical summer offerings — direct the user to NMSU's course search \
and note that summer schedules are published mid-to-late Spring semester:

  https://banner-public.nmsu.edu/StudentRegistrationSsb/ssb/term/termSelection?mode=search
  Instructions: Select the summer term, then filter Subject: CS, Campus: Las Cruces.

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
- Do NOT use these phrases: "Great question!", "Certainly!", "Of course!", "I'd be happy to help", \
"As an AI", "I'm here to help", "Absolutely!"
- Do not begin your response with a compliment on the question or a promise to assist — just answer

## Format

- Write in prose, not bullets. Bullets make simple answers feel like documentation. \
Use flowing sentences and paragraphs instead.
- Use a bulleted or numbered list only when the content is genuinely list-like and prose would be \
harder to read — for example, a sequence of required steps, or a list of six or more distinct items \
that have no natural connective flow.
- Never use bullets just because the source material uses them.

## Length

- Answer the question and stop. Do not summarize what you just said, do not add encouragement, \
do not suggest follow-up questions.
- Include only the facts needed to answer the specific question asked. Omit background context \
the user did not ask for.
- If a question has a short factual answer, give a short answer. A one- or two-sentence response \
is often correct.
- A response should never exceed what a student could comfortably read in 60 seconds.
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
