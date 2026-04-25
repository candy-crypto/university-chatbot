# retrieval.py

"""Hybrid retrieval utilities for the university chatbot backend.

This module handles query embedding, Weaviate native hybrid search
(BM25 + vector via RRF), optional metadata boosting, and prompt
construction for grounded answer generation.
"""

import os
import re
from datetime import date
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


# ── Token-level constants ─────────────────────────────────────────────────────

# Common English words that carry no domain meaning and cause spurious
# metadata boost matches (e.g. "of" in "Bachelor of Science" matching "of"
# in a query about an unrelated topic).
_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "it", "its", "this", "that",
    "i", "me", "my", "we", "our", "you", "your",
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "can",
    "what", "which", "who", "when", "where", "how", "if",
})

_UNDERGRAD_TERMS = frozenset({
    "undergraduate", "undergrad", "bachelor", "bachelors",
    "freshman", "freshmen", "sophomore", "junior", "senior",
})
_GRADUATE_TERMS = frozenset({
    "graduate", "grad", "master", "masters", "ms", "phd",
    "doctoral", "doctorate", "thesis", "dissertation",
})
_SCHEDULE_TERMS = frozenset({
    "offered", "offer", "offering", "schedule", "rotation",
    "semester", "fall", "spring", "opportunity",
    # Availability vocabulary not caught by the terms above.
    "next",         # "next time it's offered", "next fall"
    "available",    # "is CSCI 4120 available?"
    "availability", # "course availability"
})
# "when" is in _STOPWORDS so tokenize() drops it; check the raw query instead.
# Covers "when is X offered?", "when will X run?", "when does X meet?" without
# removing "when" from the stopword list (which would cause false boosts elsewhere).
_WHEN_RE = re.compile(r'\bwhen\b', re.IGNORECASE)
# Degree/requirement trigger terms — includes comparison vocabulary so the
# degree_requirement boost fires for "difference between X and Y" queries.
_REQUIREMENT_TERMS = frozenset({
    "required", "requirement", "requirements", "must", "need", "needs",
    "difference", "differences", "compare", "comparing", "comparison",
    "distinguish", "between", "versus", "vs",
})
# Topic-verb terms that signal "what courses cover X?" queries.
_COURSE_TOPIC_TERMS = frozenset({
    "address", "addresses", "cover", "covers", "covering",
    "about", "related", "focus", "focuses", "teach", "teaches",
    "include", "includes", "involve", "involves", "topics",
    "difference", "differences", "compare", "comparing",
    "distinguish", "between", "versus", "vs",
})
_POLICY_QUERY_TERMS = frozenset({
    "vww", "viewing", "wider", "world",
    "policy", "policies", "rule", "rules",
    "procedure", "procedures", "process",
    "deadline", "deadlines", "hold", "holds",
})
# Gen Ed / VWW terms — used to (a) boost policy/grad_program_info chunks and
# (b) penalize study_plan chunks that mention VWW courses but are not the
# authoritative policy source.
_GEN_ED_TERMS = frozenset({
    "vww", "viewing", "wider", "world",
    "gened", "gen", "education", "general",
})
# Faculty-query terms — signal that the user is asking about a specific person
# or looking for faculty directory information.
_FACULTY_TERMS = frozenset({
    "faculty", "professor", "instructor", "dr", "doctor",
    "who", "office", "email", "contact", "research", "teaches",
    "advisor", "adviser",
})
# Enrollment / application terms — trigger boost for enrollment chunk_type.
_ENROLLMENT_TERMS = frozenset({
    "apply", "application", "applications",
    "admission", "admissions", "admit", "admitted",
    "enroll", "enrollment", "register", "registration",
})
# "between X and Y", "X vs Y" — student is comparing named specific items;
# default TOP_K is sufficient since the named chunks will score highly.
_SPECIFIC_COMPARISON_TERMS = frozenset({"between", "vs", "versus"})


def tokenize(text: str) -> List[str]:
    """Normalize text, extract lowercase tokens, and filter stopwords.

    Stopword filtering prevents common English words ("of", "a", "in", …)
    from creating spurious intersections between query tokens and chunk
    metadata fields, which previously caused inflated boosts on unrelated
    chunks (e.g. study_plan chunks receiving 0.18–0.28 boost from shared
    preposition tokens).
    """
    return [t for t in re.findall(r"[a-z0-9]+", (text or "").lower())
            if t not in _STOPWORDS]


# ── Acronym expansion ─────────────────────────────────────────────────────────
# Common CS/NMSU acronyms that students use but that don't appear verbatim in
# catalog headings. We APPEND the expansion rather than replace so both forms
# are present for BM25 matching.

_ACRONYM_MAP = {
    r"\bAI\b":    "Artificial Intelligence",
    r"\bML\b":    "Machine Learning",
    r"\bHCI\b":   "Human Computer Interaction",
    r"\bNLP\b":   "Natural Language Processing",
    r"\bOS\b":    "Operating Systems",
    r"\bSE\b":    "Software Engineering",
    r"\bDS\b":    "Data Science",
    r"\bDB\b":    "Database",
    r"\bMAP\b":   "Masters Accelerated Program",
    # CAASS has no BM25 hits without expansion — the acronym never appears in the
    # catalog; only the full name "Center for Academic Advising and Student Support"
    # appears on the CS advising FAQ and the College of A&S intro pages.
    r"\bCAASS\b": "Center for Academic Advising and Student Support",
    # VWW never appears verbatim in catalog headings — only the full phrase does.
    r"\bVWW\b":   "Viewing a Wider World",
}


def expand_acronyms(query: str) -> str:
    """Append full forms of recognized acronyms so BM25 can match catalog headings.

    Appends rather than replaces so both the acronym and the expansion are
    present in the query string. E.g.:
        'difference between Cybersecurity and AI'
        → 'difference between Cybersecurity and AI Artificial Intelligence'
    """
    additions = []
    for pattern, expansion in _ACRONYM_MAP.items():
        if re.search(pattern, query):
            additions.append(expansion)
    if additions:
        return query + " " + " ".join(additions)
    return query


# ── Temporal query expansion ───────────────────────────────────────────────────
# NMSU approximate semester boundaries (month, day):
#   Spring:  Jan 1  – May 19
#   Summer:  May 20 – Aug 14
#   Fall:    Aug 15 – Dec 31

_SEASON_ORDER = {"Spring": 1, "Summer": 2, "Fall": 3}


def _semester_for_date(d: date) -> tuple[str, int]:
    """Return (semester_name, year) for a given date."""
    m, day, y = d.month, d.day, d.year
    if (m < 5) or (m == 5 and day < 20):
        return ("Spring", y)
    elif (m < 8) or (m == 8 and day < 15):
        return ("Summer", y)
    else:
        return ("Fall", y)


def _next_occurrence(target: str, current_sem: str, current_year: int) -> tuple[str, int]:
    """Earliest occurrence of target season that is current or future."""
    if _SEASON_ORDER[target] >= _SEASON_ORDER[current_sem]:
        return (target, current_year)
    return (target, current_year + 1)


def _strictly_next(target: str, current_sem: str, current_year: int) -> tuple[str, int]:
    """First occurrence of target season strictly after the current semester."""
    if _SEASON_ORDER[target] > _SEASON_ORDER[current_sem]:
        return (target, current_year)
    return (target, current_year + 1)


def _last_occurrence(target: str, current_sem: str, current_year: int) -> tuple[str, int]:
    """Most recent past occurrence of target season."""
    if _SEASON_ORDER[target] < _SEASON_ORDER[current_sem]:
        return (target, current_year)
    return (target, current_year - 1)


def _next_academic_semester(current_sem: str, current_year: int) -> tuple[str, int]:
    """Next academic semester, skipping Summer (students say 'next semester' to mean Fall/Spring)."""
    if current_sem == "Fall":
        return ("Spring", current_year + 1)
    return ("Fall", current_year)


def _prev_academic_semester(current_sem: str, current_year: int) -> tuple[str, int]:
    """Previous academic semester, skipping Summer."""
    if current_sem == "Spring":
        return ("Fall", current_year - 1)
    return ("Spring", current_year)


def expand_temporal_query(query: str, today: date | None = None) -> str:
    """Rewrite relative temporal references to absolute semester/year strings.

    This ensures that BM25 scoring can match the right calendar or schedule
    chunks when a student asks about 'this semester', 'next fall', etc.

    Examples (assuming today = April 24, 2026 → Spring 2026):
        'this semester'   → 'Spring 2026'
        'next semester'   → 'Fall 2026'   (academic — skips Summer)
        'last semester'   → 'Fall 2025'   (academic — skips Summer)
        'this summer'     → 'Summer 2026'
        'next fall'       → 'Fall 2026'
        'next spring'     → 'Spring 2027'
        'last spring'     → 'Spring 2025'
        'this year'       → '2026'
        'next year'       → '2027'
        'last year'       → '2025'
    """
    d = today or date.today()
    cur_sem, cur_year = _semester_for_date(d)
    q = query

    # "this/current semester"
    q = re.sub(
        r'\b(this|current)\s+semester\b',
        f"{cur_sem} {cur_year}",
        q, flags=re.IGNORECASE,
    )
    # "next semester" (academic — skips Summer)
    ns, ny = _next_academic_semester(cur_sem, cur_year)
    q = re.sub(r'\bnext\s+semester\b', f"{ns} {ny}", q, flags=re.IGNORECASE)

    # "last/previous semester" (academic — skips Summer)
    ps, py = _prev_academic_semester(cur_sem, cur_year)
    q = re.sub(r'\b(last|previous)\s+semester\b', f"{ps} {py}", q, flags=re.IGNORECASE)

    # Year terms
    q = re.sub(r'\b(this|current)\s+year\b', str(cur_year), q, flags=re.IGNORECASE)
    q = re.sub(r'\bnext\s+year\b', str(cur_year + 1), q, flags=re.IGNORECASE)
    q = re.sub(r'\b(last|previous)\s+year\b', str(cur_year - 1), q, flags=re.IGNORECASE)

    # Season terms: this/next/last fall|spring|summer
    for season in ("Fall", "Spring", "Summer"):
        sl = season.lower()
        this_sem, this_year = _next_occurrence(season, cur_sem, cur_year)
        q = re.sub(rf'\bthis\s+{sl}\b', f"{season} {this_year}", q, flags=re.IGNORECASE)

        next_sem, next_year = _strictly_next(season, cur_sem, cur_year)
        q = re.sub(rf'\bnext\s+{sl}\b', f"{season} {next_year}", q, flags=re.IGNORECASE)

        last_sem, last_year = _last_occurrence(season, cur_sem, cur_year)
        q = re.sub(
            rf'\b(last|previous)\s+{sl}\b',
            f"{season} {last_year}",
            q, flags=re.IGNORECASE,
        )

    return q


def metadata_boost(query: str, chunk: Dict[str, Any]) -> float:
    """Add a small score boost when query terms appear in chunk metadata fields."""
    query_tokens = set(tokenize(query))
    boost = 0.0

    # Exclude bare "cs" from heading matches — it is a department abbreviation
    # that appears in nearly every FAQ chunk heading, causing spurious +0.08
    # boosts on off-topic chunks (e.g. "Questions related to CS minor" ranking
    # high for "How do I apply to the CS graduate program?").
    heading_tokens = set(tokenize(chunk.get("heading", ""))) - {"cs"}
    if query_tokens.intersection(heading_tokens):
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

    # Glossary chunks preferred for definitional queries — but only when the
    # query actually contains a term from the glossary entry's heading.
    # (Unconditional +0.06 was causing the advisor glossary entry to rank #1
    # for "What is CAASS?" even though CAASS is not about advisors.)
    if chunk.get("chunk_type") == "glossary":
        glossary_heading_tokens = set(tokenize(chunk.get("heading", "")))
        if query_tokens.intersection(glossary_heading_tokens):
            boost += 0.06

    # Course schedule chunks are authoritative for offering/scheduling queries.
    # _SCHEDULE_TERMS covers most vocabulary; _WHEN_RE catches "when is/will/does"
    # whose key word "when" is a stopword and never survives tokenize().
    if chunk.get("chunk_type") == "course_schedule" and (
        query_tokens.intersection(_SCHEDULE_TERMS)
        or _WHEN_RE.search(query)
    ):
        boost += 0.15

    # Minor requirement chunks are preferred when the query is about a minor.
    if chunk.get("chunk_type") == "minor_requirement" and "minor" in query_tokens:
        boost += 0.15

    # Degree/concentration requirement chunks are authoritative for "what is
    # required" and comparison queries ("difference between X and Y").
    if (chunk.get("chunk_type") in (
            "degree_core_requirement", "degree_requirement", "concentration_requirement")
            and query_tokens.intersection(_REQUIREMENT_TERMS)):
        boost += 0.10

    # Course description chunks are authoritative for "what courses cover topic X".
    if (chunk.get("chunk_type") == "course_description"
            and "courses" in query_tokens
            and query_tokens.intersection(_COURSE_TOPIC_TERMS)):
        boost += 0.15

    # Policy chunks are preferred for VWW, Gen Ed, and explicit policy queries.
    if (chunk.get("chunk_type") in ("policy", "grad_program_info")
            and query_tokens.intersection(_POLICY_QUERY_TERMS)):
        boost += 0.12

    # Extra boost for Gen Ed / VWW policy chunks: study_plan chunks that list VWW
    # courses can outscore the policy definition pages on phrase match.  Raise the
    # policy/grad_program_info signal higher, and suppress study_plan chunks that
    # are not answering a degree-plan question.
    if (chunk.get("chunk_type") in ("policy", "grad_program_info")
            and query_tokens.intersection(_GEN_ED_TERMS)):
        boost += 0.08
    if (chunk.get("chunk_type") == "study_plan"
            and query_tokens.intersection(_GEN_ED_TERMS)):
        boost -= 0.12

    # Faculty chunks are preferred when the query is about a specific person or
    # asking for faculty directory information.
    if (chunk.get("chunk_type") == "faculty"
            and query_tokens.intersection(_FACULTY_TERMS)):
        boost += 0.15

    # Enrollment chunks are preferred for apply/admission/registration queries.
    # Vocabulary mismatch: "apply" in a user query rarely appears verbatim in
    # formal catalog enrollment chunks, so BM25 misses them without this boost.
    if (chunk.get("chunk_type") == "enrollment"
            and query_tokens.intersection(_ENROLLMENT_TERMS)):
        boost += 0.12

    # Level match — boost chunks whose level matches the audience implied by the
    # query, and penalize chunks whose level contradicts it.
    chunk_level = chunk.get("level", "")
    if chunk_level == "undergraduate" and query_tokens.intersection(_UNDERGRAD_TERMS):
        boost += 0.08
    elif chunk_level == "graduate" and query_tokens.intersection(_GRADUATE_TERMS):
        boost += 0.08
    elif chunk_level == "graduate" and query_tokens.intersection(_UNDERGRAD_TERMS):
        boost -= 0.20   # penalize graduate chunks for explicit undergrad queries
    elif chunk_level == "undergraduate" and query_tokens.intersection(_GRADUATE_TERMS):
        boost -= 0.20   # penalize undergrad chunks for explicit graduate queries

    if query_tokens.intersection(tokenize(chunk.get("lab", ""))):
        boost += 0.05

    # Penalize study_plan chunks whose concentration is not mentioned in the query.
    # Without this, all 11 concentration study plans flood results for a plain
    # "BS in Computer Science" query because their first-year content is identical.
    # The core BS CS plan has concentration="" and is not penalized.
    if chunk.get("chunk_type") == "study_plan":
        concentration = chunk.get("concentration") or ""
        if concentration:
            concentration_tokens = set(tokenize(concentration))
            if not query_tokens.intersection(concentration_tokens):
                boost -= 0.20

    # Penalize concentration_requirement chunks when the parent degree is not
    # mentioned in the query and the query doesn't explicitly say "concentration".
    # This distinguishes "BS in Cybersecurity" (standalone degree) from
    # "Computer Science Cybersecurity concentration" — both use the word
    # "Cybersecurity" so the metadata boost alone cannot differentiate them.
    # E.g. "specialty BS in Cybersecurity" → parent degree is "Computer Science"
    # which is not in the query → concentration chunk penalized → standalone
    # Cybersecurity degree_requirement chunk wins instead.
    # Exception: if the student explicitly says "concentration", the query is
    # unambiguously about a concentration track, so no penalty.
    if chunk.get("chunk_type") == "concentration_requirement":
        if "concentration" not in query_tokens:
            parent_degree_tokens = set(tokenize(chunk.get("degree_full_title", "")))
            if parent_degree_tokens and not query_tokens.intersection(parent_degree_tokens):
                boost -= 0.15

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


def _build_hard_filters(query: str, tokens: set) -> "Filter | None":
    """
    Build optional Weaviate hard filters based on query signals.

    Hard filters narrow the candidate pool *before* hybrid scoring, ensuring
    that the right chunk types can enter top-K even when their BM25/vector
    scores are weaker than irrelevant chunks from other categories.

    Priority:
      1. Level filter (always applied when audience is explicit, independent
         of chunk_type filters).
      2. Minor filter — when "minor" in query: restrict to minor + supporting
         chunk types so minor chunks aren't displaced by BS/study_plan chunks.
      3. Degree-exclude-minor — when degree vocabulary present without "minor":
         exclude minor_requirement chunks that pollute degree queries.
      4. Course-description filter — for "what courses teach X?" queries with no
         specific course code: restrict to course_description and raise TOP_K.

    Returns a combined Filter, or None if no hard filtering is warranted.
    """
    filters = []
    minor_query = "minor" in tokens

    # ── Level filter ──────────────────────────────────────────────────────────
    # Include chunks whose level matches the query audience OR whose level is
    # unset (level == "" means the chunk applies to all audiences).
    # NOTE: Filter.by_property("level").equal("") causes Weaviate hybrid BM25
    # to fail with "only stopwords provided". Use not_equal on the opposite
    # level instead — semantically equivalent and avoids the empty-string bug.
    if tokens.intersection(_UNDERGRAD_TERMS):
        filters.append(
            Filter.by_property("level").not_equal("graduate")
        )
    elif tokens.intersection(_GRADUATE_TERMS):
        filters.append(
            Filter.by_property("level").not_equal("undergraduate")
        )

    # ── chunk_type filter ─────────────────────────────────────────────────────
    if minor_query:
        # Minor queries: widen the candidate pool to include minor chunks and
        # the context types needed to answer fully (general advising pages,
        # degree overviews, course descriptions for "what courses are in the minor?").
        filters.append(
            Filter.by_property("chunk_type").contains_any([
                "minor_requirement", "minor_index",
                "degree_overview", "general", "advising", "course_description",
            ])
        )
    elif tokens.intersection({"degree", "major", "program", "bachelor", "master", "phd"}):
        # Degree queries without "minor": exclude minor chunks that match on
        # shared vocabulary (e.g. Biology minor appearing for Biology degree queries).
        filters.append(
            Filter.by_property("chunk_type").contains_none([
                "minor_requirement", "minor_index",
            ])
        )
    elif ("courses" in tokens
          and tokens.intersection(_COURSE_TOPIC_TERMS)
          and not tokens.intersection(_POLICY_QUERY_TERMS)
          and not _COURSE_CODE_RE.search(query.upper())):
        # Topic queries ("what courses teach X?" — no specific course code):
        # restrict to course_description so degree/minor chunks don't displace
        # relevant courses. TOP_K is raised in search_chunks() for these queries.
        # Exception: when policy/VWW terms are present (e.g. "what courses meet
        # my VWW requirements?"), do NOT restrict — the policy chunk must be
        # reachable even though "courses" and a topic verb are in the query.
        filters.append(Filter.by_property("chunk_type").equal("course_description"))

    if not filters:
        return None

    result = filters[0]
    for f in filters[1:]:
        result = result & f
    return result


def search_chunks(query: str, department_id: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    """
    Hybrid retrieval using Weaviate's native hybrid search.

    Weaviate fuses BM25 keyword scores and vector similarity scores using
    Reciprocal Rank Fusion (RRF) internally, then returns results ranked by
    the combined score. Optional hard Weaviate filters narrow the candidate
    pool by chunk_type and level before scoring. A metadata boost is applied
    as a final re-ranking step.
    """
    query = expand_temporal_query(query)
    query = expand_acronyms(query)
    query_vector = embed_text(query)
    tokens = set(tokenize(query))

    client = get_weaviate_client()
    try:
        ensure_collection(client)
        collection = get_collection(client)

        # Always filter by department. Add optional hard filters for level
        # and chunk_type based on query signals.
        base_filter = Filter.by_property("department_id").equal(department_id)
        extra = _build_hard_filters(query, tokens)
        weaviate_filter = base_filter if extra is None else base_filter & extra

        # Raise TOP_K when the query mentions a multi-item category without
        # naming specific items. The presence of "between/vs/versus" indicates
        # a specific comparison (2-3 named items) — keep default TOP_K in that
        # case since the relevant chunks will score highly on their own.
        #
        # Ceilings set by item counts in the CS department:
        #   concentrations: 11  → TOP_K 15
        #   minors:          6  → TOP_K 12
        #   degrees/programs ~5 → TOP_K 10
        #   scholarships    ~5+ → TOP_K 10
        #   courses: not raised here — handled by topic-verb detection below
        #   roadmaps: only 2 — default TOP_K=5 is sufficient
        effective_top_k = top_k
        is_specific = bool(tokens.intersection(_SPECIFIC_COMPARISON_TERMS))

        if not is_specific:
            if tokens.intersection({"concentration", "concentrations",
                                    "track", "tracks", "specialization"}):
                effective_top_k = max(top_k, 15)
            elif tokens.intersection({"minor", "minors"}):
                effective_top_k = max(top_k, 12)
            elif tokens.intersection({"degree", "degrees", "program", "programs"}):
                effective_top_k = max(top_k, 10)
            elif tokens.intersection({"scholarship", "scholarships",
                                      "financial", "funding", "aid"}):
                effective_top_k = max(top_k, 10)

        if ("courses" in tokens
                and tokens.intersection(_COURSE_TOPIC_TERMS)
                and not _COURSE_CODE_RE.search(query.upper())):
            effective_top_k = max(effective_top_k, 12)

        # Policy/VWW queries: raise TOP_K so policy chunks have room to surface
        # before metadata re-ranking. With alpha=0.75 (vector-dominant), the
        # extra words in a VWW query (year, courses, when) can push the policy
        # chunk below rank 5 even when BM25 matches on the heading.
        if tokens.intersection(_POLICY_QUERY_TERMS):
            effective_top_k = max(effective_top_k, 10)

        response = collection.query.hybrid(
            query=query,
            vector=query_vector,
            filters=weaviate_filter,
            limit=effective_top_k,
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
        # Re-assign rank to reflect final post-boost order.
        for i, item in enumerate(results, start=1):
            item["rank"] = i
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

    _today = date.today()
    _cur_sem, _cur_year = _semester_for_date(_today)
    system_prompt = f"""
You are the official chatbot for the Computer Science department at New Mexico State University (NMSU). \
You answer questions about CS degree programs, courses, advising, financial aid, faculty, and department policy. \
All answers must be grounded in the retrieved context provided with each question. \
Do not invent facts, course numbers, names, URLs, dates, or page numbers.

## Current Date
Today is {_today.strftime("%B %d, %Y")}. The current academic semester is {_cur_sem} {_cur_year}. \
Use this to interpret relative time references (e.g. "this semester", "next fall", "last spring") \
and to flag when a retrieved calendar chunk covers a different semester than the one the student asked about.

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

## Degree vs. Concentration Disambiguation

When a **current student** identifies their major by name (e.g., "I am a Cybersecurity major", \
"I'm in the Data Analytics program"), take it at face value — current students know their own enrollment. \
Do not suggest they might mean a concentration instead.

When a **prospective student** asks about a program by name, they may not know the full landscape. \
If the named program is a concentration within a broader degree (e.g., Cybersecurity is both a standalone \
BS and a CS concentration), briefly clarify both options and ask which they are interested in.

## Course Disambiguation

If the retrieved context contains two courses with the same or very similar name but different levels \
(e.g., CSCI 4250 Human-Computer Interaction and CSCI 5250 Advanced HCI), ask the student which one they \
mean before giving a detailed answer. Briefly name both options and ask whether they are looking for the \
undergraduate or graduate version.

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
