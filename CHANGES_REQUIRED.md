# Required Changes — University Chatbot Refactor
## NMSU CS Department Chatbot
### Version: 2025-04-13

---

## Background and Motivation

This document describes all changes required to align the chatbot's ingestion,
retrieval, and storage systems around a unified schema. The changes address
three root problems discovered during schema review:

### Problem 1: Two Incompatible Data Sources
The catalog chunkers and the web ingestion script were built independently.
They produce chunks with different field names for the same concepts, use
different Weaviate collection names, and use different embedding models.
As a result, catalog data and web data cannot be queried together — the
retrieval system currently sees only web content.

### Problem 2: Mismatched Embedding Models
The catalog chunkers use SentenceTransformer (`all-MiniLM-L6-v2`) while the
web ingestion and retrieval scripts use OpenAI (`text-embedding-3-small`).
Semantic search compares the query vector against stored document vectors.
If they come from different models, the comparison is meaningless — like
measuring distance in miles on one side and kilometers on the other.
Since the retrieval script embeds queries with OpenAI and cannot change
without redesigning retrieval, all documents must also use OpenAI.

### Problem 3: Retrieval Design Inefficiency
The retrieval script currently fetches up to 200 raw chunks from Weaviate
just to re-rank them locally using a custom Python BM25 implementation.
For a live chatbot, this adds significant latency per query. Weaviate's
native hybrid search performs BM25 + vector search + RRF fusion at the
database level in a single query, which is faster and simpler.

---

## File-by-File Changes

---

### cs.yaml (and future department YAML files)

**Changes required:**
1. Add `campus` field at the top level (e.g. `"las_cruces"`)
2. Add `page_types` mapping: one entry per web page, URL → chunk_type value
3. Add `page_levels` mapping: URL → level value (`"undergraduate"`, `"graduate"`, `"both"`)
4. Add `page_degree_types` mapping: URL → degree_type value where applicable

**Why:**
Web pages do not carry structured metadata about what type of content they
contain or which student audience they target. Rather than trying to infer
this automatically from page titles (unreliable), these values are assigned
manually per URL in the YAML. With only 55-57 pages this is practical and
produces more accurate classification than any automated approach.

**Who does this:**
Barbara — manual review of all web pages required to assign correct values.
This is task-dependent on seeing the full page list from the crawl.

**Note on chunk_type values:**
Use the values defined in UNIFIED_SCHEMA.md as a starting point. These cover
catalog content types and the web page types identified so far. However, the
manual page review will likely reveal additional content categories not yet
represented — for example, advising information, enrollment procedures,
scholarship and financial aid pages, contact and office hours pages, and
others. Any new chunk_type values discovered during the page review must be:
  1. Added to the chunk_type list in UNIFIED_SCHEMA.md and UNIFIED_SCHEMA.xlsx
  2. Documented here so the development team is aware
  3. Considered for whether a catalog equivalent exists that should share the
     same chunk_type value for consistent cross-source filtering

The chunk_type list should be treated as a living document until the page
review is complete.

---

### nmsu_catalog_chunker.py

**Changes required:**

1. **Switch embedding model from SentenceTransformer to OpenAI**
   - Remove: `from sentence_transformers import SentenceTransformer`
   - Add: OpenAI client initialization (same pattern as ingest.py)
   - Update `upload_to_weaviate` to call OpenAI embeddings API in batches
   - Why: See Problem 2 above. Consistency with the retrieval query embedder
     is non-negotiable for semantic search to produce valid results.

2. **Change Weaviate collection name**
   - Change `WEAVIATE_CLASS = "CatalogChunk"` to `WEAVIATE_CLASS = "DepartmentChunk"`
   - Why: All content must live in one collection for unified retrieval.
     The retrieval script only queries `DepartmentChunk`.

3. **Rename `dept_name` to `department_id`, use short code**
   - Change stored value from `"Computer Science"` to `"cs"`
   - Why: The retrieval script filters all queries by `department_id`.
     This filter will never match catalog chunks unless the field name
     and value format match what the retrieval script expects.

4. **Add `source` field**
   - Constructed as: `f"NMSU Academic Catalog {CATALOG_YEAR}, p.{catalog_page}"`
   - Why: The unified schema uses `source` as the citation field for all
     chunks. For catalog chunks there is no URL, so the catalog year and
     page number serve as the human-readable citation shown to users.

5. **Add `heading` field**
   - Add `derive_heading()` function that maps chunk_type to the appropriate
     descriptive field (see UNIFIED_SCHEMA.md Heading Derivation table)
   - Why: The unified schema requires a single consistent field for the
     human-readable label of each chunk, used for keyword scoring boost
     in retrieval. The catalog stores this information in different fields
     depending on chunk type; `heading` normalizes this.

6. **Rename `degree_level` to `level`, update values**
   - Change `"ms"` and `"phd"` → `"graduate"`
   - Keep `"undergraduate"`
   - Add `"both"` for chunks relevant to all students
   - Why: `degree_level` and `course_number_level` served overlapping
     purposes. A single `level` field with consistent values works for
     degrees, courses, policies, and web pages alike. The chunk_type
     provides the context needed to interpret what level means for a
     given chunk.

7. **Drop `program_family` field**
   - Why: Redundant with `department_id`. Adds no retrieval value.

8. **Drop `course_number_level` field**
   - Why: Consolidated into `level`. For course_description chunks,
     `level` is set to `"undergraduate"` or `"graduate"` based on
     course number (below or above 5000), same logic as before.

9. **Drop `course_title` from Weaviate storage**
   - Note: `course_title` is still extracted during parsing and written
     to the course lookup table in db.py. It is not stored as a Weaviate
     field because it would be empty on nearly all non-course chunks,
     and course identity resolution is handled by the lookup table.
   - Why: See course lookup table rationale below.

10. **Add `campus` field**
    - Hardcoded as `"las_cruces"` for this catalog
    - Why: Future-proofing for multi-campus and multi-catalog support.
      A schema migration later is expensive; adding the field now costs nothing.

11. **Update `_parse_degree_meta` function**
    - Produce `level` (`"undergraduate"` or `"graduate"`) instead of
      `degree_level` with ms/phd values
    - Ensure `degree_type` values are: `"ba"`, `"bs"`, `"ms"`, `"phd"`,
      `"minor"`, `"certificate"`, or empty string (not None)
    - Why: Cleaner separation — `level` answers "who is this for",
      `degree_type` answers "which specific program".

12. **Update Weaviate collection schema in `upload_to_weaviate`**
    - Remove: `program_family`, `course_number_level`, `course_title`, `dept_name`
    - Add: `heading`, `source`, `campus`
    - Rename: `degree_level` → `level`
    - Why: Schema must match the unified field list in UNIFIED_SCHEMA.md

---

### nmsu_course_chunker.py

**Changes required:**

1. **Align with CatalogChunk dataclass changes**
   - Since this script imports `CatalogChunk` from `nmsu_catalog_chunker.py`,
     most field changes flow through automatically once the dataclass is updated
   - Verify that `level` is set correctly for each chunk:
     course number >= 5000 → `"graduate"`, otherwise `"undergraduate"`
   - Verify `department_id` is set to short code (e.g. `"cs"`), not full name
   - Verify `campus` is set to `"las_cruces"`

2. **Write course_title to lookup table during pipeline run**
   - After chunks are created, extract `course_code` + `course_title` pairs
     and write to the course lookup table in db.py
   - Why: The lookup table is the designated home for course identity data.

---

### db.py

**Changes required:**

1. **Add course lookup table**
   - Table name: `course_lookup`
   - Columns: `course_code` (PK), `course_title`, `department_id`, `campus`, `catalog_year`
   - Add to `init_db()` so it is created alongside existing tables

2. **Add `upsert_course_lookup(courses)` function**
   - Accepts a list of (course_code, course_title, department_id, campus, catalog_year) tuples
   - Inserts or replaces on conflict with course_code
   - Called during catalog ingestion after chunks are parsed

3. **Add `lookup_course_by_code(code)` function**
   - Returns course_title for a given course_code, or None if not found

4. **Add `lookup_course_by_title(title)` function**
   - Returns course_code for an exact or close title match

**Why:**
Course title to code (and vice versa) resolution needs a fast, exact lookup
that does not require a full Weaviate vector search. A relational table is
the appropriate tool. Semantic queries about course topics ("courses about
algorithms") go directly to Weaviate against course_description chunk text,
since course titles alone are poor descriptors of actual content (e.g.,
"Artificial Intelligence I" covers algorithms extensively but the title
gives no clue).

---

### ingest.py

**Changes required:**

1. **Read new YAML fields during config load**
   - Read `campus`, `page_types`, `page_levels`, `page_degree_types`
     from the department YAML
   - Pass these through to the crawl and upsert functions

2. **Assign `chunk_type` from YAML during crawl**
   - Look up each page URL in `page_types` mapping
   - Assign matched value; fall back to `"general"` if URL not in mapping
   - Why: Web pages do not carry machine-readable content type signals.
     Manual classification via YAML is more reliable than automatic inference
     from page titles, and is practical given the small number of pages.

3. **Assign `level` and `degree_type` from YAML**
   - Look up URL in `page_levels` and `page_degree_types`
   - Store empty string if URL not in mapping
   - Why: Consistent with catalog fields, enables audience-based filtering
     in retrieval queries.

4. **Use h1 heading as `heading` field**
   - The page's first h1 element (already extracted into `headings` list)
     becomes the `heading` field value
   - Fall back to the first h2 if no h1 is present
   - Why: The browser tab title (currently stored as `title`) is cluttered
     with site name and department name repeated on every page. The h1
     banner contains the meaningful page label and produces better keyword
     matching signal in retrieval.

5. **Use URL as `source` field (not a separate `url` field)**
   - Store the page URL in `source`
   - Remove the separate `url` field
   - Why: The unified schema uses `source` as the citation field for all
     chunks. Having both `url` and `source` is redundant.

6. **Drop `title` field**
   - Replaced by `heading`

7. **Drop `section` field**
   - Replaced by `chunk_type` and `heading`

8. **Replace character-based chunking with structure-aware chunking**

   The current `chunk_text()` function cuts page text into 1000-character
   windows with 200-character overlap, with no awareness of headings,
   paragraphs, or topic boundaries. A chunk can easily span two unrelated
   sections or split mid-sentence. This produces low-quality retrieval
   results because a retrieved chunk may contain content irrelevant to
   the query simply because it fell within the character window.

   Replace with a chunking strategy driven by page structure and chunk_type:

   **Primary rule — split on h2/h3 heading boundaries:**
   Each h2 or h3 heading starts a new chunk. The heading text becomes
   the first line of the chunk, preserving context. This keeps related
   content together and aligns chunk boundaries with the page's own
   organizational structure.

   **Maximum length guardrail:**
   If a section exceeds a maximum character limit (suggested: 2000
   characters), split at the nearest paragraph boundary. Prevents
   oversized chunks on pages with long sections and no sub-headings.

   **Minimum length guardrail:**
   If a section is shorter than a minimum character limit (suggested:
   150 characters), merge it with the following section rather than
   creating a stub chunk. Prevents near-empty chunks from short
   transitional headings.

   **chunk_type aware strategy:**

   | chunk_type | Chunking approach |
   |---|---|
   | dept_intro, policy, grad_info, advising, general | Split on h2/h3 headings |
   | degree_requirement, study_plan, minor_requirement | Split on h2/h3 headings (semester or requirement group boundaries) |
   | course_schedule | Single chunk unless over maximum; split by academic year grouping if present |
   | faculty | One chunk per faculty member if structured; otherwise single chunk |

   **Why:**
   Arbitrary character windows produce chunks whose boundaries are
   meaningless to retrieval. Structure-aware chunking ensures each chunk
   represents a coherent unit of information — a policy, a requirement
   group, a schedule — that can stand alone as a useful answer to a
   student's question. The three-year course rotation page is a good
   example: it is one coherent reference document and should be stored
   as such, not arbitrarily split mid-row.

   **Note:** The h2/h3 headings extracted during crawl are already
   available in `extract_page_data`. No additional crawl changes are
   needed — only the chunking logic needs to change.

9. **Drop `tags` field**
   - Why: Most university web pages do not set HTML meta keywords tags.
     The fallback (words extracted from page title) is low-quality signal
     already covered by `heading`. Removing simplifies ingestion without
     meaningful retrieval loss.

9. **Rename `course_number` to `course_code`**
   - Why: Consistent naming with catalog field. Enables unified filtering
     and display across both sources.

10. **Fix course_code extraction regex**
    - Current pattern misses: 4-digit course numbers (e.g. `1115`),
      alphabetic suffixes (e.g. `G` in `CSCI 1115G`),
      space-separated prefixes (e.g. `E E 1110`)
    - Update to: `r"\b([A-Z][A-Z\s]{0,5}\s+\d{3,4}[A-Z]?)\b"`
    - Why: Real NMSU course codes include all these patterns. The current
      regex silently misses graduate courses (5000+ level) and some
      department prefixes.

11. **Extract `referenced_courses` from page text**
    - Apply updated course_code regex to full page text
    - Store all matches as a list in `referenced_courses`
    - Why: Enables queries like "which pages reference CSCI 1115G" and
      gives the LLM structured data to verify whether a course code
      appears as a formal requirement or a passing mention.

12. **Add `campus` field**
    - Read from YAML config
    - Why: Multi-campus support requires campus identification on every chunk.

13. **Add empty fields for catalog-specific schema fields**
    - Set `catalog_year`, `catalog_page`, `degree_full_title`,
      `concentration`, `credits`, `has_prerequisites`,
      `policy_topic`, `lab`, `research` to empty/null
    - Why: All chunks in the unified collection share the same schema.
      Empty fields on web chunks allow the collection to remain consistent
      without requiring separate handling in retrieval.

---

### retrieval.py

**Changes required:**

1. **Replace custom Python BM25 with Weaviate native hybrid search**
   - Remove: `bm25_score_documents()`, `normalize_scores()`,
     `reciprocal_rank_fusion()`, `fetch_objects` call, and all manual
     score fusion logic
   - Replace the two-query approach (near_vector + fetch_objects) with
     a single `collection.query.hybrid()` call
   - Why: The current approach fetches up to 200 chunks from Weaviate
     over the network, tokenizes them in Python, and scores them locally.
     For a live chatbot this adds latency on every query. Weaviate's
     native hybrid search performs BM25 + vector search + RRF fusion
     at the database level in a single round trip. Control over BM25
     parameters is not justified for a chatbot where response latency
     directly affects user experience.

2. **Update citation to use `source` field**
   - Change `chunk.get("url")` → `chunk.get("source")`
   - Why: The `url` field has been removed from the schema; `source`
     is now the citation field for all chunks.

3. **Update `RETURN_PROPERTIES` to unified schema fields**
   - Remove: `url`, `title`, `section`, `tags`, `course_number`, `crawl_version`
   - Add: `heading`, `level`, `degree_type`, `campus`, `catalog_year`,
     `catalog_page`, `course_code`, `referenced_courses`
   - Why: Must match the unified schema in UNIFIED_SCHEMA.md.

4. **Update `metadata_boost` function**
   - Replace `title` with `heading`
   - Replace `course_number` with `course_code`
   - Remove `tags` and `section` boost terms (fields no longer exist)
   - Why: Boost fields must exist in the schema to have any effect.

5. **Update `build_context` function**
   - Replace `title`, `section`, `url` references with `heading`,
     `chunk_type`, `source`
   - Why: Field names have changed; the context string passed to the
     LLM must reflect actual field values.

---

### weaviate_client.py

**Changes required:**

1. **Update `ensure_collection` property list**
   - Remove: `url`, `title`, `section`, `tags`, `course_number`, `crawl_version`
   - Add: `heading` (TEXT), `level` (TEXT), `degree_type` (TEXT),
     `campus` (TEXT), `catalog_year` (TEXT), `catalog_page` (INT),
     `course_code` (TEXT), `referenced_courses` (TEXT_ARRAY),
     `degree_full_title` (TEXT), `concentration` (TEXT),
     `credits` (TEXT), `has_prerequisites` (BOOL),
     `policy_topic` (TEXT), `lab` (TEXT), `research` (BOOL),
     `crawl_version` (TEXT)
   - Why: The collection schema must match the unified field list exactly.
     Fields missing from the schema cannot be stored or queried.

---

## Order of Implementation

The changes are interdependent. Recommended sequence:

1. `db.py` — add course lookup table (no dependencies)
2. `nmsu_catalog_chunker.py` — core dataclass and upload changes
3. `nmsu_course_chunker.py` — verify alignment with updated dataclass
4. `cs.yaml` — manual page review and mapping (Barbara)
5. `ingest.py` — web ingestion aligned to unified schema
6. `weaviate_client.py` — update collection schema
7. `retrieval.py` — update retrieval to unified schema and native hybrid search

**Important:** Steps 2-5 can proceed in parallel. Steps 6 and 7 depend on
the schema being finalized. The Weaviate collection should be dropped and
recreated after schema changes — do not attempt to migrate in place.
