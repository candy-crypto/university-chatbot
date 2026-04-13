# NMSU CS Department Chatbot — Catalog Ingestion Scripts
## Handoff Documentation

---

## Overview

Three Python scripts parse the NMSU 2025–2026 Academic Catalog PDF and upload
structured text chunks to a Weaviate vector database for use by the CS
Department chatbot. Two scripts do the parsing and uploading; the third is a
Jupyter-based inspection tool used during development and validation.

**Scripts:**

| File | Purpose |
|------|---------|
| `nmsu_catalog_chunker.py` | Parses structural content (degrees, study plans, policy, research, etc.) |
| `nmsu_course_chunker.py` | Parses end-of-catalog course descriptions (pages 1380–2067) |
| `nmsu_catalog_chunk_inspector.py` | Jupyter inspection helper — not part of the ingestion pipeline |

**Input:** `25-26_New_Mexico_State_University_-_Las_Cruces.pdf` (2068 pages)

**Output:** ~5,700 `CatalogChunk` objects uploaded to a Weaviate collection
named `CatalogChunk`, each with text content and structured metadata.

---

## Dependencies

```
pdfplumber>=0.10
weaviate-client>=4.0
```

Install:
```bash
pip install pdfplumber "weaviate-client>=4.0"
```

An embedding model is also required at upload time. The scripts accept any
function with the signature `fn(list[str]) -> list[list[float]]`. The project
used `sentence-transformers` during development:

```bash
pip install sentence-transformers
```

---

## Page Offset Convention

The catalog PDF has an unnumbered cover page at index 0. Catalog page
numbering begins on the following page:

```
pdf.pages[0]  →  cover (unnumbered)
pdf.pages[1]  →  catalog page 1
pdf.pages[N]  →  catalog page N
```

The 0-based `pdfplumber` index equals the catalog page number directly.
All page references throughout the codebase use catalog page numbers.

---

## Script 1 — `nmsu_catalog_chunker.py`

### What it does

Parses structural content from specific page ranges throughout the 2068-page
catalog and produces 195 chunks covering the CS department's full academic
programs, plus supporting university content. It reads the catalog Table of
Contents (pages 2–13) to build a page-range map, then applies targeted
parsers to each designated section.

**Content covered and chunk counts:**

| Chunk type | Count | Catalog pages | Content |
|---|---|---|---|
| `policy` | 78 | 60–73 | Transfer students, veterans, international, financial aid |
| `grad_program_info` | 37 | 74–79 | Graduate School admissions, MAP program, assistantships |
| `research` | 35 | 80–85 | All NMSU research facilities and labs (one per facility) |
| `dept_intro` | 1 | 565–580 | CS department narrative, MAP description |
| `program_index` | 1 | 567 | Full list of CS programs with online URLs |
| `minor_index` | 1 | 567 | CS minor eligibility rules and listing |
| `faculty` | 1 | 567 | CS faculty roster with research interests |
| `degree_core_requirement` | 1 | 584–585 | Shared CS BS requirements (gen-ed, core CSCI, non-dept) |
| `concentration_requirement` | 7 | 586–608 | Concentration-specific courses for each of 7 BS concentrations |
| `degree_requirement` | 9 | various | Full requirements for CS BA, Secondary Ed BA, Cybersecurity BS, CS MS, Data Analytics MDA, Bioinformatics MS, CS PhD, EE AI BS |
| `study_plan` | 14 | various | Semester roadmaps (one per degree/concentration; Bioinformatics has both tracks in one chunk) |
| `second_language` | 2 | 583–586 | Second language requirement statements for BA and BS |
| `minor_requirement` | 8 | 608–609, 230 | 4 CS undergraduate minors + CS, Bioinformatics (with CS), Communication Studies, and Economics graduate minors |

**Key design decisions:**

- The CS BS core requirements (gen-ed, shared CSCI core, math/science requirements)
  are stored once as `degree_core_requirement`. Each concentration stores only
  its specific courses as `concentration_requirement`. This avoids 7 near-identical
  large chunks that would produce near-duplicate retrieval results for queries
  about CS BS requirements.

- Study plans include their heading as the first line of text so each chunk is
  self-labeling. The Bioinformatics study plan merges both the CS-background and
  non-computing-background tracks into one chunk to prevent duplicate-metadata
  retrieval waste.

- The `referenced_courses` metadata field on every chunk captures all course
  codes (regex `[A-Z]+ NNNN`) appearing anywhere in the text, enabling queries
  like "which degrees require CSCI 4405?"

### Usage

```python
from nmsu_catalog_chunker import run_pipeline
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
embed_fn = lambda texts: model.encode(texts).tolist()

# Dry run — parse only, no upload
chunks = run_pipeline(
    pdf_path="25-26_New_Mexico_State_University_-_Las_Cruces.pdf",
    dry_run=True,
)

# Full run — embed and upload to Weaviate
chunks = run_pipeline(
    pdf_path="25-26_New_Mexico_State_University_-_Las_Cruces.pdf",
    weaviate_url="http://localhost:8080",
    weaviate_api_key=None,          # set if using Weaviate Cloud
    embed_fn=embed_fn,
    dry_run=False,
)
```

The Weaviate collection `CatalogChunk` is created automatically on first run
if it does not already exist. See the `CatalogChunk` dataclass and
`upload_to_weaviate()` in the script for the full schema.

**Expected runtime:** 2–4 minutes on a standard machine.

---

## Script 2 — `nmsu_course_chunker.py`

### What it does

Parses end-of-catalog course descriptions (catalog pages 1380–2067, ~688 pages)
into approximately 5,500 `course_description` chunks, one per course entry.
Every course in the catalog is included regardless of department prefix —
relevance filtering happens at query time in the vector store, not at ingestion.

Each chunk contains the course code, title, credit hours, description text,
prerequisites, and learning outcomes exactly as written in the catalog. The
`dept_prefix` metadata field (e.g. `CSCI`, `MATH`, `E E`) enables
prefix-filtered queries.

**Why all courses, not just CS-prefix courses:**

CS degree programs require courses across many departments — MATH, STAT, BIOL,
PHYS, CHEM, COMM, HNRS, and others. A student asking about prerequisites or
course content for a required non-CSCI course needs those descriptions too.
Filtering at ingestion silently removes content that students will legitimately
ask about.

**Processing approach:**

The 688-page course section is divided into blocks of 50 pages. Each block
opens and closes the PDF independently to keep peak memory usage flat (one
block in memory at a time, not the full section). Course prefix attribution
comes from the course code in each entry, not from block position.

Pages 1377–1379 are the course section preamble/index and are skipped.
Clean course entries begin at page 1380.

### Usage

```python
from nmsu_course_chunker import run_course_pipeline
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
embed_fn = lambda texts: model.encode(texts).tolist()

# Dry run — parse only, no upload (~6 minutes)
chunks = run_course_pipeline(
    pdf_path="25-26_New_Mexico_State_University_-_Las_Cruces.pdf",
    dry_run=True,
)

# Full run — embed and upload
chunks = run_course_pipeline(
    pdf_path="25-26_New_Mexico_State_University_-_Las_Cruces.pdf",
    weaviate_url="http://localhost:8080",
    weaviate_api_key=None,
    embed_fn=embed_fn,
    dry_run=False,
    block_size=50,      # increase to 100 if your environment has ample RAM
)
```

**Expected runtime:** 6–8 minutes (parse only) or 15–20 minutes (parse + embed
+ upload) on a standard machine.

---

## Running Both Scripts Together

Both scripts write to the same Weaviate collection (`CatalogChunk`) using the
same schema. Run the structural chunker first (it creates the collection), then
the course chunker. Duplicate detection uses a SHA-256 hash of chunk text, so
re-running either script is safe — existing chunks are skipped.

```python
from nmsu_catalog_chunker import run_pipeline
from nmsu_course_chunker import run_course_pipeline
from sentence_transformers import SentenceTransformer

PDF  = "25-26_New_Mexico_State_University_-_Las_Cruces.pdf"
URL  = "http://localhost:8080"
model    = SentenceTransformer("all-MiniLM-L6-v2")
embed_fn = lambda texts: model.encode(texts).tolist()

# Step 1: structural content (~2-4 min)
structural = run_pipeline(
    pdf_path=PDF, weaviate_url=URL,
    embed_fn=embed_fn, dry_run=False,
)

# Step 2: course descriptions (~15-20 min)
courses = run_course_pipeline(
    pdf_path=PDF, weaviate_url=URL,
    embed_fn=embed_fn, dry_run=False,
)

print(f"Total uploaded: {len(structural) + len(courses)} chunks")
```

---

## Weaviate Collection Schema

Both scripts share the `CatalogChunk` dataclass defined in
`nmsu_catalog_chunker.py`. The collection is created with these properties:

| Property | Type | Description |
|---|---|---|
| `text` | TEXT | Full chunk text content |
| `chunk_type` | TEXT | See chunk types below |
| `catalog_page` | INT | Starting catalog page number |
| `catalog_page_end` | INT | Ending catalog page number |
| `chunk_id` | TEXT | SHA-256 hash of text (dedup key) |
| `dept_name` | TEXT | Department name |
| `program_family` | TEXT[] | e.g. `["computer_science"]` |
| `degree_level` | TEXT | `undergraduate`, `ms`, `phd` |
| `degree_type` | TEXT | `ba`, `bs`, `ms`, `phd`, `minor` |
| `concentration` | TEXT | e.g. `cybersecurity`, `general` |
| `degree_full_title` | TEXT | Full degree title as in catalog |
| `course_code` | TEXT | e.g. `CSCI 4405` |
| `course_title` | TEXT | Course title |
| `credits` | TEXT | Credit hours |
| `dept_prefix` | TEXT | e.g. `CSCI`, `MATH`, `E E` |
| `course_number_level` | TEXT | `undergraduate` or `graduate` |
| `has_prerequisites` | BOOL | Whether prerequisites exist |
| `policy_topic` | TEXT | Section heading for policy chunks |
| `lab_name` | TEXT | Facility name for research chunks |
| `referenced_courses` | TEXT[] | All course codes found in text |
| `catalog_year` | TEXT | `2025-2026` |
| `source_scope` | TEXT | `catalog` |
| `is_research_related` | BOOL | True for research and thesis chunks |

**Chunk types:** `policy`, `grad_program_info`, `research`, `dept_intro`,
`program_index`, `minor_index`, `faculty`, `degree_core_requirement`,
`concentration_requirement`, `degree_requirement`, `study_plan`,
`second_language`, `minor_requirement`, `course_description`

---

## Script 3 — `nmsu_catalog_chunk_inspector.py`

### What it does

A Jupyter notebook helper for validating and exploring parsed chunks. It is
**not** part of the production ingestion pipeline and does not need to be
deployed. It calls `run_pipeline()` automatically on import and exposes
convenience functions for browsing the output.

Useful functions:

```python
from nmsu_catalog_chunk_inspector import *

# Already called on import:
# chunks = run_pipeline(pdf_path=PDF, dry_run=True, include_courses=False)

# Show all chunks of a type
show_all("degree_requirement")
show_all("study_plan")

# Find a specific degree
show_degree("Cybersecurity")
show_degree("Doctor of Philosophy")

# Find a specific course (after running with include_courses=True)
show_courses("CSCI", n=5)

# Which degrees reference a particular course?
find_by_course_code("CSCI 4405")
find_by_course_code("MATH 1511G")

# Quick health check — flags missing labels, suspiciously short chunks
check_quality()
```

---

## Environment Notes for Docker Integration

- The PDF must be accessible at the path passed to `pdf_path`. Mount it as a
  volume or copy it into the container.
- `weaviate_url` should point to the Weaviate container's internal address,
  e.g. `http://weaviate:8080` if both containers are on the same Docker network.
- `weaviate_api_key` is only needed for Weaviate Cloud; leave as `None` for a
  local Docker deployment.
- The vectorizer is set to `none` — the scripts supply their own vectors via
  `embed_fn`. Weaviate does not need a vectorizer module enabled.
- Both scripts are safe to re-run: chunk IDs are content hashes, so re-running
  against an existing collection produces no duplicates.

---

## File Summary

```
nmsu_catalog_chunker.py        1331 lines   Structural content parser + Weaviate uploader
nmsu_course_chunker.py          305 lines   Course description parser + Weaviate uploader
nmsu_catalog_chunk_inspector.py  ~80 lines  Jupyter validation helper (not for deployment)
```

Catalog PDF: `25-26_New_Mexico_State_University_-_Las_Cruces.pdf` (2068 pages)
