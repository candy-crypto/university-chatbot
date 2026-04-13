# Unified Weaviate Schema — DepartmentChunk Collection
## University Chatbot — NMSU CS Department
### Version: 2025-04-13

---

## Weaviate Collection Name
`DepartmentChunk`

All content — catalog chunks and web page chunks — is stored in this single collection.
Fields not applicable to a given source are stored as empty/null.

---

## Core Fields — Every Chunk

| Field | Type | Source | Description |
|---|---|---|---|
| `chunk_id` | text | both | Unique identifier. Catalog: SHA-256 hash of text. Web: URL + chunk index. |
| `chunk_type` | text | both | Content category. See Chunk Type Values below. |
| `department_id` | text | both | Short department code, e.g. `"cs"` |
| `campus` | text | both | Institution identifier, e.g. `"las_cruces"`. Reserved for multi-campus expansion. |
| `text` | text | both | The full chunk content. |
| `heading` | text | both | Human-readable label for this chunk. Web: h1 banner. Catalog: derived from chunk_type (see Heading Derivation below). |
| `source` | text | both | Citation identifier. Web: full URL. Catalog: `"NMSU Academic Catalog 2025-2026, p.568"` |
| `level` | text | both | Audience or content level: `"undergraduate"`, `"graduate"`, `"both"`, or empty. Interpretation depends on chunk_type. |
| `degree_type` | text | both | Specific program type when applicable: `"ba"`, `"bs"`, `"ms"`, `"phd"`, `"minor"`, `"certificate"`, or empty. |
| `course_code` | text | both | Full course code including prefix, e.g. `"CSCI 1115G"`. Empty if chunk does not represent a single course. |
| `referenced_courses` | text array | both | All course codes mentioned anywhere in the chunk text, e.g. `["CSCI 1115G", "MATH 1511G"]`. Empty array if none. |

---

## Catalog-Specific Fields
Empty for web chunks unless page review determines otherwise.

| Field | Type | Description |
|---|---|---|
| `catalog_year` | text | Catalog edition, e.g. `"2025-2026"` |
| `catalog_page` | int | PDF page number where chunk begins. Used for citation construction and debugging. |
| `catalog_page_end` | int | PDF page number where chunk ends. Equal to `catalog_page` for single-page chunks. Used for debugging multi-page chunks. |
| `degree_full_title` | text | Complete degree title, e.g. `"Computer Science (Cybersecurity) - Bachelor of Science"` |
| `concentration` | text | Degree concentration, e.g. `"cybersecurity"`, `"general"`, `"hci"` |
| `credits` | text | `course_description` chunks only. Credit hours, e.g. `"3"`. |
| `has_prerequisites` | bool | `course_description` chunks only. True if the course lists prerequisites. |
| `policy_topic` | text | `policy` and `grad_info` chunks only. Subject label, e.g. `"Academic Probation"`. |
| `lab` | text | `research` chunks only. Facility name, e.g. `"Innovative Computing Laboratory"`. |
| `research` | bool | True if chunk relates to research activities. |

---

## Web-Specific Fields
Empty for catalog chunks.

| Field | Type | Description |
|---|---|---|
| `crawl_version` | text | Timestamp of the ingestion run that produced this chunk, e.g. `"20250413_143022"`. Used for replacing stale chunks on re-crawl and debugging. |

---

## Chunk Type Values

| Value | Description | Primary Source |
|---|---|---|
| `course_description` | Individual course entry with description, credits, prerequisites | Catalog |
| `degree_requirement` | Formal course requirements for a specific degree | Catalog |
| `study_plan` | Semester-by-semester degree roadmap | Catalog |
| `minor_requirement` | Requirements for a minor | Catalog |
| `minor_index` | Listing of available minors and eligibility | Catalog |
| `program_index` | Listing of degrees offered by the department | Catalog |
| `dept_intro` | Department overview and mission narrative | Catalog / Web |
| `policy` | University or department policy | Catalog / Web |
| `grad_info` | Graduate school information and requirements | Catalog / Web |
| `research` | Research facilities and labs | Catalog / Web |
| `course_schedule` | Course offering schedule or rotation | Web |
| `faculty` | Faculty and staff listings | Web |
| `general` | Fallback for pages not matching a specific type | Web |

---

## Heading Derivation (Catalog Chunks)

The `heading` field is populated based on `chunk_type`:

| chunk_type | heading source |
|---|---|
| `course_description` | `course_code` + course name from text |
| `degree_requirement` | `degree_full_title` |
| `study_plan` | `degree_full_title` |
| `minor_requirement` | `degree_full_title` |
| `policy` | `policy_topic` |
| `grad_info` | `policy_topic` |
| `research` | `lab` |
| `dept_intro`, `program_index`, `minor_index` | `dept_name` (full department name) |

---

## Course Lookup Table (SQLite — db.py)

A separate relational table for fast bidirectional course identity resolution.
Used for: code → title lookup, title → code lookup, and as the source for
semantic course search (searching by topic against titles is supplemented by
full semantic search against course_description chunks in Weaviate).

| Column | Type | Description |
|---|---|---|
| `course_code` | text (PK) | e.g. `"CSCI 1115G"` |
| `course_title` | text | e.g. `"Introduction to Computer Science I"` |
| `department_id` | text | e.g. `"cs"` |
| `campus` | text | e.g. `"las_cruces"` |
| `catalog_year` | text | e.g. `"2025-2026"` |

Populated during catalog ingestion from `course_description` chunks.

---

## Fields Deliberately Excluded

| Field | Reason |
|---|---|
| `url` | Redundant — URL is now stored in `source` |
| `title` | Replaced by `heading` (h1 banner, more meaningful than browser tab title) |
| `section` | Replaced by `chunk_type` and `heading` |
| `tags` | Unreliable (most pages lack meta keywords); signal already covered by `heading` |
| `course_title` | Moved to course lookup table; would be empty on most chunks |
| `dept_prefix` | Derivable from `course_code`; redundant with `department_id` |
| `program_family` | Redundant with `department_id` |
| `degree_level` | Renamed and consolidated into `level` |
| `course_number_level` | Consolidated into `level` |
| `course_number` | Renamed to `course_code` for consistency with catalog |
| `is_research_related` | Renamed to `research` |
| `lab_name` | Renamed to `lab` |

---

## cs.yaml — New Fields Required

```yaml
campus: "las_cruces"

page_types:
  "https://computerscience.nmsu.edu/example-url": "degree_requirement"
  # ... one entry per page, assigned manually

page_levels:
  "https://computerscience.nmsu.edu/example-url": "undergraduate"
  # undergraduate | graduate | both

page_degree_types:
  "https://computerscience.nmsu.edu/example-url": "bs"
  # ba | bs | ms | phd | minor | certificate | (omit if not applicable)
```
