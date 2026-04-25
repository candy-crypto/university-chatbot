# Open Issues — University Chatbot

Last updated: 2026-04-25  
Branch: BarbaraA-Reed-retrieval-annotate

Items marked **[done]** were implemented in code but not yet re-ingested.
Items marked **[re-ingest needed]** require a pipeline re-run to take effect.

---

## retrieval.py

- [x] **Hard Weaviate level filter** — `Filter.by_property('level')` OR `level==''` added. **[done]**

- [x] **Minor_requirement hard Weaviate filter** — `contains_any` filter for minor queries. **[done]**

- [x] **Exclude minor_requirement for non-minor queries** — `contains_none` filter for degree queries. **[done]**

- [x] **Bidirectional chunk_type filtering + dynamic TOP_K for course topic queries** — TOP_K raised to 12 for "what courses teach X?" queries. **[done]**

- [x] **Dynamic TOP_K for category enumeration queries** — TOP_K raised by category when no specific comparison signal present: concentrations→15, minors→12, degrees/scholarships→10. Fires on category term alone; suppressed when "between/vs/versus" present (specific comparison). **[done]**

- [x] **Study_plan concentration penalty** — study_plan chunks whose concentration is not mentioned in the query receive -0.20 boost penalty, preventing all 11 concentration plans from flooding results for a plain "BS in CS" query. **[done]**

- [x] **Add "difference" / "compare" / "between" to `_requirement_terms`**. **[done]**

- [x] **Fix stopword contamination in `tokenize()`** — `_STOPWORDS` frozenset added. **[done]**

- [x] **Fix "cs" abbreviation in heading boost** — excluded from heading token matching (same as course code matching) to prevent spurious +0.08 on off-topic FAQ chunks. **[done]**

- [x] **Temporal query expansion** — `expand_temporal_query()` rewrites relative time references to absolute semester/year strings before retrieval ("this summer" → "Summer 2026", "next semester" → "Fall 2026", etc.). Current date and semester injected into system prompt. **[done]**

- [x] **Acronym expansion** — `expand_acronyms()` appends full forms of common CS acronyms before retrieval (AI→Artificial Intelligence, ML→Machine Learning, HCI→Human Computer Interaction, etc.) so BM25 matches catalog headings. **[done]**

- [x] **CAASS / glossary boost fix (partial)** — glossary boost now conditional on `query_tokens ∩ glossary_heading_tokens`. Prevents advisor glossary chunk boosting for "What is CAASS?". Full acronym expansion for CAASS still needed. *(adv_001)*

- [x] **course_schedule retrieval gap** — `_SCHEDULE_TERMS` extended with "next", "available", "availability"; `_WHEN_RE` regex added to catch "when is/will/does" whose key word "when" is a stopword that `tokenize()` drops. Both checks combined in the course_schedule boost condition. **[done]**

- [x] **Disambiguation — course level** — system prompt instructs LLM to ask which version when two similarly-named courses at different levels appear in context. **[done]**

- [x] **Disambiguation — degree vs. concentration** — system prompt instructs LLM to take current student's stated major at face value; for prospective students, clarify when a name applies to both a standalone degree and a concentration. **[done]**

- [x] **`policy` chunk_type boost** — `_GEN_ED_TERMS` frozenset added; policy/grad_program_info chunks get extra +0.08 for Gen Ed/VWW queries; study_plan chunks get -0.12 penalty when Gen Ed/VWW terms present (to suppress study plans that list VWW courses from outranking the policy definition pages). **[done]** *(req_001, req_002)*

- [ ] **OECS false positive (avail_006)** — root cause identified: OECS 125 "Operating Systems" has exact BM25 phrase match. Existing course_schedule boost (+0.15 when "offered"/"semester" in query) already promotes the rotation table to rank 1. OECS chunk may still appear in top 5 but LLM disambiguation handles it. No retrieval fix needed.

- [x] **Faculty chunk_type boost** — `_FACULTY_TERMS` frozenset added; faculty chunks get +0.15 when query contains faculty/person/contact terms. *(fac_001, fac_003)* **[done]**

- [x] **CAASS acronym expansion (full fix)** — `CAASS → "Center for Academic Advising and Student Support"` added to `_ACRONYM_MAP`. BM25 will now match the full name in the College of A&S intro and advising FAQ chunks. **[done]** *(adv_001)*

- [x] **Enrollment chunk_type boost for apply/application queries** — `_ENROLLMENT_TERMS` frozenset added; enrollment chunks get +0.12 when apply/admission/registration terms present. **[done]** *(pol_003)*

---

## ingest.py

- [x] **FAQ Q&A chunking** — `_expand_qa_sections()` splits sections with 2+ Q\d+ markers into one chunk per Q&A pair. **[ingested 2026-04-24]**

- [x] **Faculty directory dynamic rendering** — Playwright `wait_for_selector` loop added for faculty CSS selectors. **[ingested 2026-04-24]**

- [x] **Whitelist-only crawl** — BFS-discovered links no longer queued; only `pages` + `seed_urls` visited. **[ingested 2026-04-24]**

- [x] **Per-page browser isolation** — fresh Playwright page opened and closed per URL so a failed/redirected navigation cannot corrupt subsequent crawl requests. **[ingested 2026-04-24]**

---

## nmsu_catalog_chunker.py

- [x] **Two-column PDF parsing — heading fragment fix** — character midpoint used for column detection; fixes CSCI 5250 parsed as "SCI 4250". **[ingested 2026-04-24]**

- [ ] **Two-column PDF parsing — column-order section boundary** — pp.41-42: when a section heading appears in the right column while the left column still has content below that y-position, `get_page_lines()` emits left-column lines first, so the preceding chunk absorbs a few lines that visually belong to the next section. Requires restructuring to process lines in y-bands across both columns — deferred. pp.213-214 (Range Science PhD) cited in earlier notes are **not processed** and not affected. *(req_012, deg_009)*

- [x] **Bioinformatics entrance requirements not chunked** — `chunk_dept_intro()` now splits on five additional headings (MAP, Graduate Program Information, Entrance Requirements for CS, Entrance Requirements for Bioinformatics, Graduate Assistantships) mapped to `grad_program_info`. Multi-line headings buffered and joined before lookup; Unicode apostrophe in "Master's" handled. **[re-ingest needed]** *(pol_003)*

---

## cs.yaml

- [x] **Whitelist-only crawl** — renamed `page_types:` → `pages:`. **[done]**

- [x] **non-majors/minors.html level tag** — removed `undergraduate` level tag so graduate students are not penalized when asking about minor value/options. **[ingested 2026-04-24]**

---

## Pipeline re-runs needed

- [x] **Re-run web ingest** — completed 2026-04-24 (43 pages, 172 chunks)

- [x] **Re-run catalog ingest** — completed 2026-04-24 (5,901 chunks)

- [x] **Update TBD chunk IDs in ground_truth.yaml** — resolved 2026-04-24; 3 remain as known chunker gaps (adv_001, pol_003 p.566, fac_003)

- [x] **Re-run catalog ingest** — completed 2026-04-25 (5,906 chunks; +5 from p.566 grad_program_info splits)

---

## Ground truth review

- [x] **Full review pass completed** — pol_001 through other_001 reviewed and recorded (2026-04-24)

- [x] **TBD chunk IDs** — resolved 2026-04-24; 3 remain as known chunker gaps (adv_001 p.447 CAASS, pol_003 p.566 bioinformatics entrance, fac_003 dynamic faculty rendering)
