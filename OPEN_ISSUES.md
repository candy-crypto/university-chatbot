# Open Issues — University Chatbot

Last updated: 2026-04-24  
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

- [x] **course_schedule retrieval gap** — "opportunity", "next time", "when is", "when will" added to `_schedule_terms`. **[done]**

- [x] **Disambiguation — course level** — system prompt instructs LLM to ask which version when two similarly-named courses at different levels appear in context. **[done]**

- [x] **Disambiguation — degree vs. concentration** — system prompt instructs LLM to take current student's stated major at face value; for prospective students, clarify when a name applies to both a standalone degree and a concentration. **[done]**

- [ ] **`policy` chunk_type boost** — policy chunks losing to study_plan chunks for VWW / Gen Ed queries (pp.241-244). Needs targeted boost trigger. *(req_001, req_002)*

- [ ] **OECS false positive (avail_006)** — non-CS course outranking CSCI 4120 for availability queries. Root cause not yet identified.

- [ ] **Faculty chunk_type boost for proper-name queries** — Playwright wait added but boost not yet implemented. *(fac_001, fac_003)*

- [ ] **CAASS acronym expansion (full fix)** — CAASS has no BM25 hits; needs acronym expansion or query expansion. Glossary conditional boost is partial fix only. *(adv_001)*

- [ ] **Enrollment chunk_type boost for apply/application queries** — pol_003 retrieval failure: catalog enrollment chunks not surfacing for "how do I apply" queries due to vocabulary mismatch. *(pol_003)*

---

## ingest.py

- [x] **FAQ Q&A chunking** — `_expand_qa_sections()` splits sections with 2+ Q\d+ markers into one chunk per Q&A pair. **[ingested 2026-04-24]**

- [x] **Faculty directory dynamic rendering** — Playwright `wait_for_selector` loop added for faculty CSS selectors. **[ingested 2026-04-24]**

- [x] **Whitelist-only crawl** — BFS-discovered links no longer queued; only `pages` + `seed_urls` visited. **[ingested 2026-04-24]**

- [x] **Per-page browser isolation** — fresh Playwright page opened and closed per URL so a failed/redirected navigation cannot corrupt subsequent crawl requests. **[ingested 2026-04-24]**

---

## nmsu_catalog_chunker.py

- [x] **Two-column PDF parsing — heading fragment fix** — character midpoint used for column detection; fixes CSCI 5250 parsed as "SCI 4250". **[ingested 2026-04-24]**

- [ ] **Two-column PDF parsing — cross-page section boundaries** — pp.41-42 and pp.213-214 split across page boundaries. Requires restructuring `lines_to_text()` — deferred. *(req_012, deg_009)*

---

## cs.yaml

- [x] **Whitelist-only crawl** — renamed `page_types:` → `pages:`. **[done]**

- [x] **non-majors/minors.html level tag** — removed `undergraduate` level tag so graduate students are not penalized when asking about minor value/options. **[ingested 2026-04-24]**

---

## Pipeline re-runs needed

- [x] **Re-run web ingest** — completed 2026-04-24 (43 pages, 172 chunks)

- [x] **Re-run catalog ingest** — completed 2026-04-24 (5,901 chunks)

- [ ] **Update TBD chunk IDs in ground_truth.yaml** — re-run `export_chunk_ids.py` and fill in TBD entries now that ingests are complete

---

## Ground truth review

- [x] **Full review pass completed** — pol_001 through other_001 reviewed and recorded (2026-04-24)

- [ ] **TBD chunk IDs** — multiple entries have TBD chunk IDs; ingests now complete, ready to update
