# Evaluation Component — Task Reference

## Overview

A 5-phase plan to evaluate the CS department chatbot RAG pipeline.  
**Tool stack:** pytest + OpenAI SDK (already installed) + jsonlines. No RAGAS, no LangChain.  
**Total test questions:** 50–60, created manually with human-verified key facts.

---

## Phase 1 — Ground Truth and Test Set
*No code. Most important phase. Do not skip or auto-generate with LLM.*

### Tasks
- [ ] Create `backend/eval/ground_truth.yaml` with 50–60 questions (see categories below)
- [ ] For each question, write 3–6 **key facts** (not a full answer — just the facts the answer must contain)
- [ ] Run `search_chunks()` directly against the live Weaviate collection to find actual chunk IDs for each question
- [ ] Record expected chunk IDs in `ground_truth.yaml`
- [ ] Validate that all expected chunk IDs exist in the collection before locking the file

### Question Categories

| # | Category | Count | Expected Source | Notes |
|---|----------|-------|-----------------|-------|
| 1 | Course descriptions | 10 | catalog | Credits, prereqs, lab component, course content. Include grad-only and prereq-heavy courses. |
| 2 | Degree requirements | 10 | catalog | Total credits, required core courses, elective hours, GPA, thesis vs non-thesis. Include study_plan and degree_requirements chunks. |
| 3 | Advising / prospective students | 8 | web | Who to contact, how to apply, transfer credits, undergrad and grad advising. |
| 4 | Financial aid & assistantships | 6 | web | Scholarships, TAs, RAs, undergrad and grad. Include TPAL and ECE external pages. |
| 5 | Faculty & research | 6 | web | Research groups, research areas, contact info. |
| 6 | Banner redirect — enrollment availability | 5 | **redirect only** | Current seats, open sections, registration. Correct answer = Banner URL, not content. |
| 7 | Policy | 5 | either | Academic integrity, grade appeal, transfer credit, prereq waivers. |
| 8 | Cross-source questions | 5 | both | Questions requiring both catalog and web content (e.g., requirements + how to apply). |
| 9 | Catalog metadata | 5 | catalog | Tests citation format: catalog year, page range. |

### Audience Groups

The system prompt must list these groups explicitly so the LLM knows to cover all applicable ones when a question is ambiguous. For persona-sensitive questions in the ground truth, record which groups apply.

| Group | Typical Information Needs |
|-------|--------------------------|
| Prospective undergraduates | Admissions, program overview, why CS at NMSU |
| Current undergraduates | Degree requirements, advising, registration, financial aid |
| Transfer students | Credit transfer rules, NMSU-affiliated community college articulation |
| Prospective MAP students | Eligibility for the accelerated BS→MS path, how to apply |
| Current MAP students | Combined program requirements, timeline, advising |
| Prospective MS students | Admissions, thesis vs coursework tracks, program overview |
| Current MS students — coursework track | Requirements, electives, graduation |
| Current MS students — thesis track | Requirements, thesis process, advisor, graduation |
| Prospective PhD students | Admissions, funding, research fit |
| Current PhD students | Requirements, dissertation, funding, graduation |
| Other | Non-majors, faculty, staff |

**MAP** = Masters Accelerated Program — qualified undergraduates continue into an MS degree in a shorter time period (combined BS→MS track). Distinct from both standard undergrads and standard MS students.

**System prompt instruction:** *"If the question clearly identifies the user type, respond specifically for that type. If the question is ambiguous, address all applicable groups using clear headers. Select only the groups relevant to the question — do not list all groups for every answer."*

**Ground truth note:** For ambiguous questions, the `key_facts` list must include facts for every applicable group, and `applicable_groups` must be filled in so the judge knows what completeness means for that question.

### Ground Truth File Format

```yaml
- question_id: "deg_req_001"
  category: "degree_requirements"
  question: "How many credit hours are required for the CS BS degree?"
  key_facts:
    - "128 total credit hours required"
    - "Catalog year 2025-2026"
    - "Includes general education requirements"
  expected_chunk_ids:
    - "catalog::2025-2026::degree::bachelor-of-science-computer-science"
  expected_source_type: "catalog"       # "catalog", "web", "either", "redirect"
  applicable_groups:                    # omit if question is not persona-sensitive
    - "current_undergrad"
    - "prospective_undergrad"
  banner_redirect_expected: false
  notes: "Answer should cite catalog page range, not web sources"
```

---

## System Prompt Notes
*Collect observations here during Phase 1. These become the brief for writing the prompt after Phase 1 is complete.*

### Audience
- List all 11 audience groups explicitly so the LLM knows the full range of who may be asking
- If question identifies the user type → respond specifically for that type
- If question is ambiguous → address all applicable groups with clear headers; select only relevant groups, do not list all groups for every answer

### Long Catalog Lists
- For lengthy catalog lists (gen ed requirements, course sequences, elective lists): summarize the structure, categories, and total credits — do not reproduce multi-page lists verbatim
- Always cite the catalog page(s) where the full list can be found

### Source Preference
- Course descriptions, prerequisites, degree requirements, gen ed, VWW → prefer **catalog**
- Advising contacts, financial aid, assistantships, faculty directory → prefer **web**
- Faculty information also exists in the catalog but is harder to parse; if web answers prove insufficient, investigate catalog chunks as a fallback
- Either source is acceptable when the question spans both (e.g., requirements + how to apply)

### Course Codes and Prerequisites
- Course codes ending in **G** (e.g. CSCI 1115G) contribute to General Education Requirements
- Course codes ending in **V** (e.g. ASTR 308V — Into the Final Frontier) contribute to Viewing a Wider World Requirements
- Include these definitions in the prompt so the LLM can explain the suffix meaning when it appears in a response

### Cross-Source Questions
- Some questions require both catalog content (requirements) and web content (how to apply, contacts)
- When both sources are needed, answer both parts and cite each source separately
- Do not omit the web portion just because the catalog chunk ranked higher

### Banner Redirect
- For any question about current semester enrollment, open seats, or section availability: do not answer from context — redirect to Banner URL with instructions
- Banner URL: `https://banner-public.nmsu.edu/StudentRegistrationSsb/ssb/term/termSelection?mode=search`
- Instructions: select the semester, then filter Subject: CS, Campus: Las Cruces

### Graduate and MAP Application Redirects
- For questions about applying to the CS graduate program: direct to the Graduate School application at `https://apply.nmsu.edu/apply`
- For questions about the MAP pre-application form: the form is on the CS department intranet (requires NMSU login); direct the user to `https://computerscience.nmsu.edu/grad-students/graduate-degrees.html` for the link and for full program details
- The MAP intranet link and some Graduate School links may be dead or behind login — do not present them as directly accessible; tell the user to start from the graduate degrees page
- If a URL in the retrieved context is known to be a login-gated or intranet page, note that the user will need to be logged in to access it

### Thesis and Dissertation Formatting
- When a question involves thesis or dissertation formatting requirements, always direct the user to work with their advisor
- Note that detailed guidelines are maintained by the Graduate School at the SharePoint links in the graduate student FAQ
- Do not attempt to answer formatting specifics from context — the SharePoint page is behind a login and was not crawled

### Partially Answerable Questions
- When something IS known but a specific detail is not in the sources, state what is known first, then note that the specific detail is not available in department sources
- Reserve "I could not find that information" for questions with zero relevant context
- Example: faculty recommendations are required for MAP but the count is not specified → say so, and direct to the department

### Other
- *(to be filled in)*

---

## Phase 2 — Deterministic Metrics Harness
*Build the eval loop. No LLM judge yet.*

### Tasks
- [ ] Create `backend/eval/` directory
- [ ] Create `backend/eval/harness.py` — main evaluation script
- [ ] Implement: load ground truth → call `generate_grounded_answer()` per question → compute metrics → write output files
- [ ] Compute these metrics from the returned chunks (no LLM needed):

| Metric | Type | Description |
|--------|------|-------------|
| `retrieval_hit` | bool | At least one expected chunk_id in top-5 |
| `top1_match` | bool | Expected chunk_id is rank 1 |
| `source_type_correct` | bool | Dominant content_source in top-3 matches expected |
| `banner_redirect_triggered` | bool / null | True if Banner URL appears in answer; null for non-enrollment questions |
| `citation_format_valid` | bool | Catalog chunks cite year + page; web chunks cite URL |
| `latency_ms` | int | Wall-clock time for `generate_grounded_answer()` |

- [ ] Write per-question records to `eval_results_{run_id}.jsonl` (one JSON object per line)
- [ ] Write run summary to `eval_summary_{run_id}.json`
- [ ] Run baseline against full question set and save results

### Per-Question Output Record (key fields)

```
question_id, category, question, run_id
ground_truth_answer (key facts), ground_truth_chunks
system_answer, system_sources
retrieved_chunks (all 5, with rank / hybrid_score / metadata_boost / final_score / chunk_type / content_source)
retrieval_hit, top1_match, source_type_correct
banner_redirect_triggered, citation_format_valid
retrieval_score  →  (hit × 0.4) + (top1 × 0.3) + (source_correct × 0.3)
latency_ms, error
```

### Run Summary Output (key fields)

```
run_id, run_timestamp, total_questions, total_passed, pass_rate
per-category breakdown: count, passed, pass_rate, avg_retrieval_score, source_pref_accuracy
avg_latency_ms, p90_latency_ms
failed_questions, low_retrieval_ids
```

---

## Phase 3 — LLM-as-Judge Integration
*Add one OpenAI call per question to score qualitative criteria.*

### Tasks
- [ ] Create `backend/eval/judge.py` — judge prompt builder and caller
- [ ] Implement `judge_question()` function: takes question + ground truth key facts + retrieved chunks + system answer → returns 5 scores + reasoning string
- [ ] Use `response_format={"type": "json_object"}` to enforce parseable output
- [ ] Track `judge_tokens_used` per question from `response.usage.total_tokens`
- [ ] Add error handling: if judge call fails, set all scores to -1 and record error; do not abort run
- [ ] Integrate judge call into `harness.py` after each pipeline call
- [ ] Add judge fields to per-question output record
- [ ] Re-run full eval; manually review ~10 questions to calibrate judge against your own assessment
- [ ] Adjust rubric wording if judge is systematically too lenient or strict on any criterion

### Judge Criteria (0–3 scale each)

| Criterion | What it Measures |
|-----------|------------------|
| **Faithfulness** | Every factual claim traces to a retrieved chunk. 3 = all claims supported. 0 = invented facts. |
| **Completeness** | Key facts from ground truth are present in the answer. 3 = all covered. 0 = core question missed. |
| **Source preference** | Correct source type used: catalog for requirements/courses; web for advising/financial aid/contacts. |
| **Citation quality** | Citations correctly formatted and accurate. Catalog → year + page. Web → URL. 0 = missing or fabricated. |
| **Hallucination** | No invented specifics (course numbers, names, URLs, page numbers). 3 = none. 0 = multiple. Inverted scale. |

### Judge Prompt Structure

The judge receives in one call:
1. The student's question
2. Ground truth key facts (bullet list)
3. Retrieved context (verbatim from `build_context()`)
4. System answer (verbatim)
5. Domain rules (when to prefer catalog vs web; Banner redirect rule)
6. Scoring rubric with concrete anchors for each level

### Judge Output Format

```json
{
  "faithfulness": 3,
  "completeness": 2,
  "source_preference": 3,
  "citation_quality": 2,
  "hallucination": 3,
  "reasoning": "Plain-English explanation of scores for human review."
}
```

### Passing Threshold

`passed = (retrieval_score >= 0.7) AND (judge_total >= 0.7)`  
where `judge_total = mean of 5 judge scores, normalized to 0–1`

---

## Phase 4 — Banner Redirect Implementation and Test
*Implement the redirect, then verify with eval.*

### Tasks
- [ ] Implement Banner redirect logic in `router.py` (intent/keyword check before calling `generate_grounded_answer()`)
- [ ] Redirect response must include: the Banner URL + instructions (select semester → Subject: CS → Campus: Las Cruces)
- [ ] Re-run full eval
- [ ] Verify all 5 enrollment questions now show `banner_redirect_triggered = True`
- [ ] Verify judge scores faithfulness = 3 and hallucination = 3 for those questions (redirecting is correct behavior)

**Banner URL:**  
`https://banner-public.nmsu.edu/StudentRegistrationSsb/ssb/term/termSelection?mode=search`

---

## Phase 5 — Reporting and Iteration Loop
*Make eval results usable for ongoing improvement.*

### Tasks
- [ ] Add stdout summary table at end of each run (pass rate by category, avg judge scores, top-5 failing questions)
- [ ] Commit `ground_truth.yaml` and each run's JSONL + summary JSON to git so score changes across iterations are visible
- [ ] Use eval results to tune retrieval parameters: `HYBRID_ALPHA`, `metadata_boost` weights, `TOP_K`
- [ ] Re-run eval after any change to `retrieval.py`, `ingest.py`, or `cs.yaml` to catch regressions

---

## File Layout

```
backend/
  eval/
    ground_truth.yaml          ← manually curated, human-verified
    harness.py                 ← main eval script
    judge.py                   ← LLM-as-judge prompt + caller
    results/
      eval_results_{run_id}.jsonl
      eval_summary_{run_id}.json
```

---

## Key Decisions and Rationale

| Decision | Rationale |
|----------|-----------|
| No RAGAS or DeepEval | Those frameworks require LangChain wrappers; this codebase calls OpenAI and Weaviate directly. Adapting them would be more work than writing targeted judge prompts. |
| pytest as runner | Free test filtering, skip markers, familiar mental model for a Python project. No infrastructure overhead. |
| Key facts, not full answers, as ground truth | Prevents evaluation from rewarding verbatim copying; focuses on whether the substance is correct. |
| No auto-generated ground truth | Circular: the same model that generates answers would be judging against LLM-generated facts. For a chatbot students use for academic decisions, human verification is essential. |
| One judge call per question (not 5) | Reduces cost and latency; structured JSON output enforces all 5 scores in one response. |
| Judge explicitly told source preference rules | Without this, the judge cannot assess whether the system correctly preferred catalog over web or vice versa — it would not know the rule exists. |
| Commit results to git | Gives permanent record of how retrieval and prompt changes affect scores. Regressions are visible in diffs. |
