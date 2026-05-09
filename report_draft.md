# A Retrieval-Augmented Chatbot for CS Department Advising: Design, Implementation, and Evaluation

**New Mexico State University — Department of Computer Science**
Draft — May 2026

---

## 1. Introduction

University students routinely need information about course offerings, degree requirements, prerequisites, financial aid, and faculty — information that is documented in official sources but can be time-consuming to locate. This project explores whether a retrieval-augmented generation (RAG) chatbot can serve as a reliable first point of contact for such questions, drawing answers directly from the NMSU Academic Catalog and the Department of Computer Science website. The goal is an assistant that responds accurately, cites its sources, and recognizes the boundaries of what it can and cannot answer from static documents.

This report describes the design, implementation, and evaluation of that prototype. The system ingests two primary sources, the NMSU Academic Catalog (PDF) and department web pages, stores them as semantically searchable chunks in a vector database, and uses a large language model (LLM) to compose answers grounded in the retrieved text. A systematic evaluation across 75 representative questions demonstrates the system's strengths and identifies where further work is warranted.

---

## 2. System Architecture

The chatbot follows a standard RAG pattern: a user question triggers retrieval of relevant text passages from a pre-indexed knowledge base, and the passages are passed as context to an LLM that generates the answer. The system is designed to answer only from what it retrieves — not from general knowledge the model may have acquired during training.

The major components are:

**Knowledge base.** Text chunks are stored in Weaviate, an open-source vector database that supports hybrid search (combined semantic vector similarity and BM25 keyword matching). Each chunk carries metadata, including source type (catalog vs. web), chunk type (course description, degree requirement, faculty entry, etc.), catalog page range or URL, and department identifier — that enables precise filtering and source attribution.

**Ingestion pipeline.** Two parallel pipelines populate the knowledge base: a catalog pipeline that parses the 2025-2026 NMSU Academic Catalog PDF and a web crawler that scrapes department pages. Both pipelines normalize, chunk, embed, and upsert their output into Weaviate. Embeddings are produced using OpenAI's `text-embedding-3-small` model.

**Retrieval layer.** At query time, the system pre-processes the question (acronym expansion, synonym mapping, query classification) and issues a hybrid Weaviate query. Where the query signals a specific audience or content type, hard filters narrow the candidate pool before scoring: a level filter restricts results to undergraduate, graduate, or non-major content as appropriate, and a chunk-type filter limits candidates to the most relevant content categories (e.g., course descriptions, degree requirements, or minor information). Retrieved chunks are ranked by a composite score, and the top-k passages are assembled into the prompt context.

**Answer generation.** An LLM (accessed via the OpenAI API) receives a system prompt, the assembled context, and the user question. The system prompt instructs the model to answer from context only and to cite sources. It also governs several categories of redirect, where the correct response is to point the student to an external resource rather than attempt an answer from static documents. Enrollment and seat-availability queries are intercepted before retrieval and redirected to the NMSU course search tool. Application queries (undergraduate, graduate, and the MAP pre-application) are redirected to the appropriate admissions URLs. Thesis and dissertation formatting questions are redirected to the Graduate School's SharePoint pages, since those guidelines are maintained externally and change independently of the catalog.

**User interface.** A web front end built with Next.js and React provides a conversational chat interface. The interface maintains a scrolling message history, shows a loading indicator while the backend processes the query, and includes a department selector currently set to the CS department. The interface communicates with the backend via a single REST endpoint (`/chat`).

**Evaluation harness.** A Python evaluation framework runs the retrieval and generation pipeline against a curated ground-truth dataset, computes deterministic retrieval metrics, invokes an LLM-as-judge (ChatGPT-5.4-mini) for quality scoring, and produces structured output for human review and reporting.

---

## 3. Data Sources and Ingestion

### 3.1 NMSU Academic Catalog

The primary authoritative source for course descriptions, degree requirements, prerequisites, minors, and academic policies is the NMSU Academic Catalog (2025–2026 edition, PDF). The catalog PDF spans more than 2,000; the CS-relevant sections cover undergraduate and graduate course listings, B.S. and M.S. degree requirements, and minor programs.

A custom catalog chunker (`nmsu_catalog_chunker.py`) parses the PDF and produces structured chunks, each assigned a type: `course_description`, `degree_requirement`, `degree_core_requirement`, `minor_index`, `minor_requirement`, `policy`, and others. All chunks share the same set of fields, and the chunk type determines which fields are populated. Fields not relevant to a given chunk type are left empty.

As a by-product of catalog ingestion, a lightweight SQLite course lookup table is populated with one row per course: course code, title, credit count, and a suffix flag — `G` for General Education courses, `V` for Viewing a Wider World (VWW) courses, and blank otherwise. This table supports fast, exact lookups without going through the vector database and is used by the pre-retrieval shortcuts described in Section 4.3.

### 3.2 Department Web Pages

The department website supplements the catalog with additional information and, in some areas, duplicates it. Information available only on the web includes the three-year course offering rotation, advising contacts, financial aid and assistantship details, and current-semester announcements. The web is also the sole source for the information about the Bachelor of Science in Artificial Intelligence, which was introduced after the current catalog went to press. A focused web crawler (`ingest.py`) retrieves specific pages selected from various domains, including Advising, Computer Science, and Data Analytics. It extracts the main content, and produces web chunks with URL-based IDs.

The crawler applies a faculty-entry splitter that segments multi-faculty directory pages into per-person chunks, enabling retrieval of an individual's specific details.

### 3.3 Chunking Design Choices

Chunks in this system are not uniform in length. Rather than splitting text at fixed token or character counts, a common approach in general-purpose RAG frameworks, the pipelines segment content at natural semantic boundaries: one chunk per course description, one per degree program, one per faculty entry, one per policy topic. This content-driven approach means a chunk for a one-paragraph course description may be a few hundred tokens, while a chunk covering a full degree's course requirements may be several times longer. The trade-off is intentional: splitting a degree requirement block at an arbitrary token limit would sever the logical unit a student needs to see whole, producing incomplete and potentially misleading answers. Embedding quality and retrieval precision are better served by coherent semantic units than by size uniformity. 

A hard upper limit of 8,192 tokens — the maximum accepted by the OpenAI embedding model — is enforced by a preflight check in the catalog ingestion pipeline; any chunk exceeding that limit causes the run to abort before any data is written. In practice, the longest chunk in the current database is approximately 5,100 tokens (a policy section covering doctoral degree requirements), leaving comfortable headroom below the ceiling.

Several additional chunking decisions reflect lessons learned during development:

- **Two-column PDF layout.** The course description section of the NMSU catalog is typeset in two columns. Standard PDF text extractors read across the full page width, interleaving left- and right-column text into garbled output. The catalog chunker detects the two-column region and processes each column independently before merging, preserving coherent course entries.
- **Heading prepended to chunk text.** For web chunks, the h2/h3 section heading is prepended to the body text before embedding, not stored only as a metadata field. This ensures the heading participates in both BM25 keyword matching and vector similarity, so a student's question that echoes the heading term retrieves the correct chunk even when the body text alone would not rank highly. For pages whose URL does not convey department context — the Data Analytics faculty directory in particular — an additional department label ("Data Analytics faculty.") is prepended to each chunk for the same reason.
- **Overlap filtering.** Faculty chunks include a short text overlap with adjacent chunks to preserve context at boundaries, but the splitter filters duplicate sentences that would otherwise appear twice.
- **Heading bleed prevention.** The web crawler's faculty-entry splitter applies a blocklist of words that cannot be a person's first name (e.g., "Introduction", "Expertise", "Research") to prevent section headings on faculty directory pages from being mistakenly split out as person entries.
- **Course offering rotation table.** The three-year course offering rotation is published as an HTML table with courses as rows and semesters as columns. A plain text extraction would discard the column structure, making it impossible to determine which semester a given course is offered. The web crawler serializes each table row into a text representation that explicitly maps each course to its offering semesters by reading the column headers from the table header row. This preserves the course-semester relationships that are essential for answering offering-frequency questions.
- **Calendar table chunking.** The academic calendar is chunked as a unit rather than split by row, since queries about deadlines typically need the full calendar context.
- **Enrollment redirect.** Real-time seat-availability data is not ingested. Instead, the system detects enrollment queries and redirects students to the NMSU course search tool, avoiding stale data entirely.

---

## 4. Retrieval System Design

### 4.1 Hybrid Search

Weaviate's hybrid search combines a dense vector similarity score (cosine similarity over OpenAI embeddings) with a sparse BM25 keyword score. The blend is controlled by `HYBRID_ALPHA`, set to 0.75, meaning vector similarity contributes 75% of the ranking signal and BM25 contributes 25%. This weighting favors semantic matching — essential when students phrase questions differently from the catalog's formal language — while BM25 ensures that exact course codes and program names still rank highly.

### 4.2 Query Pre-Processing

Before issuing the Weaviate query, the system applies several transformations to the raw question:

- **Acronym expansion.** Common abbreviations are expanded (e.g., "AI" → "artificial intelligence", "ML" → "machine learning", "Generative AI" → "generative artificial intelligence"). The expanded form is appended to the end of the query string, so both the abbreviated and expanded forms participate in retrieval. Because the expansion appears at the end rather than adjacent to the original term, it does not always carry full BM25 weight — a known limitation of this approach.
- **Synonym and stem mapping.** A curated synonym map adds discipline-specific equivalents (e.g., "neural net" → "neural network", "ethics" → "ethical").
- **Query classification.** Patterns in the question trigger classification as a banner-redirect query (enrollment/seat questions), an offering-frequency query (rotation table questions), a comparison query, or a course-topic query. Classification controls which chunk types are boosted, how many results are fetched (top-k), and what source type the answer is expected to draw from.
- **Temporal grounding.** Questions about semester availability require different handling depending on what the student is actually asking. "Is CSCI 4110 offered in the fall?" is an offering-frequency question answered from the static three-year course rotation page; the correct retrieved source is the rotation table, not real-time enrollment data. "Are there seats available this fall?" is a live enrollment question that cannot be answered from any ingested source and is redirected to Banner. The query classifier distinguishes these two cases explicitly, preventing offering-frequency questions from being incorrectly redirected and ensuring rotation-table chunks are retrieved rather than seat-count data that does not exist in the knowledge base.

### 4.3 Pre-Retrieval Shortcuts

Before issuing any Weaviate query, the system checks whether the question can be answered directly from the course lookup table. Several question types fall into this category:

- **Gen Ed status.** "Does CSCI 1115G count for General Education?" is answered by looking up the course code in the lookup table and checking its suffix — `G` confirms it is a General Education course, no chunk retrieval needed.
- **VWW status.** The same mechanism applies to Viewing a Wider World questions: the `V` suffix answers the question definitively.
- **Listing Gen Ed or VWW courses.** "What courses satisfy my VWW requirement?" is answered by querying the lookup table for all courses with a `V` suffix, optionally filtered by department keyword if the student names a subject area.
- **Credit hours.** "How many credits is CSCI 3110?" is answered directly from the credit count stored in the lookup table.

These shortcuts avoid the latency and noise of vector search for questions that are fully answered by structured catalog data. They also sidestep a common RAG failure mode: when the correct answer is a simple fact (a suffix or a number), retrieval can return related but imprecise content that misleads the LLM.

### 4.4 Chunk-Type Filtering and Boosting

Rather than searching the entire collection for every query, the system applies Weaviate filters and post-retrieval boosts:

- A **department filter** restricts results to CS department content, isolating the system from other departments' data in the same Weaviate instance.
- **Chunk-type boosts** elevate chunks whose type matches the inferred query intent (e.g., `course_description` chunks receive a boost for questions about course content; `degree_requirement` chunks are boosted for program-level questions).
- **Enrollment queries** short-circuit retrieval entirely and return the Banner course search URL without consulting the vector database.

### 4.5 Result Set Size (k)

The number of chunks retrieved — k — is not fixed. The default is 5, which is sufficient for straightforward single-topic queries. Query classification raises k when the answer is expected to require more content: questions about all available concentrations retrieve up to 15 chunks (there are roughly 11 concentration options); questions about minors retrieve up to 12; questions comparing two degree programs retrieve up to 20 so that relevant chunks for both programs have room to surface in the ranked list. Policy and course-topic queries are similarly expanded to 10–15. The reasoning in all cases is the same: a fixed small k risks cutting off relevant content when the answer spans multiple catalog sections or web pages, while an unnecessarily large k adds noise and increases LLM context length without improving accuracy for simple queries.

### 4.6 Context Assembly

The top-ranked chunks are assembled into a prompt context block and passed to the LLM along with the user's question. Each chunk carries its source metadata, which the LLM uses to produce citations in its answer. Citations take different forms depending on the source: catalog chunks are cited as "NMSU Academic Catalog 2025–2026, pp. X–Y" (using the page range stored with the chunk); web chunks are cited by URL. All citations are collected into a "Sources:" section at the end of the answer rather than appearing inline, keeping the body of the response readable. A context-recall metric measures how much of the retrieved content overlaps with the expected chunks for each question, complementing the binary recall-at-k metric.

---

## 5. Evaluation Framework

### 5.1 Ground Truth Dataset

The evaluation dataset comprises 75 questions distributed across 13 categories representing the full range of student advising queries. The table below shows the number of questions per category; category counts are unequal and in several cases small, reflecting the initial scope of the benchmark rather than the relative importance of each topic.

[TABLE: Question counts by category (from eval_combined_report.csv or summary)]

Questions were written to reflect realistic student phrasing, including informal language, abbreviations, and multi-part queries. For each question, the ground truth specifies: the expected answer's key facts, the chunk IDs expected to appear in the retrieved context, the expected source type (catalog, web, or either), whether the question requires a Banner redirect, and a natural-language retrieval note explaining what a correct retrieval should look like.

### 5.2 Deterministic Retrieval Metrics

The evaluation harness computes several metrics that do not require an LLM and are fully reproducible:

- **Recall@k** — whether all expected chunk IDs appear in the top-k retrieved results.
- **Precision@1** — whether the top-ranked chunk is one of the expected chunks.
- **Source type correctness** — whether the retrieved chunks come from the expected source type.
- **Citation format validity** — whether citations in the answer follow the expected format (page range for catalog, URL for web).
- **Context recall** — the proportion of expected chunk text covered by the retrieved context.
- **Retrieval score** — a composite of the above metrics, weighted to reflect their relative importance.

A question is counted as **passed** if its composite retrieval score meets or exceeds a threshold of 0.7.

### 5.3 LLM-as-Judge

For qualitative evaluation, an LLM judge (OpenAI gpt-5.4-mini) scores each answer on six criteria using a 0–3 integer scale:

- **Faithfulness** — every factual claim traces to the retrieved context.
- **Completeness** — the answer covers the ground-truth key facts.
- **Source preference** — the answer draws from the correct source type for the question.
- **Citation quality** — citations are present, correctly formatted, and accurate.
- **Hallucination** (inverted) — no invented specifics (course numbers, dates, URLs).
- **Response quality** — the answer is direct, professional, and free of filler phrases.

The six scores are averaged and normalized to a 0.0–1.0 composite judge score. The judge prompt embeds a detailed rubric that includes domain-specific guidance: degree name aliases (e.g., "MS in Data Analytics" and "Professional M.S. in Computational Data Analytics" are treated as equivalent), semester abbreviation conventions (SP26 = Spring 2026, FA26 = Fall 2026, etc.), and citation format expectations that differ for catalog vs. web sources.

The judge also receives a computed temporal context derived from the current date at evaluation time: the current semester and the upcoming semester are identified and injected into the rubric. This prevents the judge from penalizing an answer for naming a specific semester (e.g., "Fall 2026") when the student's question used relative language ("this fall") — inferring the referent from the student's own words is valid reasoning, not hallucination. Without this grounding, answers to time-sensitive availability questions would be systematically downgraded on the hallucination criterion.

### 5.4 Human Annotation

A human reviewer independently assessed each answer as Pass or Fail, with optional notes. This provides a calibration check on the LLM judge and surfaces cases where automated metrics diverge from expert judgment.

Inter-rater agreement between the human reviewer and the LLM judge was measured using Cohen's Kappa, a standard statistic for comparing two raters on a binary outcome. Kappa ranges from −1 to +1: a value of 1 means perfect agreement, 0 means no better than what two raters would achieve by chance alone, and negative values indicate systematic disagreement. Crucially, kappa adjusts for the base rate — when both raters say "pass" most of the time, a high percentage of raw agreement is expected just by chance, and kappa discounts it.

The resulting kappa of 0.18 (slight agreement by conventional benchmarks) requires interpretation in context. In raw terms, the human reviewer and the judge agreed on the vast majority of questions — they disagreed on roughly 15 of 75. However, because the judge passed 74 of 75 questions at the 0.7 threshold, random chance would already predict high raw agreement, and kappa reflects little room left to agree meaningfully on failures. A more diagnostic view comes from splitting the disagreements by direction: the judge correctly identified 85% of the questions the human passed (true positive rate), but caught only 38% of the questions the human failed (true negative rate). In other words, the judge was well-calibrated on good answers and too lenient on weak ones — tending to pass answers that were partially correct or missing secondary facts rather than penalizing incompleteness. This finding is informative for future judge prompt design: tightening the completeness rubric and raising the scoring threshold for partial answers would likely bring judge and human assessments into closer alignment.

---

## 6. Results

### 6.1 Overall Performance

The evaluation was conducted on the system state after a full re-ingest incorporating all chunking improvements described in Section 3. Retrieval scores improved over multiple development cycles as chunking strategies, query classification, and judge calibration were refined; the figures reported here reflect the final system state. Earlier evaluation snapshots from the development period are available in the accompanying presentation.

[TABLE: Overall pass rates — retrieval score, judge total, human pass — across all 75 questions]

The system achieved a retrieval-based pass rate of **85.3%** (64 of 75 questions). The average retrieval score across all questions was 0.72, and the average judge composite score was 0.94.

A qualitative pattern is visible across question types, though category-level pass rates should not be read as statistically reliable given the small and unequal sample sizes in the current benchmark. Questions for which answers draw primarily from well-structured catalog content — course descriptions, prerequisites, degree requirements, financial aid — tended to perform better than questions requiring synthesis across multiple sources or across both catalog and web. The latter pattern, seen in advising and faculty questions, reflects the retrieval challenge discussed in Section 7.3.

### 6.2 Latency

The system's average end-to-end latency was approximately 11 seconds per question, with a 90th-percentile latency of approximately 19 seconds. This latency includes the retrieval query, context assembly, and LLM generation. For a synchronous chat interface, this is at the high end of acceptable; it suggests that LLM generation time dominates and that streaming responses would significantly improve perceived responsiveness.

### 6.3 Notable Cases

**deg_007 / deg_008 (thesis and project advisor timing).** These paired questions ask when a graduate student must have secured a thesis or project advisor. For deg_007 ("When must I have a thesis advisor?"), the system failed entirely — retrieval score 0.0, human-marked Fail. The critical information — that an advisor is required at the point of enrolling in CSCI 5999 (Master's Thesis) — is embedded in that course's description, not in any policy or degree-requirement chunk. Because the question is phrased as an advising policy question, the system's query classification did not direct it toward course descriptions, and the relevant chunk was never retrieved. For deg_008 ("When must I have a project advisor?"), the system happened to retrieve the CSCI 5994 course description — the analogous course for the project track — and produced a correct answer (human-marked Pass), despite a low retrieval score because not all expected chunks were found. The pair illustrates a structural limitation: advising constraints embedded inside course descriptions are not reliably surfaced by policy-oriented queries. A future improvement would be to index such constraints separately, or to broaden retrieval to include course descriptions for questions containing enrollment-trigger language.

**road_002 (course planning for Spring 2027 entry).** A student asking which CS courses to take at the start of a B.S. program received an answer that correctly identified the early courses from the recommended roadmap (CSCI 1720, 2210, 2230) but stopped there. A complete answer would also cross-reference the three-year course offering rotation to confirm which of those courses are available in Spring specifically, and would surface the prerequisite chain — that CSCI 1720 must come first because it unlocks both 2210 and 2230 — from the course descriptions. The system retrieved the roadmap chunk but did not retrieve the rotation table or the individual course descriptions, so the prerequisite logic and semester-availability check were absent from the context. This is a multi-hop reasoning gap: the question implicitly requires three separate lookups (roadmap, rotation, prerequisites) that the single-pass retrieval does not chain together. The human reviewer passed the answer as useful but noted these gaps.

**deg_013 (comparison of a B.S. in Artificial Intelligence vs B.S. in Computer Science with Artificial Intelligence Concentration).** The system retrieved the correct chunks for both programs and produced a structurally accurate comparison. However, the answer's middle paragraphs enumerated individual courses from each program — information that was not fully grounded in the retrieved context and that introduced hallucination risk. The first and last paragraphs of the answer correctly captured the structural distinction between programs (coursework-only master's vs. dissertation-based doctoral program). This case illustrates a general principle: the more specific and enumerative an answer becomes, the greater the risk of hallucinated course numbers or titles. Future system prompt versions should instruct the model to describe structural differences rather than list individual courses for program-comparison questions.

**adv_003 (advising contact details).** This question was marked as Pass by the human reviewer and Fail by the judge. The judge applied a strict completeness standard, penalizing the answer for not including every detail in the key facts list. The human reviewer judged the answer as complete for a student's practical purposes. This divergence illustrates the known limitation of LLM judges that are calibrated toward precision at the expense of practical utility.

---

## 7. Limitations and Future Work

### 7.1 Data Currency

The catalog content reflects the 2025–2026 academic year. Course descriptions, prerequisites, and degree requirements change each catalog cycle. The ingestion pipeline must be re-run each year against the new catalog PDF, and the web crawler must be re-run whenever department pages are updated. There is currently no automated scheduling or change-detection mechanism; re-ingestion is a manual process.

### 7.2 Real-Time Data

Seat availability, waitlist status, and current enrollment figures are not available through this system by design. The system redirects enrollment queries to the NMSU course search tool. If students require deeper integration with registration data, a Banner API connection would be needed — a substantially more complex integration than the current approach.

### 7.3 Retrieval Failures and Multi-Hop Queries

The system's most consistent failure mode is multi-source synthesis: questions that require combining information from, for example, a faculty directory page and a degree requirement chunk, or from two separate degree program pages. Current retrieval returns the top-k chunks by similarity score, without any mechanism to ensure diversity across source types. A re-ranking step that explicitly diversifies source coverage could reduce this failure mode.

Some retrieval failures reflect genuinely sparse coverage in the source documents. When neither the catalog nor the website contains the specific information a student asks for, the system either retrieves loosely related content or returns a general disclaimer. Expanding the crawl scope to include additional department resources (syllabi, lab pages, advising FAQs) would improve coverage.

### 7.4 Judge Calibration

As discussed in Section 5.4, the LLM judge's low true negative rate (38%) means it cannot reliably detect failures in an automated pipeline. Several rubric adjustments were made during development — adding domain-specific instructions around citation format, semester abbreviation conventions, and source-type preferences — and these improved consistency. However, further calibration with a larger labeled dataset would be needed before the judge score could be used as a reliable automated regression test.

### 7.5 Benchmark Expansion and Balance

The current evaluation set of 75 questions is sufficient to guide development but not to support reliable category-level performance analysis. Question counts per category range from 2 to 11, making percentage-based comparisons across categories misleading. A larger benchmark — with a minimum of 10–15 questions per category and proportional representation of the question types students ask most frequently — would allow meaningful measurement of where the system is weakest and would provide a more stable regression baseline as the system evolves.

### 7.6 Multi-Department Extensibility

The system's architecture is designed for a single department but not fundamentally limited to one. The knowledge base schema includes a `department_id` field that filters retrieval to a specific department's content. Extending the system to a second department would require: (1) running the ingestion pipeline against that department's catalog sections and web pages, (2) configuring department-specific chunking parameters if the catalog structure differs, and (3) writing a department-specific system prompt. Retrieval logic would require minimal changes — primarily verifying that department-specific terminology is captured in the acronym and synonym maps. The bulk of the extension effort lies in ingestion configuration and data coverage verification. On the front end, the department selector already passes a department identifier to the backend with every request; adding a new department would require only populating the selector dynamically and updating the page title and welcome message to reflect the selected department.

### 7.7 Response Quality

The evaluation confirms that the system reliably avoids common LLM failure modes (filler phrases, excessive preamble, evasive non-answers) when the retrieval is successful. When retrieval fails to surface the correct content, the LLM's response degrades gracefully — it tends to provide related but incomplete information rather than fabricating specific course numbers or requirements. This behavior is attributable to the system prompt's emphasis on grounding and to the retrieval layer's tendency to return related content even when the exact match is missing.

---

## 8. Conclusion

This prototype demonstrates that a retrieval-augmented chatbot can answer the large majority of routine CS department advising queries accurately and with correct source attribution, given a well-structured knowledge base and a carefully tuned retrieval layer. The system achieves an 85% pass rate on a 75-question evaluation set spanning 13 question categories, with 100% pass rates in the highest-volume categories (course descriptions, prerequisites, degree choice, financial aid).

The most significant engineering challenges were in the ingestion layer — particularly designing a catalog chunker that reliably segments a complex PDF into semantically coherent, correctly attributed pieces — and in the retrieval layer, where query classification, synonym expansion, and chunk-type boosting required iterative tuning against the evaluation dataset. The evaluation framework itself, including the LLM-as-judge rubric and human annotation pipeline, proved essential for detecting subtle failures that aggregate metrics would miss.

Recommended next steps are: (1) streaming response delivery to reduce perceived latency, (2) a re-ranking pass that diversifies retrieved source types for multi-source queries, (3) annual re-ingestion against updated catalog and web content, and (4) expansion of the crawl scope to cover additional department resources. With these improvements, the system would be well-positioned for a limited pilot deployment with real students.

---

## Acknowledgments

**AI-assisted development tools.** Portions of this project were implemented with the assistance of AI coding tools. Design decisions, evaluation criteria, ground-truth authorship, and analytical conclusions are the authors'.

*Barbara Reed* used Claude Code (Anthropic) throughout the development of the ingestion pipeline, retrieval system, and evaluation framework. In the ingestion layer, Claude Code implemented the catalog PDF chunker — including detection and separate processing of the two-column course description layout — and the web crawler's faculty-entry splitter, rotation table serializer, and heading-prepending logic, iterating on each as edge cases emerged from test runs. In the retrieval layer, Claude Code implemented the hybrid Weaviate search, query pre-processing pipeline (acronym expansion, synonym mapping, query classification), dynamic result-set sizing, hard filters, and chunk-type boosting, with each component tuned against evaluation results through repeated cycles of test, diagnose, and adjust. In the evaluation framework, Claude Code built the harness, the LLM-as-judge prompt and rubric (revised multiple times to address calibration issues identified through human annotation), the annotation analysis tools, and the combined results CSV used in this report. Throughout, Claude Code explained trade-offs, flagged edge cases, and surfaced implementation details that informed design decisions — but the direction, priorities, and judgment calls were the author's.

*Candy [Last Name]* — [Description of Codex use in frontend development — please fill in.]

*Luis [Last Name]* — [Description of AI tool use, if applicable — please fill in.]

---

## Appendices

### Appendix A: Evaluation Question Set

[TABLE: Full question list with IDs, categories, and questions — from eval_combined_report.csv]

### Appendix B: Per-Question Evaluation Results

[TABLE: eval_combined_report.csv — all 20 columns — or a subset selected for the report]

### Appendix C: Retrieval Score Metric Definitions

| Metric | Description | Weight in Composite |
|---|---|---|
| Recall@k | All expected chunks appear in top-k results | High |
| Precision@1 | Top-ranked chunk is an expected chunk | Medium |
| Source type correctness | Retrieved source type matches expected | Medium |
| Citation format validity | Citations follow expected format | Low |
| Context recall | Coverage of expected chunk text in retrieved context | Medium |

### Appendix D: LLM Judge Rubric Criteria

| Criterion | Scale | Description |
|---|---|---|
| Faithfulness | 0–3 | Every claim traces to retrieved context |
| Completeness | 0–3 | Key facts from ground truth are covered |
| Source preference | 0–3 | Correct source type used (catalog vs. web vs. Banner redirect) |
| Citation quality | 0–3 | Citations present, correctly formatted, and accurate |
| Hallucination (inverted) | 0–3 | No invented specifics; higher = fewer hallucinations |
| Response quality | 0–3 | Direct, professional, no filler phrases |

---

*Draft prepared May 2026. All evaluation figures are from run bb9ebc (May 8, 2026) supplemented by run 9c0d25 (May 9, 2026) for seven questions; human annotation from run 6815e2 review.*
