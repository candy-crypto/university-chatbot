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

**User interface.** A web front end built with Next.js and React provides a conversational chat interface. The interface accumulates a scrolling display of questions and answers within the session, shows a loading indicator while the backend processes the query, and includes a department selector currently set to the CS department. Each question is handled independently — prior exchanges are displayed but not passed to the backend. The interface communicates with the backend via a single REST endpoint (`/chat`).

**Evaluation harness.** A Python evaluation framework runs the retrieval and generation pipeline against a curated ground-truth dataset, computes deterministic retrieval metrics, invokes an LLM-as-judge (ChatGPT-5.4-mini) for quality scoring, and produces structured output for human review and reporting.

---

## 3. Data Sources and Ingestion

### 3.1 NMSU Academic Catalog

The primary authoritative source for course descriptions, degree requirements, prerequisites, minors, and academic policies is the NMSU Academic Catalog (2025–2026 edition, PDF). The catalog PDF spans more than 2,000 pages; the CS-relevant sections cover undergraduate and graduate course listings, B.S. and M.S. degree requirements, and minor programs.

A custom catalog chunker (`nmsu_catalog_chunker.py`) parses the PDF and produces structured chunks, each assigned a type: `course_description`, `degree_requirement`, `minor_index`, `minor_requirement`, `policy`, and others. All chunks share the same set of fields, and the chunk type determines which fields are populated. Fields not relevant to a given chunk type are left empty.

The combined knowledge base contains 6,108 chunks across 19 chunk types, drawn from both the catalog (5,906 chunks, 97%) and department web pages (202 chunks, 3%). Course descriptions alone account for 5,435 chunks — 89% of the total — reflecting the catalog's density of individual course entries. The table below shows the full distribution by chunk type and source.

[TABLE: Chunk counts by type and source (catalog / web / total) — 19 rows + total row]

As a by-product of catalog ingestion, a PostgreSQL course lookup table is populated with one row per course: course code, title, credit count, and a suffix flag: `G` for General Education courses, `V` for Viewing a Wider World (VWW) courses, and blank otherwise. Support for `H` (Honors) is planned for a future round. The table supports lookups by exact course code, by exact title, or by approximate title using full-text search. A student can ask about "Data Structures" and the system resolves it to CSCI 3110 without being told the course code, and vice versa. The canonical code and title retrieved from the table are also injected into the LLM's context, ensuring that course references in responses use the correct, consistent identifiers rather than relying on the model to reproduce them accurately. This table is used by the pre-retrieval shortcuts described in Section 4.3.

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
- **Temporal grounding.** Questions about semester availability require different handling depending on what the student is actually asking. "Is CSCI 4110 offered in the fall?" is an offering-frequency question answered from the static three-year course rotation page; the correct retrieved source is the rotation table, not real-time registration data. "Are there seats available this fall?" is a live registration question that cannot be answered from any ingested source and is redirected to Banner. The query classifier distinguishes these two cases explicitly, preventing offering-frequency questions from being incorrectly redirected and ensuring rotation-table chunks are retrieved rather than seat-count data that does not exist in the knowledge base.

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
- **Registration/Seat availability queries** short-circuit retrieval entirely and return the Banner course search URL without consulting the vector database.

### 4.5 Result Set Size (k)

The number of chunks retrieved (k) is not fixed. The default is 5, which is sufficient for straightforward single-topic queries. Query classification raises k when the answer is expected to require more content: questions about all available concentrations retrieve up to 15 chunks (there are roughly 11 concentration options); questions about minors retrieve up to 12; questions comparing two degree programs retrieve up to 20 so that relevant chunks for both programs have room to surface in the ranked list. Policy and course-topic queries are similarly expanded to 10–15. The reasoning in all cases is the same: a fixed small k risks cutting off relevant content when the answer spans multiple catalog sections or web pages, while an unnecessarily large k adds noise and increases LLM context length without improving accuracy for simple queries.

### 4.6 Context Assembly

The top-ranked chunks are assembled into a prompt context block and passed to the LLM along with the user's question. Each chunk carries its source metadata, which the LLM uses to produce citations in its answer. Citations take different forms depending on the source: catalog chunks are cited as "NMSU Academic Catalog 2025–2026, pp. X–Y" (using the page range stored with the chunk); web chunks are cited by URL. All citations are collected into a "Sources:" section at the end of the answer rather than appearing inline, keeping the body of the response readable. A context-recall metric measures how much of the retrieved content overlaps with the expected chunks for each question, complementing the hit@k metric.

---

## 5. Evaluation Framework

### 5.1 Ground Truth Dataset

The evaluation dataset comprises 75 questions distributed across 13 categories representing a range of student advising queries. The table below shows the number of questions per category; category counts are unequal and in several cases small, reflecting the initial scope of the benchmark rather than the relative importance of each topic.

[TABLE: Question counts by category (from eval_combined_report.csv or summary)]

Questions were written to reflect realistic student phrasing, including informal language, abbreviations, and multi-part queries. For each question, the ground truth specifies: the expected answer's key facts, the chunk IDs expected to appear in the retrieved context, the expected source type (catalog, web, or either), whether the question requires a Banner redirect, and a natural-language retrieval note explaining what a correct retrieval should look like.

### 5.2 Deterministic Retrieval Metrics

The evaluation harness computes several metrics that do not require an LLM and are fully reproducible:

- **Hit@k** (binary: pass/fail) — whether at least one expected chunk appears in the top-k retrieved results. Weighted at 40% in the composite. Note: this metric is sometimes called hit rate; true recall@k — the proportion of all expected chunks retrieved — is a planned improvement.
- **Precision@1** (binary: pass/fail) — whether the top-ranked chunk is one of the expected chunks. Weighted at 30%.
- **Source type correctness** (binary: pass/fail) — whether the dominant source type in the top-3 results matches the expected source type (catalog, web, or either). Weighted at 30%.
- **Retrieval score** (0.0–1.0) — the weighted sum of the three binary metrics above. Because each component is binary, the score can only take the values 0.0, 0.3, 0.4, 0.6, 0.7, or 1.0.
- **Citation format validity** (binary: pass/fail) — whether the answer includes a citation in the format appropriate for the sources that were actually retrieved: a catalog page reference (e.g., "NMSU Academic Catalog 2025–2026, pp. X–Y") if catalog chunks were retrieved, and a URL if web chunks were retrieved. When both source types were retrieved, both citation formats must be present. Tracked separately; not included in the retrieval score composite.
- **Context recall** (0.0–1.0) — the proportion of ground-truth key facts covered by the answer. Tracked separately; not included in the retrieval score composite.

A question is counted as **passed** if its composite retrieval score meets or exceeds 0.7 AND its judge composite score meets or exceeds 0.7. Questions with no expected chunk IDs (unanswerable questions) pass on judge score alone; Banner-redirect questions pass if the redirect URL is present and the judge score meets the threshold.

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

The resulting kappa of 0.18 (slight agreement by conventional benchmarks) requires interpretation in context. In raw terms, the human reviewer and the judge agreed on the majority of questions — they disagreed on roughly 15 of 75 (20%). However, because the judge passed 74 of 75 questions at the 0.7 threshold, random chance would already predict high raw agreement, and kappa reflects little room left to agree meaningfully on failures. A more diagnostic view comes from splitting the disagreements by direction: the judge correctly identified 85% of the questions the human passed (true positive rate), but caught only 38% of the questions the human failed (true negative rate). In other words, the judge was well-calibrated on good answers and too lenient on weak ones — tending to pass answers that were partially correct or missing secondary facts rather than penalizing incompleteness. This finding is informative for future judge prompt design: tightening the completeness rubric and raising the scoring threshold for partial answers would likely bring judge and human assessments into closer alignment.

---

## 6. Results

### 6.1 Overall Performance

The evaluation was conducted on the system state after a full re-ingest incorporating all chunking improvements described in Section 3. Retrieval scores improved over multiple development cycles as chunking strategies, query classification, and judge calibration were refined; the figures reported here reflect the final system state. Earlier evaluation snapshots from the development period are available in the accompanying presentation.

[TABLE: Overall pass rates — retrieval score, judge total, human pass — across all 75 questions]

The system achieved a judge-based pass rate of **85.3%** (64 of 75 questions), using a judge composite threshold of 0.70. The average retrieval score across all questions was 0.72, and the average judge composite score was 0.94.

A qualitative pattern is visible across question types, though category-level pass rates should not be read as statistically reliable given the small and unequal sample sizes in the current benchmark. Questions for which answers draw primarily from well-structured catalog content — course descriptions, prerequisites, degree requirements, financial aid — tended to perform better than questions requiring synthesis across multiple sources or across both catalog and web. The latter pattern, seen in advising and faculty questions, reflects the retrieval challenge discussed in Section 7.3.

### 6.2 Latency

The system's average end-to-end latency was approximately 11 seconds per question, with a 90th-percentile latency of approximately 19 seconds. This latency includes the retrieval query, context assembly, and LLM generation. For a synchronous chat interface, this is at the high end of acceptable; it suggests that LLM generation time dominates and that streaming responses would significantly improve perceived responsiveness.

### 6.3 Notable Cases

**deg_007 / deg_008 (thesis and project advisor timing).** These paired questions ask when a graduate student must have secured a thesis or project advisor. For deg_007 ("When must I have a thesis advisor?"), the system failed entirely — retrieval score 0.0, human-marked Fail. The critical information — that an advisor is required at the point of enrolling in CSCI 5999 (Master's Thesis) — is embedded in that course's description, not in any policy or degree-requirement chunk. Because the question is phrased as an advising policy question, the system's query classification did not direct it toward course descriptions, and the relevant chunk was never retrieved. For deg_008 ("When must I have a project advisor?"), the system happened to retrieve the CSCI 5994 course description — the analogous course for the project track — and produced a correct answer (human-marked Pass), despite a low retrieval score because not all expected chunks were found. The pair illustrates a structural limitation: advising constraints embedded inside course descriptions are not reliably surfaced by policy-oriented queries. A future improvement would be to index such constraints separately, or to broaden retrieval to include course descriptions for questions containing enrollment-trigger language.

**road_002 (course planning for Spring 2027 entry).** A student asking which CS courses to take at the start of a B.S. program received an answer that correctly identified the early courses from the recommended roadmap (CSCI 1720, 2210, 2230) but stopped there. A complete answer would also cross-reference the three-year course offering rotation to confirm which of those courses are available in Spring specifically, and would surface the prerequisite chain — that CSCI 1720 must come first because it unlocks both 2210 and 2230 — from the course descriptions. The system retrieved the roadmap chunk but did not retrieve the rotation table or the individual course descriptions, so the prerequisite logic and semester-availability check were absent from the context. This is a multi-hop reasoning gap: the question implicitly requires three separate lookups (roadmap, rotation, prerequisites) that the single-pass retrieval does not chain together. The human reviewer passed the answer as useful but noted these gaps.

**deg_013 (comparison of a B.S. in Artificial Intelligence vs B.S. in Computer Science with Artificial Intelligence Concentration).** The system retrieved the correct chunks for both programs and produced a structurally accurate high-level comparison. However, the answer enumerated individual required courses from each program in detail — information that was only partially grounded in the retrieved context and introduced hallucination risk on specific course numbers and titles. The judge failed the answer (composite 0.56) on faithfulness, completeness, and hallucination. This case illustrates a general principle: the more specific and enumerative an answer becomes, the greater the risk of hallucinated course numbers or titles. Future system prompt versions should instruct the model to describe structural differences at a high level rather than enumerate individual courses for program-comparison questions.

**adv_003 (advising contact details).** This question was marked as Pass by the human reviewer and Fail by the judge. The judge applied a strict completeness standard, penalizing the answer for not including every detail in the key facts list. The human reviewer judged the answer as complete for a student's practical purposes. This divergence illustrates the known limitation of LLM judges that are calibrated toward precision at the expense of practical utility.

---

## 7. Limitations and Future Work

### 7.1 Data Currency

The catalog content reflects the 2025–2026 academic year. Course descriptions, prerequisites, and degree requirements change each catalog cycle. The ingestion pipeline must be re-run each year against the new catalog PDF, and the web crawler must be re-run whenever department pages are updated. There is currently no automated scheduling or change-detection mechanism; re-ingestion is a manual process.

### 7.2 Real-Time Data

Seat availability, waitlist status, and current enrollment figures are not available through this system by design. The system redirects enrollment queries to the NMSU course search tool. If students require deeper integration with registration data, a Banner API connection would be needed — a substantially more complex integration than the current approach.

### 7.3 Retrieval Failures and Multi-Hop Queries

The system's most consistent failure mode is multi-source synthesis: questions that require combining information from, for example, a faculty directory page and a degree requirement chunk, or from two separate degree program pages. Current retrieval returns the top-k chunks by similarity score, without any mechanism to ensure diversity across source types. A re-ranking step that explicitly diversifies source coverage could reduce this failure mode.

Some retrieval failures reflect genuinely sparse coverage in the source documents — for example, questions about job opportunities and the market value of specific degrees, which appear in the benchmark but have limited/no content in the crawled pages. However, because the benchmark was constructed by someone with knowledge of the corpus, questions for which no answer exists are underrepresented relative to what real students would ask. Questions about current office hours, scholarship application deadlines, or TA openings are examples that would expose coverage gaps not exercised by the current benchmark. This is one motivation for stress-testing the system with actual students before any deployment, as discussed in Section 7.6.

### 7.4 Judge Calibration

As discussed in Section 5.4, the LLM judge's low true negative rate (38%) means it cannot reliably detect failures in an automated pipeline. Several rubric adjustments were made during development to improve consistency, including domain-specific instructions around citation format, semester abbreviation conventions, and source-type preferences. However, the judge's leniency bias likely reflects a model-level tendency to confirm passes rather than a gap in rubric content.

One low-risk experiment worth running is adding a system-role message to the judge call. Currently the rubric, question, ground truth, retrieved context, and answer are all assembled into a single user message with no system framing. A short system prompt — establishing the model's role as a rigorous evaluator and instructing it to actively look for failures rather than confirm plausible passes — could reduce leniency without changing the rubric itself. The experiment would involve re-running the judge on the known human-annotated failures and measuring whether the true negative rate improves. Further calibration with a larger labeled dataset would be needed before the judge score could serve as a reliable automated regression gate.

### 7.5 Retrieval Metric Refinement

The current retrieval score composite uses hit@k — a binary flag that fires as soon as any one expected chunk appears in the top-k results — weighted at 40%. For questions that require only a single chunk, this is a reasonable measure. For questions that require multiple chunks (a degree comparison, a roadmap question, a multi-part policy query), hit@k is too lenient: retrieving 1 of 5 expected chunks earns the same score as retrieving all 5, yet the LLM can only construct a complete answer if most or all of the relevant chunks are present. The 40% weight amplifies this distortion, allowing multi-chunk questions to pass the 0.7 threshold on the strength of hit@k alone even when retrieval coverage is poor. Replacing hit@k with recall@k — the proportion of expected chunks actually retrieved — would give partial credit proportional to coverage and make the composite score more diagnostic for complex queries. This change would require re-running the full evaluation to obtain corrected scores.

### 7.6 Benchmark Expansion and Balance

The current evaluation set of 75 questions is sufficient to guide development but not to support reliable category-level performance analysis. Question counts per category range from 2 to 11, making percentage-based comparisons across categories misleading. A larger benchmark — with a minimum of 10–15 questions per category and proportional representation of the question types students ask most frequently — would allow meaningful measurement of where the system is weakest and would provide a more stable regression baseline as the system evolves. Collecting questions from actual students in a stress-testing phase before deployment would also surface coverage gaps and question patterns that a curated benchmark is unlikely to anticipate.

### 7.7 Multi-Department Extensibility

The system's architecture is designed for a single department but not fundamentally limited to one. The knowledge base schema includes a `department_id` field that is stored with every chunk and applied as a hard filter at retrieval time, so content from different departments coexists in the same Weaviate collection without cross-contamination. The front end already passes a department identifier to the backend with every request. Adding a second department would involve the following steps:

- **Department configuration file.** Each department is defined by a YAML configuration file (analogous to `cs.yaml`) that specifies which catalog page ranges to ingest, which web pages to crawl, and any department-specific metadata tags (level, degree types, etc.). Creating this file is the first task for a new department and determines the scope of everything that follows.

- **Catalog ingestion.** The catalog chunker must be run against the new department's catalog sections. The chunker is driven by section headings and page ranges, so it is largely catalog-structure-agnostic; however, if a department uses unusual heading conventions or has a significantly different course-listing format, the heading-detection logic may require adjustment. The resulting chunks are upserted into Weaviate tagged with the new department's identifier.

- **Web crawl.** The web crawler must be configured with the new department's page URLs. Faculty directory pages, advising pages, and any department-specific resources (rotation tables, scholarship pages) need to be listed in the configuration. If the faculty directory uses a different HTML structure than the CS site, the faculty-entry splitter may need to be adapted.

- **System prompt.** The system prompt contains department-specific content: the department name, relevant acronyms and their expansions, redirect URLs for enrollment, applications, and graduate school resources, and scope instructions. For a second department, the prompt should be templated — with department name, acronym list, and redirect URLs drawn from the department's configuration file at request time — rather than duplicating and hardcoding a separate prompt. This is the highest-leverage change for making the system genuinely multi-department.

- **Retrieval tuning.** The retrieval layer includes term sets and synonym maps (e.g., for course-availability, faculty, and minor queries) that were built from CS-specific vocabulary. These would need to be reviewed and extended for a new department's terminology. The chunk-type boost weights were also calibrated on CS data and may need adjustment if a new department's content is distributed differently across chunk types.

- **Evaluation benchmark.** A ground-truth question set would need to be developed for the new department before any meaningful evaluation could be run. Without it, there is no systematic way to verify that retrieval is working correctly or to detect regressions.

- **Front-end update.** Adding the new department to the selector and updating the welcome message is a small change. If the department selector is eventually populated dynamically from the backend, no front-end code change would be required at all.

### 7.8 Multi-Turn Conversation Support

The current system treats each question as independent: no conversation history is passed to the retrieval or generation step. This means follow-up questions — "What about the MS instead?", "Where is her office?" — cannot refer back to what was just discussed, and the system will either misinterpret them or ask the student to restate the question in full. Adding multi-turn support would require maintaining a conversation history in the front end, passing relevant prior turns to the backend with each request, and incorporating prior context into the retrieval step. The last point is non-trivial: a follow-up like "What are its prerequisites?" is ambiguous without knowing which course was just discussed, so the retrieval query must be resolved against the conversation history before hitting Weaviate. The generation step is simpler — the LLM already handles context well — but the growing context window as conversations lengthen requires a truncation or summarization strategy.

### 7.9 Hybrid Search Balance

The hybrid search alpha — which controls the weight given to semantic vector similarity versus BM25 keyword matching — is currently fixed at 0.75 (75% vector, 25% BM25). This value was set as a reasonable default and has not been systematically varied. The optimal balance likely differs by query type: exact course code lookups ("Is CSCI 4700 a G course?") benefit from higher BM25 weight, while conceptual or thematic queries ("courses that cover machine learning") benefit from higher vector weight. A structured experiment varying alpha across query categories, or implementing a query-adaptive alpha that shifts the balance based on whether the query contains exact identifiers, could improve retrieval precision without any changes to the underlying index or chunking.

### 7.10 Response Quality

The evaluation confirms that the system reliably avoids common LLM failure modes (filler phrases, excessive preamble, evasive non-answers) when the retrieval is successful. When retrieval fails to surface the correct content, the LLM's response degrades. It tends to provide related but incomplete information rather than fabricating specific course numbers or requirements. This behavior is attributable to the system prompt's emphasis on grounding and to the retrieval layer's tendency to return related content even when the exact match is missing.

---

## 8. Conclusion

This prototype demonstrates that a retrieval-augmented chatbot can answer the large majority of routine CS department advising queries accurately and with correct source attribution, given a well-structured knowledge base and a carefully tuned retrieval layer. The system achieves an 85% pass rate on a 75-question evaluation set spanning 13 question categories, with 100% pass rates in the highest-volume categories (course descriptions, prerequisites, degree choice, financial aid).

The most significant engineering challenges were in the ingestion layer — particularly designing a catalog chunker that reliably segments a complex PDF into semantically coherent, correctly attributed pieces — and in the retrieval layer, where query classification, synonym expansion, and chunk-type boosting required iterative tuning against the evaluation dataset. The evaluation framework itself, including the LLM-as-judge rubric and human annotation pipeline, proved essential for detecting subtle failures that aggregate metrics would miss.

Recommended next steps are: (1) streaming response delivery to reduce perceived latency, (2) a re-ranking pass that diversifies retrieved source types for multi-source queries, (3) annual re-ingestion against updated catalog and web content, and (4) expansion of the crawl scope to cover additional department resources. With these improvements, the system would be well-positioned for a limited pilot deployment with real students.

---

## Acknowledgments

The initial development of the university chatbot project, including the overall project structure, system architecture, and foundational codebase, was supported with the assistance of ChatGPT 5.4 OpenAI. Guidance was provided for designing the frontend and backend workflow, retrieval-augmented generation (RAG) pipeline, database integration, website crawling and ingestion process, vector search implementation, and deployment planning. The generated materials served as a foundational starting point and were later reviewed, modified, expanded, and customized throughout the development process to meet the specific requirements and goals of the project.

**AI-assisted development.** All scripts in the catalog ingestion and evaluation were drafted and refined using Claude Code (Anthropic). Claude Code also supported the refinement of retrieval and webpage ingestion scripts. The author directed the work through an iterative dialogue: specifying requirements, reviewing outputs, checking logic, identifying errors and edge cases, and determining when a component was ready to move forward. This guidance required understanding the data well enough to write the ground-truth benchmark, recognize when retrieval was misbehaving, correct factual errors in generated documentation, and make judgment calls about design trade-offs — for example, choosing between chunking strategies, deciding which retrieval failures warranted a code fix versus a ground-truth correction, and calibrating the LLM judge rubric through human annotation. Claude Code generated and revised code in response to that direction but did not determine priorities, scope, or analytical conclusions.

The team also used Codex (under ChatGPT 5.4) in developing this project, including assistance with the code implementation, project structure, and debugging process. The guidance helped organize the application into a clearer and more maintainable structure, improve the logic and functionality of the code, and identify and resolve errors throughout development. This support contributed significantly to the completion and refinement of the project.

*Luis * — [Description of AI tool use, if applicable — please fill in.]

---

*Draft prepared May 2026. All evaluation figures are from evaluation runs completed May 5, 2026; human annotation from run 6815e2 review.*
