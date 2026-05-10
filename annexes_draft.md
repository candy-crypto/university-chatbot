# Appendices — RAG Chatbot for CS Department Advising

---

## Appendix A: Evaluation Question Set

75 questions across 13 categories. Questions were authored by a team member with knowledge of the corpus; the benchmark covers the most common student advising query types but underrepresents questions for which no answer exists in the source documents.

| ID | Category | Question |
|---|---|---|
| adv_001 | advising | What is CAASS? |
| adv_002 | advising | How do I get an advisor? |
| adv_003 | advising | As a Computer Science major, what advising am I required to get? |
| avail_001 | course availability | What CS courses are commonly offered during summer sessions? |
| avail_002 | course availability | What CS courses will be offered next semester? |
| avail_003 | course availability | When is the next time Algorithm Design & Implementation will be offered? |
| avail_004 | course availability | When is the next time CSCI 4120 will be offered? |
| avail_005 | course availability | When is the next opportunity to take Algorithm Design and Implementation? |
| avail_006 | course availability | Is the Operating Systems course offered every semester? |
| avail_007 | course availability | On what schedule will CHEM 1215G be offered this Summer? |
| avail_008 | course availability | I would like to take Calculus based Physics this Fall. Are seats still available? |
| avail_009 | course availability | Is COMM 115G offered every semester? |
| avail_010 | course availability | When is the next time that CSCI 5215 will be offered? |
| desc_001 | course description | What computer science courses address computing ethics and governance? |
| desc_002 | course description | What undergraduate courses have content related to artificial intelligence? |
| desc_003 | course description | What is MATH 1521H? |
| desc_004 | course description | What is the difference between the Practical Programming and Software Development courses? |
| desc_005 | course description | What is covered in CJUS 412? |
| desc_006 | course description | What is the difference between CSCI 1720 and CSCI 1210? |
| desc_007 | course description | What courses teach cybersecurity and what are their prerequisites? |
| req_001 | course requirement | What are Viewing a Wider World Requirements? |
| req_002 | course requirement | Does the History of Food course contribute to my general education requirements? |
| req_003 | course requirement | Can I take the course about death and dying to get credits toward my VWW requirements? |
| req_004 | course requirement | What are the General Education requirements for undergraduates? |
| req_005 | course requirement | In which year of my studies should I be taking courses to meet the Viewing the Wider World requirements? |
| req_006 | course requirement | What math courses can I take to meet my VWW requirements? |
| req_007 | course requirement | What is the difference between the BS in Cybersecurity and a BS in Human Computer Interaction? |
| req_008 | course requirement | Is Computer Security/CSCI 4205 required for a CS minor? |
| req_009 | course requirement | Must I take a course in Biology to get a BS in CS? |
| req_010 | course requirement | Which statistics course is required for Computer Science majors? |
| req_011 | course requirement | I am in my second semester in computer science this Fall. I need to take an English general education course. Which are offered? |
| deg_001 | degree choice | What is the difference between a BS in Computer Science and a BS in Cybersecurity? |
| deg_002 | degree choice | Why would I want a specialty BS in Cybersecurity instead of a standard BS? |
| deg_003 | degree choice | When should one consider a BS in Computer Science instead of a BA? |
| deg_004 | degree choice | What is the difference between the BS and BA degrees in CS? |
| deg_010 | degree choice | What Computer Science graduate degrees are offered at NMSU? |
| deg_011 | degree choice | What computer science minors can I pursue at NMSU? |
| deg_005 | degree requirements | What is the recommended number of credits I should take in a regular semester? |
| deg_006 | degree requirements | Explain and compare the different tracks for an M.S.: Coursework only, Project, or Thesis. |
| deg_007 | degree requirements | When must I have a thesis advisor? |
| deg_008 | degree requirements | When must I have a Project advisor? |
| deg_009 | degree requirements | When must I have a doctoral advisor? |
| deg_012 | degree requirements | What kinds of course credits can I transfer in from another institution? |
| deg_013 | degree requirements | How do the requirements for the BS in Artificial Intelligence differ from the requirements for the BS in Computer Science Artificial Intelligence concentration? |
| fac_001 | faculty | Which faculty are affiliated with the data analytics concentration? |
| fac_002 | faculty | Which computer science faculty members focus on logic programming, and what are their email addresses? |
| fac_003 | faculty | Where is Dr. Wayllace's office? |
| fac_004 | faculty | Which professors have interests in cryptography? |
| fac_005 | faculty | Who is the head of the computer science department, and what is his/her field of expertise? |
| fin_001 | financial aid | What percentage of the current CS MS students receive financial aid? |
| fin_002 | financial aid | What types of financial aid are available for Computer Science students? |
| fin_003 | financial aid | When and how can I apply for a Graduate Assistantship? |
| fin_004 | financial aid | What do Teaching Assistantships involve, e.g., number of hours weekly and types of work required? |
| fin_005 | financial aid | How can I get a Research Assistantship in computer science? |
| minor_001 | minors | What are my options for a CS minor, and what are the differences? |
| minor_002 | minors | I am a graduate studying engineering. What would be the value of my getting a CS minor? |
| other_001 | other | What is the difference in job opportunities between a BS in Cybersecurity and one in AI? |
| other_002 | other | What is the main phone number for the computer science department? |
| pol_001 | policy | How do I get off academic probation? |
| pol_002 | policy | What are grounds for dismissal? |
| pol_003 | policy | How do I apply to the CS graduate program? |
| pol_004 | policy | Where can I get information about the required format for the submission of my dissertation? |
| pol_005 | policy | Tell me about PhD qualifying exams. Who takes them and when? What are they like? |
| pol_006 | policy | I am in my second year as a Cybersecurity major and want to apply for MAP. What is the process? |
| pol_007 | policy | When is the deadline to apply to get my graduate diploma this summer? |
| pol_008 | policy | How many faculty references do I need in my MAP application? |
| pol_009 | policy | How do I apply to NMSU's Graduate School? |
| pol_010 | policy | Give me information about dissertation formatting requirements. |
| prereq_001 | prerequisites | What are the prerequisites for Programming Language Structure? |
| prereq_002 | prerequisites | What are the prerequisites for CSCI 4110? |
| prereq_003 | prerequisites | What are the prerequisites for CSCI 4130? |
| prereq_004 | prerequisites | What are the prerequisites for the Compilers course? |
| prereq_005 | prerequisites | What are the prerequisites for Practical Programming? |
| road_001 | roadmap | What courses should I take in my first semester as an undergraduate in Cybersecurity? |
| road_002 | roadmap | I am beginning my studies for a BS in Computer Science in the Spring of 2027. In what CS courses should I consider enrolling? |

---

## Appendix B: Per-Question Evaluation Results

Scores from run bb9ebc (May 8, 2026), supplemented by run 9c0d25 (May 9, 2026) for seven questions not in bb9ebc. Human pass/fail from manual annotation of run 6815e2. Judge pass threshold: 0.70. Retrieval score pass threshold: 0.70.

Hit@k, Precision@1, and Source type correct are binary (TRUE/FALSE). Retrieval score and judge total are 0.0–1.0. Judge criteria (faithfulness through response quality) are 0–3.

| ID | Human | Judge | Ret. Score | Judge Total | Hit@k | Prec@1 | Src Type | Faith. | Compl. | Src Pref. | Citation | Halluc. | Resp. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| adv_001 | ✓ | ✓ | 1.00 | 1.00 | ✓ | ✓ | ✓ | 3 | 3 | 3 | 3 | 3 | 3 |
| adv_002 | ✗ | ✓ | 0.70 | 1.00 | ✓ | ✗ | ✓ | 3 | 3 | 3 | 3 | 3 | 3 |
| adv_003 | ✓ | ✓ | 0.00 | 0.78 | ✗ | ✗ | ✗ | 3 | 1 | 1 | 3 | 3 | 3 |
| avail_001 | ✓ | ✓ | 0.30 | 1.00 | ✗ | ✗ | ✓ | 3 | 3 | 3 | 3 | 3 | 3 |
| avail_002 | ✓ | ✓ | 1.00 | 0.78 | ✓ | ✓ | ✓ | 2 | 2 | 3 | 3 | 1 | 3 |
| avail_003 | ✓ | ✓ | 0.40 | 0.89 | ✓ | ✗ | ✗ | 2 | 3 | 3 | 3 | 2 | 3 |
| avail_004 | ✓ | ✓ | 0.70 | 1.00 | ✓ | ✓ | ✗ | 3 | 3 | 3 | 3 | 3 | 3 |
| avail_005 | ✓ | ✓ | 0.40 | 0.89 | ✓ | ✗ | ✗ | 2 | 3 | 3 | 3 | 2 | 3 |
| avail_006 | ✓ | ✓ | 0.70 | 1.00 | ✓ | ✓ | ✗ | 3 | 3 | 3 | 3 | 3 | 3 |
| avail_007 | ✓ | ✓ | 0.30 | 1.00 | ✗ | ✗ | ✓ | 3 | 3 | 3 | 3 | 3 | 3 |
| avail_008 | ✓ | ✓ | 0.30 | 1.00 | ✗ | ✗ | ✓ | 3 | 3 | 3 | 3 | 3 | 3 |
| avail_009 | ✓ | ✓ | 0.30 | 0.89 | ✗ | ✗ | ✓ | 3 | 3 | 1 | 3 | 3 | 3 |
| avail_010 | ✓ | ✓ | 0.70 | 1.00 | ✓ | ✓ | ✗ | 3 | 3 | 3 | 3 | 3 | 3 |
| desc_001 | ✗ | ✓ | 1.00 | 0.78 | ✓ | ✓ | ✓ | 2 | 3 | 2 | 2 | 2 | 3 |
| desc_002 | ✓ | ✓ | 0.70 | 0.72 | ✓ | ✗ | ✓ | 2 | 2 | 3 | 2 | 1 | 3 |
| desc_003 | ✓ | ✓ | 1.00 | 1.00 | ✓ | ✓ | ✓ | 3 | 3 | 3 | 3 | 3 | 3 |
| desc_004 | ✓ | ✓ | 1.00 | 0.72 | ✓ | ✓ | ✓ | 1 | 2 | 3 | 3 | 1 | 3 |
| desc_005 | ✓ | ✓ | 1.00 | 1.00 | ✓ | ✓ | ✓ | 3 | 3 | 3 | 3 | 3 | 3 |
| desc_006 | ✓ | ✓ | 1.00 | 1.00 | ✓ | ✓ | ✓ | 3 | 3 | 3 | 3 | 3 | 3 |
| desc_007 | ✗ | ✓ | 1.00 | 0.83 | ✓ | ✓ | ✓ | 2 | 2 | 3 | 3 | 2 | 3 |
| req_001 | ✓ | ✓ | 1.00 | 1.00 | ✓ | ✓ | ✓ | 3 | 3 | 3 | 3 | 3 | 3 |
| req_002 | ✓ | ✓ | 0.70 | 1.00 | ✓ | ✗ | ✓ | 3 | 3 | 3 | 3 | 3 | 3 |
| req_003 | ✓ | ✓ | 0.30 | 0.83 | ✗ | ✗ | ✓ | 3 | 3 | 3 | 1 | 2 | 3 |
| req_004 | ✓ | ✓ | 1.00 | 1.00 | ✓ | ✓ | ✓ | 3 | 3 | 3 | 3 | 3 | 3 |
| req_005 | ✓ | ✓ | 0.30 | 1.00 | ✗ | ✗ | ✓ | 3 | 3 | 3 | 3 | 3 | 3 |
| req_006 | ✓ | ✓ | 0.00 | 0.83 | ✗ | ✗ | ✗ | 3 | 3 | 3 | 0 | 3 | 3 |
| req_007 | ✓ | ✓ | 1.00 | 1.00 | ✓ | ✓ | ✓ | 3 | 3 | 3 | 3 | 3 | 3 |
| req_008 | ✓ | ✓ | 1.00 | 1.00 | ✓ | ✓ | ✓ | 3 | 3 | 3 | 3 | 3 | 3 |
| req_009 | ✓ | ✓ | 0.70 | 1.00 | ✓ | ✗ | ✓ | 3 | 3 | 3 | 3 | 3 | 3 |
| req_010 | ✓ | ✓ | 0.30 | 1.00 | ✗ | ✗ | ✓ | 3 | 3 | 3 | 3 | 3 | 3 |
| req_011 | ✓ | ✓ | 0.70 | 0.72 | ✓ | ✗ | ✓ | 2 | 2 | 1 | 3 | 2 | 3 |
| deg_001 | ✓ | ✓ | 0.70 | 0.83 | ✓ | ✗ | ✓ | 2 | 2 | 3 | 3 | 2 | 3 |
| deg_002 | ✓ | ✓ | 0.70 | 1.00 | ✓ | ✗ | ✓ | 3 | 3 | 3 | 3 | 3 | 3 |
| deg_003 | ✓ | ✓ | 0.70 | 1.00 | ✓ | ✗ | ✓ | 3 | 3 | 3 | 3 | 3 | 3 |
| deg_004 | ✓ | ✓ | 1.00 | 0.94 | ✓ | ✓ | ✓ | 3 | 3 | 3 | 3 | 2 | 3 |
| deg_005 | ✓ | ✓ | 1.00 | 0.94 | ✓ | ✓ | ✓ | 3 | 2 | 3 | 3 | 3 | 3 |
| deg_006 | ✓ | ✓ | 1.00 | 0.94 | ✓ | ✓ | ✓ | 3 | 3 | 3 | 3 | 2 | 3 |
| deg_007 | ✗ | ✓ | 0.00 | 1.00 | ✗ | ✗ | ✗ | 3 | 3 | 3 | 3 | 3 | 3 |
| deg_008 | ✓ | ✓ | 0.40 | 0.94 | ✓ | ✗ | ✗ | 3 | 2 | 3 | 3 | 3 | 3 |
| deg_009 | ✗ | ✓ | 0.70 | 0.94 | ✓ | ✗ | ✓ | 3 | 2 | 3 | 3 | 3 | 3 |
| deg_010 | ✗ | ✓ | 1.00 | 0.94 | ✓ | ✓ | ✓ | 3 | 2 | 3 | 3 | 3 | 3 |
| deg_011 | ✓ | ✓ | 0.70 | 1.00 | ✓ | ✗ | ✓ | 3 | 3 | 3 | 3 | 3 | 3 |
| deg_012 | ✗ | ✓ | 1.00 | 1.00 | ✓ | ✓ | ✓ | 3 | 3 | 3 | 3 | 3 | 3 |
| deg_013 | ✓ | ✗ | 1.00 | 0.56 | ✓ | ✓ | ✓ | 1 | 1 | 2 | 2 | 1 | 3 |
| fac_001 | ✓ | ✓ | 1.00 | 1.00 | ✓ | ✓ | ✓ | 3 | 3 | 3 | 3 | 3 | 3 |
| fac_002 | ✓ | ✓ | 1.00 | 0.94 | ✓ | ✓ | ✓ | 3 | 3 | 3 | 2 | 3 | 3 |
| fac_003 | ✓ | ✓ | 1.00 | 1.00 | ✓ | ✓ | ✓ | 3 | 3 | 3 | 3 | 3 | 3 |
| fac_004 | ✓ | ✓ | 0.40 | 1.00 | ✓ | ✗ | ✗ | 3 | 3 | 3 | 3 | 3 | 3 |
| fac_005 | ✗ | ✓ | 0.30 | 0.94 | ✗ | ✗ | ✓ | 3 | 2 | 3 | 3 | 3 | 3 |
| fin_001 | ✓ | ✓ | 0.30 | 1.00 | ✗ | ✗ | ✓ | 3 | 3 | 3 | 3 | 3 | 3 |
| fin_002 | ✓ | ✓ | 1.00 | 1.00 | ✓ | ✓ | ✓ | 3 | 3 | 3 | 3 | 3 | 3 |
| fin_003 | ✓ | ✓ | 0.70 | 0.89 | ✓ | ✗ | ✓ | 3 | 2 | 2 | 3 | 3 | 3 |
| fin_004 | ✓ | ✓ | 0.70 | 1.00 | ✓ | ✓ | ✗ | 3 | 3 | 3 | 3 | 3 | 3 |
| fin_005 | ✓ | ✓ | 0.70 | 0.94 | ✓ | ✗ | ✓ | 3 | 2 | 3 | 3 | 3 | 3 |
| minor_001 | ✓ | ✓ | 1.00 | 1.00 | ✓ | ✓ | ✓ | 3 | 3 | 3 | 3 | 3 | 3 |
| minor_002 | ✓ | ✓ | 1.00 | 1.00 | ✓ | ✓ | ✓ | 3 | 3 | 3 | 3 | 3 | 3 |
| other_001 | ✓ | ✓ | 0.40 | 1.00 | ✓ | ✗ | ✗ | 3 | 3 | 3 | 3 | 3 | 3 |
| other_002 | ✓ | ✓ | 0.70 | 1.00 | ✓ | ✓ | ✗ | 3 | 3 | 3 | 3 | 3 | 3 |
| pol_001 | ✓ | ✓ | 1.00 | 0.94 | ✓ | ✓ | ✓ | 3 | 2 | 3 | 3 | 3 | 3 |
| pol_002 | ✓ | ✓ | 0.30 | 0.89 | ✗ | ✗ | ✓ | 3 | 1 | 3 | 3 | 3 | 3 |
| pol_003 | ✓ | ✓ | 0.70 | 1.00 | ✓ | ✗ | ✓ | 3 | 3 | 3 | 3 | 3 | 3 |
| pol_004 | ✓ | ✓ | 1.00 | 1.00 | ✓ | ✓ | ✓ | 3 | 3 | 3 | 3 | 3 | 3 |
| pol_005 | ✓ | ✓ | 0.70 | 1.00 | ✓ | ✗ | ✓ | 3 | 3 | 3 | 3 | 3 | 3 |
| pol_006 | ✓ | ✓ | 0.70 | 0.72 | ✓ | ✓ | ✗ | 2 | 2 | 2 | 2 | 2 | 3 |
| pol_007 | ✓ | ✓ | 0.40 | 1.00 | ✓ | ✗ | ✗ | 3 | 3 | 3 | 3 | 3 | 3 |
| pol_008 | ✓ | ✓ | 1.00 | 1.00 | ✓ | ✓ | ✓ | 3 | 3 | 3 | 3 | 3 | 3 |
| pol_009 | ✓ | ✓ | 0.30 | 1.00 | ✗ | ✗ | ✓ | 3 | 3 | 3 | 3 | 3 | 3 |
| pol_010 | ✓ | ✓ | 1.00 | 1.00 | ✓ | ✓ | ✓ | 3 | 3 | 3 | 3 | 3 | 3 |
| prereq_001 | ✓ | ✓ | 1.00 | 1.00 | ✓ | ✓ | ✓ | 3 | 3 | 3 | 3 | 3 | 3 |
| prereq_002 | ✓ | ✓ | 1.00 | 1.00 | ✓ | ✓ | ✓ | 3 | 3 | 3 | 3 | 3 | 3 |
| prereq_003 | ✓ | ✓ | 1.00 | 1.00 | ✓ | ✓ | ✓ | 3 | 3 | 3 | 3 | 3 | 3 |
| prereq_004 | ✓ | ✓ | 0.70 | 1.00 | ✓ | ✗ | ✓ | 3 | 3 | 3 | 3 | 3 | 3 |
| prereq_005 | ✓ | ✓ | 1.00 | 1.00 | ✓ | ✓ | ✓ | 3 | 3 | 3 | 3 | 3 | 3 |
| road_001 | ✓ | ✓ | 0.70 | 1.00 | ✓ | ✗ | ✓ | 3 | 3 | 3 | 3 | 3 | 3 |
| road_002 | ✓ | ✓ | 0.70 | 0.78 | ✓ | ✗ | ✓ | 2 | 2 | 1 | 3 | 3 | 3 |

---

## Appendix C: Retrieval Score Metric Definitions

The retrieval score is a weighted composite of three binary metrics. Its possible values are 0.0, 0.3, 0.4, 0.6, 0.7, and 1.0. The pass threshold is 0.70.

| Metric | Description | Range | Weight |
|---|---|---|---|
| Hit@k | True if any expected chunk ID appears in the top-k retrieved results. Binary — does not distinguish retrieving 1 of 5 expected chunks from retrieving all 5. | Binary | 40% |
| Precision@1 | True if the rank-1 chunk is one of the expected chunk IDs. | Binary | 30% |
| Source type correctness | True if the dominant source type in the top-3 results matches the expected source (catalog, web, or either). Always True for redirect questions. | Binary | 30% |

Two additional metrics are tracked but not included in the retrieval score:

| Metric | Description | Range |
|---|---|---|
| Citation format validity | Heuristic check: catalog sources should produce a year-tagged catalog citation; web sources should produce a URL. | Binary |
| Context recall | Fraction of ground-truth key facts whose significant terms appear in the system answer. | 0.0–1.0 |

---

## Appendix D: LLM Judge Rubric Criteria

The judge (gpt-5.4-mini) scores each response on six criteria. The composite judge total is the mean of all six scores normalized to 0.0–1.0 (each criterion max is 3). The pass threshold is 0.70.

| Criterion | Scale | Description |
|---|---|---|
| Faithfulness | 0–3 | Every factual claim in the answer traces to the retrieved context or the ground-truth key facts. |
| Completeness | 0–3 | The answer covers the key facts from the ground truth. |
| Source preference | 0–3 | The answer draws from the correct source type for the question (catalog for requirements/descriptions; web for advising/faculty; Banner redirect for enrollment). |
| Citation quality | 0–3 | Citations are present, correctly formatted, and accurate. Catalog sources cite year and page range; web sources cite URL. |
| Hallucination (inverted) | 0–3 | No invented specifics (course numbers, names, URLs, dates). Higher score = fewer hallucinations. |
| Response quality | 0–3 | Answer leads with content; professional tone; no filler phrases ("Great question!", "Certainly!", etc.). |

*Data from evaluation runs bb9ebc and 9c0d25 (May 8–9, 2026). Human annotation from run 6815e2.*
