# Recommendation: Catalog Format for AI-Assisted Retrieval
**Prepared for:** NMSU Department of Computer Science  
**Date:** April 25, 2026  
**Context:** Findings from implementation of an AI-powered departmental chatbot

---

## Executive Summary

The NMSU Academic Catalog is currently published as a two-column, paginated PDF designed for print. Building an AI retrieval system on top of this format is technically feasible but requires significant custom engineering, produces imperfect results in edge cases, and imposes a recurring annual maintenance burden. The same catalog content, published as structured HTML web pages — one page per section — would eliminate most of that complexity, improve answer accuracy, and make the system nearly self-maintaining as the catalog changes year to year. The underlying catalog management system almost certainly already supports HTML output; the ask is to turn it on, not to build something new.

---

## 1. The Problem with the Current Format

### 1.1 The catalog is designed for print, not retrieval

The PDF is a rendering artifact. The actual structured data — degree requirements, admission policies, course descriptions — exists somewhere upstream, in a catalog management system or in Banner. What gets published is a flattened, paginated, visually formatted snapshot of that data. Retrieving information from it means working backward from the rendered output to reconstruct the structure that was there before rendering.

### 1.2 Specific technical obstacles encountered

The following problems were encountered and required custom engineering to resolve. Each is a direct consequence of the print-oriented format.

**Two-column layout.** The PDF uses a two-column layout on every page. Characters must be detected by their horizontal position and grouped into lines before any text can be read. When a section boundary falls in the right column while the left column on the same page still has content below it, lines are returned in column order (all left, then all right) rather than reading order. A few lines from one section are silently appended to the adjacent section.

**Heading detection by font size.** Section boundaries are identified by font size — headings are 12pt or larger, body text is 8pt. This required calibrating size thresholds against the actual PDF and building a heading-line buffer to join headings that wrap across two lines. It also required special-casing the column midpoint to prevent a heading that straddles the column boundary from being read as two garbled fragments (e.g., "CSCI 5250" parsed as "SCI 4250" because the leading "C" fell in the left column).

**No stable identifiers.** Sections have no permanent IDs. Chunk identifiers are SHA-256 hashes of the text content, which change whenever wording is revised. A test suite that verifies retrieval accuracy must be manually updated after every catalog edition.

**Hardcoded page ranges.** The system must be explicitly told where each section begins and ends in the PDF (e.g., "the CS MS degree requirements are on pages 128–130"). These ranges shift every year. If a new policy section is inserted before page 128, every subsequent range must be reviewed and corrected.

**Duplicate content.** Course descriptions appear twice in the catalog: once embedded within the department section (pages 568–580 for CS) and again in the master course listing at the end (~pages 1377–2067). The duplicate must be detected and skipped during parsing.

**Unicode and encoding artifacts.** The PDF encodes certain characters differently from plain text. The ligature "fi" (as in "specified") appears as a single glyph rather than two characters. Apostrophes are "curly" (Unicode U+2019) rather than straight (U+0027). The section heading "Master's Accelerated Program (MAP)" fails to match a lookup table if the apostrophe is typed as a straight quote. These require explicit handling.

**Dynamic web rendering.** The faculty directory page loads faculty entries via JavaScript after the initial page load. A standard web crawler retrieves only the static shell and misses all faculty names, office locations, research interests, and contact information.

### 1.3 Maintenance burden

Each annual catalog update requires:

- Reviewing all hardcoded page ranges to determine which shifted
- Testing for new heading fragmentation or column-order edge cases
- Re-running the full ingest pipeline (~5,900 chunks)
- Manually updating the retrieval accuracy test suite where chunk content changed

This is not a one-time setup cost. It recurs every year the catalog is in its current form.

---

## 2. What Structured HTML Would Look Like

A structured HTML catalog means one web page per logical section, accessible at a stable URL, with content organized using standard HTML heading tags. Visually, it is a plain informational page — a title, body paragraphs, course requirement tables, and hyperlinks to related sections. No popups, no JavaScript required for the core content.

### 2.1 Example: CS MS Degree Requirements page

**URL:** `https://catalogs.nmsu.edu/las-cruces/computer-science/ms/requirements/`

```
Computer Science — Master of Science

Admission Requirements
  Applicants must hold a BS in Computer Science or equivalent.
  The GRE is not required. Strong applications include...

Degree Requirements
  Total credits: 30
  Thesis option: CSCI 598 (6 credits) + 24 hours coursework
  Non-thesis option: CSCI 599 (3 credits) + 27 hours coursework

  Required Courses
    CSCI 505   Theory of Computation     3 cr
    CSCI 526   Advanced Algorithms       3 cr

  Electives (choose 12 credits)
    CSCI 530   Machine Learning          3 cr
    CSCI 555   Computer Networks II      3 cr

  See also: Graduate School academic policies
            Application dates and deadlines
```

The links to "Graduate School academic policies" and "Application dates and deadlines" are ordinary HTML hyperlinks pointing to those section pages — no popups or modal dialogs.

### 2.2 Why the structure matters more than the appearance

The visual layout above is nearly identical to what a student sees in the current catalog. The difference is what is underneath it. In the current PDF:

```
[16pt bold text]  Degree Requirements
[8pt body text]   Total credits: 30 ...
[16pt bold text]  Required Courses
[8pt body text]   CSCI 505  Theory of Computation ...
```

The only signal that "Degree Requirements" is a heading is its font size. In well-structured HTML:

```html
<h1>Computer Science — Master of Science</h1>
<h2>Degree Requirements</h2>
<p>Total credits: 30...</p>
<h3>Required Courses</h3>
<table>
  <tr><td>CSCI 505</td><td>Theory of Computation</td><td>3</td></tr>
</table>
```

The `<h2>` tag is an unambiguous machine-readable declaration that "Degree Requirements" is a section heading. No font-size calibration, no heading-line buffer, no column-detection logic is needed. The retrieval system reads `<h2>` and knows with certainty where a new section begins.

---

## 3. What Changes — and What Does Not

### 3.1 What is eliminated

| Current requirement | Why it exists | Eliminated by HTML |
|---|---|---|
| Two-column character detection | PDF column layout | Yes — HTML has no columns |
| Font-size heading detection | PDF uses size as semantic signal | Yes — replaced by heading tags |
| Hardcoded page range table | PDF has no section URLs | Yes — each section has a URL |
| Annual page range audit | Ranges shift with each edition | Yes — URLs are stable |
| Heading-line join buffer | Multi-line headings fragment across lines | Yes — HTML headings are single elements |
| Unicode apostrophe handling | PDF encoding artifacts | Yes — HTML uses standard UTF-8 |
| Duplicate course description skip | Content appears twice in PDF | Yes — each description has one URL |

### 3.2 What remains

Some complexity is inherent to the retrieval problem and does not depend on the source format:

- **Semantic chunking.** A 3,000-word degree requirements page should not be one chunk. It still needs to be split into meaningful sub-sections. With HTML, heading tags make this straightforward and reliable.
- **Query understanding.** Recognizing that "when is CSCI 4120 offered?" is a scheduling question and "what are VWW requirements?" is a policy question requires query analysis regardless of how the catalog is formatted.
- **Dynamic faculty directory.** If the faculty directory page continues to use JavaScript rendering for individual faculty entries, those entries remain difficult to index regardless of the overall catalog format.

### 3.3 What improves beyond eliminating existing problems

- **New programs are auto-discovered.** When a new degree program is added, the catalog management system creates a new page. The web crawler follows links from the department page and indexes it automatically. No code change is required.
- **Content changes propagate automatically.** Revised admission requirements, updated course prerequisites, and changed deadlines are indexed on the next scheduled crawl without any manual intervention.
- **Chunk IDs can be stable.** If each section has a permanent URL (e.g., `cs/ms/requirements/`), the retrieval system can use the URL as a stable identifier. The test suite does not need to be updated when wording changes.

---

## 4. The Path Forward

### 4.1 The catalog management system likely already supports this

Most universities that produce a paginated PDF catalog use a catalog management system — Acalog, Courseleaf, Curricunet, or a custom system — that stores content in structured form and generates the PDF as one of several output formats. HTML output is typically a standard feature of these systems, not an additional development effort. The request to the university is to enable and publish the HTML output at stable URLs, not to rebuild how the catalog is authored or maintained.

If NMSU uses Acalog (common for institutions of its size), section-level HTML pages are available through Acalog's standard web portal. The department would need to work with the registrar or catalog office to confirm this and request that the HTML output be published at a stable public URL.

### 4.2 If the HTML catalog is not immediately available

If moving to a structured HTML output is not feasible in the near term, the following PDF-level changes would reduce (though not eliminate) the maintenance burden, roughly in order of impact:

1. **Single-column layout.** Removes the column-order boundary problem and the character-midpoint detection logic. The two most-documented retrieval errors in the current system trace directly to two-column layout.

2. **PDF heading tags.** Tagged PDFs (PDF/UA standard) include explicit structural markup for headings, paragraphs, and tables. This allows reading `<H2>` tags from the PDF directly rather than inferring heading level from font size. Most catalog management systems can export tagged PDFs.

3. **Remove embedded course descriptions from department sections.** The duplicate course descriptions on pages 568–580 require the parser to detect and skip them. Removing this duplication simplifies parsing and reduces catalog length.

4. **Spell out abbreviations on first use.** CAASS, MAP, VWW, and similar abbreviations currently require hardcoded expansion logic in the retrieval system. Spelling out the full name at first use in each section would eliminate this dependency.

### 4.3 Longer-term: structured course data

Course descriptions are the largest single component of the catalog (~5,400 entries, 700 pages). They are almost certainly maintained as structured records in Banner or an equivalent SIS. If those records are made available as a data export or API endpoint, the course description section of the PDF becomes unnecessary for retrieval purposes. Queries about course content, prerequisites, and credit hours would be answered from structured data rather than parsed text, with higher accuracy and no annual re-ingest required.

---

## 5. Summary of Recommendations

**Primary recommendation:** Request that the catalog office enable and publish the HTML output of the existing catalog management system at stable, public, section-level URLs. This eliminates the PDF parsing layer entirely and reduces annual maintenance from days to near zero.

**If HTML output is not immediately available:**
1. Convert to single-column layout
2. Enable PDF heading tags (PDF/UA)
3. Remove duplicate course descriptions from department sections
4. Spell out abbreviations on first use in each section

**Longer-term:** Expose course data from Banner as a structured export or API to replace parsed course descriptions with authoritative structured records.

None of these recommendations require changes to how the catalog is authored or reviewed. They are publishing and formatting choices that affect what is delivered to readers — and to retrieval systems — without changing the content itself.

---

*This recommendation is based on direct experience building and maintaining a retrieval system against the 2025–2026 NMSU Academic Catalog PDF. The technical obstacles described in Section 1.2 are documented in the system's issue log with specific page references and code-level evidence.*
