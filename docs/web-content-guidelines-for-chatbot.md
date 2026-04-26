# Web Content Guidelines for Chatbot Retrieval

These guidelines help ensure that department web pages can be fully and accurately indexed
by the CS department chatbot. The chatbot uses a Playwright-based crawler that renders
pages in a headless browser, followed by a DOM walker that extracts visible text content.

## Core Principle

**Content must be present in the DOM on initial page load.** The crawler loads each page
once and extracts whatever is rendered at that point. Content that requires user interaction
to appear — clicking tabs, advancing pages, scrolling — may not be captured.

---

## Specific Recommendations

### Avoid JavaScript-only pagination

When a directory or list is split across multiple pages using client-side pagination
(e.g., AngularJS or React rendering page 2 on button click without changing the URL),
only the first page is captured.

**Preferred alternatives:**
- Display all entries on a single scrollable page
- Use URL-based pagination (`?page=2`, `/page/2`) so each page has a distinct URL the
  crawler can visit independently

*Example of current problem:* The faculty directory at
`computerscience.nmsu.edu/facultydirectory/faculty-staff-directory.html` displays 12 of
31 faculty on the initial load; pages 2 and 3 require clicking pagination buttons and
are not automatically captured.

### Avoid tab-switching that hides content

If a page organizes content into tabs (e.g., "Tenure Track", "Emeriti", "Staff"), only
the active tab's content is in the DOM at load time. Inactive tab content is typically
hidden or not yet rendered.

**Preferred alternative:** Place all content on a single page with headings, or use
separate pages per category linked from a navigation menu.

### Use descriptive headings

The chunking pipeline splits content at `<h2>` and `<h3>` boundaries. Pages with clear,
descriptive headings produce well-scoped chunks that retrieve accurately. Pages with no
headings produce a single large chunk that is harder to match to specific queries.

**Recommendation:** Use `<h2>` and `<h3>` headings to organize sections (e.g.,
"Admission Requirements", "Degree Requirements", "Financial Aid").

### Keep contact information in a consistent format

The crawler captures structured text as-is. Consistent formatting makes it easier for
the language model to extract specific fields.

The faculty directory format works well:

```
Name  Title  email  (phone) | SH room
```

For example: `hcao@nmsu.edu (575) 646-4600 | SH 171`

Maintain this format across all faculty entries so phone numbers and office locations
can be reliably identified.

### Avoid iframes and embedded PDFs for key content

Content inside `<iframe>` elements and embedded PDFs is not accessible to the DOM
walker. Important information (degree requirements, policies, schedules) should be in
the page's own HTML, not loaded from a separate document.

### Prefer static HTML over client-rendered content

Content that exists in the raw HTML response is always available to the crawler.
Content that appears only after JavaScript executes depends on the crawler's ability
to fully render the page. While the chatbot crawler does use a full browser (Playwright),
purely static content is more robust and faster to index.

### Keep URLs stable

The chatbot indexes content at specific URLs. If a page moves, its content disappears
from the index until the next re-crawl, and any direct links included in chatbot
responses become broken. Use stable, predictable URL structures and set up redirects
when pages must move.

---

## Summary Table

| Practice | Impact on Chatbot |
|---|---|
| Client-side pagination (no URL change) | Only page 1 indexed |
| Tab-switching to reveal content | Only active tab indexed |
| No `<h2>`/`<h3>` headings | One large imprecise chunk |
| Consistent contact info format | Phone/room reliably extracted |
| Content in iframes or PDFs | Not indexed |
| Stable URLs | Links in answers remain valid |
