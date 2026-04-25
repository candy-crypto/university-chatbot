"""
catalog_chunker.py  —  NMSU Academic Catalog 2025-2026
=======================================================
PDF Parser, Chunker, and Weaviate Ingestion for the NMSU CS Department chatbot.

Key design decisions:
  - pdfplumber throughout for column-aware, font-size-aware extraction
  - pdfplumber 0-based page index == catalog page number
      pdf.pages[0]   = cover (unnumbered)
      pdf.pages[1]   = catalog page 1
      pdf.pages[567] = catalog page 567
  - Two-column layout on every page; split at x=306 (half of 612pt page)
  - Heading detection via font size: body=8pt, headings >= 12pt
  - TOC parsed first for page-range ground truth
  - Footnotes stay with their parent chunk (not separated)
  - All course codes mentioned in requirements captured in referenced_courses
  - Course descriptions parsed ONLY from end-of-catalog section (~p.1380+)
    The duplicate embedded course descriptions in dept sections are skipped.

Usage (dry run, Jupyter):
    from catalog_chunker import run_pipeline, filter_chunks, show_chunk
    chunks = run_pipeline(dry_run=True)
    degrees = filter_chunks(chunks, chunk_type="degree_requirement")
    show_chunk(degrees[0])

Usage (upload):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    chunks = run_pipeline(
        embed_fn=lambda texts: model.encode(texts).tolist(),
        dry_run=False,
    )
"""

import re
import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from typing import Optional
import unicodedata
import pdfplumber

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

PDF_PATH       = "25-26_New_Mexico_State_University_-_Las_Cruces.pdf"
CATALOG_YEAR   = "2025-2026"
WEAVIATE_URL   = "http://localhost:8080"
WEAVIATE_CLASS = "CatalogChunk"

# Font size thresholds (confirmed by probe of actual PDF)
BODY_SIZE          = 8.0   # Regular body text
SUBHEAD_SIZE       = 9.0   # Bold sub-labels ("Eligibility Requirements", etc.) — stay in chunk
HEADING_SIZE_MIN   = 12.0  # Subsection headings and above — chunk boundaries
PAGE_WIDTH         = 612.0
COL_SPLIT_X        = PAGE_WIDTH / 2  # 306pt — midpoint of letter page
HEADER_STRIP_Y     = 45.0  # Lines with top < this are running headers (actual y ≈ 36pt)

# CS embedded course description range to SKIP for course chunking
CS_COURSES_SKIP_START = 568
CS_COURSES_SKIP_END   = 580

# Fallback start for end-of-catalog course descriptions if TOC parse misses it
COURSE_DESC_FALLBACK_START = 1380

# ═══════════════════════════════════════════════════════════════════════════════
# DATA MODEL
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CatalogChunk:
    text:                str
    chunk_type:          str   # see CHUNK TYPES below
    catalog_page:        int
    catalog_page_end:    int  = -1   # same as catalog_page if single page

    # Identity hash (set automatically)
    chunk_id:            str  = ""

    # Program context
    dept_name:           str  = ""
    program_family:      list = field(default_factory=list)

    # Degree/minor context
    degree_level:        str  = ""   # undergraduate | ms | phd
    degree_type:         str  = ""   # bs | ba | ms | phd | minor | certificate
    concentration:       str  = ""   # general | cybersecurity | hci | ai | ...
    degree_full_title:   str  = ""

    # Course context
    course_code:         str  = ""
    course_title:        str  = ""
    credits:             str  = ""
    dept_prefix:         str  = ""
    course_number_level: str  = ""   # undergraduate | graduate
    has_prerequisites:   bool = False

    # Topic labels
    policy_topic:        str  = ""
    lab_name:            str  = ""

    # Cross-reference: every course code mentioned anywhere in the text
    referenced_courses:  list = field(default_factory=list)

    # Provenance
    catalog_year:        str  = CATALOG_YEAR
    source_scope:        str  = "catalog"
    is_research_related: bool = False

    def __post_init__(self):
        if not self.chunk_id:
            self.chunk_id = hashlib.sha256(self.text.encode()).hexdigest()[:16]
        if self.catalog_page_end == -1:
            self.catalog_page_end = self.catalog_page

# CHUNK TYPES:
#   dept_intro               Department overview / mission narrative
#   program_index            "Degrees for the Department" full listing
#   minor_index              "Minors for the Department" listing + eligibility rules
#   faculty                  Faculty roster with research interests
#   degree_core_requirement  Shared requirements (gen-ed, core courses, non-dept)
#                            for an entire degree family — one chunk used by all
#                            concentrations; avoids redundant retrieval
#   concentration_requirement Concentration-specific courses only — small, focused,
#                            uniquely identifies one concentration
#   degree_requirement       Full requirements for standalone degrees (BA, MS, PhD,
#                            and non-concentrated BS degrees)
#   study_plan               Suggested semester-by-semester roadmap
#   second_language          Second language requirement paragraph
#   minor_requirement        Individual minor / certificate requirements
#   course_description       Single course entry (end-of-catalog section only)
#   policy                   University-wide policy section
#   glossary                 Catalog glossary definitions
#   grad_program_info        Grad school entrance reqs, assistantships, funding
#   research                 Research facility or lab description

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE STRUCTURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def get_page_lines(page) -> list[dict]:
    """
    Extract all lines from a pdfplumber page object, annotated with:
      text, top (y-position), x0 (left edge), size (avg font size), col ('L'|'R')

    Lines are produced by grouping chars within 3pt vertical tolerance.
    Left column:  x0 < COL_SPLIT_X
    Right column: x0 >= COL_SPLIT_X
    Running-header lines (top < HEADER_STRIP_Y) are tagged is_header=True.
    """
    chars = page.chars
    if not chars:
        return []

    # Group chars by (column, y-bucket) so same-y headings in different
    # columns never merge into a single garbled line.
    #
    # Use the character horizontal midpoint (not its left edge x0) to decide
    # which column it belongs to. A heading that starts just left of COL_SPLIT_X
    # (e.g. "CSCI 5250" at x0=304, x1=312) has its centre in the right column
    # and should be classified as "R". Using x0 alone caused the leading character
    # ("C") to land in the left column, producing garbled headings like "SCI 4250".
    raw_lines = defaultdict(list)
    for c in chars:
        if not c["text"]:
            continue
        x_mid    = (c["x0"] + c.get("x1", c["x0"] + c.get("size", 6) * 0.55)) / 2
        col_key  = "L" if x_mid < COL_SPLIT_X else "R"
        y_bucket = round(c["top"] / 3) * 3
        raw_lines[(col_key, y_bucket)].append(c)

    lines = []
    for (col_key, y_bucket) in sorted(raw_lines.keys()):
        lc = sorted(raw_lines[(col_key, y_bucket)], key=lambda c: c["x0"])
        # Build text with gap-based space detection.
        # PDFs often encode spaces as advances (no space char), so we infer
        # them from horizontal gaps between consecutive character x-positions.
        text = ""
        for i, c in enumerate(lc):
            if i > 0:
                gap = c["x0"] - lc[i - 1].get("x1", lc[i - 1]["x0"] + 3)
                if gap > 1.5:
                    text += " "
            text += c["text"]
        text = text.strip()
        if not text:
            continue
        avg_size = sum(c["size"] for c in lc) / len(lc)
        avg_top  = sum(c["top"]  for c in lc) / len(lc)
        x0       = lc[0]["x0"]
        is_hdr   = avg_top < HEADER_STRIP_Y and _is_running_header(text)
        lines.append({
            "text":      text,
            "top":       avg_top,
            "x0":        x0,
            "size":      avg_size,
            "col":       col_key,
            "is_header": is_hdr,
        })
    return lines


# ── Remaining two-column limitation: section boundary mixing ──────────────────
#
# get_page_lines() sorts by (col_key, y_bucket), processing all left-column
# lines before any right-column lines. This is correct within a section, but
# breaks at section boundaries that fall mid-page.
#
# When a section ends partway down the left column and the NEXT section begins
# in the right column (above the left-column section end), the right-column
# tail of the ending section is read AFTER all left-column content of the new
# section. A few lines from section A's right column tail get appended to
# section B's chunk, and section A's chunk is missing those lines.
#
# Known affected pages: pp.41-42 (Prerequisites/Corequisites), pp.213-214
# (doctoral milestones). Effect is small — only lines near the column boundary
# are affected.
#
# Correct fix: process lines in y-bands spanning both columns so headings are
# seen in reading order rather than column order. This requires restructuring
# lines_to_text() and all downstream chunkers; deferred.
# ──────────────────────────────────────────────────────────────────────────────


def lines_to_text(lines: list[dict], skip_headers: bool = True) -> str:
    """
    Flatten a list of annotated lines to a plain text string.
    Reads left column top-to-bottom, then right column top-to-bottom.
    """
    left  = [l for l in lines if l["col"] == "L" and not (skip_headers and l["is_header"])]
    right = [l for l in lines if l["col"] == "R" and not (skip_headers and l["is_header"])]
    left  = sorted(left,  key=lambda l: l["top"])
    right = sorted(right, key=lambda l: l["top"])
    all_lines = left + right
    return "\n".join(l["text"] for l in all_lines)


def _is_running_header(text: str) -> bool:
    """True if the line looks like a page-number / department running header."""
    # "New Mexico State University - Las Cruces  567"
    if re.match(r'^New Mexico State University', text):
        return True
    # "567  Computer Science"  (double-space variant)
    if re.match(r'^\d{1,4}\s{2,}', text):
        return True
    # "584 Computer Science - Bachelor of Science"  (single-space, 3-4 digit page no.)
    if re.match(r'^\d{3,4}\s+[A-Z]', text):
        return True
    # "Computer Science - Bachelor of Arts  581"  (right-page header)
    if re.match(r'^[A-Z][A-Za-z ,\-]+\s{2,}\d{1,4}$', text):
        return True
    return False


def is_heading(line: dict, threshold: float = HEADING_SIZE_MIN) -> bool:
    """True if this line's font size marks it as a section heading."""
    return line["size"] >= threshold


# ═══════════════════════════════════════════════════════════════════════════════
# COURSE CODE UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

COURSE_CODE_RE = re.compile(r'\b([A-Z]{1,4}(?:\s[A-Z]{1,4})?)\s+(\d{3,4}[A-Z]?)\b')
COURSE_ENTRY_RE = re.compile(
    r'^([A-Z]{1,4}(?:\s[A-Z]{1,4})?\s+\d{3,4}[A-Z]?)\.\s+(.+)$'
)


def find_referenced_courses(text: str) -> list[str]:
    """Extract all course codes (PREFIX NNNN) mentioned anywhere in text."""
    found = set()
    for m in COURSE_CODE_RE.finditer(text):
        code = f"{m.group(1)} {m.group(2)}".strip()
        found.add(code)
    return sorted(found)


# ═══════════════════════════════════════════════════════════════════════════════
# TOC PARSER
# ═══════════════════════════════════════════════════════════════════════════════

TOC_ENTRY_RE = re.compile(r'^(.+?)\s*\.{2,}\s*(\d+)\s*$')


def parse_toc(pdf, toc_start: int = 2, toc_end: int = 13) -> list[dict]:
    """
    Parse the two-column TOC (catalog pages 2-13) into a sorted list:
      [{"catalog_page": int, "title": str, "indent": float, "section_type": str}, ...]

    indent is the x0 position of the entry's first character — used as a proxy
    for hierarchical level (lower x0 = higher level).
    """
    entries = []
    for pg in range(toc_start, min(toc_end + 1, len(pdf.pages))):
        lines = get_page_lines(pdf.pages[pg])
        for line in lines:
            if line["is_header"]:
                continue
            m = TOC_ENTRY_RE.match(line["text"])
            if not m:
                continue
            title    = m.group(1).strip()
            page_num = int(m.group(2))
            if not title or page_num < 1:
                continue
            entries.append({
                "catalog_page": page_num,
                "title":        title,
                "indent":       line["x0"],   # raw x0 as indent proxy
                "section_type": _infer_section_type(title),
            })

    entries.sort(key=lambda e: e["catalog_page"])
    return entries


def build_page_range_map(entries: list[dict]) -> dict[int, dict]:
    """
    Compute end_page for each TOC entry based on the next entry at the same
    or higher (lower indent) level.  Returns {start_page: {**entry, end_page}}.
    """
    result = {}
    for i, entry in enumerate(entries):
        end = entry["catalog_page"]
        for j in range(i + 1, len(entries)):
            nxt = entries[j]
            if nxt["indent"] <= entry["indent"]:
                end = nxt["catalog_page"] - 1
                break
        else:
            end = entries[-1]["catalog_page"]
        result[entry["catalog_page"]] = {**entry, "end_page": max(end, entry["catalog_page"])}
    return result


def find_course_section_start(entries: list[dict]) -> int:
    """Return the catalog page where the end-of-catalog course listings begin."""
    for e in entries:
        t = e["title"].lower()
        if "course" in t and e["catalog_page"] > 1000:
            return e["catalog_page"]
    return COURSE_DESC_FALLBACK_START


def _infer_section_type(title: str) -> str:
    t = title.lower()
    if "bachelor of arts"        in t: return "degree_ba"
    if "bachelor of science"     in t: return "degree_bs"
    if "master of"               in t: return "degree_ms"
    if "doctor of"               in t: return "degree_phd"
    if "undergraduate minor"     in t: return "minor_undergrad"
    if "graduate minor"          in t: return "minor_grad"
    if "graduate certificate"    in t: return "certificate"
    if "research facilit"        in t: return "research"
    if "graduate school"         in t: return "grad_program_info"
    if "transfer"                in t: return "policy"
    if "veteran" in t or "military" in t: return "policy"
    if "international student"   in t: return "policy"
    if "financial aid"           in t: return "policy"
    if "tuition"                 in t: return "policy"
    return "general"


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION-SPECIFIC CHUNKERS
# ═══════════════════════════════════════════════════════════════════════════════

def chunk_generic_pages(
    pdf,
    start: int,
    end:   int,
    chunk_type:  str,
    dept_name:   str = "",
    lab_name:    str = "",
    topic:       str = "",
    degree_level: str = "",
    split_at_size: float = HEADING_SIZE_MIN,
) -> list[CatalogChunk]:
    """
    General-purpose chunker for policy, grad_program_info, and research pages.
    Splits into a new chunk whenever a heading (size >= split_at_size) is found.
    Heading text becomes the policy_topic / lab_name of the new chunk.

    Consecutive heading lines are buffered and joined before being committed as
    the topic label, so multi-line titles like:
      "Research Initiatives in the College of Health, Education"
      "and Social Transformation"
    are stored as one label rather than the second line overwriting the first.
    """
    chunks     = []
    cur_lines  = []
    cur_page   = start
    cur_topic  = topic
    hbuf       = []   # accumulates consecutive heading lines
    hbuf_page  = start

    def flush():
        text = "\n".join(cur_lines).strip()
        if len(text) < 120:
            return
        # Drop preamble fragments before the first heading in policy/research
        if chunk_type in ("research", "policy", "grad_program_info") and not cur_topic:
            return
        c = CatalogChunk(
            text=text,
            chunk_type=chunk_type,
            catalog_page=cur_page,
            dept_name=dept_name,
            policy_topic=cur_topic if chunk_type in ("policy", "grad_program_info") else "",
            lab_name=cur_topic if chunk_type == "research" else lab_name,
            degree_level=degree_level,
            referenced_courses=find_referenced_courses(text),
            is_research_related=(chunk_type == "research"),
        )
        chunks.append(c)

    def commit_hbuf():
        """Resolve buffered heading lines → flush previous chunk, set new topic."""
        nonlocal cur_topic, cur_page, cur_lines
        if not hbuf:
            return
        full_topic = " ".join(hbuf)
        flush()
        cur_lines  = []
        cur_topic  = full_topic
        cur_page   = hbuf_page
        # Include the heading text in the chunk body so it is retrievable
        cur_lines.append(full_topic)

    for pg in range(start, min(end + 1, len(pdf.pages))):
        for line in get_page_lines(pdf.pages[pg]):
            if line["is_header"]:
                continue
            if is_heading(line, split_at_size):
                # Accumulate consecutive heading lines
                if not hbuf:
                    hbuf_page = pg
                hbuf.append(line["text"])
            else:
                # Body line — commit any pending heading first
                if hbuf:
                    commit_hbuf()
                    hbuf.clear()
                cur_lines.append(line["text"])

    # End of range — commit any trailing heading, then flush
    if hbuf:
        commit_hbuf()
        hbuf.clear()
    flush()
    return chunks


def chunk_dept_intro(pdf, start: int, end: int, dept_name: str,
                     start_heading: str = "") -> list[CatalogChunk]:
    """
    Parse department intro page(s).  Splits on 16pt+ headings into:
      program_index  ("Degrees for the Department")
      minor_index    ("Minors for the Department")
      faculty        ("Faculty", "College Faculty")
      dept_intro     (any narrative before the first major heading)
    Skips the embedded "XYZ Courses" section entirely.
    """
    DEPT_CHUNK_TYPES = {
        "Degrees for the Department": "program_index",
        "Minors for the Department":  "minor_index",
        "Faculty":                    "faculty",
        "College Faculty":            "faculty",
    }
    COURSES_HEADING_RE = re.compile(r'.+\sCourses?$', re.IGNORECASE)

    chunks      = []
    cur_lines   = []
    cur_type    = "dept_intro"
    cur_page    = start
    in_courses  = False
    found_start = (start_heading == "")
    norm_fn     = lambda s: re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s.strip().lower()))
    norm_start  = norm_fn(start_heading)
    hbuf_dept   = []   # buffer for multi-line 18pt dept heading

    def flush():
        text = "\n".join(cur_lines).strip()
        if len(text) < 30:
            return
        c = CatalogChunk(
            text=text,
            chunk_type=cur_type,
            catalog_page=cur_page,
            dept_name=dept_name,
            program_family=[_dept_to_family(dept_name)],
            referenced_courses=find_referenced_courses(text),
        )
        chunks.append(c)

    for pg in range(start, min(end + 1, len(pdf.pages))):
        lines = get_page_lines(pdf.pages[pg])
        for line in lines:
            if line["is_header"]:
                continue
            txt = line["text"]

            # Skip content before the department heading
            if not found_start:
                if line["size"] >= 18:
                    hbuf_dept.append(txt)
                    if norm_fn(" ".join(hbuf_dept)) == norm_start:
                        found_start = True
                        cur_page    = pg
                        hbuf_dept   = []
                else:
                    hbuf_dept = []
                continue

            # Detect start of embedded course descriptions — skip this section
            if is_heading(line) and COURSES_HEADING_RE.match(txt):
                flush()
                cur_lines  = []
                in_courses = True
                continue

            # Detect a known major heading — ends the course section if we were in it
            if is_heading(line) and txt in DEPT_CHUNK_TYPES:
                flush()
                cur_lines  = []
                cur_type   = DEPT_CHUNK_TYPES[txt]
                cur_page   = pg
                in_courses = False
                continue   # heading text is captured as chunk_type, not content

            if not in_courses:
                cur_lines.append(txt)

    flush()
    return chunks



def chunk_cs_bs_core(
    pdf,
    start:             int,
    end:               int,
    degree_full_title: str,
    dept_name:         str = "",
    start_heading:     str = "Computer Science - Bachelor of Science",
) -> list[CatalogChunk]:
    """
    Extract the shared CS BS core requirements as a single
    degree_core_requirement chunk (pages 584-585).

    Skips content before the CS BS heading (which is the CS BA tail),
    then collects all content through the end of the page range, stopping
    if an 18pt concentration heading appears.
    """
    degree_type, concentration, degree_level = _parse_degree_meta(degree_full_title)
    family = _dept_to_family(dept_name)
    norm   = lambda s: re.sub(r'\s+', ' ',
                              unicodedata.normalize('NFKC', s.strip().lower()))
    norm_start = norm(start_heading)

    cur_lines   = []
    cur_page    = start
    found_start = False
    hbuf        = []   # accumulate multi-line 18pt headings to detect start

    for pg in range(start, min(end + 1, len(pdf.pages))):
        stop_page = False
        for line in get_page_lines(pdf.pages[pg]):
            if line["is_header"]:
                continue
            if line["size"] >= 18:
                if not found_start:
                    # Accumulate to detect our start heading
                    hbuf.append(line["text"])
                    combined = ' '.join(hbuf)
                    if norm(combined) == norm_start:
                        found_start = True
                        cur_page    = pg
                        cur_lines   = [combined]
                        hbuf        = []
                    continue
                else:
                    # A new concentration heading — stop
                    stop_page = True
                    break
            else:
                hbuf = []   # reset if non-18pt line interrupts
                if found_start:
                    cur_lines.append(line["text"])
        if stop_page:
            break

    text = "\n".join(cur_lines).strip()
    if len(text) < 100:
        return []
    return [CatalogChunk(
        text=text,
        chunk_type="degree_core_requirement",
        catalog_page=cur_page,
        catalog_page_end=end,
        degree_full_title=degree_full_title,
        degree_type=degree_type,
        concentration="general",
        degree_level=degree_level,
        dept_name=dept_name,
        program_family=[family] if family else [],
        referenced_courses=find_referenced_courses(text),
    )]

def chunk_degree_section(
    pdf,
    start:             int,
    end:               int,
    degree_full_title: str,
    dept_name:         str = "",
    start_heading:     str = "",
    is_concentration:  bool = False,
) -> list[CatalogChunk]:
    """
    Parse one degree program section (requirements + study plan + second language).

    Handles multi-line headings by buffering consecutive heading lines and
    dispatching the combined text once a non-heading line appears.  This is
    necessary because large degree titles wrap across two pdfplumber lines.

    start_heading: normalized-fuzzy match against the accumulated heading buffer;
    all content before the first match is skipped (handles shared transition pages).
    """
    degree_type, concentration, degree_level = _parse_degree_meta(degree_full_title)
    family = _dept_to_family(dept_name)
    norm = lambda s: re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', s.strip().lower()))
    norm_title = norm(degree_full_title)
    norm_start = norm(start_heading) if start_heading else ""

    OTHER_DEGREE_RE = re.compile(
        r'.+\s[-–]\s*(Bachelor of (Arts|Science)|Master of|Doctor of)',
        re.IGNORECASE,
    )
    STUDY_PLAN_RE   = re.compile(r'^A Suggested Plan of Study', re.IGNORECASE)
    SECOND_LANG_RE  = re.compile(r'^Second Language Requirement', re.IGNORECASE)
    # Marks the transition from shared requirements to concentration-specific courses.
    # The ligature 'fi' -> 'ﬁ' is normalized by NFKC; the regex handles both forms.
    CONC_SPEC_RE    = re.compile(
        r'The spe[ck]i.ic requirements for the concentration in',
        re.IGNORECASE,
    )

    chunks       = []
    cur_lines    = []
    cur_type     = "degree_requirement"
    cur_page     = start
    found_start  = (start_heading == "")
    in_conc_spec    = False   # True once we pass the concentration-specific trigger
    post_conc_skip = False   # True after conc chunk done — skip until next heading
    hbuf           = []   # (text, page) for consecutive heading lines

    def flush():
        text = "\n".join(cur_lines).strip()
        if len(text) < 40:
            return
        c = CatalogChunk(
            text=text,
            chunk_type=cur_type,
            catalog_page=cur_page,
            catalog_page_end=end,
            degree_full_title=degree_full_title,
            degree_type=degree_type,
            concentration=concentration,
            degree_level=degree_level,
            dept_name=dept_name,
            program_family=[family] if family else [],
            referenced_courses=find_referenced_courses(text),
            is_research_related=bool(
                re.search(r'research|thesis|dissertation', text, re.IGNORECASE)
            ),
        )
        chunks.append(c)

    def dispatch_hbuf():
        """Flush the heading buffer.  Returns True if processing should stop."""
        nonlocal cur_lines, cur_type, cur_page, found_start, hbuf, post_conc_skip
        if not hbuf:
            return False
        combined = ' '.join(t for t, _ in hbuf)
        pg       = hbuf[-1][1]
        hbuf     = []
        nc       = norm(combined)

        # Still searching for our start heading?
        if not found_start:
            if nc == norm_start:
                found_start = True
            return False

        # Internal chunk boundaries
        if STUDY_PLAN_RE.match(combined):
            if cur_type == "study_plan":
                # Already inside a study_plan chunk (e.g. Bioinformatics has two
                # tracks on consecutive pages). Including the second heading as
                # content merges both tracks into one chunk, avoiding duplicate
                # metadata that wastes top-k slots at retrieval time.
                cur_lines.append(combined)
            else:
                flush()
                # Include the heading as the first line so the chunk opens with
                # its own label (e.g. "A Suggested Plan of Study for Students
                # (with a Computer Science background)").
                cur_lines      = [combined]
                cur_type       = "study_plan"
                cur_page       = pg
                post_conc_skip = False
            return False
        if SECOND_LANG_RE.match(combined):
            flush()
            cur_lines = [combined]   # include heading for same reason
            cur_type  = "second_language"
            cur_page  = pg
            return False

        # Another degree starting?
        if OTHER_DEGREE_RE.match(combined) and nc != norm_title:
            flush()
            return True  # stop

        # Minor/certificate title headings → we've left the degree section
        if re.search(r'(Undergraduate|Graduate)\s+(Minor|Certificate)', combined, re.IGNORECASE):
            flush()
            return True  # stop

        # Sub-heading within our section — include as content
        cur_lines.append(combined)
        return False

    for pg in range(start, min(end + 1, len(pdf.pages))):
        for line in get_page_lines(pdf.pages[pg]):
            if line["is_header"]:
                continue
            if is_heading(line):
                hbuf.append((line["text"], pg))
            else:
                if hbuf:
                    if dispatch_hbuf():
                        return chunks
                if found_start:
                    txt = line["text"]
                    # Routing logic for concentration mode:
                    #   Phase 1 (not in_conc_spec, not post_conc_skip):
                    #     skip shared req table, wait for CONC_SPEC_RE trigger
                    #   Phase 2 (in_conc_spec):
                    #     collect concentration-specific courses until Total Credits
                    #   Phase 3 (post_conc_skip=True): skip footnotes
                    #   Phase 4 (post_conc_skip=False, cur_type==study_plan or later):
                    #     collect normally
                    if is_concentration and not in_conc_spec and post_conc_skip:
                        pass   # Phase 3: skip footnotes between conc block and study plan
                    elif is_concentration and not in_conc_spec and cur_type not in (
                            "study_plan", "second_language", "concentration_done"):
                        # Phase 1: skip shared requirements table, watch for trigger
                        norm_txt = re.sub(r'\s+', ' ',
                                          unicodedata.normalize('NFKC', txt).strip())
                        if CONC_SPEC_RE.search(norm_txt):
                            in_conc_spec = True
                            cur_type     = "concentration_requirement"
                            cur_lines    = [degree_full_title, txt]
                            cur_page     = pg
                        # else: skip — this is the repeated shared requirements table
                    elif is_concentration and in_conc_spec:
                        # Collecting concentration-specific courses.
                        # Stop at "Total Credits" only while still in concentration_requirement
                        # (after a study_plan transition, Total Credits is semester subtotals).
                        if re.match(r'^Total Credits', txt) and cur_type == "concentration_requirement":
                            cur_lines.append(txt)
                            flush()
                            # Done with concentration courses.
                            # Skip body text until the next heading (study_plan).
                            in_conc_spec    = False
                            post_conc_skip  = True
                            cur_lines       = []
                            cur_type        = "concentration_done"
                        else:
                            cur_lines.append(txt)
                    else:
                        if not post_conc_skip:
                            # "A Suggested Plan of Study" sometimes prints at
                            # sub-heading font size and is not routed through
                            # dispatch_hbuf(). Catch it here so the chunk type
                            # still transitions from degree_requirement to study_plan.
                            if STUDY_PLAN_RE.match(txt) and cur_type != "study_plan":
                                flush()
                                cur_lines      = [txt]
                                cur_type       = "study_plan"
                                cur_page       = pg
                                post_conc_skip = False
                            elif STUDY_PLAN_RE.match(txt) and cur_type == "study_plan":
                                # Second study plan heading (e.g. Bioinformatics two-track)
                                # — merge into current chunk as dispatch_hbuf() would.
                                cur_lines.append(txt)
                            else:
                                cur_lines.append(txt)

    if hbuf:
        dispatch_hbuf()
    flush()
    return chunks


def chunk_minor_pages(pdf, start: int, end: int, dept_name: str,
                      start_heading: str = "") -> list[CatalogChunk]:
    """
    Parse minor/certificate requirement pages.

    Minor titles appear at 18pt and are sometimes split across two pdfplumber
    lines (e.g. "Algorithm Theory - Undergraduate" / "Minor").  We accumulate
    consecutive 18pt heading lines into a buffer and commit them as the chunk
    title when body text (< 18pt) appears.

    Any heading at < 18pt is a department-level heading (e.g. "Undergraduate
    Program Information" at 16pt) indicating a different department has started
    — we stop immediately and return collected chunks.
    """
    MINOR_HDG_SIZE = 18.0   # minor titles are 18pt throughout the catalog

    chunks       = []
    cur_lines    = []
    cur_title    = ""
    cur_page     = start
    family       = _dept_to_family(dept_name)
    hbuf         = []   # accumulates multi-line 18pt heading fragments
    hbuf_page    = start

    def flush():
        nonlocal cur_lines
        text = "\n".join(cur_lines).strip()
        if len(text) < 40:
            cur_lines = []
            return
        level = "graduate" if re.search(r"\bgraduate\b", cur_title, re.IGNORECASE) else "undergraduate"
        c = CatalogChunk(
            text=text,
            chunk_type="minor_requirement",
            catalog_page=cur_page,
            degree_full_title=cur_title,
            degree_type="minor",
            degree_level=level,
            dept_name=dept_name,
            program_family=[family] if family else [],
            referenced_courses=find_referenced_courses(text),
        )
        chunks.append(c)
        cur_lines = []

    def commit_hbuf():
        """Resolve buffered heading lines into cur_title and start a new chunk."""
        nonlocal cur_title, cur_page, cur_lines
        if not hbuf:
            return
        full_title = " ".join(hbuf)
        flush()                   # save previous chunk
        cur_title = full_title
        cur_page  = hbuf_page
        cur_lines = [full_title]  # include heading in content for retrieval

    found_minor_start = (start_heading == "")
    norm_minor_start  = re.sub(r'\s+', ' ', start_heading.strip().lower())

    for pg in range(start, min(end + 1, len(pdf.pages))):
        for line in get_page_lines(pdf.pages[pg]):
            if line["is_header"]:
                continue
            size = line["size"]
            txt  = line["text"]

            # Skip pre-content until our expected first minor heading appears
            if not found_minor_start:
                # Buffer 18pt lines; commit when we find the start heading
                if size >= MINOR_HDG_SIZE:
                    hbuf.append(txt)
                    combined = " ".join(hbuf)
                    if re.sub(r'\s+', ' ', combined.strip().lower()) == norm_minor_start:
                        found_minor_start = True
                        # heading text begins the first chunk
                        cur_title = combined
                        cur_page  = pg
                        cur_lines = [combined]
                        hbuf.clear()
                else:
                    hbuf.clear()   # reset on body line before start
                continue

            if size >= MINOR_HDG_SIZE:
                # Accumulate multi-line minor title
                if not hbuf:
                    hbuf_page = pg     # page of first line of this heading
                hbuf.append(txt)

            elif is_heading(line):
                # 12–17 pt = department-level heading → this is a different dept, stop
                commit_hbuf()
                hbuf.clear()
                flush()
                return chunks

            else:
                # Body text — commit any pending heading before adding content
                if hbuf:
                    commit_hbuf()
                    hbuf.clear()
                cur_lines.append(txt)

    if hbuf:
        commit_hbuf()
        hbuf.clear()
    flush()
    return chunks


def chunk_course_descriptions(pdf, start: int, end: int) -> list[CatalogChunk]:
    """
    Parse end-of-catalog course description pages.
    Each course entry begins with the pattern "PREFIX NNNN[G]. Course Title".

    Streams page-by-page to avoid accumulating all 700+ pages in memory at once.
    Content accumulation per entry is capped at COURSE_CHUNK_MAX_CHARS to prevent
    non-course tail content (indexes, appendices) at the end of the catalog from
    being absorbed into the last matched course entry.
    """
    # Cap at 8000 tokens worth of characters (well under the 8192 embedding limit).
    COURSE_CHUNK_MAX_CHARS = 8000 * 4

    chunks         = []
    cur_lines      = []
    cur_page       = start
    cur_code       = ""
    cur_title_text = ""

    def flush():
        text = "\n".join(cur_lines).strip()
        # Truncate to cap before building the chunk — anything beyond is non-course content.
        if len(text) > COURSE_CHUNK_MAX_CHARS:
            print(f"  [warn] {cur_code} p{cur_page}: truncated from {len(text)} to "
                  f"{COURSE_CHUNK_MAX_CHARS} chars (non-course tail content discarded)")
            text = text[:COURSE_CHUNK_MAX_CHARS]
        if len(text) < 30 or not cur_code:
            return
        pm = re.match(r'^([A-Z]{1,4}(?:\s[A-Z]{1,4})?)', cur_code)
        prefix = pm.group(1).strip() if pm else ""
        nm = re.search(r'\d+', cur_code)
        level = "graduate" if nm and int(nm.group()) >= 5000 else "undergraduate"
        cm = re.search(r'(\d[\d\-]+)\s*Credits?', text, re.IGNORECASE)
        credits = cm.group(1) if cm else ""
        has_prereq = bool(re.search(r'Prerequisite', text, re.IGNORECASE))
        chunks.append(CatalogChunk(
            text=text,
            chunk_type="course_description",
            catalog_page=cur_page,
            course_code=cur_code,
            course_title=cur_title_text,
            credits=credits,
            dept_prefix=prefix,
            dept_name=prefix,
            course_number_level=level,
            has_prerequisites=has_prereq,
            referenced_courses=find_referenced_courses(text),
        ))

    # Stream one page at a time — never loads more than one page in memory
    for pg in range(start, min(end + 1, len(pdf.pages))):
        for line in get_page_lines(pdf.pages[pg]):
            if line["is_header"]:
                continue
            line_text = line["text"]
            m = COURSE_ENTRY_RE.match(line_text)
            if m:
                flush()
                cur_lines      = [line_text]
                cur_page       = pg
                cur_code       = m.group(1).strip()
                cur_title_text = m.group(2).strip()
            else:
                cur_lines.append(line_text)

    flush()
    return chunks


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_degree_meta(title: str) -> tuple[str, str, str]:
    """
    Extract (degree_type, concentration, degree_level) from a degree title.
    "Computer Science (Cybersecurity) - Bachelor of Science"
      → ("bs", "cybersecurity", "undergraduate")
    """
    t = title.lower()
    if   "bachelor of arts"    in t: degree_type = "ba";  degree_level = "undergraduate"
    elif "bachelor of science" in t: degree_type = "bs";  degree_level = "undergraduate"
    elif "master of"           in t: degree_type = "ms";  degree_level = "ms"
    elif "doctor of"           in t: degree_type = "phd"; degree_level = "phd"
    else:                            degree_type = "";     degree_level = ""
    conc_m = re.search(r'\(([^)]+)\)', title)
    concentration = conc_m.group(1).lower().replace(" ", "_") if conc_m else "general"
    return degree_type, concentration, degree_level


def _dept_to_family(dept_name: str) -> str:
    d = dept_name.lower()
    if "computer science"       in d: return "computer_science"
    if "data analytics"         in d: return "data_analytics"
    if "bioinformatics"         in d: return "bioinformatics"
    if "electrical engineering" in d: return "electrical_engineering"
    return re.sub(r'\W+', '_', d)[:30].strip('_')


# ═══════════════════════════════════════════════════════════════════════════════
# WEAVIATE UPLOAD
# ═══════════════════════════════════════════════════════════════════════════════

def upload_to_weaviate(
    chunks:        list[CatalogChunk],
    weaviate_url:  str,
    embed_fn,
    api_key:       str  = None,
    batch_size:    int  = 50,
) -> int:
    """
    Upload chunks to Weaviate v4.  embed_fn(list[str]) -> list[list[float]].
    Creates the collection if it doesn't exist.  Returns upload count.
    """
    import weaviate
    from weaviate.classes.config import Configure, Property, DataType

    kwargs = {"url": weaviate_url}
    if api_key:
        kwargs["auth_credentials"] = weaviate.auth.AuthApiKey(api_key)

    with weaviate.connect_to_custom(**kwargs) as client:
        if not client.collections.exists(WEAVIATE_CLASS):
            client.collections.create(
                name=WEAVIATE_CLASS,
                vectorizer_config=Configure.Vectorizer.none(),
                properties=[
                    Property(name="text",                data_type=DataType.TEXT),
                    Property(name="chunk_type",          data_type=DataType.TEXT),
                    Property(name="catalog_page",        data_type=DataType.INT),
                    Property(name="catalog_page_end",    data_type=DataType.INT),
                    Property(name="chunk_id",            data_type=DataType.TEXT),
                    Property(name="dept_name",           data_type=DataType.TEXT),
                    Property(name="program_family",      data_type=DataType.TEXT_ARRAY),
                    Property(name="degree_level",        data_type=DataType.TEXT),
                    Property(name="degree_type",         data_type=DataType.TEXT),
                    Property(name="concentration",       data_type=DataType.TEXT),
                    Property(name="degree_full_title",   data_type=DataType.TEXT),
                    Property(name="course_code",         data_type=DataType.TEXT),
                    Property(name="course_title",        data_type=DataType.TEXT),
                    Property(name="credits",             data_type=DataType.TEXT),
                    Property(name="dept_prefix",         data_type=DataType.TEXT),
                    Property(name="course_number_level", data_type=DataType.TEXT),
                    Property(name="has_prerequisites",   data_type=DataType.BOOL),
                    Property(name="policy_topic",        data_type=DataType.TEXT),
                    Property(name="lab_name",            data_type=DataType.TEXT),
                    Property(name="referenced_courses",  data_type=DataType.TEXT_ARRAY),
                    Property(name="catalog_year",        data_type=DataType.TEXT),
                    Property(name="source_scope",        data_type=DataType.TEXT),
                    Property(name="is_research_related", data_type=DataType.BOOL),
                ],
            )

        col      = client.collections.get(WEAVIATE_CLASS)
        uploaded = 0
        with col.batch.dynamic() as batch:
            for i in range(0, len(chunks), batch_size):
                sub = chunks[i : i + batch_size]
                vectors = embed_fn([c.text for c in sub])
                for chunk, vec in zip(sub, vectors):
                    props = {k: v for k, v in asdict(chunk).items() if k != "text"}
                    batch.add_object(
                        properties={**props, "text": chunk.text},
                        vector=vec,
                        uuid=chunk.chunk_id,
                    )
                    uploaded += 1
    return uploaded


# ═══════════════════════════════════════════════════════════════════════════════
# EXPLICIT SECTION TABLE
# ═══════════════════════════════════════════════════════════════════════════════
# Each entry: (start_cat_page, end_cat_page, handler, config_dict)
# 'start_heading' in config: skip content before this heading (transition pages)

EXPLICIT_SECTIONS = [
    # ── Glossary ─────────────────────────────────────────────────────────────
    # topic pre-set so content before the first heading is not dropped.
    (22,  31,  "glossary", {"topic": "Glossary"}),
    # ── University policies ──────────────────────────────────────────────────
    (31,  55,  "policy",   {"dept_name": ""}),
    (60,  73,  "policy",   {"dept_name": ""}),
    # ── General Education & VWW (undergraduate only) ─────────────────────────
    # p241: Gen Ed intro paragraph + Area I-IV course lists
    # p242: Area V-VI + 9-credit rule + VWW definition paragraph + alternatives
    # pp243-244: VWW course lists by college
    # Course lists are redundant with G/V-suffix course descriptions but the
    # definitional paragraphs are essential for "What are VWW requirements?" queries.
    (241, 244, "policy",   {"topic": "General Education and Viewing a Wider World Requirements",
                            "degree_level": "undergraduate"}),
    # ── Academic Advising / CAASS ─────────────────────────────────────────────
    # p41: CAASS mentioned in passing within registration policy — keep for retrieval.
    # p447-449: College of Arts and Sciences intro — contains the full CAASS name in
    #   the college header. Ends just before "Bachelor Degrees" heading on p449.
    # p955 and p1129 are incidental mentions in programs of no interest — omitted.
    (41,  41,  "policy",   {"dept_name": "", "topic": "Center for Academic Advising and Student Support (CAASS)"}),
    (447, 449, "policy",   {"dept_name": "", "topic": "College of Arts and Sciences",
                            "start_heading": "College of Arts and Sciences"}),
    # ── Graduate School intro, entrance reqs, assistantships ─────────────────
    # p86-90: alphabetical master's degree index — skip (no retrieval value)
    # p91-105: graduate programs not of interest — skip
    (74,  79,  "grad_info",{}),   # graduate school intro
    # ── Research Facilities (all labs — no CS filter) ─────────────────────────
    (80,  85,  "research", {}),   # all research facilities; heading buffer handles p85 R-col
    # ── CS Department ─────────────────────────────────────────────────────────
    # Intro page: program_index + minor_index + faculty
    # (568-580 embedded course descriptions are skipped inside chunk_dept_intro)
    # CS dept intro: starts at p565 but skip until "Computer Science" 18pt heading
    (565, 580, "dept_intro",  {"dept_name": "Computer Science",
                               "start_heading": "Computer Science"}),
    # CS BA — heading appears at bottom of p581 R-col; everything above is
    # Communication Studies tail content, so skip until the BA heading.
    (581, 584, "degree", {
        "title":        "Computer Science - Bachelor of Arts",
        "dept_name":    "Computer Science",
        "start_heading":"Computer Science - Bachelor of Arts",
    }),
    # CS BS shared core requirements (pages 584-585) — one chunk for all concentrations
    (584, 585, "degree_core", {
        "title":    "Computer Science - Bachelor of Science",
        "dept_name":"Computer Science",
    }),
    # CS BS second language + study plan (skip BA tail via start_heading)
    (584, 587, "degree", {
        "title":        "Computer Science - Bachelor of Science",
        "dept_name":    "Computer Science",
        "start_heading":"Computer Science - Bachelor of Science",
    }),
    # CS concentrations (BS)
    (586, 588, "degree", {"title": "Computer Science (Algorithm Theory) - Bachelor of Science", "is_concentration": True,
                          "dept_name": "Computer Science",
                          "start_heading": "Computer Science (Algorithm Theory) - Bachelor of Science"}),
    (589, 591, "degree", {"title": "Computer Science (Artificial Intelligence) - Bachelor of Science", "is_concentration": True,
                          "dept_name": "Computer Science",
                          "start_heading": "Computer Science (Artificial Intelligence) - Bachelor of Science"}),
    (591, 593, "degree", {"title": "Computer Science (Big Data and Data Science) - Bachelor of Science", "is_concentration": True,
                          "dept_name": "Computer Science",
                          "start_heading": "Computer Science (Big Data and Data Science) - Bachelor of Science"}),
    (594, 596, "degree", {"title": "Computer Science (Computer Networking) - Bachelor of Science", "is_concentration": True,
                          "dept_name": "Computer Science",
                          "start_heading": "Computer Science (Computer Networking) - Bachelor of Science"}),
    (596, 599, "degree", {"title": "Computer Science (Cybersecurity) - Bachelor of Science", "is_concentration": True,
                          "dept_name": "Computer Science",
                          "start_heading": "Computer Science (Cybersecurity) - Bachelor of Science"}),
    (599, 601, "degree", {"title": "Computer Science (Human Computer Interaction) - Bachelor of Science", "is_concentration": True,
                          "dept_name": "Computer Science",
                          "start_heading": "Computer Science (Human Computer Interaction) - Bachelor of Science"}),
    (601, 604, "degree", {"title": "Computer Science (Secondary Education) - Bachelor of Arts",
                          "dept_name": "Computer Science",
                          "start_heading": "Computer Science (Secondary Education) - Bachelor of Arts"}),
    (604, 606, "degree", {"title": "Computer Science (Software Development) - Bachelor of Science", "is_concentration": True,
                          "dept_name": "Computer Science",
                          "start_heading": "Computer Science (Software Development) - Bachelor of Science"}),
    (606, 608, "degree", {"title": "Cybersecurity - Bachelor of Science",
                          "dept_name": "Computer Science",
                          "start_heading": "Cybersecurity - Bachelor of Science"}),
    # CS undergraduate minors — start_heading skips Cybersecurity BS study-plan tail on p608
    (608, 610, "minor",  {"dept_name": "Computer Science",
                          "start_heading": "Algorithm Theory - Undergraduate Minor"}),
    # ── Graduate degrees ──────────────────────────────────────────────────────
    # CS MS heading appears mid-page 128 — MAP section tail occupies the top of the page
    (128, 130, "degree", {"title": "Computer Science - Master of Science",
                          "dept_name": "Computer Science",
                          "start_heading": "Computer Science - Master of Science"}),
    (132, 134, "degree", {"title": "Data Analytics - Master of Data Analytics",     "dept_name": "Data Analytics"}),
    (106, 108, "degree", {"title": "Bioinformatics - Master of Science",
                          "dept_name": "Bioinformatics",
                          "start_heading": "Bioinformatics - Master of Science"}),
    (191, 194, "degree", {"title": "Computer Science - Doctor of Philosophy",
                          "dept_name": "Computer Science",
                          "start_heading": "Computer Science - Doctor of Philosophy"}),
    # ── Graduate minors relevant to CS programs (p230) ──────────────────────────
    # Both minors are in p230 R-col; chunk_minor_pages handles multi-line 18pt titles
    # and stops at the next non-18pt heading (Economics at y=702).
    # start_heading skips the L-col content (other dept minors) on the same page.
    (230, 230, "minor", {
        "dept_name":    "Computer Science",
        "start_heading":"Bioinformatics (with Computer Science) - Graduate Minor",
    }),
    # ── EE AI/ML/DS concentration ─────────────────────────────────────────────
    (1015, 1016, "degree", {
        "title":        "Electrical Engineering (Artificial Intelligence, Machine Learning, & Data Science) - Bachelor of Science in Electrical Engineering",
        "dept_name":    "Electrical Engineering",
        "start_heading":"Electrical Engineering (Artificial Intelligence, Machine Learning, & Data Science) - Bachelor of Science in Electrical Engineering",
    }),
]


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline(
    pdf_path:       str  = PDF_PATH,
    weaviate_url:   str  = WEAVIATE_URL,
    weaviate_api_key: str = None,
    embed_fn             = None,
    dry_run:        bool = True,
    include_courses: bool = True,
    course_prefixes: list = None,
) -> list[CatalogChunk]:
    """
    Full ingestion pipeline.

    dry_run=True  → parse and return chunks without uploading (default)
    dry_run=False → requires embed_fn; uploads to Weaviate after parsing

    include_courses: if True, chunk ALL course descriptions (pages ~1379–end of PDF).
    course_prefixes: unused — kept for call-site compatibility. All courses are included.
    """

    all_chunks: list[CatalogChunk] = []
    seen_ids: set[str] = set()

    def add(chunks):
        for c in chunks:
            if c.chunk_id not in seen_ids:
                all_chunks.append(c)
                seen_ids.add(c.chunk_id)

    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        print(f"Opened: {total} pages  (catalog pages 0–{total - 1})")

        # ── TOC ──────────────────────────────────────────────────────────────
        print("Parsing TOC …")
        toc = parse_toc(pdf)
        print(f"  {len(toc)} entries found")
        course_start = find_course_section_start(toc)
        print(f"  Course descriptions start at catalog page {course_start}")

        # ── Explicit sections ─────────────────────────────────────────────────
        for (start, end, handler, cfg) in EXPLICIT_SECTIONS:
            label = cfg.get("title", cfg.get("dept_name", handler))
            print(f"  p{start}–{end}  {handler:10s}  {label[:60]}")

            if handler == "policy":
                add(chunk_generic_pages(pdf, start, end, "policy",
                                        dept_name=cfg.get("dept_name", ""),
                                        topic=cfg.get("topic", ""),
                                        degree_level=cfg.get("degree_level", "")))

            elif handler == "glossary":
                add(chunk_generic_pages(pdf, start, end, "glossary",
                                        topic=cfg.get("topic", "Glossary")))

            elif handler == "grad_info":
                add(chunk_generic_pages(pdf, start, end, "grad_program_info"))

            elif handler == "research":
                add(chunk_generic_pages(pdf, start, end, "research"))

            elif handler == "dept_intro":
                add(chunk_dept_intro(pdf, start, end, cfg["dept_name"],
                                    start_heading=cfg.get("start_heading", "")))

            elif handler == "degree_core":
                add(chunk_cs_bs_core(
                    pdf, start, end,
                    degree_full_title=cfg["title"],
                    dept_name=cfg.get("dept_name", ""),
                ))

            elif handler == "degree":
                add(chunk_degree_section(
                    pdf, start, end,
                    degree_full_title=cfg["title"],
                    dept_name=cfg.get("dept_name", ""),
                    start_heading=cfg.get("start_heading", ""),
                    is_concentration=cfg.get("is_concentration", False),
                ))

            elif handler == "minor":
                add(chunk_minor_pages(pdf, start, end, cfg.get("dept_name", ""),
                                      start_heading=cfg.get("start_heading", "")))

    # ── End-of-catalog course descriptions (batched, outside main PDF handle) ──
    # Fresh PDF handle per batch so pdfplumber's per-page cache is released
    # between batches — avoids OOM on the 700-page course section.
    # ALL course descriptions are included — no prefix filter — per original design.
    if include_courses:
        COURSE_BATCH = 80
        course_total = 0
        print(f"  Parsing ALL course descriptions from p{course_start} "
              f"in batches of {COURSE_BATCH} pages …")
        for b_start in range(course_start, total, COURSE_BATCH):
            b_end = min(b_start + COURSE_BATCH - 1, total - 1)
            with pdfplumber.open(pdf_path) as pdf_b:
                batch = chunk_course_descriptions(pdf_b, b_start, b_end)
            add(batch)
            course_total += len(batch)
            if batch:
                print(f"    p{b_start}–{b_end}: {len(batch)} entries")
        print(f"  {course_total} course chunks total")

    print(f"\nTotal chunks: {len(all_chunks)}")
    _print_summary(all_chunks)

    if not dry_run:
        if embed_fn is None:
            raise ValueError("embed_fn is required when dry_run=False")
        print(f"\nUploading to Weaviate at {weaviate_url} …")
        n = upload_to_weaviate(all_chunks, weaviate_url, embed_fn, weaviate_api_key)
        print(f"Uploaded {n} chunks.")

    return all_chunks


def _print_summary(chunks):
    counts = Counter(c.chunk_type for c in chunks)
    print("\nChunk type breakdown:")
    for ct, n in sorted(counts.items()):
        print(f"  {ct:<25s} {n:4d}")


# ═══════════════════════════════════════════════════════════════════════════════
# INSPECTION UTILITIES  (for Jupyter)
# ═══════════════════════════════════════════════════════════════════════════════

def filter_chunks(
    chunks:        list[CatalogChunk],
    chunk_type:    str  = None,
    dept_name:     str  = None,
    degree_type:   str  = None,
    degree_level:  str  = None,
    concentration: str  = None,
    program_family:str  = None,
    is_research:   bool = None,
    course_prefix: str  = None,
) -> list[CatalogChunk]:
    r = chunks
    if chunk_type:    r = [c for c in r if c.chunk_type    == chunk_type]
    if dept_name:     r = [c for c in r if dept_name.lower() in c.dept_name.lower()]
    if degree_type:   r = [c for c in r if c.degree_type   == degree_type]
    if degree_level:  r = [c for c in r if c.degree_level  == degree_level]
    if concentration: r = [c for c in r if concentration.lower() in c.concentration.lower()]
    if program_family:r = [c for c in r if program_family in c.program_family]
    if is_research is not None: r = [c for c in r if c.is_research_related == is_research]
    if course_prefix: r = [c for c in r if c.dept_prefix.startswith(course_prefix)]
    return r


def show_chunk(chunk: CatalogChunk, preview: int = 600):
    """Pretty-print a chunk for notebook inspection."""
    sep = "─" * 64
    print(sep)
    print(f"type        : {chunk.chunk_type}")
    print(f"pages       : {chunk.catalog_page}–{chunk.catalog_page_end}")
    print(f"title       : {chunk.degree_full_title or '—'}")
    print(f"dept        : {chunk.dept_name or '—'}")
    print(f"degree      : {chunk.degree_type or '—'}  level: {chunk.degree_level or '—'}  conc: {chunk.concentration or '—'}")
    if chunk.course_code:
        print(f"course      : {chunk.course_code}  ({chunk.credits} cr)  prereq={chunk.has_prerequisites}")
    if chunk.policy_topic:
        print(f"topic       : {chunk.policy_topic}")
    if chunk.lab_name:
        print(f"lab         : {chunk.lab_name}")
    refs = chunk.referenced_courses
    print(f"ref_courses : {refs[:12]}{'…' if len(refs) > 12 else ''}")
    print(f"\ntext ({len(chunk.text)} chars):\n")
    print(chunk.text[:preview])
    if len(chunk.text) > preview:
        print(f"\n  … [{len(chunk.text) - preview} chars omitted]")
    print(sep)


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    chunks = run_pipeline(dry_run=True)
