"""
nmsu_course_chunker.py
======================
Chunks end-of-catalog course descriptions (catalog pages 1377–2067) from the
NMSU 2025-2026 Academic Catalog into CatalogChunk objects for Weaviate ingestion.

PAGE OFFSET NOTE
----------------
The catalog PDF has an unnumbered cover as page 0 (0-based index). Catalog
page numbering begins on the next page:
    pdf.pages[0]   → cover (no catalog number)
    pdf.pages[1]   → catalog page 1
    pdf.pages[N]   → catalog page N

The 0-based pdfplumber index equals the catalog page number directly.
All page references in this script use catalog page numbers.

DESIGN
------
The course section (pages 1377–2067, ~690 pages) is divided into blocks of
BLOCK_SIZE pages. Each block is opened and closed independently so peak
memory is proportional to one block, not the entire section.

Prefix attribution is derived from the course code in each entry
(e.g. "CSCI 4405. AI I" → prefix "CSCI"), not from block boundaries.
No prefix table is needed; all courses carry their own identity.

USAGE (Jupyter)
---------------
    from nmsu_course_chunker import run_course_pipeline, filter_chunks, show_chunk

    # Parse all courses, no upload (~6 min in a normal Jupyter environment)
    chunks = run_course_pipeline(
        pdf_path="25-26_New_Mexico_State_University_-_Las_Cruces.pdf",
        dry_run=True,
    )

    # Upload (requires embed_fn)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    chunks = run_course_pipeline(
        pdf_path="...",
        embed_fn=lambda texts: model.encode(texts).tolist(),
        dry_run=False,
    )

    # Inspect
    show_chunk(filter_chunks(chunks, prefix="CSCI")[0])
    find_course(chunks, "CSCI 4405")
"""

import re
from collections import defaultdict
from typing import Optional
import pdfplumber

try:
    from nmsu_catalog_chunker import CatalogChunk, upload_to_weaviate
except ImportError:
    from catalog_chunker import CatalogChunk, upload_to_weaviate


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

PDF_PATH          = "25-26_New_Mexico_State_University_-_Las_Cruces.pdf"
CATALOG_YEAR      = "2025-2026"
WEAVIATE_URL      = "http://localhost:8080"
WEAVIATE_CLASS    = "CatalogChunk"
COURSE_START_PAGE = 1380   # p1377-1379 are the course index/preamble; clean entries start at 1380
COURSE_END_PAGE   = 2067
BLOCK_SIZE        = 50     # pages per block; ~25 s each in a typical environment
COL_SPLIT_X       = 306.0  # midpoint of 612pt letter page

# Running-header pattern (y < 45pt matching this → strip)
HEADER_RE = re.compile(
    r'^(?:New Mexico State University\s*[-–]\s*Las Cruces\s*\d+|\d{3,4}\s+[A-Z].*)$'
)

# Course entry: "CSCI 4405. Artificial Intelligence I"
# Group 1 = code, Group 2 = title
COURSE_ENTRY_RE = re.compile(
    r'^([A-Z][A-Z\s]{0,5}\s*\d{3,4}[A-Z]?)\.\s+(.+)$'
)

# Any course code appearing in text (for referenced_courses metadata)
COURSE_CODE_RE = re.compile(r'\b([A-Z][A-Z\s]{0,5}\s+\d{3,4}[A-Z]?)\b')

# Text patterns that indicate the regex-matched "title" is actually the tail of the
# previous course's description (e.g. "EPWS 471. May be repeated..." after a line
# ending with "Same as HORT 471 and").  These strings never begin a real course title.
_NONTITLE_RE = re.compile(
    r'^(?:May\s+be\b|Cannot\s+be\b|\d+\s*(?:to|[-\u2013])\s*\d+\s+Credit|'
    r'Prerequisite[s]?\b|Cross[\s-]listed|Formerly\b)',
    re.IGNORECASE,
)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def _page_lines(page) -> list[tuple[str, bool]]:
    """
    Text lines from one page in reading order: L-col top→bottom, then R-col.
    Running headers stripped.

    Returns a list of (text, bold_start) tuples where bold_start is True when
    the first character of the line is in a bold font.  Course-entry headings in
    the NMSU catalog are always bold; cross-reference text ("Same as …") is not.
    """
    raw = defaultdict(list)
    for c in page.chars:
        if not c["text"]:
            continue
        col      = "L" if c["x0"] < COL_SPLIT_X else "R"
        y_bucket = round(c["top"] / 3) * 3
        raw[(col, y_bucket)].append(c)

    lines = []
    for (col, y_bucket) in sorted(raw.keys()):
        lc   = sorted(raw[(col, y_bucket)], key=lambda c: c["x0"])
        text = "".join(c["text"] for c in lc).strip()
        if not text:
            continue
        avg_top = sum(c["top"] for c in lc) / len(lc)
        if avg_top < 45 and HEADER_RE.match(text):
            continue
        bold_start = "bold" in (lc[0].get("fontname") or "").lower()
        lines.append((text, bold_start))
    return lines


# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK PROCESSOR
# ═══════════════════════════════════════════════════════════════════════════════

def _chunk_block(pdf_path: str, start: int, end: int) -> list[CatalogChunk]:
    """
    Parse catalog pages start–end into course chunks.
    Opens and closes the PDF once per block.
    """
    chunks    = []
    cur_lines = []
    cur_page  = start
    cur_code  = ""
    cur_title = ""

    def flush():
        text = "\n".join(cur_lines).strip()
        if len(text) < 30 or not cur_code:
            return

        code_clean = re.sub(r'[\xa0\s]+', ' ', cur_code).strip()

        # Prefix = everything before the first digit
        pm     = re.match(r'^([A-Z][A-Z\s]*?)\s*\d', code_clean)
        prefix = pm.group(1).strip() if pm else code_clean.split()[0]

        nm     = re.search(r'(\d+)', code_clean)
        level  = "graduate" if nm and int(nm.group()) >= 5000 else "undergraduate"

        cm      = re.search(r'([\d][\d\-]*)\s*Credits?', text, re.IGNORECASE)
        credits = cm.group(1) if cm else ""

        has_prereq  = bool(re.search(r'Prerequisite', text, re.IGNORECASE))
        ref_courses = sorted({
            re.sub(r'[\xa0\s]+', ' ', m.group(1)).strip()
            for m in COURSE_CODE_RE.finditer(text)
        })

        chunks.append(CatalogChunk(
            text=text,
            chunk_type="course_description",
            catalog_page=cur_page,
            catalog_page_end=cur_page,
            course_code=code_clean,
            course_title=re.sub(r'[\xa0]+', ' ', cur_title).strip(),
            credits=credits,
            dept_prefix=prefix,
            dept_name=prefix,
            course_number_level=level,
            has_prerequisites=has_prereq,
            referenced_courses=ref_courses,
            catalog_year=CATALOG_YEAR,
            source_scope="catalog",
        ))

    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        for pg in range(start, min(end + 1, total)):
            for line, bold_start in _page_lines(pdf.pages[pg]):
                m = COURSE_ENTRY_RE.match(line)
                # A genuine course entry heading is bold AND matches COURSE_ENTRY_RE.
                # Neither condition alone is sufficient:
                #   • bold alone: "Prerequisite:" / "Corequisite:" labels are also bold
                #   • regex alone: cross-reference tails like "EPWS 471. May be repeated…"
                #                  match the DEPT NNN. pattern but are not bold
                # Two additional text-based guards catch edge cases where font metadata
                # may be unreliable:
                #   (a) previous line ends with a conjunction — we are mid-sentence
                #   (b) the matched "title" starts with a known non-title fragment
                mid_sentence = bool(
                    cur_lines and
                    cur_lines[-1].rstrip().endswith((' and', ' or', ' &'))
                )
                if (m and bold_start
                        and not mid_sentence
                        and not _NONTITLE_RE.match(m.group(2))):
                    flush()
                    cur_lines = [line]
                    cur_page  = pg
                    cur_code  = m.group(1)
                    cur_title = m.group(2)
                else:
                    cur_lines.append(line)

    flush()
    return chunks


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_course_pipeline(
    pdf_path:         str  = PDF_PATH,
    weaviate_url:     str  = WEAVIATE_URL,
    weaviate_api_key: str  = None,
    embed_fn               = None,
    dry_run:          bool = True,
    block_size:       int  = BLOCK_SIZE,
    verbose:          bool = True,
) -> list[CatalogChunk]:
    """
    Parse all course descriptions and optionally upload to Weaviate.

    dry_run=True  → return chunks without uploading (default)
    dry_run=False → requires embed_fn; embeds and uploads all chunks
    """
    starts = list(range(COURSE_START_PAGE, COURSE_END_PAGE + 1, block_size))
    blocks = [(s, min(s + block_size - 1, COURSE_END_PAGE)) for s in starts]

    if verbose:
        total_pages = COURSE_END_PAGE - COURSE_START_PAGE + 1
        print(f"Course description pipeline")
        print(f"  Pages  : {COURSE_START_PAGE}–{COURSE_END_PAGE} ({total_pages} pages)")
        print(f"  Blocks : {len(blocks)} × {block_size} pages")
        print(f"  Mode   : {'dry run' if dry_run else 'embed + upload'}")
        print()

    all_chunks = []
    seen_ids   = set()

    for i, (start, end) in enumerate(blocks):
        for c in _chunk_block(pdf_path, start, end):
            if c.chunk_id not in seen_ids:
                all_chunks.append(c)
                seen_ids.add(c.chunk_id)
        if verbose:
            print(f"  Block {i+1:3d}/{len(blocks)}  "
                  f"p{start}–{end}  "
                  f"running total: {len(all_chunks)}")

    if verbose:
        from collections import Counter
        print(f"\nTotal: {len(all_chunks)} course chunks")
        print("\nTop 20 prefixes:")
        for p, n in Counter(c.dept_prefix for c in all_chunks).most_common(20):
            print(f"  {p:<10} {n:4d}")

    if not dry_run:
        if embed_fn is None:
            raise ValueError("embed_fn required when dry_run=False")
        print(f"\nUploading to {weaviate_url} …")
        n = upload_to_weaviate(all_chunks, weaviate_url, embed_fn, weaviate_api_key)
        print(f"Uploaded {n} chunks.")

    return all_chunks


# ═══════════════════════════════════════════════════════════════════════════════
# INSPECTION UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def filter_chunks(
    chunks:         list[CatalogChunk],
    prefix:         str  = None,
    level:          str  = None,    # "undergraduate" | "graduate"
    has_prereqs:    bool = None,
    title_contains: str  = None,
    code_contains:  str  = None,
) -> list[CatalogChunk]:
    r = [c for c in chunks if c.chunk_type == "course_description"]
    if prefix:
        r = [c for c in r if c.dept_prefix == prefix.strip()]
    if level:
        r = [c for c in r if c.course_number_level == level]
    if has_prereqs is not None:
        r = [c for c in r if c.has_prerequisites == has_prereqs]
    if title_contains:
        r = [c for c in r if title_contains.lower() in c.course_title.lower()]
    if code_contains:
        r = [c for c in r if code_contains.upper() in c.course_code.upper()]
    return r


def find_course(chunks: list[CatalogChunk], code: str) -> Optional[CatalogChunk]:
    """Find a course by code, e.g. find_course(chunks, 'CSCI 4405')."""
    target = re.sub(r'[\xa0\s]+', ' ', code).strip().upper()
    for c in chunks:
        if re.sub(r'[\xa0\s]+', ' ', c.course_code).strip().upper() == target:
            return c
    return None


def show_chunk(chunk: CatalogChunk, preview: int = 500):
    sep = "─" * 64
    print(sep)
    print(f"code        : {chunk.course_code}")
    print(f"title       : {chunk.course_title}")
    print(f"prefix      : {chunk.dept_prefix}")
    print(f"credits     : {chunk.credits or '—'}")
    print(f"level       : {chunk.course_number_level}")
    print(f"prereqs     : {chunk.has_prerequisites}")
    print(f"page        : {chunk.catalog_page}")
    if chunk.referenced_courses:
        print(f"ref_courses : {chunk.referenced_courses[:12]}")
    print(f"\ntext ({len(chunk.text)} chars):\n")
    print(chunk.text[:preview])
    if len(chunk.text) > preview:
        print(f"\n  … [{len(chunk.text) - preview} chars omitted]")
    print(sep)
