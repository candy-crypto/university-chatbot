"""
nmsu_catalog_chunk_inspector.py
================================
Jupyter helper for exploring and validating catalog chunks produced by
nmsu_catalog_chunker.py.  Run all cells top-to-bottom after chunking.

Usage:
    # In Jupyter (same directory as nmsu_catalog_chunker.py):
    %run nmsu_catalog_chunk_inspector.py
    # or import individual helpers:
    from nmsu_catalog_chunk_inspector import *
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from nmsu_catalog_chunker import (
    run_pipeline, filter_chunks, show_chunk, CatalogChunk
)
from collections import Counter

PDF_PATH = "25-26_New_Mexico_State_University_-_Las_Cruces.pdf"

# ── 1. Run the pipeline (dry run — no embedding/upload) ──────────────────────
# Set include_courses=True for the full run; False for quick structural check.
# On first use, include_courses=False (~2 min) to validate structure.
# include_courses=True adds ~1000+ course chunks but takes ~8 min.

chunks = run_pipeline(
    pdf_path=PDF_PATH,
    dry_run=True,
    include_courses=False,   # ← change to True for full ingest
)

# ── 2. Summary ───────────────────────────────────────────────────────────────
print("\n=== CHUNK TYPE BREAKDOWN ===")
for ct, n in sorted(Counter(c.chunk_type for c in chunks).items()):
    print(f"  {ct:<25s} {n:4d}")

# ── 3. Per-type quick views ───────────────────────────────────────────────────

def show_all(chunk_type: str, preview: int = 300):
    """Print preview of every chunk of a given type."""
    found = filter_chunks(chunks, chunk_type=chunk_type)
    print(f"\n=== {chunk_type.upper()} ({len(found)} chunks) ===")
    for c in found:
        show_chunk(c, preview=preview)

def show_degree(title_fragment: str, preview: int = 500):
    """Find and display a degree_requirement chunk by partial title match."""
    found = [c for c in chunks
             if c.chunk_type == "degree_requirement"
             and title_fragment.lower() in c.degree_full_title.lower()]
    if not found:
        print(f"No degree_requirement found matching {title_fragment!r}")
        return
    for c in found:
        show_chunk(c, preview=preview)

def show_courses(prefix: str, n: int = 5):
    """Show first n course_description chunks for a given prefix."""
    found = filter_chunks(chunks, chunk_type="course_description", course_prefix=prefix)
    print(f"\n=== {prefix} COURSES ({len(found)} total, showing first {n}) ===")
    for c in found[:n]:
        show_chunk(c, preview=300)

def find_by_course_code(code: str):
    """Find all degree chunks that reference a given course code."""
    results = [c for c in chunks if code in c.referenced_courses]
    print(f"\nChunks referencing {code}: {len(results)}")
    for c in results:
        title = c.degree_full_title or c.policy_topic or c.lab_name
        print(f"  [{c.chunk_type}]  {title}  (p{c.catalog_page})")

def check_quality():
    """Quick health check — flags obvious problems."""
    issues = []
    for c in chunks:
        if c.chunk_type == "research" and not c.lab_name:
            issues.append(f"research chunk p{c.catalog_page} has no lab_name")
        if c.chunk_type in ("policy", "grad_program_info") and not c.policy_topic:
            issues.append(f"{c.chunk_type} chunk p{c.catalog_page} has no topic")
        if c.chunk_type == "degree_requirement" and not c.degree_full_title:
            issues.append(f"degree_requirement p{c.catalog_page} has no title")
        if len(c.text) < 80:
            issues.append(f"{c.chunk_type} p{c.catalog_page} suspiciously short ({len(c.text)} chars)")

    if not issues:
        print("✓ No quality issues found")
    else:
        print(f"⚠ {len(issues)} issues:")
        for i in issues:
            print(f"  {i}")

# ── 4. Common inspection calls (uncomment as needed) ─────────────────────────

# show_all("degree_requirement")
# show_all("study_plan")
# show_all("minor_requirement")
# show_all("policy", preview=200)
# show_all("research", preview=200)
# show_degree("Cybersecurity")
# show_degree("Master of Science")
# show_degree("Doctor of Philosophy")
# show_degree("Electrical Engineering")
# show_courses("CSCI")
# show_courses("E E")
# find_by_course_code("CSCI 4540")
# find_by_course_code("MATH 1511G")
check_quality()
