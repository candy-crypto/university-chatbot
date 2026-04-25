"""
ingest.py — Web crawler and ingestion pipeline for the university chatbot.

Crawls department web pages using Playwright, extracts structured sections,
chunks content by heading boundaries, embeds with OpenAI, and stores in
Weaviate under the unified DepartmentChunk schema.
"""

import os
import re
import uuid
import time
import yaml
from datetime import datetime, timezone
from collections import deque
from urllib.parse import urlparse

from dotenv import load_dotenv
from openai import OpenAI
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
from weaviate.classes.query import Filter
from weaviate.classes.data import DataObject

from weaviate_client import get_weaviate_client, ensure_collection, get_collection
from db import init_db, log_crawl_run

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")

openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Chunking guardrails — tunable via environment variables
CHUNK_MIN_LEN = int(os.getenv("CHUNK_MIN_LEN", "150"))
CHUNK_MAX_LEN = int(os.getenv("CHUNK_MAX_LEN", "2000"))

# Course code pattern: supports 4-digit numbers, alphabetic suffixes, space-separated prefixes
# e.g. CSCI 1115G, MATH 1511G, E E 1110
COURSE_CODE_RE = re.compile(r'\b([A-Z][A-Z\s]{0,5}\s+\d{3,4}[A-Z]?)\b')

# FAQ Q-marker pattern: matches Q1, Q7, Q101, Q201 etc. at a word boundary.
# Used to split FAQ-style pages into one chunk per Q&A pair.
_QA_MARKER_RE = re.compile(r'(?<!\w)Q\d+(?=[\s.])')


# ── Config ────────────────────────────────────────────────────────────────────

def load_department_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Text utilities ─────────────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


# ── URL filtering ──────────────────────────────────────────────────────────────

def is_allowed_url(url: str, config: dict) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False
    if parsed.netloc not in set(config.get("allowed_domains", [])):
        return False
    path = parsed.path or "/"
    if not any(path.startswith(p) for p in config.get("allowed_path_prefixes", ["/"])):
        return False
    for pattern in config.get("deny_url_patterns", []):
        if pattern in url:
            return False
    return True


# ── Page extraction ────────────────────────────────────────────────────────────

def extract_page_data(page, url: str, config: dict) -> dict:
    """
    Render the page with Playwright and extract:
      - h1: first h1 heading, used as the chunk heading field
      - text: full visible body text, used for course code scanning
      - sections: list of {heading, text} dicts split on h2/h3 boundaries, used for chunking
      - links: discovered hrefs for crawl queue
      - page_timestamp: last-modified metadata if present
    """
    page.goto(url, wait_until="domcontentloaded", timeout=30000)
    try:
        page.wait_for_load_state("networkidle", timeout=5000)
    except PlaywrightTimeoutError:
        pass

    # Faculty directory pages load their content via JavaScript.
    # Wait for a faculty entry element to appear so we capture the full list
    # rather than the static shell. The selectors cover the CS department
    # directory and the Data Analytics faculty directory.
    _FACULTY_URLS = {
        "computerscience.nmsu.edu/facultydirectory/faculty-staff-directory.html",
        "dataanalytics.nmsu.edu/facultydirectory/index.html",
    }
    if any(furl in url for furl in _FACULTY_URLS):
        for selector in (".fac-card", ".faculty-card", ".staff-member",
                         "[class*='faculty']", "[class*='staff']", ".views-row"):
            try:
                page.wait_for_selector(selector, timeout=8000)
                break
            except PlaywrightTimeoutError:
                continue

    # h1 for the heading field
    h1_list = page.locator("h1").evaluate_all(
        "elements => elements.map(el => el.innerText && el.innerText.trim()).filter(Boolean)"
    )

    # Last-modified metadata for provenance
    page_timestamp = page.locator(
        "meta[property='article:modified_time'], "
        "meta[name='last-modified'], "
        "meta[property='article:published_time']"
    ).evaluate_all(
        """elements => {
            const vals = elements.map(el => el.getAttribute('content')).filter(Boolean);
            return vals.length ? vals[0] : null;
        }"""
    )

    # Full body text for course code regex scanning
    body_text = normalize_text(page.locator("body").inner_text(timeout=5000))

    # Structured sections split on h2/h3 for chunking
    sections = page.evaluate("""
        () => {
            const SKIP  = new Set(['script','style','nav','footer','header','noscript']);
            const SPLIT = new Set(['h2','h3']);
            const root  = document.querySelector('main')
                       || document.querySelector('article')
                       || document.querySelector('#content')
                       || document.body;

            const result = [];
            let curHeading = '';
            let curParts   = [];

            function flush() {
                const t = curParts.join(' ').replace(/\\s+/g, ' ').trim();
                if (t) result.push({ heading: curHeading, text: t });
                curParts = [];
            }

            function walk(node) {
                if (node.nodeType === 3) {
                    const t = node.textContent.replace(/\\s+/g, ' ').trim();
                    if (t) curParts.push(t);
                } else if (node.nodeType === 1) {
                    const tag = node.tagName.toLowerCase();
                    if (SKIP.has(tag)) return;
                    if (SPLIT.has(tag)) {
                        flush();
                        curHeading = node.innerText.replace(/\\s+/g, ' ').trim();
                        return;
                    }
                    for (const child of node.childNodes) walk(child);
                }
            }

            for (const child of root.childNodes) walk(child);
            flush();
            return result;
        }
    """)

    # Collect allowed links for the crawl queue
    hrefs = page.locator("a[href]").evaluate_all(
        "elements => elements.map(el => el.href).filter(Boolean)"
    )
    links = set()
    for href in hrefs:
        clean = href.split("#")[0].strip()
        if clean and is_allowed_url(clean, config):
            links.add(clean)

    return {
        "url":            url,
        "h1":             h1_list[0] if h1_list else "",
        "text":           body_text,
        "sections":       sections,
        "links":          links,
        "page_timestamp": page_timestamp,
    }


# ── Chunking ───────────────────────────────────────────────────────────────────

def _expand_qa_sections(sections: list) -> list:
    """
    Split FAQ-style sections into one section per Q&A pair.

    If a section's text contains 2 or more Q\\d+ markers (e.g. Q1, Q101),
    it is split at each marker boundary so every question and its answer
    become a separate section.  The question text (up to 200 chars) is
    used as the heading so downstream retrieval can match on question wording.
    Sections with fewer than 2 markers are returned unchanged.
    """
    expanded = []
    for section in sections:
        text = section.get("text", "")
        parent_heading = section.get("heading", "")
        positions = [m.start() for m in _QA_MARKER_RE.finditer(text)]

        if len(positions) < 2:
            expanded.append(section)
            continue

        # Keep any preamble before the first Q-marker as its own section
        if positions[0] > 50:
            preamble = text[:positions[0]].strip()
            if preamble:
                expanded.append({"heading": parent_heading, "text": preamble})

        for i, start in enumerate(positions):
            end = positions[i + 1] if i + 1 < len(positions) else len(text)
            qa_text = text[start:end].strip()
            # Use the question portion as the heading (first 200 chars, trimmed
            # at the last space so we don't cut mid-word)
            raw_heading = qa_text[:200]
            cutoff = raw_heading.rfind(" ")
            qa_heading = raw_heading[:cutoff].strip() if cutoff > 0 else raw_heading
            expanded.append({"heading": qa_heading or parent_heading, "text": qa_text})

    return expanded


def chunk_page(sections: list, chunk_type: str,
               min_len: int = CHUNK_MIN_LEN,
               max_len: int = CHUNK_MAX_LEN) -> list:
    """
    Convert structured sections into chunks respecting min/max length guardrails.

    Returns a list of dicts: [{"heading": str, "text": str}]
      - heading: the h2/h3 section heading for this chunk (used as metadata field)
      - text: section heading prepended to body text (used for BM25 and embedding)

    Strategies by chunk_type:
      - course_schedule: keep as single chunk if under max_len
      - all others: split on h2/h3 section boundaries, merge short sections,
        split long sections at paragraph boundaries
    """
    if not sections:
        return []

    # Expand FAQ-style sections: if a section contains multiple Q\d+ markers,
    # split it into one section per Q&A pair before applying size guardrails.
    sections = _expand_qa_sections(sections)

    # course_schedule: one coherent document, keep whole if possible
    if chunk_type == "course_schedule":
        full = " ".join(
            (f"{s['heading']}\n{s['text']}" if s.get("heading") else s["text"]).strip()
            for s in sections
        ).strip()
        if len(full) <= max_len:
            return [{"heading": sections[0].get("heading", ""), "text": full}] if full else []
        # Fall through to normal splitting if over max

    chunks = []
    pending = ""
    pending_heading = ""

    for section in sections:
        heading = section.get("heading", "")
        text = section.get("text", "").strip()
        section_text = f"{heading}\n{text}".strip() if heading else text

        if not section_text:
            continue

        # Section exceeds max — split at paragraph or sentence boundaries
        if len(section_text) > max_len:
            if pending:
                chunks.append({"heading": pending_heading, "text": pending.strip()})
                pending = ""
                pending_heading = ""
            parts = re.split(r'\n\n+|\.\s+', section_text)
            current = ""
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                if current and len(current) + len(part) + 1 > max_len:
                    chunks.append({"heading": heading, "text": current.strip()})
                    current = part
                else:
                    current = (current + " " + part).strip() if current else part
            if current:
                chunks.append({"heading": heading, "text": current.strip()})
            continue

        # Section is too short — merge into pending
        if len(section_text) < min_len:
            if not pending:
                pending_heading = heading
            pending = (pending + "\n\n" + section_text).strip() if pending else section_text
            if len(pending) >= max_len:
                chunks.append({"heading": pending_heading, "text": pending})
                pending = ""
                pending_heading = ""
            continue

        # Normal section — flush pending first
        if pending:
            combined = (pending + "\n\n" + section_text).strip()
            if len(combined) <= max_len:
                chunks.append({"heading": pending_heading, "text": combined})
            else:
                chunks.append({"heading": pending_heading, "text": pending})
                chunks.append({"heading": heading, "text": section_text})
            pending = ""
            pending_heading = ""
        else:
            chunks.append({"heading": heading, "text": section_text})

    if pending:
        chunks.append({"heading": pending_heading, "text": pending.strip()})

    return [c for c in chunks if c.get("text")]


# ── Course code extraction ─────────────────────────────────────────────────────

def extract_course_code(heading: str, text: str) -> str:
    """Extract the primary course code from the page heading or top of body text."""
    match = COURSE_CODE_RE.search(f"{heading}\n{text[:500]}")
    if not match:
        return ""
    return re.sub(r'\s+', ' ', match.group(1)).strip()


def extract_referenced_courses(text: str) -> list:
    """Extract all unique course codes mentioned in the text."""
    return sorted({
        re.sub(r'\s+', ' ', m.group(1)).strip()
        for m in COURSE_CODE_RE.finditer(text)
    })


# ── Embedding ──────────────────────────────────────────────────────────────────

def embed_batch(texts: list) -> list:
    response = openai_client.embeddings.create(
        model=OPENAI_EMBED_MODEL,
        input=texts,
    )
    return [item.embedding for item in response.data]


# ── Crawl ──────────────────────────────────────────────────────────────────────

def crawl_site(config: dict, max_pages: int = 60) -> list:
    """
    Whitelist-only crawl of the configured department site.

    Only visits URLs explicitly listed in the config's 'pages' mapping (the
    page-type/level/degree-type classification table) plus any 'seed_urls'.
    BFS-discovered links are ignored — they are not added to the queue.

    This prevents the crawler from ingesting unlisted pages that happen to be
    linked from listed pages (e.g. login forms, news articles, staff-only docs).

    The 'pages' key replaces the old 'page_types' key; either is accepted for
    backwards compatibility during the transition.
    """
    root_url  = config["root_url"]
    campus    = config.get("campus", "")

    # Support both 'pages' (new) and 'page_types' (legacy) key names.
    pages_cfg         = config.get("pages") or config.get("page_types", {})
    page_levels       = config.get("page_levels", {})
    page_degree_types = config.get("page_degree_types", {})

    # The crawl queue is the union of:
    #   1. root_url (always visited)
    #   2. seed_urls listed in config
    #   3. all URLs explicitly listed in the 'pages' mapping
    # BFS-discovered links are crawled but NOT added to the queue.
    explicitly_listed = set(pages_cfg.keys())
    seed_urls = (
        [root_url]
        + [u for u in config.get("seed_urls", []) if u != root_url]
        + [u for u in explicitly_listed if u not in {root_url}]
    )
    # Deduplicate while preserving order (root first, seeds next, listed last).
    seen_order: dict[str, None] = {}
    for u in seed_urls:
        seen_order[u] = None
    seed_urls = list(seen_order.keys())

    visited = set()
    queued  = set(seed_urls)
    queue   = deque(seed_urls)
    pages   = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(user_agent="UniversityChatbotBot/1.0")
        page    = context.new_page()

        try:
            while queue and len(visited) < max_pages:
                url = queue.popleft()
                if url in visited:
                    continue

                try:
                    data = extract_page_data(page, url, config)
                    visited.add(url)

                    if len(data["text"]) > 100:
                        pages.append({
                            "url":                url,
                            "heading":            data["h1"],
                            "text":               data["text"],
                            "sections":           data["sections"],
                            "page_timestamp":     data.get("page_timestamp"),
                            "chunk_type":         pages_cfg.get(url, "general"),
                            "level":              page_levels.get(url, ""),
                            "degree_type":        page_degree_types.get(url, ""),
                            "campus":             campus,
                            "course_code":        extract_course_code(data["h1"], data["text"]),
                            "referenced_courses": extract_referenced_courses(data["text"]),
                        })

                    # Whitelist-only: do NOT add BFS-discovered links to the queue.
                    # (Links are still extracted for reference but not followed.)

                    time.sleep(0.3)

                except Exception as e:
                    print(f"Skipping {url}: {e}")
                    visited.add(url)

        finally:
            context.close()
            browser.close()

    return pages


# ── Weaviate upsert ────────────────────────────────────────────────────────────

def delete_department_chunks(collection, department_id: str):
    """Remove all existing web chunks for this department before re-ingesting."""
    collection.data.delete_many(
        where=(
            Filter.by_property("department_id").equal(department_id)
            & Filter.by_property("content_source").equal("web")
        )
    )


def upsert_pages_to_weaviate(pages: list, department_id: str, crawl_version: str) -> int:
    """
    Chunk pages by section boundaries, embed with OpenAI, and store in Weaviate.
    Falls back to full-text single chunk if structured chunking yields nothing.
    """
    normalized_department_id = (department_id or "").strip().lower()

    client = get_weaviate_client()
    try:
        ensure_collection(client)
        collection = get_collection(client)

        delete_department_chunks(collection, normalized_department_id)

        objects = []

        for page in pages:
            chunks = chunk_page(page["sections"], page["chunk_type"])

            # Fallback: if section-based chunking yields nothing, use full text
            if not chunks:
                text = page.get("text", "").strip()
                if text:
                    chunks = [{"heading": page.get("heading", ""), "text": text[:CHUNK_MAX_LEN]}]

            if not chunks:
                continue

            embeddings = embed_batch([c["text"] for c in chunks])

            for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                chunk_text = chunk["text"]
                # Use the section's own h2/h3 heading; fall back to page h1 only
                # if the section has no heading (e.g. pre-heading content).
                chunk_heading = chunk.get("heading") or page.get("heading", "")
                obj = DataObject(
                    uuid=str(uuid.uuid4()),
                    properties={
                        # ── Core fields ───────────────────────────────────────
                        "chunk_id":            f"{page['url']}#chunk-{i}",
                        "chunk_type":          page["chunk_type"],
                        "department_id":       normalized_department_id,
                        "campus":              page.get("campus", ""),
                        "text":                chunk_text,
                        "heading":             chunk_heading,
                        "source":              page["url"],
                        "level":               page.get("level", ""),
                        "degree_type":         page.get("degree_type", ""),
                        "course_code":         page.get("course_code", ""),
                        "referenced_courses":  page.get("referenced_courses", []),
                        "content_source":      "web",

                        # ── Catalog-specific — empty for web ──────────────────
                        "catalog_year":        "",
                        "catalog_page":        0,
                        "catalog_page_end":    0,
                        "degree_full_title":   "",
                        "concentration":       "",
                        "credits":             "",
                        "has_prerequisites":   False,
                        "policy_topic":        "",
                        "lab":                 "",
                        "research":            False,

                        # ── Web-specific ──────────────────────────────────────
                        "crawl_version":       crawl_version,
                    },
                    vector=emb,
                )
                objects.append(obj)

        if objects:
            collection.data.insert_many(objects)

        return len(objects)

    finally:
        client.close()


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    config_path = "configs/departments/cs.yaml"
    config      = load_department_config(config_path)

    department_id = config["department_id"]
    crawl_version = time.strftime("%Y%m%d_%H%M%S")

    init_db()

    pages = crawl_site(config, max_pages=60)

    chunks_indexed = upsert_pages_to_weaviate(
        pages=pages,
        department_id=department_id,
        crawl_version=crawl_version,
    )

    log_crawl_run(
        department_id=department_id,
        crawl_version=crawl_version,
        pages_scraped=len(pages),
        chunks_indexed=chunks_indexed,
    )

    print(f"Scraped {len(pages)} pages")
    print(f"Indexed {chunks_indexed} chunks into Weaviate")
    print(f"Crawl version: {crawl_version}")


if __name__ == "__main__":
    main()
