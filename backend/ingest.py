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

def chunk_page(sections: list, chunk_type: str,
               min_len: int = CHUNK_MIN_LEN,
               max_len: int = CHUNK_MAX_LEN) -> list:
    """
    Convert structured sections into chunks respecting min/max length guardrails.

    Strategies by chunk_type:
      - course_schedule: keep as single chunk if under max_len
      - all others: split on h2/h3 section boundaries, merge short sections,
        split long sections at paragraph boundaries
    """
    if not sections:
        return []

    # course_schedule: one coherent document, keep whole if possible
    if chunk_type == "course_schedule":
        full = " ".join(
            (f"{s['heading']}\n{s['text']}" if s.get("heading") else s["text"]).strip()
            for s in sections
        ).strip()
        if len(full) <= max_len:
            return [full] if full else []
        # Fall through to normal splitting if over max

    chunks = []
    pending = ""

    for section in sections:
        heading = section.get("heading", "")
        text = section.get("text", "").strip()
        section_text = f"{heading}\n{text}".strip() if heading else text

        if not section_text:
            continue

        # Section exceeds max — split at paragraph or sentence boundaries
        if len(section_text) > max_len:
            if pending:
                chunks.append(pending.strip())
                pending = ""
            parts = re.split(r'\n\n+|\.\s+', section_text)
            current = ""
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                if current and len(current) + len(part) + 1 > max_len:
                    chunks.append(current.strip())
                    current = part
                else:
                    current = (current + " " + part).strip() if current else part
            if current:
                chunks.append(current.strip())
            continue

        # Section is too short — merge into pending
        if len(section_text) < min_len:
            pending = (pending + "\n\n" + section_text).strip() if pending else section_text
            if len(pending) >= max_len:
                chunks.append(pending)
                pending = ""
            continue

        # Normal section — flush pending first
        if pending:
            combined = (pending + "\n\n" + section_text).strip()
            if len(combined) <= max_len:
                chunks.append(combined)
            else:
                chunks.append(pending)
                chunks.append(section_text)
            pending = ""
        else:
            chunks.append(section_text)

    if pending:
        chunks.append(pending.strip())

    return [c for c in chunks if c]


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
    Breadth-first crawl of the configured department site.
    Reads chunk_type, level, degree_type, and campus from the YAML config
    for each page URL. Falls back to 'general' for unlisted pages.
    """
    root_url          = config["root_url"]
    campus            = config.get("campus", "")
    page_types        = config.get("page_types", {})
    page_levels       = config.get("page_levels", {})
    page_degree_types = config.get("page_degree_types", {})

    visited = set()
    queued  = {root_url}
    queue   = deque([root_url])
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
                            "chunk_type":         page_types.get(url, "general"),
                            "level":              page_levels.get(url, ""),
                            "degree_type":        page_degree_types.get(url, ""),
                            "campus":             campus,
                            "course_code":        extract_course_code(data["h1"], data["text"]),
                            "referenced_courses": extract_referenced_courses(data["text"]),
                        })

                    for link in data["links"]:
                        if link not in visited and link not in queued:
                            queue.append(link)
                            queued.add(link)

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
                    chunks = [text[:CHUNK_MAX_LEN]]

            if not chunks:
                continue

            embeddings = embed_batch(chunks)

            for i, (chunk_text, emb) in enumerate(zip(chunks, embeddings)):
                obj = DataObject(
                    uuid=str(uuid.uuid4()),
                    properties={
                        # ── Core fields ───────────────────────────────────────
                        "chunk_id":            f"{page['url']}#chunk-{i}",
                        "chunk_type":          page["chunk_type"],
                        "department_id":       normalized_department_id,
                        "campus":              page.get("campus", ""),
                        "text":                chunk_text,
                        "heading":             page.get("heading", ""),
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
