import os
import re
import uuid
import time
import yaml
from datetime import datetime, timezone
from collections import deque
from urllib.parse import urljoin, urlparse

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


def load_department_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_text(text: str) -> str:
    """
    Collapse repeated whitespace so stored chunks are cleaner.
    """
    return re.sub(r"\s+", " ", text).strip()


def is_allowed_url(url: str, config: dict) -> bool:
    """
    Restrict crawling to the configured department site and allowed paths.
    """
    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https"):
        return False

    allowed_domains = set(config.get("allowed_domains", []))
    if parsed.netloc not in allowed_domains:
        return False

    path = parsed.path or "/"
    allowed_prefixes = config.get("allowed_path_prefixes", ["/"])
    if not any(path.startswith(prefix) for prefix in allowed_prefixes):
        return False

    for pattern in config.get("deny_url_patterns", []):
        if pattern in url:
            return False

    return True


def extract_page_data(page, url: str, config: dict) -> dict:
    """
    Load a page with Playwright, wait for rendering, then extract:
      - title
      - visible text from body
      - discovered links from rendered DOM
    """
    page.goto(url, wait_until="domcontentloaded", timeout=30000)

    try:
        page.wait_for_load_state("networkidle", timeout=5000)
    except PlaywrightTimeoutError:
        # Some pages never become fully idle; continue anyway.
        pass

    title = page.title().strip() if page.title() else ""

    headings = page.locator("h1, h2").evaluate_all(
        """
        elements => elements
            .map(el => el.innerText && el.innerText.trim())
            .filter(Boolean)
        """
    )

    meta_timestamp = page.locator("meta[property='article:modified_time'], meta[name='last-modified'], meta[property='article:published_time']").evaluate_all(
        """
        elements => {
            const values = elements
                .map(el => el.getAttribute('content'))
                .filter(Boolean);
            return values.length ? values[0] : null;
        }
        """
    )

    keyword_tags = page.locator("meta[name='keywords']").evaluate_all(
        """
        elements => {
            const values = elements
                .map(el => el.getAttribute('content'))
                .filter(Boolean);
            return values.length ? values[0] : null;
        }
        """
    )

    # Pull visible text from the rendered page.
    body_text = page.locator("body").inner_text(timeout=5000)
    text = normalize_text(body_text)

    # Collect all href values from rendered anchor tags.
    hrefs = page.locator("a[href]").evaluate_all(
        """
        elements => elements
            .map(el => el.href)
            .filter(Boolean)
        """
    )

    links = set()
    for href in hrefs:
        clean = href.split("#")[0].strip()
        if clean and is_allowed_url(clean, config):
            links.add(clean)

    return {
        "url": url,
        "title": title,
        "text": text,
        "links": links,
        "headings": headings,
        "page_timestamp": meta_timestamp,
        "keyword_tags": keyword_tags,
    }


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200):
    """
    Simple character-based chunking.
    """
    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = end - overlap

    return chunks


def derive_source(url: str) -> str:
    parsed = urlparse(url)
    return parsed.netloc.lower()


def derive_document_id(url: str) -> str:
    parsed = urlparse(url)
    path = (parsed.path or "/").strip("/")
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", path).strip("-").lower() or "home"
    return f"{parsed.netloc.lower()}::{slug}"


def derive_section(page_title: str, headings: list[str]) -> str:
    for heading in headings:
        clean = normalize_text(heading)
        if clean and clean.lower() != page_title.lower():
            return clean
    return page_title


def derive_tags(title: str, keyword_tags) -> list[str]:
    if isinstance(keyword_tags, list):
        raw_tags = keyword_tags[0] if keyword_tags else ""
    else:
        raw_tags = keyword_tags or ""

    tags = [tag.strip().lower() for tag in raw_tags.split(",") if tag.strip()]
    if tags:
        return tags[:10]

    title_tags = [
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", title)
        if token.lower() not in {"nmsu", "university", "department"}
    ]
    return title_tags[:6]


def derive_course_metadata(title: str, text: str) -> tuple[str, str]:
    course_match = re.search(r"\b([A-Z]{2,4}\s?-?\d{3}[A-Z]?)\b", f"{title}\n{text[:500]}")
    if not course_match:
        return "", ""

    course_number = re.sub(r"\s+", " ", course_match.group(1)).replace("-", " ").strip()
    title_match = re.search(
        rf"{re.escape(course_match.group(1))}\s*[:\-]\s*([^\n|]{{3,100}})",
        title,
        flags=re.IGNORECASE,
    )
    if title_match:
        course_title = normalize_text(title_match.group(1))
    else:
        course_title = ""

    return course_number, course_title


def derive_timestamp(page_timestamp, crawl_version: str) -> str:
    if isinstance(page_timestamp, list):
        raw_timestamp = page_timestamp[0] if page_timestamp else ""
    else:
        raw_timestamp = page_timestamp or ""

    if raw_timestamp:
        return raw_timestamp

    try:
        parsed = datetime.strptime(crawl_version, "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
        return parsed.isoformat()
    except ValueError:
        return datetime.now(timezone.utc).isoformat()


def embed_batch(texts):
    """
    Batch embed chunks with OpenAI.
    """
    response = openai_client.embeddings.create(
        model=OPENAI_EMBED_MODEL,
        input=texts
    )
    return [item.embedding for item in response.data]


def crawl_site(config: dict, max_pages: int = 30):
    """
    Breadth-first crawl of the configured department site using Playwright.
    """
    root_url = config["root_url"]
    visited = set()
    queued = {root_url}
    queue = deque([root_url])
    pages = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent="UniversityChatbotBot/1.0"
        )
        page = context.new_page()

        try:
            while queue and len(visited) < max_pages:
                url = queue.popleft()

                if url in visited:
                    continue

                try:
                    data = extract_page_data(page, url, config)
                    visited.add(url)

                    if len(data["text"]) > 100:
                        section = derive_section(data["title"], data.get("headings", []))
                        tags = derive_tags(data["title"], data.get("keyword_tags"))
                        course_number, course_title = derive_course_metadata(data["title"], data["text"])
                        pages.append({
                            "url": data["url"],
                            "title": data["title"],
                            "document_id": derive_document_id(data["url"]),
                            "source": derive_source(data["url"]),
                            "section": section,
                            "page_timestamp": data.get("page_timestamp"),
                            "tags": tags,
                            "course_number": course_number,
                            "course_title": course_title,
                            "text": data["text"],
                        })

                    for link in data["links"]:
                        if link not in visited and link not in queued:
                            queue.append(link)
                            queued.add(link)

                    # Be polite to the site.
                    time.sleep(0.3)

                except Exception as e:
                    print(f"Skipping {url}: {e}")
                    visited.add(url)

        finally:
            context.close()
            browser.close()

    return pages


def delete_department_chunks(collection, department_id: str):
    """
    Optional cleanup before reindexing a department.
    Keeps week 2 simple by replacing old chunks.
    """
    collection.data.delete_many(
        where=Filter.by_property("department_id").equal(department_id)
    )


def upsert_pages_to_weaviate(pages, department_id: str, crawl_version: str):
    """
    Chunk pages, embed them, and insert into Weaviate.
    """
    normalized_department_id = (department_id or "").strip().lower()

    client = get_weaviate_client()
    try:
        ensure_collection(client)
        collection = get_collection(client)

        delete_department_chunks(collection, normalized_department_id)

        objects = []

        for page in pages:
            chunks = chunk_text(page["text"])
            if not chunks:
                continue

            embeddings = embed_batch(chunks)

            for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                obj = DataObject(
                    uuid=str(uuid.uuid4()),
                    properties={
                        "department_id": normalized_department_id,
                        "document_id": page.get("document_id", page["url"]),
                        "url": page["url"],
                        "source": page.get("source", ""),
                        "title": page["title"],
                        "section": page.get("section", page["title"]),
                        "timestamp": derive_timestamp(page.get("page_timestamp"), crawl_version),
                        "tags": page.get("tags", []),
                        "course_number": page.get("course_number", ""),
                        "course_title": page.get("course_title", ""),
                        "text": chunk,
                        "chunk_id": f"{page['url']}#chunk-{i}",
                        "crawl_version": crawl_version,
                    },
                    vector=emb,
                )
                objects.append(obj)

        if objects:
            collection.data.insert_many(objects)

        return len(objects)

    finally:
        client.close()


def main():
    config_path = "configs/departments/cs.yaml"
    config = load_department_config(config_path)

    department_id = config["department_id"]
    crawl_version = time.strftime("%Y%m%d_%H%M%S")

    # Ensure relational tables exist when running ingestion directly.
    init_db()

    pages = crawl_site(config, max_pages=30)

    chunks_indexed = upsert_pages_to_weaviate(
        pages=pages,
        department_id=department_id,
        crawl_version=crawl_version
    )

    log_crawl_run(
        department_id=department_id,
        crawl_version=crawl_version,
        pages_scraped=len(pages),
        chunks_indexed=chunks_indexed
    )

    print(f"Scraped {len(pages)} pages")
    print(f"Indexed {chunks_indexed} chunks into Weaviate")
    print(f"Crawl version: {crawl_version}")


if __name__ == "__main__":
    main()
