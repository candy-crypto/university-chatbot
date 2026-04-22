# db.py

import os
from contextlib import contextmanager
import psycopg
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")


@contextmanager
def get_conn():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not set")
    with psycopg.connect(DATABASE_URL) as conn:
        yield conn


def init_db():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS courses (
                    id           BIGSERIAL PRIMARY KEY,
                    course_code  TEXT NOT NULL,
                    course_title TEXT NOT NULL,
                    credits      TEXT NOT NULL DEFAULT '',
                    suffix       TEXT NOT NULL DEFAULT '',
                    department_id TEXT NOT NULL,
                    campus       TEXT NOT NULL DEFAULT 'las_cruces',
                    catalog_year TEXT NOT NULL,
                    UNIQUE(course_code, catalog_year)
                );
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS courses_title_fts_idx
                ON courses USING gin(to_tsvector('english', course_title));
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chat_logs (
                    id BIGSERIAL PRIMARY KEY,
                    department_id TEXT NOT NULL,
                    user_question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    sources JSONB NOT NULL DEFAULT '[]'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS crawl_runs (
                    id BIGSERIAL PRIMARY KEY,
                    department_id TEXT NOT NULL,
                    crawl_version TEXT NOT NULL,
                    pages_scraped INT NOT NULL DEFAULT 0,
                    chunks_indexed INT NOT NULL DEFAULT 0,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
            """)
        conn.commit()


def upsert_courses(courses: list):
    """Insert or update course records from catalog ingest.

    Each item is a dict with keys: course_code, course_title, credits,
    suffix, department_id, campus, catalog_year.
    """
    if not courses:
        return
    with get_conn() as conn:
        with conn.cursor() as cur:
            for c in courses:
                cur.execute("""
                    INSERT INTO courses
                        (course_code, course_title, credits, suffix,
                         department_id, campus, catalog_year)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (course_code, catalog_year) DO UPDATE SET
                        course_title  = EXCLUDED.course_title,
                        credits       = EXCLUDED.credits,
                        suffix        = EXCLUDED.suffix,
                        department_id = EXCLUDED.department_id,
                        campus        = EXCLUDED.campus
                """, (
                    c["course_code"], c["course_title"], c.get("credits", ""),
                    c.get("suffix", ""), c["department_id"],
                    c.get("campus", "las_cruces"), c["catalog_year"],
                ))
        conn.commit()


def lookup_courses_by_suffix(suffix: str, dept_prefix: str = "") -> list:
    """Return all courses with the given suffix ('G' or 'V'), optionally filtered
    by department code prefix (e.g. 'MATH'). Results ordered by course_code."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            if dept_prefix:
                cur.execute("""
                    SELECT course_code, course_title, credits, suffix, department_id, catalog_year
                    FROM courses
                    WHERE suffix = %s
                      AND UPPER(course_code) LIKE UPPER(%s)
                    ORDER BY course_code
                """, (suffix, f"{dept_prefix}%"))
            else:
                cur.execute("""
                    SELECT course_code, course_title, credits, suffix, department_id, catalog_year
                    FROM courses
                    WHERE suffix = %s
                    ORDER BY course_code
                """, (suffix,))
            rows = cur.fetchall()
    return [{"course_code": r[0], "course_title": r[1], "credits": r[2],
             "suffix": r[3], "department_id": r[4], "catalog_year": r[5]}
            for r in rows]


def lookup_course_by_code(course_code: str) -> dict | None:
    """Find a course by exact course code (case-insensitive). Returns most recent catalog year."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT course_code, course_title, credits, suffix, department_id, catalog_year
                FROM courses
                WHERE UPPER(REPLACE(course_code, ' ', '')) = UPPER(REPLACE(%s, ' ', ''))
                ORDER BY catalog_year DESC
                LIMIT 1
            """, (course_code,))
            row = cur.fetchone()
    if not row:
        return None
    return {"course_code": row[0], "course_title": row[1], "credits": row[2],
            "suffix": row[3], "department_id": row[4], "catalog_year": row[5]}


def lookup_course_by_title(title_fragment: str) -> list:
    """Full-text search for courses whose title matches the fragment.
    Returns up to 5 best matches ordered by relevance."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT course_code, course_title, credits, suffix, department_id, catalog_year,
                       ts_rank(to_tsvector('english', course_title),
                               plainto_tsquery('english', %s)) AS rank
                FROM courses
                WHERE to_tsvector('english', course_title)
                      @@ plainto_tsquery('english', %s)
                ORDER BY rank DESC, catalog_year DESC
                LIMIT 5
            """, (title_fragment, title_fragment))
            rows = cur.fetchall()
    return [{"course_code": r[0], "course_title": r[1], "credits": r[2],
             "suffix": r[3], "department_id": r[4], "catalog_year": r[5],
             "rank": float(r[6])}
            for r in rows]


def log_chat(department_id: str, user_question: str, answer: str, sources_json: str):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO chat_logs (department_id, user_question, answer, sources)
                VALUES (%s, %s, %s, %s::jsonb)
            """, (department_id, user_question, answer, sources_json))
        conn.commit()


def log_crawl_run(department_id: str, crawl_version: str, pages_scraped: int, chunks_indexed: int):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO crawl_runs (department_id, crawl_version, pages_scraped, chunks_indexed)
                VALUES (%s, %s, %s, %s)
            """, (department_id, crawl_version, pages_scraped, chunks_indexed))
        conn.commit()