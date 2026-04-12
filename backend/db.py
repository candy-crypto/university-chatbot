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