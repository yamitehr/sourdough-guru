"""Auto-create Supabase tables on startup via direct Postgres connection."""

import psycopg2

from app.config import get_settings

SQL = """
CREATE TABLE IF NOT EXISTS chat_sessions (
    session_id TEXT PRIMARY KEY,
    messages JSONB NOT NULL DEFAULT '[]'::jsonb,
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS bake_sessions (
    session_id TEXT PRIMARY KEY,
    plan_data JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMP DEFAULT NOW(),
    active BOOLEAN DEFAULT TRUE
);
"""


def ensure_tables():
    """Create tables if they don't exist."""
    settings = get_settings()
    conn = psycopg2.connect(settings.DATABASE_URL)
    try:
        with conn.cursor() as cur:
            cur.execute(SQL)
        conn.commit()
        print("Supabase tables verified/created.")
    finally:
        conn.close()
