"""
Centralized connection management for Redis and PostgreSQL.

Production uses Redis + PostgreSQL (Railway).
Tests and local dev fall back to in-memory / SQLite.
"""

import os
import logging

logger = logging.getLogger(__name__)


def get_redis_client(url: str = None):
    """
    Get a Redis client instance.

    Args:
        url: Redis URL. Defaults to REDIS_URL env var.

    Returns:
        redis.Redis instance, or None if Redis is unavailable.
    """
    url = url or os.environ.get('REDIS_URL')
    if not url:
        logger.warning("REDIS_URL not set — Redis unavailable, callers should use in-memory fallback")
        return None

    try:
        import redis
        client = redis.Redis.from_url(url, decode_responses=True)
        client.ping()
        logger.info("Redis connection established")
        return client
    except ImportError:
        logger.warning("redis package not installed — Redis unavailable")
        return None
    except Exception as e:
        logger.warning(f"Redis connection failed: {e} — callers should use in-memory fallback")
        return None


def get_postgres_connection(url: str = None):
    """
    Get a PostgreSQL connection.

    Args:
        url: PostgreSQL URL. Defaults to DATABASE_URL env var.

    Returns:
        psycopg2 connection, or None if PostgreSQL is unavailable.
    """
    url = url or os.environ.get('DATABASE_URL')
    if not url:
        logger.warning("DATABASE_URL not set — PostgreSQL unavailable, callers should use SQLite fallback")
        return None

    try:
        import psycopg2
        conn = psycopg2.connect(url)
        conn.autocommit = True
        logger.info("PostgreSQL connection established")
        return conn
    except ImportError:
        logger.warning("psycopg2 package not installed — PostgreSQL unavailable")
        return None
    except Exception as e:
        logger.warning(f"PostgreSQL connection failed: {e} — callers should use SQLite fallback")
        return None


def ensure_agent_schema(conn) -> bool:
    """
    Ensure agent tables exist in PostgreSQL.

    Runs migration 002_agent_schema.sql if tables don't exist.
    Idempotent — safe to call multiple times.

    Args:
        conn: psycopg2 connection

    Returns:
        True if schema is ready, False on failure.
    """
    if conn is None:
        return False

    try:
        cur = conn.cursor()

        # Check if agent tables already exist
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = 'agent_runs'
            );
        """)
        exists = cur.fetchone()[0]

        if exists:
            logger.debug("Agent schema already exists")
            cur.close()
            return True

        # Read and execute migration
        migration_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'migrations', '002_agent_schema.sql'
        )

        if not os.path.exists(migration_path):
            logger.error(f"Migration file not found: {migration_path}")
            cur.close()
            return False

        with open(migration_path) as f:
            sql = f.read()

        cur.execute(sql)
        logger.info("Agent schema migration applied successfully")
        cur.close()
        return True

    except Exception as e:
        logger.error(f"Failed to apply agent schema: {e}")
        return False
