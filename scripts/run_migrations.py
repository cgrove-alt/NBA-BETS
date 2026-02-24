"""
Run all SQL migrations against PostgreSQL.

Reads migration files from migrations/ directory in order (001, 002, ...)
and executes them. All migrations use CREATE TABLE IF NOT EXISTS, making
this script fully idempotent — safe to run repeatedly.

Usage:
    python scripts/run_migrations.py              # Uses DATABASE_URL env var
    python scripts/run_migrations.py --url <url>  # Explicit PostgreSQL URL
    python scripts/run_migrations.py --check      # Check if migrations are needed (dry run)

Called automatically by the API on startup (see backend/api.py lifespan).
"""

import argparse
import glob
import logging
import os
import sys

logger = logging.getLogger(__name__)

# Project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MIGRATIONS_DIR = os.path.join(PROJECT_ROOT, 'migrations')


def get_migration_files() -> list[str]:
    """Return sorted list of migration SQL files."""
    pattern = os.path.join(MIGRATIONS_DIR, '*.sql')
    files = sorted(glob.glob(pattern))
    return files


def run_migrations(database_url: str = None, check_only: bool = False) -> bool:
    """
    Run all SQL migrations against PostgreSQL.

    Args:
        database_url: PostgreSQL connection URL. Defaults to DATABASE_URL env var.
        check_only: If True, only check if migrations are needed (don't execute).

    Returns:
        True if all migrations succeeded (or check passed), False otherwise.
    """
    url = database_url or os.environ.get('DATABASE_URL')
    if not url:
        logger.warning("DATABASE_URL not set — skipping migrations (SQLite fallback mode)")
        return True  # Not an error — local dev uses SQLite

    migration_files = get_migration_files()
    if not migration_files:
        logger.warning(f"No migration files found in {MIGRATIONS_DIR}")
        return True

    try:
        import psycopg2
    except ImportError:
        logger.error("psycopg2 not installed — cannot run PostgreSQL migrations")
        return False

    try:
        conn = psycopg2.connect(url)
        conn.autocommit = True
        cur = conn.cursor()
    except Exception as e:
        logger.error(f"Failed to connect to PostgreSQL: {e}")
        return False

    # Create migrations tracking table
    try:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS _migrations (
                id SERIAL PRIMARY KEY,
                filename VARCHAR(255) UNIQUE NOT NULL,
                applied_at TIMESTAMP DEFAULT NOW()
            );
        """)
    except Exception as e:
        logger.error(f"Failed to create migrations tracking table: {e}")
        cur.close()
        conn.close()
        return False

    # Check which migrations have been applied
    cur.execute("SELECT filename FROM _migrations ORDER BY filename;")
    applied = {row[0] for row in cur.fetchall()}

    pending = []
    for filepath in migration_files:
        filename = os.path.basename(filepath)
        if filename not in applied:
            pending.append((filename, filepath))

    if not pending:
        logger.info("All migrations already applied")
        cur.close()
        conn.close()
        return True

    if check_only:
        logger.info(f"{len(pending)} migration(s) pending: {[f for f, _ in pending]}")
        cur.close()
        conn.close()
        return False  # Indicates work is needed

    # Apply pending migrations
    success = True
    for filename, filepath in pending:
        logger.info(f"Applying migration: {filename}")
        try:
            with open(filepath, 'r') as f:
                sql = f.read()
            cur.execute(sql)
            cur.execute(
                "INSERT INTO _migrations (filename) VALUES (%s) ON CONFLICT (filename) DO NOTHING;",
                (filename,)
            )
            logger.info(f"Migration {filename} applied successfully")
        except Exception as e:
            logger.error(f"Migration {filename} failed: {e}")
            success = False
            break  # Stop on first failure — don't skip migrations

    cur.close()
    conn.close()
    return success


def main():
    parser = argparse.ArgumentParser(description='Run database migrations')
    parser.add_argument('--url', help='PostgreSQL URL (default: DATABASE_URL env var)')
    parser.add_argument('--check', action='store_true', help='Check if migrations are needed (dry run)')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )

    files = get_migration_files()
    print(f"Found {len(files)} migration file(s) in {MIGRATIONS_DIR}:")
    for f in files:
        print(f"  - {os.path.basename(f)}")

    if args.check:
        needed = not run_migrations(database_url=args.url, check_only=True)
        if needed:
            print("\nMigrations are pending — run without --check to apply.")
            sys.exit(1)
        else:
            print("\nAll migrations are up to date.")
            sys.exit(0)

    ok = run_migrations(database_url=args.url)
    if ok:
        print("\nAll migrations applied successfully.")
    else:
        print("\nMigration failed — check logs above.")
        sys.exit(1)


if __name__ == '__main__':
    main()
