"""
Migration script: SQLite to PostgreSQL for betting_market_features

This script migrates the odds_history database from SQLite to PostgreSQL
as specified in the task requirements.

Usage:
    python migrate_to_postgres.py --db-url postgresql://user:pass@host:5432/dbname
"""

import os
import sys
import sqlite3
import argparse

try:
    import psycopg2
    from psycopg2.extras import execute_batch
    HAS_POSTGRES = True
except ImportError:
    HAS_POSTGRES = False
    print("Warning: psycopg2 not installed. Install with: pip install psycopg2-binary")


# PostgreSQL schema (matches plan.md specification)
POSTGRES_SCHEMA = """
-- Games table
CREATE TABLE IF NOT EXISTS games (
    game_id TEXT PRIMARY KEY,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    commence_time TIMESTAMP NOT NULL,
    sport TEXT DEFAULT 'basketball_nba',
    created_at TIMESTAMP DEFAULT NOW()
);

-- Odds history table
CREATE TABLE IF NOT EXISTS odds_history (
    id SERIAL PRIMARY KEY,
    game_id TEXT NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    book_name TEXT NOT NULL,
    market TEXT NOT NULL,
    home_odds FLOAT,
    away_odds FLOAT,
    home_line FLOAT,
    away_line FLOAT,
    total FLOAT,
    over_odds FLOAT,
    under_odds FLOAT,
    is_opening BOOLEAN DEFAULT FALSE,
    is_closing BOOLEAN DEFAULT FALSE,
    FOREIGN KEY (game_id) REFERENCES games(game_id),
    UNIQUE(game_id, timestamp, book_name, market)
);

-- Line movements table
CREATE TABLE IF NOT EXISTS line_movements (
    id SERIAL PRIMARY KEY,
    game_id TEXT NOT NULL,
    market TEXT NOT NULL,
    opening_line FLOAT,
    closing_line FLOAT,
    movement FLOAT,
    opening_time TIMESTAMP,
    closing_time TIMESTAMP,
    num_moves INTEGER,
    max_move FLOAT,
    rlm_detected BOOLEAN DEFAULT FALSE,
    steam_detected BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW(),
    FOREIGN KEY (game_id) REFERENCES games(game_id),
    UNIQUE(game_id, market)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_odds_game_market
    ON odds_history(game_id, market, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_odds_timestamp
    ON odds_history(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_movements_game
    ON line_movements(game_id);
"""


def migrate_sqlite_to_postgres(sqlite_path: str, postgres_url: str, verbose: bool = True):
    """
    Migrate data from SQLite to PostgreSQL.

    Args:
        sqlite_path: Path to SQLite database file
        postgres_url: PostgreSQL connection URL
        verbose: Print progress messages
    """
    if not HAS_POSTGRES:
        raise RuntimeError("psycopg2 not installed. Run: pip install psycopg2-binary")

    if not os.path.exists(sqlite_path):
        raise FileNotFoundError(f"SQLite database not found: {sqlite_path}")

    # Connect to both databases
    if verbose:
        print(f"Connecting to SQLite: {sqlite_path}")
    sqlite_conn = sqlite3.connect(sqlite_path)
    sqlite_conn.row_factory = sqlite3.Row

    if verbose:
        print("Connecting to PostgreSQL...")
    pg_conn = psycopg2.connect(postgres_url)

    try:
        # Create PostgreSQL schema
        if verbose:
            print("Creating PostgreSQL schema...")
        with pg_conn.cursor() as cur:
            cur.execute(POSTGRES_SCHEMA)
        pg_conn.commit()

        # Migrate games
        if verbose:
            print("Migrating games table...")
        games_count = _migrate_table(
            sqlite_conn, pg_conn, 'games',
            ['game_id', 'home_team', 'away_team', 'commence_time', 'sport', 'created_at'],
            verbose
        )

        # Migrate odds_history
        if verbose:
            print("Migrating odds_history table...")
        odds_count = _migrate_table(
            sqlite_conn, pg_conn, 'odds_history',
            ['game_id', 'timestamp', 'book_name', 'market', 'home_odds', 'away_odds',
             'home_line', 'away_line', 'total', 'over_odds', 'under_odds',
             'is_opening', 'is_closing'],
            verbose,
            batch_size=1000
        )

        # Migrate line_movements
        if verbose:
            print("Migrating line_movements table...")
        movements_count = _migrate_table(
            sqlite_conn, pg_conn, 'line_movements',
            ['game_id', 'market', 'opening_line', 'closing_line', 'movement',
             'opening_time', 'closing_time', 'num_moves', 'max_move',
             'rlm_detected', 'steam_detected', 'created_at'],
            verbose
        )

        if verbose:
            print("\n" + "="*50)
            print("Migration completed successfully!")
            print(f"Games migrated: {games_count}")
            print(f"Odds snapshots migrated: {odds_count}")
            print(f"Line movements migrated: {movements_count}")
            print("="*50)

        return {
            'games': games_count,
            'odds_history': odds_count,
            'line_movements': movements_count
        }

    finally:
        sqlite_conn.close()
        pg_conn.close()


def _migrate_table(sqlite_conn, pg_conn, table_name: str, columns: list,
                   verbose: bool = True, batch_size: int = 100) -> int:
    """Migrate a single table from SQLite to PostgreSQL."""
    # Fetch all rows from SQLite
    sqlite_cur = sqlite_conn.cursor()

    # Check if table exists in SQLite
    sqlite_cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,)
    )
    if not sqlite_cur.fetchone():
        if verbose:
            print(f"  Table '{table_name}' not found in SQLite, skipping...")
        return 0

    # Build query
    columns_str = ', '.join(columns)
    sqlite_cur.execute(f"SELECT {columns_str} FROM {table_name}")
    rows = sqlite_cur.fetchall()

    if not rows:
        if verbose:
            print(f"  No data in '{table_name}', skipping...")
        return 0

    # Insert into PostgreSQL
    placeholders = ', '.join(['%s'] * len(columns))
    insert_query = f"""
        INSERT INTO {table_name} ({columns_str})
        VALUES ({placeholders})
        ON CONFLICT DO NOTHING
    """

    pg_cur = pg_conn.cursor()

    # Convert rows to tuples
    data = [tuple(row) for row in rows]

    # Batch insert
    if len(data) > batch_size:
        if verbose:
            print(f"  Inserting {len(data)} rows in batches of {batch_size}...")
        execute_batch(pg_cur, insert_query, data, page_size=batch_size)
    else:
        if verbose:
            print(f"  Inserting {len(data)} rows...")
        pg_cur.executemany(insert_query, data)

    pg_conn.commit()

    if verbose:
        print(f"  ✓ Migrated {len(data)} rows")

    return len(data)


def test_postgres_connection(postgres_url: str) -> bool:
    """Test PostgreSQL connection."""
    if not HAS_POSTGRES:
        print("Error: psycopg2 not installed")
        return False

    try:
        conn = psycopg2.connect(postgres_url)
        conn.close()
        print("✓ PostgreSQL connection successful")
        return True
    except Exception as e:
        print(f"✗ PostgreSQL connection failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Migrate odds_history from SQLite to PostgreSQL'
    )
    parser.add_argument(
        '--sqlite-db',
        default='odds_history.db',
        help='Path to SQLite database (default: odds_history.db)'
    )
    parser.add_argument(
        '--postgres-url',
        help='PostgreSQL connection URL (postgresql://user:pass@host:5432/dbname)'
    )
    parser.add_argument(
        '--test-connection',
        action='store_true',
        help='Only test PostgreSQL connection, do not migrate'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress messages'
    )

    args = parser.parse_args()

    # Get PostgreSQL URL from args or environment
    postgres_url = args.postgres_url or os.environ.get('DATABASE_URL')

    if not postgres_url:
        print("Error: PostgreSQL URL required")
        print("Provide via --postgres-url or set DATABASE_URL environment variable")
        print("\nExample:")
        print("  python migrate_to_postgres.py --postgres-url postgresql://user:pass@localhost:5432/nba_betting")
        print("  export DATABASE_URL=postgresql://user:pass@localhost:5432/nba_betting")
        sys.exit(1)

    # Test connection only
    if args.test_connection:
        success = test_postgres_connection(postgres_url)
        sys.exit(0 if success else 1)

    # Run migration
    try:
        migrate_sqlite_to_postgres(
            args.sqlite_db,
            postgres_url,
            verbose=not args.quiet
        )
    except Exception as e:
        print(f"Migration failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
