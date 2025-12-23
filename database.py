"""
NBA Betting Model Database Module

SQLite database backend for proper data management:
- Teams, players, games
- Point-in-time statistics (avoid look-ahead bias)
- Betting odds history
- Injury reports
- Bet tracking and performance

Usage:
    db = DatabaseManager()
    db.initialize()

    # Store game data
    db.upsert_game(game_data)

    # Get point-in-time stats for a team
    stats = db.get_team_stats_before_date(team_id, game_date)
"""

import sqlite3
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from contextlib import contextmanager


# Default database location
DEFAULT_DB_PATH = Path("nba_betting.db")


class DatabaseManager:
    """
    SQLite database manager for NBA betting model data.

    Handles:
    - Schema creation and migrations
    - CRUD operations for all entities
    - Point-in-time queries
    - Connection pooling
    """

    def __init__(self, db_path: Path = DEFAULT_DB_PATH):
        self.db_path = db_path
        self._connection = None

    @contextmanager
    def get_connection(self):
        """Get a database connection with context management."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable dict-like access
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def initialize(self):
        """Initialize database with schema."""
        with self.get_connection() as conn:
            self._create_tables(conn)
            self._create_indexes(conn)
        print(f"Database initialized at {self.db_path}")

    def _create_tables(self, conn: sqlite3.Connection):
        """Create all database tables."""
        cursor = conn.cursor()

        # Teams table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS teams (
                id INTEGER PRIMARY KEY,
                nba_id INTEGER UNIQUE NOT NULL,
                abbreviation VARCHAR(3) NOT NULL UNIQUE,
                name VARCHAR(100) NOT NULL,
                city VARCHAR(50),
                conference VARCHAR(4),
                division VARCHAR(20),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Players table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS players (
                id INTEGER PRIMARY KEY,
                nba_id INTEGER UNIQUE NOT NULL,
                name VARCHAR(100) NOT NULL,
                team_id INTEGER REFERENCES teams(id),
                position VARCHAR(10),
                height VARCHAR(10),
                weight INTEGER,
                jersey_number INTEGER,
                active BOOLEAN DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Seasons table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS seasons (
                id INTEGER PRIMARY KEY,
                season_string VARCHAR(10) UNIQUE NOT NULL,
                start_date DATE NOT NULL,
                end_date DATE,
                is_current BOOLEAN DEFAULT 0
            )
        """)

        # Games table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS games (
                id INTEGER PRIMARY KEY,
                nba_game_id VARCHAR(20) UNIQUE NOT NULL,
                season_id INTEGER REFERENCES seasons(id),
                game_date DATE NOT NULL,
                game_time TIME,
                home_team_id INTEGER REFERENCES teams(id),
                away_team_id INTEGER REFERENCES teams(id),
                home_score INTEGER,
                away_score INTEGER,
                status VARCHAR(20) DEFAULT 'scheduled',
                arena VARCHAR(100),
                attendance INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Team statistics snapshot (point-in-time)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS team_stats_snapshot (
                id INTEGER PRIMARY KEY,
                team_id INTEGER REFERENCES teams(id),
                season_id INTEGER REFERENCES seasons(id),
                snapshot_date DATE NOT NULL,
                games_played INTEGER,
                wins INTEGER,
                losses INTEGER,
                pts_avg REAL,
                opp_pts_avg REAL,
                fg_pct REAL,
                fg3_pct REAL,
                ft_pct REAL,
                reb_avg REAL,
                ast_avg REAL,
                stl_avg REAL,
                blk_avg REAL,
                tov_avg REAL,
                off_rating REAL,
                def_rating REAL,
                net_rating REAL,
                pace REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(team_id, season_id, snapshot_date)
            )
        """)

        # Player statistics snapshot (point-in-time)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS player_stats_snapshot (
                id INTEGER PRIMARY KEY,
                player_id INTEGER REFERENCES players(id),
                season_id INTEGER REFERENCES seasons(id),
                snapshot_date DATE NOT NULL,
                games_played INTEGER,
                minutes_avg REAL,
                pts_avg REAL,
                reb_avg REAL,
                ast_avg REAL,
                stl_avg REAL,
                blk_avg REAL,
                tov_avg REAL,
                fg_pct REAL,
                fg3_pct REAL,
                ft_pct REAL,
                plus_minus_avg REAL,
                per REAL,
                ts_pct REAL,
                usg_pct REAL,
                bpm REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(player_id, season_id, snapshot_date)
            )
        """)

        # Betting odds
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS betting_odds (
                id INTEGER PRIMARY KEY,
                game_id INTEGER REFERENCES games(id),
                sportsbook VARCHAR(50) NOT NULL,
                captured_at TIMESTAMP NOT NULL,
                home_moneyline INTEGER,
                away_moneyline INTEGER,
                home_spread REAL,
                home_spread_odds INTEGER,
                away_spread_odds INTEGER,
                total_line REAL,
                over_odds INTEGER,
                under_odds INTEGER,
                is_opening BOOLEAN DEFAULT 0,
                is_closing BOOLEAN DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Injuries
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS injuries (
                id INTEGER PRIMARY KEY,
                player_id INTEGER REFERENCES players(id),
                reported_date DATE NOT NULL,
                game_id INTEGER REFERENCES games(id),
                status VARCHAR(20) NOT NULL,
                injury_type VARCHAR(100),
                return_date DATE,
                source VARCHAR(50),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Bets tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS bets (
                id INTEGER PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                game_id INTEGER REFERENCES games(id),
                bet_type VARCHAR(20) NOT NULL,
                selection VARCHAR(50) NOT NULL,
                line REAL,
                odds_placed INTEGER,
                odds_closing INTEGER,
                stake REAL,
                model_probability REAL,
                implied_probability REAL,
                edge REAL,
                status VARCHAR(20) DEFAULT 'pending',
                payout REAL,
                settled_at TIMESTAMP,
                notes TEXT
            )
        """)

        # Model predictions (for backtesting)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_predictions (
                id INTEGER PRIMARY KEY,
                game_id INTEGER REFERENCES games(id),
                model_name VARCHAR(50) NOT NULL,
                prediction_type VARCHAR(20) NOT NULL,
                predicted_value REAL,
                actual_value REAL,
                confidence REAL,
                features_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Training data cache
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS training_features (
                id INTEGER PRIMARY KEY,
                game_id INTEGER REFERENCES games(id),
                feature_type VARCHAR(20) NOT NULL,
                features_json TEXT NOT NULL,
                outcome_value REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(game_id, feature_type)
            )
        """)

    def _create_indexes(self, conn: sqlite3.Connection):
        """Create database indexes for performance."""
        cursor = conn.cursor()

        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_games_date ON games(game_date)",
            "CREATE INDEX IF NOT EXISTS idx_games_teams ON games(home_team_id, away_team_id)",
            "CREATE INDEX IF NOT EXISTS idx_team_stats_date ON team_stats_snapshot(snapshot_date)",
            "CREATE INDEX IF NOT EXISTS idx_team_stats_team ON team_stats_snapshot(team_id)",
            "CREATE INDEX IF NOT EXISTS idx_player_stats_date ON player_stats_snapshot(snapshot_date)",
            "CREATE INDEX IF NOT EXISTS idx_odds_game ON betting_odds(game_id)",
            "CREATE INDEX IF NOT EXISTS idx_injuries_player ON injuries(player_id)",
            "CREATE INDEX IF NOT EXISTS idx_injuries_date ON injuries(reported_date)",
            "CREATE INDEX IF NOT EXISTS idx_bets_status ON bets(status)",
            "CREATE INDEX IF NOT EXISTS idx_predictions_game ON model_predictions(game_id)",
        ]

        for index_sql in indexes:
            cursor.execute(index_sql)

    # =========================================================================
    # Team Operations
    # =========================================================================

    def upsert_team(self, team_data: Dict) -> int:
        """Insert or update a team."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO teams (nba_id, abbreviation, name, city, conference, division)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(nba_id) DO UPDATE SET
                    abbreviation = excluded.abbreviation,
                    name = excluded.name,
                    city = excluded.city,
                    conference = excluded.conference,
                    division = excluded.division
            """, (
                team_data.get("nba_id"),
                team_data.get("abbreviation"),
                team_data.get("name"),
                team_data.get("city"),
                team_data.get("conference"),
                team_data.get("division"),
            ))
            return cursor.lastrowid

    def get_team_by_abbrev(self, abbreviation: str) -> Optional[Dict]:
        """Get team by abbreviation."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM teams WHERE abbreviation = ?",
                (abbreviation.upper(),)
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_team_by_nba_id(self, nba_id: int) -> Optional[Dict]:
        """Get team by NBA ID."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM teams WHERE nba_id = ?", (nba_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    # =========================================================================
    # Game Operations
    # =========================================================================

    def upsert_game(self, game_data: Dict) -> int:
        """Insert or update a game."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO games (
                    nba_game_id, season_id, game_date, game_time,
                    home_team_id, away_team_id, home_score, away_score,
                    status, arena, attendance
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(nba_game_id) DO UPDATE SET
                    home_score = excluded.home_score,
                    away_score = excluded.away_score,
                    status = excluded.status,
                    updated_at = CURRENT_TIMESTAMP
            """, (
                game_data.get("nba_game_id"),
                game_data.get("season_id"),
                game_data.get("game_date"),
                game_data.get("game_time"),
                game_data.get("home_team_id"),
                game_data.get("away_team_id"),
                game_data.get("home_score"),
                game_data.get("away_score"),
                game_data.get("status", "scheduled"),
                game_data.get("arena"),
                game_data.get("attendance"),
            ))
            return cursor.lastrowid

    def get_games_by_date(self, game_date: str) -> List[Dict]:
        """Get all games for a specific date."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT g.*,
                       ht.abbreviation as home_abbrev, ht.name as home_name,
                       at.abbreviation as away_abbrev, at.name as away_name
                FROM games g
                JOIN teams ht ON g.home_team_id = ht.id
                JOIN teams at ON g.away_team_id = at.id
                WHERE g.game_date = ?
            """, (game_date,))
            return [dict(row) for row in cursor.fetchall()]

    def get_games_between_dates(
        self,
        start_date: str,
        end_date: str,
        team_id: Optional[int] = None
    ) -> List[Dict]:
        """Get games between two dates, optionally filtered by team."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            if team_id:
                cursor.execute("""
                    SELECT g.*,
                           ht.abbreviation as home_abbrev,
                           at.abbreviation as away_abbrev
                    FROM games g
                    JOIN teams ht ON g.home_team_id = ht.id
                    JOIN teams at ON g.away_team_id = at.id
                    WHERE g.game_date BETWEEN ? AND ?
                    AND (g.home_team_id = ? OR g.away_team_id = ?)
                    ORDER BY g.game_date
                """, (start_date, end_date, team_id, team_id))
            else:
                cursor.execute("""
                    SELECT g.*,
                           ht.abbreviation as home_abbrev,
                           at.abbreviation as away_abbrev
                    FROM games g
                    JOIN teams ht ON g.home_team_id = ht.id
                    JOIN teams at ON g.away_team_id = at.id
                    WHERE g.game_date BETWEEN ? AND ?
                    ORDER BY g.game_date
                """, (start_date, end_date))

            return [dict(row) for row in cursor.fetchall()]

    # =========================================================================
    # Team Stats Operations (Point-in-Time)
    # =========================================================================

    def upsert_team_stats(self, stats_data: Dict) -> int:
        """Insert or update team statistics snapshot."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO team_stats_snapshot (
                    team_id, season_id, snapshot_date, games_played,
                    wins, losses, pts_avg, opp_pts_avg, fg_pct, fg3_pct, ft_pct,
                    reb_avg, ast_avg, stl_avg, blk_avg, tov_avg,
                    off_rating, def_rating, net_rating, pace
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(team_id, season_id, snapshot_date) DO UPDATE SET
                    games_played = excluded.games_played,
                    wins = excluded.wins,
                    losses = excluded.losses,
                    pts_avg = excluded.pts_avg,
                    opp_pts_avg = excluded.opp_pts_avg,
                    fg_pct = excluded.fg_pct,
                    fg3_pct = excluded.fg3_pct,
                    ft_pct = excluded.ft_pct,
                    reb_avg = excluded.reb_avg,
                    ast_avg = excluded.ast_avg,
                    stl_avg = excluded.stl_avg,
                    blk_avg = excluded.blk_avg,
                    tov_avg = excluded.tov_avg,
                    off_rating = excluded.off_rating,
                    def_rating = excluded.def_rating,
                    net_rating = excluded.net_rating,
                    pace = excluded.pace
            """, (
                stats_data.get("team_id"),
                stats_data.get("season_id"),
                stats_data.get("snapshot_date"),
                stats_data.get("games_played"),
                stats_data.get("wins"),
                stats_data.get("losses"),
                stats_data.get("pts_avg"),
                stats_data.get("opp_pts_avg"),
                stats_data.get("fg_pct"),
                stats_data.get("fg3_pct"),
                stats_data.get("ft_pct"),
                stats_data.get("reb_avg"),
                stats_data.get("ast_avg"),
                stats_data.get("stl_avg"),
                stats_data.get("blk_avg"),
                stats_data.get("tov_avg"),
                stats_data.get("off_rating"),
                stats_data.get("def_rating"),
                stats_data.get("net_rating"),
                stats_data.get("pace"),
            ))
            return cursor.lastrowid

    def get_team_stats_before_date(
        self,
        team_id: int,
        game_date: str,
        season_id: Optional[int] = None
    ) -> Optional[Dict]:
        """
        Get team stats as of the day BEFORE a game (point-in-time).

        This is critical for avoiding look-ahead bias in training.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            if season_id:
                cursor.execute("""
                    SELECT * FROM team_stats_snapshot
                    WHERE team_id = ? AND season_id = ? AND snapshot_date < ?
                    ORDER BY snapshot_date DESC
                    LIMIT 1
                """, (team_id, season_id, game_date))
            else:
                cursor.execute("""
                    SELECT * FROM team_stats_snapshot
                    WHERE team_id = ? AND snapshot_date < ?
                    ORDER BY snapshot_date DESC
                    LIMIT 1
                """, (team_id, game_date))

            row = cursor.fetchone()
            return dict(row) if row else None

    # =========================================================================
    # Betting Odds Operations
    # =========================================================================

    def insert_odds(self, odds_data: Dict) -> int:
        """Insert betting odds snapshot."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO betting_odds (
                    game_id, sportsbook, captured_at,
                    home_moneyline, away_moneyline,
                    home_spread, home_spread_odds, away_spread_odds,
                    total_line, over_odds, under_odds,
                    is_opening, is_closing
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                odds_data.get("game_id"),
                odds_data.get("sportsbook"),
                odds_data.get("captured_at", datetime.now().isoformat()),
                odds_data.get("home_moneyline"),
                odds_data.get("away_moneyline"),
                odds_data.get("home_spread"),
                odds_data.get("home_spread_odds"),
                odds_data.get("away_spread_odds"),
                odds_data.get("total_line"),
                odds_data.get("over_odds"),
                odds_data.get("under_odds"),
                odds_data.get("is_opening", False),
                odds_data.get("is_closing", False),
            ))
            return cursor.lastrowid

    def get_latest_odds(self, game_id: int, sportsbook: Optional[str] = None) -> Optional[Dict]:
        """Get the most recent odds for a game."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            if sportsbook:
                cursor.execute("""
                    SELECT * FROM betting_odds
                    WHERE game_id = ? AND sportsbook = ?
                    ORDER BY captured_at DESC
                    LIMIT 1
                """, (game_id, sportsbook))
            else:
                cursor.execute("""
                    SELECT * FROM betting_odds
                    WHERE game_id = ?
                    ORDER BY captured_at DESC
                    LIMIT 1
                """, (game_id,))

            row = cursor.fetchone()
            return dict(row) if row else None

    def get_closing_odds(self, game_id: int) -> Optional[Dict]:
        """Get closing odds for a game (for CLV calculation)."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM betting_odds
                WHERE game_id = ? AND is_closing = 1
                LIMIT 1
            """, (game_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    # =========================================================================
    # Injury Operations
    # =========================================================================

    def upsert_injury(self, injury_data: Dict) -> int:
        """Insert or update injury report."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO injuries (
                    player_id, reported_date, game_id, status,
                    injury_type, return_date, source
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                injury_data.get("player_id"),
                injury_data.get("reported_date"),
                injury_data.get("game_id"),
                injury_data.get("status"),
                injury_data.get("injury_type"),
                injury_data.get("return_date"),
                injury_data.get("source"),
            ))
            return cursor.lastrowid

    def get_injuries_for_game(self, game_id: int) -> List[Dict]:
        """Get all injuries reported for a specific game."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT i.*, p.name as player_name, t.abbreviation as team_abbrev
                FROM injuries i
                JOIN players p ON i.player_id = p.id
                JOIN teams t ON p.team_id = t.id
                WHERE i.game_id = ?
            """, (game_id,))
            return [dict(row) for row in cursor.fetchall()]

    # =========================================================================
    # Bet Tracking Operations
    # =========================================================================

    def insert_bet(self, bet_data: Dict) -> int:
        """Record a new bet."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO bets (
                    game_id, bet_type, selection, line,
                    odds_placed, stake, model_probability,
                    implied_probability, edge, notes
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                bet_data.get("game_id"),
                bet_data.get("bet_type"),
                bet_data.get("selection"),
                bet_data.get("line"),
                bet_data.get("odds_placed"),
                bet_data.get("stake"),
                bet_data.get("model_probability"),
                bet_data.get("implied_probability"),
                bet_data.get("edge"),
                bet_data.get("notes"),
            ))
            return cursor.lastrowid

    def settle_bet(self, bet_id: int, won: bool, payout: float, closing_odds: Optional[int] = None):
        """Settle a bet with outcome."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE bets
                SET status = ?,
                    payout = ?,
                    odds_closing = ?,
                    settled_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (
                "won" if won else "lost",
                payout,
                closing_odds,
                bet_id,
            ))

    def get_bet_performance(self, days: int = 30) -> Dict:
        """Get betting performance metrics for the last N days."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()

            cursor.execute("""
                SELECT
                    COUNT(*) as total_bets,
                    SUM(CASE WHEN status = 'won' THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN status = 'lost' THEN 1 ELSE 0 END) as losses,
                    SUM(stake) as total_staked,
                    SUM(payout) as total_payout,
                    AVG(edge) as avg_edge,
                    AVG(model_probability) as avg_model_prob
                FROM bets
                WHERE created_at > ? AND status IN ('won', 'lost')
            """, (cutoff_date,))

            row = cursor.fetchone()
            if not row:
                return {}

            total_bets = row[0] or 0
            wins = row[1] or 0
            losses = row[2] or 0
            total_staked = row[3] or 0
            total_payout = row[4] or 0

            return {
                "total_bets": total_bets,
                "wins": wins,
                "losses": losses,
                "win_rate": wins / total_bets if total_bets > 0 else 0,
                "total_staked": total_staked,
                "total_payout": total_payout,
                "profit_loss": total_payout - total_staked,
                "roi": (total_payout - total_staked) / total_staked if total_staked > 0 else 0,
                "avg_edge": row[5] or 0,
                "avg_model_prob": row[6] or 0,
            }

    # =========================================================================
    # Training Data Operations
    # =========================================================================

    def save_training_features(self, game_id: int, feature_type: str, features: Dict, outcome: float):
        """Save pre-computed training features for a game."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO training_features (game_id, feature_type, features_json, outcome_value)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(game_id, feature_type) DO UPDATE SET
                    features_json = excluded.features_json,
                    outcome_value = excluded.outcome_value
            """, (game_id, feature_type, json.dumps(features), outcome))

    def get_training_data(
        self,
        feature_type: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """Retrieve training data with features and outcomes."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            query = """
                SELECT tf.*, g.game_date, g.home_score, g.away_score
                FROM training_features tf
                JOIN games g ON tf.game_id = g.id
                WHERE tf.feature_type = ?
            """
            params = [feature_type]

            if start_date:
                query += " AND g.game_date >= ?"
                params.append(start_date)

            if end_date:
                query += " AND g.game_date <= ?"
                params.append(end_date)

            query += " ORDER BY g.game_date"

            if limit:
                query += f" LIMIT {limit}"

            cursor.execute(query, params)

            results = []
            for row in cursor.fetchall():
                data = dict(row)
                data["features"] = json.loads(data.get("features_json", "{}"))
                results.append(data)

            return results

    # =========================================================================
    # Model Prediction Operations
    # =========================================================================

    def save_prediction(self, prediction_data: Dict) -> int:
        """Save a model prediction for backtesting."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO model_predictions (
                    game_id, model_name, prediction_type,
                    predicted_value, actual_value, confidence, features_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                prediction_data.get("game_id"),
                prediction_data.get("model_name"),
                prediction_data.get("prediction_type"),
                prediction_data.get("predicted_value"),
                prediction_data.get("actual_value"),
                prediction_data.get("confidence"),
                json.dumps(prediction_data.get("features", {})),
            ))
            return cursor.lastrowid

    def get_prediction_accuracy(
        self,
        model_name: str,
        prediction_type: str,
        days: int = 30
    ) -> Dict:
        """Calculate prediction accuracy for a model."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()

            cursor.execute("""
                SELECT
                    COUNT(*) as total,
                    AVG(ABS(predicted_value - actual_value)) as mae,
                    AVG((predicted_value - actual_value) * (predicted_value - actual_value)) as mse,
                    AVG(confidence) as avg_confidence
                FROM model_predictions
                WHERE model_name = ?
                AND prediction_type = ?
                AND actual_value IS NOT NULL
                AND created_at > ?
            """, (model_name, prediction_type, cutoff_date))

            row = cursor.fetchone()
            if not row or row[0] == 0:
                return {}

            return {
                "total_predictions": row[0],
                "mae": row[1],
                "rmse": (row[2] ** 0.5) if row[2] else 0,
                "avg_confidence": row[3],
            }


# =============================================================================
# Utility Functions
# =============================================================================

def initialize_teams(db: DatabaseManager):
    """Initialize database with all NBA teams."""
    from fast_data_fetcher import NBA_TEAMS

    for abbrev, data in NBA_TEAMS.items():
        db.upsert_team({
            "nba_id": data["id"],
            "abbreviation": abbrev,
            "name": data["name"],
            "conference": data["conference"],
        })

    print(f"Initialized {len(NBA_TEAMS)} NBA teams")


def initialize_seasons(db: DatabaseManager):
    """Initialize seasons table with recent seasons."""
    seasons = [
        {"season_string": "2021-22", "start_date": "2021-10-19", "end_date": "2022-06-16"},
        {"season_string": "2022-23", "start_date": "2022-10-18", "end_date": "2023-06-12"},
        {"season_string": "2023-24", "start_date": "2023-10-24", "end_date": "2024-06-17"},
        {"season_string": "2024-25", "start_date": "2024-10-22", "end_date": "2025-06-15"},
        {"season_string": "2025-26", "start_date": "2025-10-21", "end_date": None, "is_current": True},
    ]

    with db.get_connection() as conn:
        cursor = conn.cursor()
        for season in seasons:
            cursor.execute("""
                INSERT OR IGNORE INTO seasons (season_string, start_date, end_date, is_current)
                VALUES (?, ?, ?, ?)
            """, (
                season["season_string"],
                season["start_date"],
                season.get("end_date"),
                season.get("is_current", False),
            ))

    print(f"Initialized {len(seasons)} NBA seasons")


if __name__ == "__main__":
    print("NBA Betting Model Database")
    print("=" * 50)

    # Initialize database
    db = DatabaseManager()
    db.initialize()

    # Initialize teams and seasons
    initialize_teams(db)
    initialize_seasons(db)

    print("\nDatabase ready for use!")
    print(f"Location: {db.db_path}")
