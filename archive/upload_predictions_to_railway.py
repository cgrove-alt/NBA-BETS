#!/usr/bin/env python3
"""
Quick-Fix Script: Upload Predictions CSV to Railway Backend

This script manually uploads a predictions CSV file to Railway by storing it in
the PostgreSQL database, making it accessible via the API immediately.

Usage:
    python upload_predictions_to_railway.py predictions_2026-01-21.csv

Requirements:
    - PostgreSQL database connection (DATABASE_URL env var or Railway CLI)
    - predictions CSV file to upload

Note: This is a TEMPORARY solution until the automated cron service is deployed.
"""

import sys
import os
import pandas as pd
import psycopg2
from datetime import datetime
from pathlib import Path

def upload_predictions_to_db(csv_path: str, database_url: str = None):
    """Upload predictions CSV to PostgreSQL database."""

    # Validate CSV exists
    if not Path(csv_path).exists():
        print(f"❌ ERROR: File not found: {csv_path}")
        sys.exit(1)

    # Get database URL
    if not database_url:
        database_url = os.getenv("DATABASE_URL")

    if not database_url:
        print("❌ ERROR: DATABASE_URL not set")
        print("\nOptions:")
        print("1. Set DATABASE_URL environment variable")
        print("2. Use Railway CLI: railway run python upload_predictions_to_railway.py")
        print("3. Pass database URL as second argument")
        sys.exit(1)

    # Load predictions CSV
    print(f"📊 Loading predictions from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
        print(f"✓ Loaded {len(df)} predictions")
    except Exception as e:
        print(f"❌ ERROR loading CSV: {e}")
        sys.exit(1)

    # Extract date from CSV filename or first row
    if 'date' in df.columns:
        prediction_date = df['date'].iloc[0]
    else:
        # Extract from filename: predictions_2026-01-21.csv
        import re
        match = re.search(r'(\d{4}-\d{2}-\d{2})', csv_path)
        prediction_date = match.group(1) if match else datetime.now().strftime('%Y-%m-%d')

    print(f"📅 Prediction date: {prediction_date}")

    # Connect to PostgreSQL
    print("\n🔌 Connecting to PostgreSQL...")
    try:
        conn = psycopg2.connect(database_url)
        cursor = conn.cursor()
        print("✓ Connected to database")
    except Exception as e:
        print(f"❌ ERROR connecting to database: {e}")
        sys.exit(1)

    # Create predictions_history table if not exists
    print("\n📋 Creating predictions_history table if needed...")
    try:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions_history (
                id SERIAL PRIMARY KEY,
                date DATE NOT NULL,
                game VARCHAR(100),
                player_name VARCHAR(100) NOT NULL,
                team VARCHAR(10),
                prop_type VARCHAR(20) NOT NULL,
                prediction FLOAT NOT NULL,
                pred_low FLOAT,
                pred_median FLOAT,
                pred_high FLOAT,
                line FLOAT NOT NULL,
                over_prob FLOAT,
                edge FLOAT,
                confidence_score FLOAT NOT NULL,
                edge_quality_tier VARCHAR(20),
                suggested_bet_size FLOAT,
                bet_recommendation VARCHAR(20),
                pick VARCHAR(10),
                uncertainty_flag VARCHAR(50),
                injury_boost BOOLEAN,
                created_at TIMESTAMP DEFAULT NOW(),
                UNIQUE(date, player_name, prop_type)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_date ON predictions_history(date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_player ON predictions_history(date, player_name)")
        conn.commit()
        print("✓ Table ready")
    except Exception as e:
        print(f"❌ ERROR creating table: {e}")
        conn.close()
        sys.exit(1)

    # Delete existing predictions for this date (if any)
    print(f"\n🗑️  Clearing existing predictions for {prediction_date}...")
    try:
        cursor.execute(
            "DELETE FROM predictions_history WHERE date = %s",
            (prediction_date,)
        )
        deleted_count = cursor.rowcount
        conn.commit()
        if deleted_count > 0:
            print(f"✓ Deleted {deleted_count} old predictions")
        else:
            print("✓ No existing predictions to delete")
    except Exception as e:
        print(f"⚠️  Warning: Could not delete old predictions: {e}")

    # Insert new predictions
    print(f"\n📤 Uploading {len(df)} predictions...")
    inserted_count = 0
    errors = []

    for idx, row in df.iterrows():
        try:
            # Handle NaN values
            def safe_val(val):
                return None if pd.isna(val) or val == '' else val

            cursor.execute("""
                INSERT INTO predictions_history (
                    date, game, player_name, team, prop_type,
                    prediction, pred_low, pred_median, pred_high,
                    line, over_prob, edge, confidence_score,
                    edge_quality_tier, suggested_bet_size, bet_recommendation,
                    pick, uncertainty_flag, injury_boost
                ) VALUES (
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s
                )
                ON CONFLICT (date, player_name, prop_type) DO UPDATE SET
                    prediction = EXCLUDED.prediction,
                    pred_low = EXCLUDED.pred_low,
                    pred_median = EXCLUDED.pred_median,
                    pred_high = EXCLUDED.pred_high,
                    line = EXCLUDED.line,
                    over_prob = EXCLUDED.over_prob,
                    edge = EXCLUDED.edge,
                    confidence_score = EXCLUDED.confidence_score,
                    edge_quality_tier = EXCLUDED.edge_quality_tier,
                    suggested_bet_size = EXCLUDED.suggested_bet_size,
                    bet_recommendation = EXCLUDED.bet_recommendation,
                    pick = EXCLUDED.pick,
                    uncertainty_flag = EXCLUDED.uncertainty_flag,
                    injury_boost = EXCLUDED.injury_boost
            """, (
                prediction_date,
                safe_val(row.get('game')),
                row['player_name'],
                safe_val(row.get('team')),
                row['prop_type'],
                row['prediction'],
                safe_val(row.get('pred_low')),
                safe_val(row.get('pred_median')),
                safe_val(row.get('pred_high')),
                row['line'],
                safe_val(row.get('over_prob')),
                safe_val(row.get('edge')),
                row['confidence_score'],
                safe_val(row.get('edge_quality_tier')),
                safe_val(row.get('suggested_bet_size')),
                safe_val(row.get('bet_recommendation')),
                safe_val(row.get('pick')),
                safe_val(row.get('uncertainty_flag')),
                safe_val(row.get('injury_boost'))
            ))
            inserted_count += 1

            # Progress indicator
            if (idx + 1) % 20 == 0:
                print(f"  Uploaded {idx + 1}/{len(df)} predictions...")

        except Exception as e:
            errors.append(f"Row {idx}: {e}")

    # Commit all inserts
    try:
        conn.commit()
        print(f"✓ Successfully uploaded {inserted_count}/{len(df)} predictions")
    except Exception as e:
        print(f"❌ ERROR committing to database: {e}")
        conn.rollback()
        conn.close()
        sys.exit(1)

    # Report errors if any
    if errors:
        print(f"\n⚠️  {len(errors)} errors occurred:")
        for error in errors[:5]:  # Show first 5 errors
            print(f"  - {error}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")

    # Close connection
    conn.close()

    print("\n✅ UPLOAD COMPLETE!")
    print("\n🔗 Test API endpoint:")
    print(f"   https://web-production-7b482.up.railway.app/api/predictions/{prediction_date}")
    print("\n🌐 Check Vercel frontend:")
    print("   https://your-vercel-site.vercel.app")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python upload_predictions_to_railway.py <predictions.csv> [database_url]")
        print("\nExample:")
        print("  python upload_predictions_to_railway.py predictions_2026-01-21.csv")
        print("\nOr with Railway CLI:")
        print("  railway run python upload_predictions_to_railway.py predictions_2026-01-21.csv")
        sys.exit(1)

    csv_path = sys.argv[1]
    database_url = sys.argv[2] if len(sys.argv) > 2 else None

    print("=" * 70)
    print("  UPLOAD PREDICTIONS TO RAILWAY")
    print("=" * 70)
    print()

    upload_predictions_to_db(csv_path, database_url)

    print("\n" + "=" * 70)
    print("  NO SHORTCUTS. NO EXCUSES!")
    print("=" * 70)


if __name__ == "__main__":
    main()
