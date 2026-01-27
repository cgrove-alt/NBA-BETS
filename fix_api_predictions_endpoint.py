#!/usr/bin/env python3
"""
Script to patch backend/api.py to read predictions from PostgreSQL instead of CSV.

This is a TEMPORARY patch until we deploy the proper database-backed solution.

Usage:
    python fix_api_predictions_endpoint.py
"""

import re

# Read the current API file
with open('backend/api.py') as f:
    content = f.read()

# New implementation that reads from database
new_implementation = '''@app.get("/api/predictions/{date}", response_model=DailyPredictionsResponse)
def get_daily_predictions(date: str):
    """Get daily predictions for a specific date.

    Args:
        date: Date string in YYYY-MM-DD format

    Returns:
        Daily predictions with confidence, bet sizing, and recommendations
    """
    import pandas as pd
    import psycopg2
    import os
    from pathlib import Path

    # Validate date format
    try:
        from datetime import datetime
        datetime.strptime(date, '%Y-%m-%d')
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid date format. Use YYYY-MM-DD"
        )

    # TRY 1: Read from PostgreSQL database (Railway production)
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        try:
            conn = psycopg2.connect(database_url)
            df = pd.read_sql(
                """
                SELECT date, game, player_name, team, prop_type,
                       prediction, pred_low, pred_median, pred_high,
                       line, over_prob, edge, confidence_score,
                       edge_quality_tier, suggested_bet_size,
                       bet_recommendation, pick, uncertainty_flag, injury_boost
                FROM predictions_history
                WHERE date = %s
                ORDER BY confidence_score DESC
                """,
                conn,
                params=(date,)
            )
            conn.close()

            if len(df) > 0:
                # Successfully loaded from database
                pass  # Continue to conversion below
            else:
                # No predictions in database, try CSV fallback
                raise ValueError("No predictions in database")

        except Exception as db_error:
            # Database read failed, try CSV fallback
            print(f"Warning: Database read failed: {db_error}")
            database_url = None  # Force CSV fallback

    # TRY 2: Read from CSV file (local development fallback)
    if not database_url or len(df) == 0:
        csv_path = Path(f"predictions_{date}.csv")

        if not csv_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"No predictions found for {date}. Generate predictions first or upload to database."
            )

        # Load predictions CSV
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error reading predictions file: {str(e)}"
            )

    # Convert to prediction objects (same for both database and CSV)
    predictions = []
    for _, row in df.iterrows():
        # Handle NaN values for string fields - pandas reads empty cells as NaN
        team = row.get('team', '')
        if pd.notna(team) and team != '':
            team = str(team)
        else:
            team = ''

        uncertainty_flag = row.get('uncertainty_flag')
        if pd.notna(uncertainty_flag) and uncertainty_flag != '':
            uncertainty_flag = str(uncertainty_flag)
        else:
            uncertainty_flag = None

        pick = row.get('pick')
        if pd.notna(pick) and pick != '':
            pick = str(pick)
        else:
            pick = None

        predictions.append(DailyPrediction(
            player_name=str(row['player_name']),
            team=team,
            prop_type=str(row['prop_type']),
            prediction=float(row['prediction']),
            pred_low=float(row.get('pred_low', 0)) if pd.notna(row.get('pred_low')) else None,
            pred_median=float(row.get('pred_median', 0)) if pd.notna(row.get('pred_median')) else None,
            pred_high=float(row.get('pred_high', 0)) if pd.notna(row.get('pred_high')) else None,
            line=float(row['line']),
            confidence_score=float(row['confidence_score']),
            edge_quality_tier=str(row['edge_quality_tier']),
            suggested_bet_size=float(row['suggested_bet_size']) if pd.notna(row.get('suggested_bet_size')) else 0.0,
            bet_recommendation=str(row['bet_recommendation']),
            uncertainty_flag=uncertainty_flag,
            pick=pick,
            edge=float(row.get('edge', 0)) if pd.notna(row.get('edge')) else 0.0
        ))

    return DailyPredictionsResponse(
        date=date,
        predictions=predictions
    )
'''

# Find the old get_daily_predictions function and replace it
pattern = r'@app\.get\("/api/predictions/\{date\}".*?\n(?:async )?def get_daily_predictions.*?(?=\n@app\.|$)'

if re.search(pattern, content, re.DOTALL):
    # Replace the function
    new_content = re.sub(pattern, new_implementation.rstrip(), content, flags=re.DOTALL)

    # Write back
    with open('backend/api.py', 'w') as f:
        f.write(new_content)

    print("✅ Successfully patched backend/api.py")
    print("\n📝 Changes made:")
    print("   - /api/predictions/{date} now reads from PostgreSQL first")
    print("   - Falls back to CSV if database unavailable (local dev)")
    print("   - Works seamlessly in both Railway (DB) and local (CSV) environments")
    print("\n🚀 Next steps:")
    print("   1. Commit and push to GitHub")
    print("   2. Railway will auto-deploy the updated API")
    print("   3. Upload predictions: python upload_predictions_to_railway.py predictions_2026-01-21.csv")
    print("   4. Verify API: curl https://web-production-7b482.up.railway.app/api/predictions/2026-01-21")
else:
    print("❌ ERROR: Could not find get_daily_predictions function to patch")
    print("   Manual edit required - see new_implementation in this script")
