#!/usr/bin/env python3
"""
Test accessing DARKO Google Spreadsheet as CSV

Google Sheets can be exported as CSV using a special URL format:
https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={GID}
"""

import requests
import csv
from io import StringIO

# DARKO spreadsheet ID (from URL)
SHEET_ID = "1mhwOLqPu2F9026EQiVxFPIN1t9RGafGpl-dokaIsm9c"

# Try to export as CSV (gid=0 is usually the first sheet)
csv_url = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid=0"

print("Attempting to fetch DARKO data from Google Sheets...")
print(f"URL: {csv_url}")
print()

try:
    response = requests.get(csv_url, timeout=10)
    response.raise_for_status()

    print(f"✓ Successfully fetched data ({len(response.text)} bytes)")
    print()

    # Parse CSV
    csv_reader = csv.DictReader(StringIO(response.text))

    # Get headers
    headers = csv_reader.fieldnames
    print(f"Columns ({len(headers)}):")
    for i, header in enumerate(headers[:20], 1):  # Show first 20 columns
        print(f"  {i:2d}. {header}")

    if len(headers) > 20:
        print(f"  ... and {len(headers) - 20} more columns")

    print()

    # Get first few rows
    print("Sample data (first 5 rows):")
    print("-" * 80)

    rows = list(csv_reader)
    for i, row in enumerate(rows[:5], 1):
        # Show key fields
        player = row.get('PLAYER', row.get('Player', row.get('player', 'N/A')))
        team = row.get('TEAM', row.get('Team', row.get('team', 'N/A')))
        dpm = row.get('DPM', row.get('dpm', 'N/A'))

        print(f"{i}. {player:<25s} | Team: {team:<5s} | DPM: {dpm}")

    print()
    print(f"✓ Total rows: {len(rows)}")
    print()

    # Check if we have DPM data
    dpm_columns = [col for col in headers if 'dpm' in col.lower() or 'plus' in col.lower()]
    if dpm_columns:
        print(f"✓ Found DPM-related columns: {dpm_columns}")
    else:
        print("⚠️  No obvious DPM columns found")
        print("   Available columns suggest this might be a different data source")

    print()
    print("SUCCESS: DARKO Google Sheet is accessible!")

except requests.exceptions.HTTPError as e:
    print(f"❌ HTTP Error: {e}")
    print("   The sheet may be private or require authentication")
except Exception as e:
    print(f"❌ Error: {e}")
    print(f"   Type: {type(e).__name__}")
