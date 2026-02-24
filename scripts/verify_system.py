#!/usr/bin/env python3
"""
System Verification Script — NBA Betting Model

Checks all components are functional before deployment.
Run: python3 scripts/verify_system.py

Each check prints PASS, FALLBACK, or FAIL.
FALLBACK means the primary service is unavailable but a local fallback works.
"""

import os
import sys
import importlib
import glob

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))


class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'


def pass_msg(label, detail=""):
    detail_str = f" — {detail}" if detail else ""
    print(f"  {Colors.GREEN}PASS{Colors.END}     {label}{detail_str}")
    return True


def fallback_msg(label, detail=""):
    detail_str = f" — {detail}" if detail else ""
    print(f"  {Colors.YELLOW}FALLBACK{Colors.END} {label}{detail_str}")
    return True


def fail_msg(label, detail=""):
    detail_str = f" — {detail}" if detail else ""
    print(f"  {Colors.RED}FAIL{Colors.END}     {label}{detail_str}")
    return False


def check_env_vars():
    """Check required and optional environment variables."""
    print(f"\n{Colors.BOLD}1. Environment Variables{Colors.END}")

    results = []

    # Required
    bdl = os.environ.get('BALLDONTLIE_API_KEY')
    if bdl and bdl != 'your_balldontlie_api_key_here':
        results.append(pass_msg("BALLDONTLIE_API_KEY", "set"))
    else:
        results.append(fail_msg("BALLDONTLIE_API_KEY", "not set — required for stats"))

    odds = os.environ.get('THE_ODDS_API_KEY')
    if odds and odds != 'your-key-here' and odds != 'your_odds_api_key_here':
        results.append(pass_msg("THE_ODDS_API_KEY", "set"))
    else:
        results.append(fallback_msg("THE_ODDS_API_KEY", "not set — odds features disabled"))

    # Infrastructure (have local fallbacks)
    db_url = os.environ.get('DATABASE_URL')
    if db_url and 'postgresql' in db_url:
        results.append(pass_msg("DATABASE_URL", "PostgreSQL configured"))
    else:
        results.append(fallback_msg("DATABASE_URL", "not set — using SQLite fallback"))

    redis_url = os.environ.get('REDIS_URL')
    if redis_url:
        results.append(pass_msg("REDIS_URL", "configured"))
    else:
        results.append(fallback_msg("REDIS_URL", "not set — using InMemoryMessageBus fallback"))

    gemini = os.environ.get('GEMINI_API_KEY')
    if gemini and gemini != 'your_gemini_api_key_here' and gemini != 'your-key-here':
        results.append(pass_msg("GEMINI_API_KEY", "set"))
    else:
        results.append(fallback_msg("GEMINI_API_KEY", "not set — agents use deterministic fallbacks"))

    return all(results)


def check_database():
    """Check database connectivity."""
    print(f"\n{Colors.BOLD}2. Database Connection{Colors.END}")

    # Try PostgreSQL first
    db_url = os.environ.get('DATABASE_URL')
    if db_url:
        try:
            import psycopg2
            conn = psycopg2.connect(db_url)
            conn.close()
            return pass_msg("PostgreSQL", "connected")
        except ImportError:
            pass_msg("psycopg2 not installed", "skipping PostgreSQL check")
        except Exception as e:
            fallback_msg("PostgreSQL", f"connection failed: {e}")

    # Try SQLite fallback
    try:
        import sqlite3
        db_path = os.path.join(PROJECT_ROOT, 'data', 'calibration.db')
        conn = sqlite3.connect(db_path)
        conn.execute("SELECT 1")
        conn.close()
        return fallback_msg("SQLite", f"available at data/calibration.db")
    except Exception as e:
        return fail_msg("Database", f"no database available: {e}")


def check_redis():
    """Check Redis connectivity."""
    print(f"\n{Colors.BOLD}3. Redis / Message Bus{Colors.END}")

    redis_url = os.environ.get('REDIS_URL')
    if redis_url:
        try:
            import redis
            client = redis.Redis.from_url(redis_url, decode_responses=True)
            client.ping()
            return pass_msg("Redis", "connected and responding")
        except ImportError:
            fallback_msg("redis package not installed")
        except Exception as e:
            fallback_msg("Redis", f"connection failed: {e}")

    # Verify InMemory fallback works
    try:
        from agents.core.message_bus import InMemoryMessageBus
        bus = InMemoryMessageBus()
        return fallback_msg("InMemoryMessageBus", "available as fallback")
    except Exception as e:
        return fail_msg("Message Bus", f"no message bus available: {e}")


def check_balldontlie_api():
    """Check BallDontLie API responds."""
    print(f"\n{Colors.BOLD}4. BallDontLie API{Colors.END}")

    api_key = os.environ.get('BALLDONTLIE_API_KEY')
    if not api_key or api_key == 'your_balldontlie_api_key_here':
        return fail_msg("BallDontLie API", "no API key set")

    try:
        import urllib.request
        import json

        req = urllib.request.Request(
            'https://api.balldontlie.io/v1/teams',
            headers={'Authorization': api_key}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            team_count = len(data.get('data', []))
            return pass_msg("BallDontLie API", f"responding ({team_count} teams)")
    except Exception as e:
        return fail_msg("BallDontLie API", f"request failed: {e}")


def check_model_files():
    """Check all required model files exist."""
    print(f"\n{Colors.BOLD}5. Model Files{Colors.END}")

    models_dir = os.path.join(PROJECT_ROOT, 'models')
    pkl_files = glob.glob(os.path.join(models_dir, '*.pkl'))

    if not pkl_files:
        return fail_msg("Model files", "no .pkl files found in models/")

    # Check key model categories
    filenames = [os.path.basename(f) for f in pkl_files]

    categories = {
        'spread': [f for f in filenames if 'spread' in f],
        'moneyline': [f for f in filenames if 'moneyline' in f],
        'player_props': [f for f in filenames if f.startswith('player_')],
        'minutes_oracle': [f for f in filenames if 'minutes' in f],
    }

    all_ok = True
    for cat, files in categories.items():
        if files:
            pass_msg(f"{cat}", f"{len(files)} model(s)")
        else:
            fail_msg(f"{cat}", "no models found")
            all_ok = False

    total = len(pkl_files)
    print(f"         Total: {total} model files in models/")
    return all_ok


def check_agent_imports():
    """Check all 6 agents can be imported."""
    print(f"\n{Colors.BOLD}6. Agent Imports{Colors.END}")

    agents = {
        'pregame':      'agents.pregame.pregame_agent',
        'postgame':     'agents.postgame.postgame_agent',
        'odds_monitor': 'agents.odds_monitor.odds_monitor_agent',
        'orchestrator': 'agents.orchestrator.orchestrator_agent',
        'watchdog':     'agents.watchdog.watchdog_agent',
        'briefing':     'agents.briefing.briefing_agent',
    }

    all_ok = True
    for name, module_path in agents.items():
        try:
            importlib.import_module(module_path)
            pass_msg(f"{name}", f"imported from {module_path}")
        except Exception as e:
            fail_msg(f"{name}", f"import failed: {e}")
            all_ok = False

    return all_ok


def check_core_imports():
    """Check core packages can be imported."""
    print(f"\n{Colors.BOLD}7. Core Package Imports{Colors.END}")

    packages = [
        ('nba_data', 'Data pipeline'),
        ('nba_models', 'Prediction engine'),
        ('nba_betting', 'Betting engine'),
        ('agents.core', 'Agent framework'),
        ('backend.api', 'Backend API'),
    ]

    all_ok = True
    for pkg, desc in packages:
        try:
            importlib.import_module(pkg)
            pass_msg(f"{pkg}", desc)
        except Exception as e:
            fail_msg(f"{pkg}", f"{desc} — {e}")
            all_ok = False

    return all_ok


def main():
    print("=" * 60)
    print("  NBA BETTING MODEL — SYSTEM VERIFICATION")
    print("=" * 60)

    checks = [
        check_env_vars,
        check_database,
        check_redis,
        check_balldontlie_api,
        check_model_files,
        check_agent_imports,
        check_core_imports,
    ]

    results = []
    for check in checks:
        try:
            results.append(check())
        except Exception as e:
            print(f"  {Colors.RED}ERROR{Colors.END}  Check crashed: {e}")
            results.append(False)

    # Summary
    passed = sum(1 for r in results if r)
    total = len(results)

    print()
    print("=" * 60)
    if passed == total:
        print(f"  {Colors.GREEN}{Colors.BOLD}ALL {total} CHECKS PASSED{Colors.END}")
        print("  System is ready. FALLBACK items work locally but need")
        print("  proper services (PostgreSQL, Redis) for production.")
    else:
        failed = total - passed
        print(f"  {Colors.RED}{Colors.BOLD}{failed} CHECK(S) FAILED{Colors.END} out of {total}")
        print("  Fix the FAIL items above before proceeding.")
    print("=" * 60)

    sys.exit(0 if passed == total else 1)


if __name__ == '__main__':
    main()
