#!/usr/bin/env python3
"""
Local Deployment Configuration Tests

Tests that all deployment files are properly configured before deploying to Railway.

Run: python test_deployment_config.py
"""

import os
import sys
import json
from pathlib import Path


def test_railway_toml_exists():
    """Test that railway.toml exists and has correct structure."""
    print("Testing railway.toml...")

    toml_path = Path("railway.toml")
    if not toml_path.exists():
        print("  ❌ railway.toml not found")
        return False

    content = toml_path.read_text()

    # Check for required sections
    required = [
        "[build]",
        "[deploy]",
        "startCommand",
        "healthcheckPath"
    ]

    for req in required:
        if req not in content:
            print(f"  ❌ Missing required section: {req}")
            return False

    print("  ✅ railway.toml configured correctly")
    return True


def test_migration_script_exists():
    """Test that migration script exists."""
    print("Testing migration script...")

    migration_path = Path("migrations/001_initial_schema.sql")
    if not migration_path.exists():
        print("  ❌ migrations/001_initial_schema.sql not found")
        return False

    content = migration_path.read_text()

    # Check for required tables
    required_tables = [
        "CREATE TABLE IF NOT EXISTS teams",
        "CREATE TABLE IF NOT EXISTS players",
        "CREATE TABLE IF NOT EXISTS games",
        "CREATE TABLE IF NOT EXISTS injuries",
        "CREATE TABLE IF NOT EXISTS odds_history",
        "CREATE TABLE IF NOT EXISTS predictions_history"
    ]

    for table in required_tables:
        if table not in content:
            print(f"  ❌ Missing table creation: {table}")
            return False

    print("  ✅ Migration script includes all required tables")
    return True


def test_env_example_exists():
    """Test that .env.example exists with all required variables."""
    print("Testing .env.example...")

    env_path = Path(".env.example")
    if not env_path.exists():
        print("  ❌ .env.example not found")
        return False

    content = env_path.read_text()

    # Check for required variables
    required_vars = [
        "BALLDONTLIE_API_KEY",
        "DATABASE_URL",
        "THE_ODDS_API_KEY",
        "AUTH_ENABLED",
        "JWT_SECRET_KEY"
    ]

    for var in required_vars:
        if var not in content:
            print(f"  ❌ Missing environment variable: {var}")
            return False

    print("  ✅ .env.example contains all required variables")
    return True


def test_requirements_txt():
    """Test that requirements.txt has all necessary packages."""
    print("Testing requirements.txt...")

    req_path = Path("requirements.txt")
    if not req_path.exists():
        print("  ❌ requirements.txt not found")
        return False

    content = req_path.read_text()

    # Check for critical packages
    required_packages = [
        "fastapi",
        "uvicorn",
        "pydantic",
        "numpy",
        "pandas",
        "scikit-learn",
        "lightgbm",
        "catboost",
        "xgboost",
        "requests",
        "apscheduler",
        "plotly",
        "jinja2"
    ]

    for pkg in required_packages:
        if pkg not in content:
            print(f"  ❌ Missing package: {pkg}")
            return False

    print("  ✅ requirements.txt includes all critical packages")
    return True


def test_api_structure():
    """Test that API structure is correct."""
    print("Testing API structure...")

    try:
        from backend.api import app

        # Check that app exists
        if not app:
            print("  ❌ FastAPI app not found")
            return False

        # Check for required endpoints
        routes = [route.path for route in app.routes if hasattr(route, 'path')]

        required_routes = [
            "/api/health",
            "/api/predictions/{date}",
            "/api/injuries/{date}",
            "/api/backtest/latest"
        ]

        for route in required_routes:
            if route not in routes:
                print(f"  ❌ Missing route: {route}")
                return False

        print("  ✅ API has all required endpoints")
        return True

    except ImportError as e:
        print(f"  ❌ Failed to import API: {e}")
        return False


def test_scheduled_scripts_exist():
    """Test that scheduled job scripts exist."""
    print("Testing scheduled job scripts...")

    required_scripts = [
        "daily_predictions.py",
        "odds_tracker_service.py",
        "scheduled_retraining.py"
    ]

    for script in required_scripts:
        if not Path(script).exists():
            print(f"  ❌ Missing script: {script}")
            return False

    print("  ✅ All scheduled job scripts exist")
    return True


def test_deployment_docs():
    """Test that deployment documentation exists."""
    print("Testing deployment documentation...")

    if not Path("RAILWAY_DEPLOYMENT.md").exists():
        print("  ❌ RAILWAY_DEPLOYMENT.md not found")
        return False

    print("  ✅ Deployment documentation exists")
    return True


def test_verify_script():
    """Test that verification script exists."""
    print("Testing verification script...")

    verify_path = Path("verify_deployment.py")
    if not verify_path.exists():
        print("  ❌ verify_deployment.py not found")
        return False

    # Check if script is executable
    if not os.access(verify_path, os.X_OK):
        print("  ⚠️  verify_deployment.py not executable (run: chmod +x verify_deployment.py)")

    print("  ✅ Verification script exists")
    return True


def main():
    """Run all tests."""
    print("=" * 70)
    print("  Railway Deployment Configuration Tests")
    print("=" * 70)
    print()

    tests = [
        test_railway_toml_exists,
        test_migration_script_exists,
        test_env_example_exists,
        test_requirements_txt,
        test_api_structure,
        test_scheduled_scripts_exist,
        test_deployment_docs,
        test_verify_script,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"  ❌ Test failed with error: {e}")
            results.append(False)
        print()

    # Summary
    print("=" * 70)
    print("  Summary")
    print("=" * 70)

    passed = sum(results)
    total = len(results)

    print(f"  Passed: {passed}/{total}")
    print(f"  Failed: {total - passed}/{total}")
    print()

    if passed == total:
        print("  🎉 All deployment configuration tests passed!")
        print("  ✅ Ready to deploy to Railway")
        return 0
    print("  ❌ Some tests failed. Fix errors above before deploying.")
    return 1


if __name__ == '__main__':
    sys.exit(main())
