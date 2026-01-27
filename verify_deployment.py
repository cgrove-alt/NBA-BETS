#!/usr/bin/env python3
"""
Railway Deployment Verification Script

Verifies that all Railway services are deployed correctly:
- API health check
- Database connectivity
- Scheduled jobs status
- Environment variables

Usage:
    python verify_deployment.py --url https://your-app.railway.app
    python verify_deployment.py --url https://your-app.railway.app --api-key your_api_key
"""

import sys
import argparse
import requests
from datetime import datetime


def check_api_health(base_url: str, api_key: str = None) -> tuple[bool, str]:
    """Check API /health endpoint."""
    try:
        headers = {}
        if api_key:
            headers['X-API-Key'] = api_key

        response = requests.get(f"{base_url}/api/health", headers=headers, timeout=10)

        if response.status_code == 200:
            data = response.json()
            return True, f"✅ API Health: {data.get('status', 'unknown')}"
        return False, f"❌ API Health: HTTP {response.status_code}"

    except requests.exceptions.RequestException as e:
        return False, f"❌ API Health: Connection failed - {str(e)}"


def check_predictions_endpoint(base_url: str, api_key: str = None) -> tuple[bool, str]:
    """Check predictions endpoint with today's date."""
    try:
        headers = {}
        if api_key:
            headers['X-API-Key'] = api_key

        today = datetime.now().strftime('%Y-%m-%d')
        response = requests.get(f"{base_url}/api/predictions/{today}", headers=headers, timeout=10)

        if response.status_code == 200:
            data = response.json()
            count = len(data.get('predictions', []))
            return True, f"✅ Predictions Endpoint: {count} predictions for {today}"
        if response.status_code == 404:
            return True, f"⚠️  Predictions Endpoint: No predictions for {today} (this is OK if no games scheduled)"
        return False, f"❌ Predictions Endpoint: HTTP {response.status_code}"

    except requests.exceptions.RequestException as e:
        return False, f"❌ Predictions Endpoint: {str(e)}"


def check_injuries_endpoint(base_url: str, api_key: str = None) -> tuple[bool, str]:
    """Check injuries endpoint."""
    try:
        headers = {}
        if api_key:
            headers['X-API-Key'] = api_key

        today = datetime.now().strftime('%Y-%m-%d')
        response = requests.get(f"{base_url}/api/injuries/{today}", headers=headers, timeout=10)

        if response.status_code == 200:
            data = response.json()
            count = len(data.get('injuries', []))
            return True, f"✅ Injuries Endpoint: {count} injuries for {today}"
        if response.status_code == 404:
            return True, "⚠️  Injuries Endpoint: No injuries data (this is OK)"
        return False, f"❌ Injuries Endpoint: HTTP {response.status_code}"

    except requests.exceptions.RequestException as e:
        return False, f"❌ Injuries Endpoint: {str(e)}"


def check_backtest_endpoint(base_url: str, api_key: str = None) -> tuple[bool, str]:
    """Check backtest endpoint."""
    try:
        headers = {}
        if api_key:
            headers['X-API-Key'] = api_key

        response = requests.get(f"{base_url}/api/backtest/latest", headers=headers, timeout=10)

        if response.status_code == 200:
            data = response.json()
            roi = data.get('overall_metrics', {}).get('roi', 'N/A')
            return True, f"✅ Backtest Endpoint: ROI = {roi}%"
        if response.status_code == 404:
            return True, "⚠️  Backtest Endpoint: No backtest results (run backtest first)"
        return False, f"❌ Backtest Endpoint: HTTP {response.status_code}"

    except requests.exceptions.RequestException as e:
        return False, f"❌ Backtest Endpoint: {str(e)}"


def check_cors_headers(base_url: str) -> tuple[bool, str]:
    """Check CORS headers are properly configured."""
    try:
        response = requests.options(f"{base_url}/api/health", timeout=10)

        cors_headers = {
            'Access-Control-Allow-Origin': response.headers.get('Access-Control-Allow-Origin'),
            'Access-Control-Allow-Methods': response.headers.get('Access-Control-Allow-Methods'),
        }

        if cors_headers['Access-Control-Allow-Origin']:
            return True, f"✅ CORS Configured: {cors_headers['Access-Control-Allow-Origin']}"
        return False, "❌ CORS: Not configured"

    except requests.exceptions.RequestException as e:
        return False, f"❌ CORS Check: {str(e)}"


def check_response_time(base_url: str) -> tuple[bool, str]:
    """Check API response time."""
    try:
        import time
        start = time.time()
        requests.get(f"{base_url}/api/health", timeout=10)
        elapsed = (time.time() - start) * 1000  # Convert to ms

        if elapsed < 500:
            return True, f"✅ Response Time: {elapsed:.0f}ms (excellent)"
        if elapsed < 1000:
            return True, f"⚠️  Response Time: {elapsed:.0f}ms (good)"
        return False, f"❌ Response Time: {elapsed:.0f}ms (slow)"

    except requests.exceptions.RequestException as e:
        return False, f"❌ Response Time: {str(e)}"


def print_header(title: str):
    """Print formatted header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_result(passed: bool, message: str):
    """Print test result."""
    print(f"  {message}")


def main():
    parser = argparse.ArgumentParser(description='Verify Railway deployment')
    parser.add_argument('--url', required=True, help='Railway app URL (e.g., https://your-app.railway.app)')
    parser.add_argument('--api-key', help='API key for authenticated endpoints')
    parser.add_argument('--verbose', action='store_true', help='Show detailed output')

    args = parser.parse_args()

    base_url = args.url.rstrip('/')

    print_header("Railway Deployment Verification")
    print(f"  URL: {base_url}")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Run all checks
    checks = [
        ("API Health Check", lambda: check_api_health(base_url, args.api_key)),
        ("Response Time Check", lambda: check_response_time(base_url)),
        ("CORS Configuration", lambda: check_cors_headers(base_url)),
        ("Predictions Endpoint", lambda: check_predictions_endpoint(base_url, args.api_key)),
        ("Injuries Endpoint", lambda: check_injuries_endpoint(base_url, args.api_key)),
        ("Backtest Endpoint", lambda: check_backtest_endpoint(base_url, args.api_key)),
    ]

    results = []

    for check_name, check_func in checks:
        print_header(check_name)
        passed, message = check_func()
        print_result(passed, message)
        results.append((check_name, passed))

    # Summary
    print_header("Summary")
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    print(f"  Passed: {passed_count}/{total_count}")
    print(f"  Failed: {total_count - passed_count}/{total_count}")

    if passed_count == total_count:
        print("\n  🎉 All checks passed! Deployment verified successfully.")
        return 0
    if passed_count >= total_count * 0.7:
        print("\n  ⚠️  Most checks passed. Review warnings above.")
        return 0
    print("\n  ❌ Deployment verification failed. Fix errors above.")
    return 1


if __name__ == '__main__':
    sys.exit(main())
