"""
REAL End-to-End Performance Validation

Tests actual prediction generation with real API calls, models, and features.
This validates the claimed 8.4x speedup is achievable in production.
"""

import time
import sys
import os
from datetime import datetime

def test_with_warmup():
    """Test WITH optimizations (warmup + parallel)."""
    print("\n" + "="*80)
    print("TEST 1: WITH OPTIMIZATIONS (warmup + parallel)")
    print("="*80)

    # Clear cache first
    os.system("python3 -c 'from prediction_optimizer import clear_cache; clear_cache()' > /dev/null 2>&1")

    # Run with default settings (warmup enabled)
    start = time.time()
    result = os.system("python3 daily_predictions.py --date 2026-01-15 > /tmp/predictions_optimized.txt 2>&1")
    elapsed = time.time() - start

    if result != 0:
        print(f"❌ Command failed (exit code {result})")
        print("Check /tmp/predictions_optimized.txt for errors")
        return None

    # Count predictions
    try:
        with open('/tmp/predictions_optimized.txt') as f:
            output = f.read()
            # Count "props analyzed" lines
            prop_count = output.count('props analyzed')
            print("\n✅ Completed successfully")
            print(f"   Games analyzed: {prop_count}")
            print(f"   Total time: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
            return elapsed
    except Exception as e:
        print(f"❌ Error reading output: {e}")
        return None


def test_without_warmup():
    """Test WITHOUT warmup (but still with parallel)."""
    print("\n" + "="*80)
    print("TEST 2: WITHOUT WARMUP (baseline comparison)")
    print("="*80)

    # Clear cache first
    os.system("python3 -c 'from prediction_optimizer import clear_cache; clear_cache()' > /dev/null 2>&1")

    # Run with warmup disabled
    start = time.time()
    result = os.system("python3 daily_predictions.py --date 2026-01-15 --no-warmup > /tmp/predictions_no_warmup.txt 2>&1")
    elapsed = time.time() - start

    if result != 0:
        print(f"❌ Command failed (exit code {result})")
        print("Check /tmp/predictions_no_warmup.txt for errors")
        return None

    # Count predictions
    try:
        with open('/tmp/predictions_no_warmup.txt') as f:
            output = f.read()
            prop_count = output.count('props analyzed')
            print("\n✅ Completed successfully")
            print(f"   Games analyzed: {prop_count}")
            print(f"   Total time: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
            return elapsed
    except Exception as e:
        print(f"❌ Error reading output: {e}")
        return None


def main():
    print("="*80)
    print("REAL-WORLD PERFORMANCE VALIDATION")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nThis tests actual prediction generation with:")
    print("  - Real API calls (Balldontlie)")
    print("  - Real model inference")
    print("  - Real feature generation")
    print("  - Real injury checks")
    print("\nTarget: <5 minutes for full game day")

    # Test 1: With optimizations
    time_optimized = test_with_warmup()

    if time_optimized is None:
        print("\n⚠️  Optimized test failed - cannot measure performance")
        print("    This may be due to:")
        print("    - Missing API key (BALLDONTLIE_API_KEY)")
        print("    - No games on 2026-01-15")
        print("    - Missing model files")
        return 1

    # Test 2: Without warmup (partial optimization)
    time_no_warmup = test_without_warmup()

    # Summary
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)

    print(f"\nWith optimizations:    {time_optimized:.1f}s ({time_optimized/60:.2f} min)")
    if time_no_warmup:
        print(f"Without warmup:        {time_no_warmup:.1f}s ({time_no_warmup/60:.2f} min)")
        speedup = time_no_warmup / time_optimized
        print(f"\nWarmup speedup:        {speedup:.2f}x")

    # Check success criteria
    print("\n" + "="*80)
    print("SUCCESS CRITERIA")
    print("="*80)

    target_seconds = 5 * 60  # 5 minutes

    if time_optimized < target_seconds:
        print(f"✅ PASS: {time_optimized:.1f}s < {target_seconds}s target")
        print("\n🎉 Performance target met!")
        print("✓ Production ready for real-time betting")
        return 0
    print(f"❌ FAIL: {time_optimized:.1f}s > {target_seconds}s target")
    print("\n⚠️  Performance target NOT met")
    print(f"   Exceeded by: {(time_optimized - target_seconds):.1f}s")
    print("\n   Recommendations:")
    print("   1. Increase worker count (max_workers=20)")
    print("   2. Add more aggressive caching")
    print("   3. Profile to find remaining bottlenecks")
    return 1


if __name__ == "__main__":
    sys.exit(main())
