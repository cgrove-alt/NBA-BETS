"""
Test script to validate Task 4.1 optimization improvements.

Compares:
- Before: Unoptimized serial execution
- After: Parallel + caching + lazy loading

Success Criteria: < 5 minutes for 15 games (30 teams + 300 players)
"""

import time
import sys
from datetime import datetime

def test_cache_performance():
    """Test cache hit/miss performance."""
    print("="*80)
    print("TEST 1: Cache Performance")
    print("="*80)

    from prediction_optimizer import cached, clear_cache

    # Clear any existing cache
    clear_cache()

    @cached(ttl_type='player_stats')
    def fetch_player_data(player_id):
        """Simulate slow API call."""
        time.sleep(0.1)
        return {'id': player_id, 'ppg': 25.0}

    # Test 100 player fetches with cache
    player_ids = list(range(50)) * 2  # 100 total (50 unique)

    start = time.time()
    for pid in player_ids:
        fetch_player_data(pid)
    elapsed = time.time() - start

    # Without cache would take ~10 seconds (100 * 0.1s)
    # With cache should take ~5 seconds (50 unique * 0.1s + 50 hits)
    expected_max = 6.0  # Allow some overhead
    expected_min = 4.5  # Should be close to 5s

    print(f"\nFetched {len(player_ids)} players in {elapsed:.2f}s")
    print(f"Expected: {expected_min:.1f}s - {expected_max:.1f}s")

    if expected_min < elapsed < expected_max:
        print("✓ Cache working efficiently")
        return True
    print("✗ Cache performance outside expected range")
    return False


def test_parallel_execution():
    """Test parallel vs serial execution speedup."""
    print("\n" + "="*80)
    print("TEST 2: Parallel Execution Speedup")
    print("="*80)

    from prediction_optimizer import get_executor

    def slow_task(x):
        """Simulate slow computation."""
        time.sleep(0.05)
        return x * 2

    items = list(range(20))

    # Serial execution
    print("\nSerial execution...")
    start = time.time()
    serial_results = [slow_task(x) for x in items]
    serial_time = time.time() - start
    print(f"  Time: {serial_time:.2f}s")

    # Parallel execution
    print("\nParallel execution (10 workers)...")
    executor = get_executor(max_workers=10)
    start = time.time()
    parallel_results = executor.map(slow_task, items, show_progress=False)
    parallel_time = time.time() - start
    print(f"  Time: {parallel_time:.2f}s")

    # Calculate speedup
    speedup = serial_time / parallel_time
    print(f"\nSpeedup: {speedup:.1f}x")

    # Results should match
    assert serial_results == parallel_results, "Results don't match!"

    # Should be at least 3x faster with 10 workers
    if speedup >= 3.0:
        print("✓ Parallel execution working efficiently")
        return True
    print(f"✗ Speedup ({speedup:.1f}x) below expected (>3x)")
    return False


def test_batch_processing():
    """Test batch API call simulation."""
    print("\n" + "="*80)
    print("TEST 3: Batch Processing")
    print("="*80)

    from prediction_optimizer import BatchProcessor

    # Simulate individual vs batch API calls
    player_ids = list(range(100))

    # Individual calls (slow)
    print("\nIndividual API calls...")
    start = time.time()
    results_individual = {}
    for pid in player_ids:
        time.sleep(0.005)  # 5ms per call
        results_individual[pid] = {'id': pid, 'ppg': 20.0}
    individual_time = time.time() - start
    print(f"  Time: {individual_time:.2f}s ({len(player_ids)} calls)")

    # Batch calls (fast)
    print("\nBatch API calls (25 per batch)...")
    def batch_api_call(ids):
        time.sleep(0.02)  # 20ms per batch
        return [{'id': pid, 'ppg': 20.0} for pid in ids]

    batch_processor = BatchProcessor()
    start = time.time()
    batch_processor.batch_fetch_player_stats(
        player_ids,
        batch_api_call,
        batch_size=25
    )
    batch_time = time.time() - start
    num_batches = (len(player_ids) + 24) // 25
    print(f"  Time: {batch_time:.2f}s ({num_batches} batches)")

    speedup = individual_time / batch_time
    print(f"\nSpeedup: {speedup:.1f}x")

    if speedup >= 4.0:
        print("✓ Batch processing working efficiently")
        return True
    print(f"✗ Speedup ({speedup:.1f}x) below expected (>4x)")
    return False


def test_lazy_loading():
    """Test lazy loading of expensive features."""
    print("\n" + "="*80)
    print("TEST 4: Lazy Loading")
    print("="*80)

    from prediction_optimizer import LazyFeatureLoader

    loader = LazyFeatureLoader()

    def expensive_load():
        """Simulate expensive tracking data fetch."""
        time.sleep(0.5)
        return {'zones': [1, 2, 3], 'shots': 100}

    # First access - slow
    print("\nFirst access (loads data)...")
    start = time.time()
    data1 = loader.get_or_load('team_123_tracking', expensive_load)
    time1 = time.time() - start
    print(f"  Time: {time1:.3f}s")

    # Second access - fast (cached)
    print("\nSecond access (cached)...")
    start = time.time()
    data2 = loader.get_or_load('team_123_tracking', expensive_load)
    time2 = time.time() - start
    print(f"  Time: {time2:.3f}s")

    assert data1 == data2, "Data doesn't match!"
    speedup = time1 / time2

    print(f"\nSpeedup: {speedup:.0f}x")

    if time2 < 0.001:  # Should be instantaneous
        print("✓ Lazy loading working efficiently")
        return True
    print(f"✗ Cached access too slow ({time2:.3f}s)")
    return False


def test_full_integration():
    """Test full prediction pipeline with optimizations."""
    print("\n" + "="*80)
    print("TEST 5: Full Integration (Simulated)")
    print("="*80)

    from prediction_optimizer import cached, get_executor, BatchProcessor

    # Simulate fetching data for a typical game day
    num_games = 10
    players_per_game = 20
    props_per_player = 3

    total_props = num_games * players_per_game * props_per_player
    print(f"\nSimulating {num_games} games:")
    print(f"  - {players_per_game} players/game")
    print(f"  - {props_per_player} props/player")
    print(f"  - {total_props} total predictions")

    @cached(ttl_type='player_features')
    def fetch_features(player_id):
        time.sleep(0.01)  # 10ms to fetch/compute features
        return {'player_id': player_id, 'ppg': 20.0}

    def predict_prop(task):
        fetch_features(task['player_id'])
        time.sleep(0.002)  # 2ms for prediction
        return {'player': task['player_id'], 'prediction': 25.0}

    # Build all tasks
    tasks = []
    for game in range(num_games):
        for player in range(players_per_game):
            player_id = game * 100 + player
            for prop_type in ['points', 'rebounds', 'assists']:
                tasks.append({'player_id': player_id, 'prop': prop_type})

    # Execute in parallel
    print("\nExecuting predictions...")
    executor = get_executor(max_workers=20)
    start = time.time()
    results = executor.map(predict_prop, tasks, desc="Predictions", show_progress=True)
    elapsed = time.time() - start

    print(f"\nCompleted {len(results)} predictions in {elapsed:.2f}s")
    print(f"Average: {elapsed/len(results)*1000:.1f}ms per prediction")

    # Success criteria: < 5 minutes for 600 predictions
    # That's 0.5s per prediction max
    max_time_per_pred = 0.5
    avg_time = elapsed / len(results)

    if avg_time < max_time_per_pred:
        print(f"✓ Performance target met ({avg_time*1000:.1f}ms < {max_time_per_pred*1000:.0f}ms)")
        return True
    print(f"✗ Performance target missed ({avg_time*1000:.1f}ms > {max_time_per_pred*1000:.0f}ms)")
    return False


def run_all_tests():
    """Run all optimization tests."""
    print("\n" + "="*80)
    print("TASK 4.1: OPTIMIZATION VALIDATION SUITE")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    tests = [
        ("Cache Performance", test_cache_performance),
        ("Parallel Execution", test_parallel_execution),
        ("Batch Processing", test_batch_processing),
        ("Lazy Loading", test_lazy_loading),
        ("Full Integration", test_full_integration),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n✗ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    passed = sum(1 for _, p in results if p)
    total = len(results)

    for name, passed_flag in results:
        status = "✓ PASS" if passed_flag else "✗ FAIL"
        print(f"{status:10s} | {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All optimizations working correctly!")
        print("✓ Ready for production use")
        return 0
    print(f"\n⚠️  {total - passed} test(s) failed")
    print("✗ Review failures before deploying")
    return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
