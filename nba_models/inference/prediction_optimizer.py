"""
Performance Optimization Module for Daily Predictions

Provides:
1. Parallel API calls using ThreadPoolExecutor
2. Advanced caching with TTL
3. Batch processing
4. Lazy loading for expensive features

Goal: Reduce prediction generation time from >10 minutes to <5 minutes
"""

import time
import hashlib
import json
from pathlib import Path
from typing import Any
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
import threading
import contextlib

# ============================================================================
# ADVANCED CACHING
# ============================================================================

CACHE_DIR = Path(__file__).parent / ".prediction_cache"
CACHE_DIR.mkdir(exist_ok=True)

# Cache TTLs (time-to-live in seconds)
CACHE_TTL = {
    'team_stats': 21600,       # 6 hours
    'player_stats': 14400,     # 4 hours
    'injury_data': 900,        # 15 minutes
    'game_features': 3600,     # 1 hour
    'player_features': 7200,   # 2 hours
    'odds_data': 300,          # 5 minutes
}


def get_cache_key(func_name: str, *args, **kwargs) -> str:
    """Generate a unique cache key from function name and arguments."""
    # Convert args/kwargs to a stable string representation
    args_str = json.dumps({
        'args': [str(a) for a in args],
        'kwargs': {k: str(v) for k, v in sorted(kwargs.items())}
    }, sort_keys=True)

    key_str = f"{func_name}:{args_str}"
    return hashlib.md5(key_str.encode()).hexdigest()


def get_cache_path(cache_key: str) -> Path:
    """Get file path for a cache entry."""
    return CACHE_DIR / f"{cache_key}.json"


def read_from_cache(cache_key: str, ttl_seconds: int = 3600) -> Any | None:
    """Read data from cache if valid."""
    cache_path = get_cache_path(cache_key)

    if not cache_path.exists():
        return None

    try:
        with open(cache_path) as f:
            cached = json.load(f)

        # Check if cache is still valid
        cached_at = cached.get('timestamp', 0)
        age = time.time() - cached_at

        if age > ttl_seconds:
            # Cache expired
            cache_path.unlink(missing_ok=True)
            return None

        return cached.get('data')

    except (OSError, json.JSONDecodeError, KeyError):
        # Invalid cache file
        cache_path.unlink(missing_ok=True)
        return None


def write_to_cache(cache_key: str, data: Any) -> None:
    """Write data to cache."""
    if data is None:
        return

    cache_path = get_cache_path(cache_key)

    try:
        with open(cache_path, 'w') as f:
            json.dump({
                'timestamp': time.time(),
                'data': data
            }, f)
    except (OSError, TypeError):
        pass  # Silently fail on cache write errors


def cached(ttl_type: str = 'player_stats'):
    """
    Decorator to cache function results.

    Usage:
        @cached(ttl_type='team_stats')
        def fetch_team_statistics(team_id, date):
            # expensive operation
            return data
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = get_cache_key(func.__name__, *args, **kwargs)
            ttl = CACHE_TTL.get(ttl_type, 3600)

            # Try to read from cache
            cached_data = read_from_cache(cache_key, ttl)
            if cached_data is not None:
                return cached_data

            # Cache miss - execute function
            result = func(*args, **kwargs)

            # Write to cache
            write_to_cache(cache_key, result)

            return result

        return wrapper
    return decorator


def clear_cache(older_than_hours: float = 0) -> int:
    """
    Clear the prediction cache.

    Args:
        older_than_hours: Only clear entries older than this. If 0, clear all.

    Returns:
        Number of entries removed.
    """
    if not CACHE_DIR.exists():
        return 0

    removed = 0
    cutoff = time.time() - (older_than_hours * 3600) if older_than_hours > 0 else float('inf')

    for cache_file in CACHE_DIR.glob("*.json"):
        try:
            if older_than_hours > 0:
                with open(cache_file) as f:
                    cached = json.load(f)
                    if cached.get('timestamp', 0) > cutoff:
                        continue
            cache_file.unlink()
            removed += 1
        except (OSError, json.JSONDecodeError):
            try:
                cache_file.unlink()
                removed += 1
            except OSError:
                pass

    return removed


# ============================================================================
# PARALLEL EXECUTION
# ============================================================================

class ParallelExecutor:
    """Execute multiple tasks in parallel using ThreadPoolExecutor."""

    def __init__(self, max_workers: int = 10):
        """
        Initialize parallel executor.

        Args:
            max_workers: Maximum number of concurrent threads (default 10)
        """
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def map(self, func: Callable, items: list[Any],
            desc: str = "Processing", show_progress: bool = True) -> list[Any]:
        """
        Execute function on multiple items in parallel.

        Args:
            func: Function to execute
            items: List of items to process
            desc: Description for progress display
            show_progress: Whether to show progress

        Returns:
            List of results in same order as items
        """
        if not items:
            return []

        results = [None] * len(items)

        # Submit all tasks
        future_to_idx = {
            self.executor.submit(func, item): idx
            for idx, item in enumerate(items)
        }

        # Collect results as they complete
        completed = 0
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                print(f"  Error processing item {idx}: {e}")
                results[idx] = None

            completed += 1
            if show_progress and completed % 5 == 0:
                print(f"  {desc}: {completed}/{len(items)}", end='\r', flush=True)

        if show_progress:
            print(f"  {desc}: {len(items)}/{len(items)} ✓")

        return results

    def map_dict(self, func: Callable, items: dict[Any, Any],
                  desc: str = "Processing", show_progress: bool = True) -> dict[Any, Any]:
        """
        Execute function on dict values in parallel.

        Args:
            func: Function to execute
            items: Dict of items to process
            desc: Description for progress display
            show_progress: Whether to show progress

        Returns:
            Dict with same keys, processed values
        """
        if not items:
            return {}

        keys = list(items.keys())
        values = list(items.values())

        # Process in parallel
        processed_values = self.map(func, values, desc, show_progress)

        # Reconstruct dict
        return dict(zip(keys, processed_values, strict=False))

    def shutdown(self):
        """Shutdown the executor."""
        self.executor.shutdown(wait=True)


# Global executor instance
_executor = None
_executor_lock = threading.Lock()


def get_executor(max_workers: int = 10) -> ParallelExecutor:
    """Get or create global parallel executor."""
    global _executor
    if _executor is None:
        with _executor_lock:
            if _executor is None:
                _executor = ParallelExecutor(max_workers=max_workers)
    return _executor


# ============================================================================
# BATCH PROCESSING
# ============================================================================

class BatchProcessor:
    """Process items in batches to optimize API calls."""

    @staticmethod
    def batch_fetch_player_stats(player_ids: list[int],
                                  api_func: Callable,
                                  batch_size: int = 25) -> dict[int, dict]:
        """
        Fetch player stats in batches.

        Args:
            player_ids: List of player IDs
            api_func: API function that accepts list of IDs
            batch_size: Number of players per batch

        Returns:
            Dict mapping player_id to stats
        """
        results = {}

        # Split into batches
        for i in range(0, len(player_ids), batch_size):
            batch = player_ids[i:i + batch_size]

            try:
                batch_results = api_func(batch)

                # Index by player_id
                for player_data in batch_results:
                    pid = player_data.get('player_id') or player_data.get('id')
                    if pid:
                        results[pid] = player_data

            except Exception as e:
                print(f"  Warning: Batch fetch failed for IDs {batch[:3]}...: {e}")
                continue

        return results

    @staticmethod
    def batch_fetch_team_stats(team_ids: list[int],
                                api_func: Callable) -> dict[int, dict]:
        """
        Fetch team stats in a single batch.

        Args:
            team_ids: List of team IDs
            api_func: API function that accepts list of IDs

        Returns:
            Dict mapping team_id to stats
        """
        try:
            results_list = api_func(team_ids)

            # Index by team_id
            results = {}
            for team_data in results_list:
                tid = team_data.get('team_id') or team_data.get('id')
                if tid:
                    results[tid] = team_data

            return results

        except Exception as e:
            print(f"  Warning: Batch team fetch failed: {e}")
            return {}


# ============================================================================
# LAZY LOADING
# ============================================================================

class LazyFeatureLoader:
    """
    Lazy load expensive features only when needed.

    Example:
        loader = LazyFeatureLoader()

        # Only fetch tracking data if confidence is borderline
        if 45 < confidence < 75:
            tracking_data = loader.get_tracking_data(team_id)
    """

    def __init__(self):
        self._cache = {}
        self._lock = threading.Lock()

    def get_or_load(self, key: str, load_func: Callable) -> Any:
        """
        Get cached value or load it.

        Args:
            key: Cache key
            load_func: Function to load data if not cached

        Returns:
            Loaded data
        """
        if key in self._cache:
            return self._cache[key]

        with self._lock:
            # Double-check pattern
            if key in self._cache:
                return self._cache[key]

            # Load data
            data = load_func()
            self._cache[key] = data
            return data

    def clear(self):
        """Clear all cached data."""
        with self._lock:
            self._cache.clear()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def timed(func: Callable) -> Callable:
    """
    Decorator to measure function execution time.

    Usage:
        @timed
        def slow_function():
            time.sleep(1)
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start

        if elapsed > 1.0:
            print(f"  [{func.__name__}] took {elapsed:.2f}s")

        return result

    return wrapper


def benchmark(func: Callable, *args, iterations: int = 1, **kwargs) -> float:
    """
    Benchmark a function's execution time.

    Args:
        func: Function to benchmark
        *args, **kwargs: Arguments to pass to function
        iterations: Number of times to run (default 1)

    Returns:
        Average execution time in seconds
    """
    total_time = 0

    for _ in range(iterations):
        start = time.time()
        func(*args, **kwargs)
        elapsed = time.time() - start
        total_time += elapsed

    avg_time = total_time / iterations

    print(f"Benchmark: {func.__name__}")
    print(f"  Iterations: {iterations}")
    print(f"  Average time: {avg_time:.3f}s")
    print(f"  Total time: {total_time:.3f}s")

    return avg_time


# ============================================================================
# CACHE WARMUP
# ============================================================================

def warmup_cache(api, game_date: str, team_ids: list[int], player_ids: list[int]):
    """
    Pre-populate cache with data for today's games.

    This runs once at startup to fetch all needed data in parallel,
    dramatically reducing individual fetch times later.

    Args:
        api: Balldontlie API instance
        game_date: Date in YYYY-MM-DD format
        team_ids: List of team IDs playing today
        player_ids: List of player IDs to analyze
    """
    print("\n  Warming up cache...")
    start = time.time()

    executor = get_executor(max_workers=5)

    # Fetch team stats in parallel
    def fetch_team(tid):
        try:
            # This will be cached by balldontlie_api.py's built-in cache
            return api.get_season_averages(team_ids=[tid])
        except Exception:
            return None

    if team_ids:
        print(f"    Fetching {len(team_ids)} team stats...", end='', flush=True)
        executor.map(fetch_team, team_ids, show_progress=False)
        print(" ✓")

    # Fetch player stats in batches (more efficient)
    if player_ids:
        print(f"    Fetching {len(player_ids)} player stats...", end='', flush=True)
        BatchProcessor()

        # Use batch API call (25 players at a time)
        for i in range(0, len(player_ids), 25):
            batch = player_ids[i:i+25]
            with contextlib.suppress(Exception):
                api.get_season_averages(player_ids=batch)
        print(" ✓")

    elapsed = time.time() - start
    print(f"  Cache warmup completed in {elapsed:.1f}s")


if __name__ == "__main__":
    # Test cache functionality
    print("Testing cache...")

    @cached(ttl_type='player_stats')
    def slow_fetch(player_id):
        time.sleep(0.1)  # Simulate API call
        return {'id': player_id, 'ppg': 25.5}

    # First call - slow
    start = time.time()
    result1 = slow_fetch(123)
    time1 = time.time() - start
    print(f"First call: {time1:.3f}s")

    # Second call - fast (cached)
    start = time.time()
    result2 = slow_fetch(123)
    time2 = time.time() - start
    print(f"Second call (cached): {time2:.3f}s")

    assert result1 == result2
    assert time2 < time1 / 10  # At least 10x faster

    print("\n✓ Cache working correctly")

    # Test parallel execution
    print("\nTesting parallel execution...")
    executor = get_executor(max_workers=5)

    def slow_task(x):
        time.sleep(0.1)
        return x * 2

    items = list(range(10))

    # Sequential
    start = time.time()
    seq_results = [slow_task(x) for x in items]
    seq_time = time.time() - start

    # Parallel
    start = time.time()
    par_results = executor.map(slow_task, items, show_progress=False)
    par_time = time.time() - start

    print(f"Sequential: {seq_time:.3f}s")
    print(f"Parallel: {par_time:.3f}s")
    print(f"Speedup: {seq_time/par_time:.1f}x")

    assert seq_results == par_results
    assert par_time < seq_time / 2  # At least 2x faster

    print("\n✓ Parallel execution working correctly")

    # Clean up
    removed = clear_cache()
    print(f"\nCleared {removed} cache entries")
