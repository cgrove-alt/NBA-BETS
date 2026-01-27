"""
Profile daily_predictions.py to identify performance bottlenecks.

Usage:
    python profile_daily_predictions.py --date 2026-01-05

Outputs:
    - profile.stats: Raw profiling data
    - performance_report.txt: Human-readable report
"""

import cProfile
import pstats
import io
from datetime import datetime

def profile_predictions():
    """Profile the prediction generation process."""

    # Create profiler
    profiler = cProfile.Profile()

    # Run predictions with profiling
    print("Starting profiling...")
    profiler.enable()

    # Import and run main (this will execute daily_predictions.py)
    from daily_predictions import main
    main()

    profiler.disable()

    # Save raw stats
    profiler.dump_stats('profile.stats')
    print("\nRaw profiling data saved to: profile.stats")

    # Generate human-readable report
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)

    # Sort by cumulative time and print top 30 functions
    ps.sort_stats('cumulative')

    with open('performance_report.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("PERFORMANCE PROFILING REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")

        f.write("TOP 30 FUNCTIONS BY CUMULATIVE TIME\n")
        f.write("-"*80 + "\n")
        ps.stream = f
        ps.print_stats(30)

        f.write("\n\n" + "="*80 + "\n")
        f.write("TOP 30 FUNCTIONS BY TIME PER CALL\n")
        f.write("-"*80 + "\n")
        ps.sort_stats('time')
        ps.print_stats(30)

        f.write("\n\n" + "="*80 + "\n")
        f.write("FUNCTION CALL COUNTS (Top 20)\n")
        f.write("-"*80 + "\n")
        ps.sort_stats('calls')
        ps.print_stats(20)

    print("Performance report saved to: performance_report.txt")

    # Print summary to console
    print("\n" + "="*80)
    print("PROFILING SUMMARY (Top 10 by cumulative time)")
    print("="*80)
    ps = pstats.Stats('profile.stats')
    ps.sort_stats('cumulative')
    ps.print_stats(10)

    print("\n" + "="*80)
    print("BOTTLENECK ANALYSIS")
    print("="*80)

    # Analyze for common bottlenecks
    bottlenecks = []

    # Parse stats to identify bottlenecks
    for func, stats in ps.stats.items():
        filename, line, funcname = func
        cc, nc, tt, ct, callers = stats

        # Flag functions that take >1s cumulative time
        if ct > 1.0:
            bottlenecks.append({
                'name': funcname,
                'file': filename.split('/')[-1],
                'cumtime': ct,
                'calls': nc,
                'percall': ct/nc if nc > 0 else 0
            })

    # Sort by cumulative time
    bottlenecks.sort(key=lambda x: x['cumtime'], reverse=True)

    print("\nFunctions consuming >1 second:\n")
    for i, b in enumerate(bottlenecks[:15], 1):
        print(f"{i:2d}. {b['name'][:40]:40s} | {b['file'][:25]:25s} | "
              f"{b['cumtime']:6.2f}s total | {b['calls']:5d} calls | "
              f"{b['percall']*1000:6.1f}ms/call")

    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    # Check specific patterns
    api_calls = sum(1 for b in bottlenecks if 'api' in b['name'].lower() or 'fetch' in b['name'].lower())
    db_calls = sum(1 for b in bottlenecks if 'sql' in b['name'].lower() or 'query' in b['name'].lower())

    print(f"\n1. API/Network calls in bottlenecks: {api_calls}")
    print("   → Consider: parallelization, caching, batch requests\n")

    print(f"2. Database calls in bottlenecks: {db_calls}")
    print("   → Consider: query optimization, connection pooling\n")

    # Check for repeated calls
    high_call_funcs = [b for b in bottlenecks if b['calls'] > 100]
    if high_call_funcs:
        print(f"3. Functions called >100 times: {len(high_call_funcs)}")
        print("   → Consider: memoization, caching")
        for f in high_call_funcs[:3]:
            print(f"      - {f['name']}: {f['calls']} calls")
        print()

    print("="*80)

if __name__ == "__main__":
    profile_predictions()
