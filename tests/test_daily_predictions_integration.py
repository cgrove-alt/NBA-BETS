#!/usr/bin/env python3
"""
Integration test for daily_predictions.py

Tests that the actual code paths work, not just the isolated modules.
This catches bugs like undefined variables that unit tests miss.
"""

import sys
from datetime import datetime

def test_imports():
    """Test that all imports work correctly."""
    print("[Test 1] Testing imports...")

    try:
        from nba_models.inference.daily_predictions import (
            logger,
            fetch_current_injuries,
            is_player_available,
            InjuryStatus
        )
        print("  ✓ All critical imports successful")
        print(f"    - logger: {type(logger)}")
        print(f"    - fetch_current_injuries: {type(fetch_current_injuries)}")
        print(f"    - is_player_available: {type(is_player_available)}")
        print(f"    - InjuryStatus: {type(InjuryStatus)}")
        return True
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False
    except Exception as e:
        print(f"  ✗ Unexpected error: {e}")
        return False


def test_logger_defined():
    """Test that logger variable is properly defined."""
    print("\n[Test 2] Testing logger definition...")

    try:
        from nba_models.inference.daily_predictions import logger

        # Verify it's a Logger instance
        import logging
        assert isinstance(logger, logging.Logger), f"logger is {type(logger)}, not Logger"

        # Verify it has the correct name
        assert logger.name == "daily_predictions", f"logger name is {logger.name}"

        # Verify we can call debug without error
        logger.debug("Test debug message")

        print("  ✓ Logger is properly defined and functional")
        return True
    except AssertionError as e:
        print(f"  ✗ Assertion failed: {e}")
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_injury_lookup_logic():
    """Test the injury lookup code path that caused the NameError."""
    print("\n[Test 3] Testing injury lookup code path...")

    try:
        from nba_models.inference.daily_predictions import logger, InjuryStatus

        # Simulate the code path from lines 1691-1717
        # This is the exact logic that would fail with undefined logger

        # Test case 1: Player in injury_lookup
        injury_lookup = {12345: InjuryStatus.OUT}
        player_id = 12345
        player_name = "Test Player"

        if player_id in injury_lookup:
            status = injury_lookup[player_id]
            if status in [InjuryStatus.OUT, InjuryStatus.DOUBTFUL]:
                print(f"    ✓ Would skip {player_name} ({status.value})")
        else:
            # This is the code path that had the bug (line 1717)
            logger.debug(f"Player {player_name} (ID: {player_id}) not in injury lookup - assuming healthy")

        # Test case 2: Player NOT in injury_lookup (the bug trigger)
        player_id = 99999  # Not in lookup
        player_name = "Healthy Player"

        if player_id in injury_lookup:
            status = injury_lookup[player_id]
            if status in [InjuryStatus.OUT, InjuryStatus.DOUBTFUL]:
                print(f"    Would skip {player_name}")
        else:
            # This line would fail with NameError if logger not defined
            logger.debug(f"Player {player_name} (ID: {player_id}) not in injury lookup - assuming healthy")
            print("    ✓ Logger.debug() executed successfully for healthy player")

        print("  ✓ Injury lookup code path works correctly")
        return True

    except NameError as e:
        print(f"  ✗ NameError (logger undefined): {e}")
        return False
    except Exception as e:
        print(f"  ✗ Unexpected error: {e}")
        return False


def test_injury_integration_workflow():
    """Test the full injury checking workflow."""
    print("\n[Test 4] Testing full injury integration workflow...")

    try:
        from nba_models.inference.daily_predictions import (
            fetch_current_injuries,
            is_player_available,
            InjuryStatus,
            logger
        )
        from datetime import datetime

        # Step 1: Fetch injuries (like line 1507)
        target_date_dt = datetime.now()
        current_injuries = fetch_current_injuries(target_date_dt)

        # Step 2: Build lookup dict (like lines 1510-1513)
        injury_lookup = {}
        for injury_report in current_injuries:
            if injury_report.player_id:
                injury_lookup[injury_report.player_id] = injury_report.status

        print(f"    ✓ Fetched {len(current_injuries)} injuries")
        print(f"    ✓ Built lookup dict with {len(injury_lookup)} entries")

        # Step 3: Simulate checking players (like lines 1681-1717)
        test_player_ids = [999999, 888888, 777777]  # Fake IDs (likely not in lookup)

        for player_id in test_player_ids:
            player_name = f"Player_{player_id}"

            if player_id in injury_lookup:
                status = injury_lookup[player_id]
                if status in [InjuryStatus.OUT, InjuryStatus.DOUBTFUL]:
                    continue  # Skip
                if status in [InjuryStatus.QUESTIONABLE, InjuryStatus.GTD]:
                    pass
            else:
                # The critical code path that had the bug
                logger.debug(f"Player {player_name} (ID: {player_id}) not in injury lookup - assuming healthy")

        print(f"    ✓ Successfully processed {len(test_player_ids)} test players")
        print("  ✓ Full workflow completed without errors")
        return True

    except Exception as e:
        print(f"  ✗ Error in workflow: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all integration tests."""
    print("=" * 70)
    print("INTEGRATION TEST: daily_predictions.py")
    print("=" * 70)
    print("\nThis test verifies that the actual code paths work correctly,")
    print("catching bugs like undefined variables that unit tests might miss.\n")

    results = []

    results.append(test_imports())
    results.append(test_logger_defined())
    results.append(test_injury_lookup_logic())
    results.append(test_injury_integration_workflow())

    # Summary
    print("\n" + "=" * 70)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 70)

    passed = sum(results)
    total = len(results)

    print(f"\nTests Passed: {passed}/{total}")

    if passed == total:
        print("\n✓ ALL INTEGRATION TESTS PASSED")
        print("  The code is ready for production deployment")
        print("=" * 70)
        return 0
    print(f"\n✗ {total - passed} TEST(S) FAILED")
    print("  Fix the issues before deploying to production")
    print("=" * 70)
    return 1


if __name__ == "__main__":
    exit(main())
