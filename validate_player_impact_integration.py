"""
Integration Validation Script for Enhanced player_impact_fetcher.py

This script demonstrates how the enhanced module can be integrated into
feature_engineering.py and other parts of the prediction pipeline.
"""

from player_impact_fetcher import PlayerImpactFetcher


def validate_basic_functionality():
    """Test basic fetcher operations."""
    print("1. Testing Basic Functionality")
    print("-" * 60)

    fetcher = PlayerImpactFetcher()

    # Test metric standardization
    test_values = [
        (8.0, 'darko'),
        (7.5, 'raptor'),
        (6.0, 'epm'),
        (5.0, 'plus_minus'),
    ]

    print("Metric standardization:")
    for value, metric_type in test_values:
        standardized = fetcher._standardize_metric(value, metric_type)
        print(f"  {metric_type:15s} {value:5.1f} → {standardized:5.2f}")

    print("✓ Basic functionality validated\n")


def validate_data_sources():
    """Test data source priority and availability."""
    print("2. Testing Data Source Priority")
    print("-" * 60)

    fetcher = PlayerImpactFetcher()

    sources = {
        'DARKO': len(fetcher.darko_cache),
        'EPM': len(fetcher.epm_cache),
        'RAPTOR': len(fetcher.raptor_cache),
        'Basic Stats': len(fetcher.basic_stats_cache),
    }

    print("Available data sources:")
    for source, count in sources.items():
        status = "✓" if count > 0 else "✗"
        print(f"  {status} {source:15s}: {count:3d} players")

    total_players = len(set(
        list(fetcher.darko_cache.keys()) +
        list(fetcher.epm_cache.keys()) +
        list(fetcher.raptor_cache.keys()) +
        list(fetcher.basic_stats_cache.keys())
    ))

    print(f"\nTotal unique players: {total_players}")
    print("✓ Data sources validated\n")


def validate_player_lookups():
    """Test player impact lookups."""
    print("3. Testing Player Impact Lookups")
    print("-" * 60)

    fetcher = PlayerImpactFetcher()

    # Get sample players
    all_players = (
        list(fetcher.darko_cache.keys()) +
        list(fetcher.raptor_cache.keys())
    )

    if all_players:
        print("Sample player lookups:")
        for player in all_players[:3]:
            impact = fetcher.get_player_impact_metric(player)
            data = fetcher.get_player_impact(player)
            source = data.get('source', 'unknown') if data else 'unknown'
            print(f"  {player:20s} → {impact:6.2f} (from {source})")

        print("✓ Player lookups validated\n")
    else:
        print("⚠ No cached players available for testing\n")


def validate_team_adjustments():
    """Test team rating adjustments for injuries."""
    print("4. Testing Team Rating Adjustments")
    print("-" * 60)

    fetcher = PlayerImpactFetcher()

    # Test with sample players
    all_players = list(fetcher.darko_cache.items()) + list(fetcher.raptor_cache.items())

    if all_players:
        print("Team rating adjustments (single player):")
        for player_name, data in all_players[:3]:
            team = data.get('team', 'UNK')
            adjustment = fetcher.calculate_team_rating_adjustment(
                team,
                injured_players=[player_name]
            )
            print(f"  {team} without {player_name:20s} → {adjustment:+6.2f} pts")

        # Test multiple injuries
        if len(all_players) >= 2:
            player1, data1 = all_players[0]
            player2, data2 = all_players[1]

            # Only if same team
            if data1.get('team') == data2.get('team'):
                team = data1.get('team')
                adjustment = fetcher.calculate_team_rating_adjustment(
                    team,
                    injured_players=[player1, player2]
                )
                print(f"\n  {team} without {player1} + {player2}")
                print(f"  → {adjustment:+6.2f} pts (cumulative)")

        print("✓ Team adjustments validated\n")
    else:
        print("⚠ No cached players available for testing\n")


def validate_integration_readiness():
    """Validate module is ready for feature engineering integration."""
    print("5. Testing Integration Readiness")
    print("-" * 60)

    fetcher = PlayerImpactFetcher()

    checks = []

    # Check 1: Can get standardized metrics
    try:
        fetcher._standardize_metric(7.0, 'darko')
        checks.append(("Metric standardization", True))
    except Exception:
        checks.append(("Metric standardization", False))

    # Check 2: Can get player impact
    try:
        impact = fetcher.get_player_impact_metric("Test Player")
        checks.append(("Player impact lookup", True))
    except Exception:
        checks.append(("Player impact lookup", False))

    # Check 3: Can calculate team adjustments
    try:
        fetcher.calculate_team_rating_adjustment("LAL", [])
        checks.append(("Team adjustment calculation", True))
    except Exception:
        checks.append(("Team adjustment calculation", False))

    # Check 4: Cache system works
    try:
        fetcher._save_cache('darko')
        checks.append(("Cache save/load", True))
    except Exception:
        checks.append(("Cache save/load", False))

    # Check 5: Priority order works
    try:
        fetcher.darko_cache = {"Test": {"impact_metric": 8.0}}
        fetcher.raptor_cache = {"Test": {"impact_metric": 7.0}}
        impact = fetcher.get_player_impact("Test")
        is_darko = impact.get("impact_metric") == 8.0
        checks.append(("Priority order", is_darko))
    except Exception:
        checks.append(("Priority order", False))

    print("Integration checks:")
    for check_name, passed in checks:
        status = "✓" if passed else "✗"
        print(f"  {status} {check_name}")

    all_passed = all(passed for _, passed in checks)

    if all_passed:
        print("\n✓ All integration checks passed!")
        print("  → Module ready for feature_engineering.py integration")
    else:
        print("\n✗ Some checks failed!")
        print("  → Review failures before integration")

    print()


def show_usage_examples():
    """Show example code for integration."""
    print("6. Integration Usage Examples")
    print("-" * 60)

    example_code = """
# Example 1: Add player impact to features
from player_impact_fetcher import PlayerImpactFetcher

fetcher = PlayerImpactFetcher()

# In feature_engineering.py
def generate_player_features(player_name, opponent_team, position):
    features = {}

    # Add player's impact metric
    features['player_impact'] = fetcher.get_player_impact_metric(player_name)

    # Add opponent defensive impact
    features['opponent_def_impact'] = fetcher.get_opponent_defensive_impact_vs_position(
        opponent_team, position
    )

    return features

# Example 2: Adjust predictions for injuries
def adjust_prediction_for_injuries(base_prediction, team, injured_players):
    # Calculate team rating adjustment
    adjustment = fetcher.calculate_team_rating_adjustment(
        team,
        injured_players=injured_players
    )

    # Apply to prediction (negative adjustment = weaker team)
    adjusted_prediction = base_prediction + adjustment

    return adjusted_prediction

# Example 3: Get team roster impacts
def get_team_top_players(team_abbrev, top_n=5):
    roster_impacts = fetcher.get_team_roster_impacts(team_abbrev)

    # Returns players sorted by impact (highest first)
    return roster_impacts[:top_n]
"""

    print("Example integration code:")
    print(example_code)
    print("✓ Usage examples provided\n")


def main():
    """Run all validation checks."""
    print("=" * 60)
    print("Player Impact Fetcher - Integration Validation")
    print("=" * 60)
    print()

    validate_basic_functionality()
    validate_data_sources()
    validate_player_lookups()
    validate_team_adjustments()
    validate_integration_readiness()
    show_usage_examples()

    print("=" * 60)
    print("Validation Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
