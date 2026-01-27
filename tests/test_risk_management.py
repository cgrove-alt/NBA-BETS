"""
Unit Tests for Risk Management Module

Tests Kelly Criterion calculations, stop-loss rules, daily exposure caps,
and correlation adjustments.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from risk_management import (
    calculate_kelly_bet_size,
    get_kelly_multiplier_for_tier,
    calculate_recommended_stake,
    DynamicKellyCalculator,
    BankrollManager,
    DrawdownProtection,
    RiskLevel,
    HaltReason,
    CLVTracker,
    calculate_risk_of_ruin,
    calculate_risk_of_ruin_monte_carlo
)


class TestKellyFormula:
    """Test Kelly Criterion formula with known inputs."""

    def test_kelly_even_money_55_percent(self):
        """Test Kelly for 55% win prob at even money (2.0 odds)."""
        # Kelly = (b*p - q) / b = (1*0.55 - 0.45) / 1 = 0.10
        # Expected: 10% of bankroll (before fractional adjustment)
        win_prob = 0.55
        decimal_odds = 2.0
        bankroll = 10000.0

        # Full Kelly (fractional=1.0, no cap)
        bet_size = calculate_kelly_bet_size(
            win_prob, decimal_odds, bankroll, fractional=1.0, max_bet_pct=1.0
        )

        # Should be ~10% of bankroll = $1000
        expected = 1000.0
        assert abs(bet_size - expected) < 10, f"Expected ~{expected}, got {bet_size}"

    def test_kelly_minus_110_odds_55_percent(self):
        """Test Kelly for 55% win prob at -110 odds (1.91 decimal)."""
        # Kelly = (0.91*0.55 - 0.45) / 0.91 ≈ 0.055
        win_prob = 0.55
        decimal_odds = 1.91
        bankroll = 10000.0

        # Full Kelly (no cap)
        bet_size = calculate_kelly_bet_size(
            win_prob, decimal_odds, bankroll, fractional=1.0, max_bet_pct=1.0
        )

        # Should be ~5.5% of bankroll = $550
        expected = 550.0
        assert abs(bet_size - expected) < 30, f"Expected ~{expected}, got {bet_size}"

    def test_kelly_no_edge_returns_zero(self):
        """Test that Kelly returns 0 when there's no edge."""
        # 50% win prob at even money = no edge
        win_prob = 0.50
        decimal_odds = 2.0
        bankroll = 10000.0

        bet_size = calculate_kelly_bet_size(win_prob, decimal_odds, bankroll)

        assert bet_size == 0.0, "Should return 0 when no edge"

    def test_kelly_negative_edge_returns_zero(self):
        """Test that Kelly returns 0 when edge is negative."""
        # 45% win prob at even money = negative edge
        win_prob = 0.45
        decimal_odds = 2.0
        bankroll = 10000.0

        bet_size = calculate_kelly_bet_size(win_prob, decimal_odds, bankroll)

        assert bet_size == 0.0, "Should return 0 when negative edge"

    def test_fractional_kelly_quarter(self):
        """Test that fractional Kelly reduces bet size correctly."""
        # 55% win prob at even money, quarter Kelly
        win_prob = 0.55
        decimal_odds = 2.0
        bankroll = 10000.0

        # Full Kelly (no cap)
        full_kelly = calculate_kelly_bet_size(
            win_prob, decimal_odds, bankroll, fractional=1.0, max_bet_pct=1.0
        )

        # Quarter Kelly
        quarter_kelly = calculate_kelly_bet_size(
            win_prob, decimal_odds, bankroll, fractional=0.25, max_bet_pct=1.0
        )

        # Quarter Kelly should be ~25% of full Kelly
        assert abs(quarter_kelly - full_kelly * 0.25) < 5

    def test_kelly_invalid_inputs(self):
        """Test Kelly with invalid inputs."""
        bankroll = 10000.0

        # Invalid win probability
        assert calculate_kelly_bet_size(0.0, 2.0, bankroll) == 0.0
        assert calculate_kelly_bet_size(1.0, 2.0, bankroll) == 0.0
        assert calculate_kelly_bet_size(-0.1, 2.0, bankroll) == 0.0
        assert calculate_kelly_bet_size(1.1, 2.0, bankroll) == 0.0

        # Invalid odds
        assert calculate_kelly_bet_size(0.55, 1.0, bankroll) == 0.0
        assert calculate_kelly_bet_size(0.55, 0.5, bankroll) == 0.0

        # Invalid bankroll
        assert calculate_kelly_bet_size(0.55, 2.0, 0.0) == 0.0
        assert calculate_kelly_bet_size(0.55, 2.0, -1000.0) == 0.0


class TestEdgeTierMultipliers:
    """Test edge quality tier Kelly multipliers."""

    def test_tier_multipliers(self):
        """Test that tier multipliers match specification."""
        assert get_kelly_multiplier_for_tier('elite') == 1.0
        assert get_kelly_multiplier_for_tier('strong') == 0.50
        assert get_kelly_multiplier_for_tier('moderate') == 0.25
        assert get_kelly_multiplier_for_tier('weak') == 0.0
        assert get_kelly_multiplier_for_tier('avoid') == 0.0

    def test_tier_case_insensitive(self):
        """Test that tier lookup is case-insensitive."""
        assert get_kelly_multiplier_for_tier('ELITE') == 1.0
        assert get_kelly_multiplier_for_tier('Elite') == 1.0
        assert get_kelly_multiplier_for_tier('eLiTe') == 1.0

    def test_unknown_tier_returns_zero(self):
        """Test that unknown tier returns 0."""
        assert get_kelly_multiplier_for_tier('unknown') == 0.0
        assert get_kelly_multiplier_for_tier('') == 0.0

    def test_elite_tier_bet_sizing(self):
        """Test bet sizing for elite tier."""
        win_prob = 0.55
        decimal_odds = 1.91
        bankroll = 10000.0

        bet_size = calculate_kelly_bet_size(
            win_prob, decimal_odds, bankroll,
            fractional=0.25, edge_tier='elite'
        )

        # Elite should use full fractional Kelly
        assert bet_size > 0, "Elite tier should recommend a bet"

    def test_strong_tier_bet_sizing(self):
        """Test bet sizing for strong tier."""
        win_prob = 0.55
        decimal_odds = 1.91
        bankroll = 10000.0

        elite_bet = calculate_kelly_bet_size(
            win_prob, decimal_odds, bankroll,
            fractional=0.25, edge_tier='elite'
        )

        strong_bet = calculate_kelly_bet_size(
            win_prob, decimal_odds, bankroll,
            fractional=0.25, edge_tier='strong'
        )

        # Strong should be ~50% of elite
        assert abs(strong_bet - elite_bet * 0.5) < 5

    def test_moderate_tier_bet_sizing(self):
        """Test bet sizing for moderate tier."""
        win_prob = 0.55
        decimal_odds = 1.91
        bankroll = 10000.0

        elite_bet = calculate_kelly_bet_size(
            win_prob, decimal_odds, bankroll,
            fractional=0.25, edge_tier='elite'
        )

        moderate_bet = calculate_kelly_bet_size(
            win_prob, decimal_odds, bankroll,
            fractional=0.25, edge_tier='moderate'
        )

        # Moderate should be ~25% of elite
        assert abs(moderate_bet - elite_bet * 0.25) < 5

    def test_weak_tier_no_bet(self):
        """Test that weak tier returns 0."""
        bet_size = calculate_kelly_bet_size(
            0.55, 1.91, 10000.0, edge_tier='weak'
        )
        assert bet_size == 0.0


class TestDrawdownAdjustments:
    """Test drawdown adjustments to Kelly sizing."""

    def test_no_drawdown_full_stakes(self):
        """Test that 0% drawdown uses full stakes."""
        win_prob = 0.55
        decimal_odds = 1.91
        bankroll = 10000.0

        no_dd = calculate_kelly_bet_size(
            win_prob, decimal_odds, bankroll,
            fractional=0.25, current_drawdown=0.0
        )

        assert no_dd > 0, "Should recommend bet with no drawdown"

    def test_drawdown_reduces_stakes(self):
        """Test that drawdown reduces bet size."""
        win_prob = 0.55
        decimal_odds = 1.91
        bankroll = 10000.0

        no_dd = calculate_kelly_bet_size(
            win_prob, decimal_odds, bankroll,
            fractional=0.25, current_drawdown=0.0
        )

        with_dd = calculate_kelly_bet_size(
            win_prob, decimal_odds, bankroll,
            fractional=0.25, current_drawdown=0.15  # 15% drawdown
        )

        assert with_dd < no_dd, "Drawdown should reduce bet size"

    def test_large_drawdown_minimal_stakes(self):
        """Test that large drawdown results in minimal stakes."""
        win_prob = 0.55
        decimal_odds = 1.91
        bankroll = 10000.0

        large_dd = calculate_kelly_bet_size(
            win_prob, decimal_odds, bankroll,
            fractional=0.25, current_drawdown=0.30  # 30% drawdown
        )

        no_dd = calculate_kelly_bet_size(
            win_prob, decimal_odds, bankroll,
            fractional=0.25, current_drawdown=0.0
        )

        # Should be significantly reduced (floor at 25% of no-drawdown bet)
        # With 30% drawdown, multiplier = max(0.25, 1.0 - 0.30*2) = 0.40
        # So large_dd ≈ no_dd * 0.40
        assert large_dd < no_dd * 0.50  # Should be less than 50% of no-drawdown


class TestCorrelationAdjustments:
    """Test correlation adjustments for same-day bets."""

    def test_single_bet_no_correlation_adjustment(self):
        """Test that single bet has no correlation adjustment."""
        win_prob = 0.55
        decimal_odds = 1.91
        bankroll = 10000.0

        single = calculate_kelly_bet_size(
            win_prob, decimal_odds, bankroll,
            num_same_day_bets=1
        )

        assert single > 0

    def test_multiple_bets_reduces_size(self):
        """Test that multiple same-day bets reduce bet size."""
        win_prob = 0.55
        decimal_odds = 1.91
        bankroll = 10000.0

        single = calculate_kelly_bet_size(
            win_prob, decimal_odds, bankroll,
            num_same_day_bets=1
        )

        multiple = calculate_kelly_bet_size(
            win_prob, decimal_odds, bankroll,
            num_same_day_bets=5
        )

        assert multiple < single, "Multiple bets should reduce size due to correlation"

    def test_correlation_floor(self):
        """Test that correlation adjustment has a floor."""
        # Even with many bets, should not go below 25% of base
        win_prob = 0.55
        decimal_odds = 1.91
        bankroll = 10000.0

        many_bets = calculate_kelly_bet_size(
            win_prob, decimal_odds, bankroll,
            num_same_day_bets=20
        )

        # Should still recommend some bet (not 0)
        assert many_bets > 0


class TestMaxBetCap:
    """Test maximum bet size cap."""

    def test_bet_capped_at_5_percent(self):
        """Test that bet is capped at 5% of bankroll."""
        # Use very high win probability to trigger large Kelly
        win_prob = 0.80
        decimal_odds = 2.0  # Even money
        bankroll = 10000.0

        bet_size = calculate_kelly_bet_size(
            win_prob, decimal_odds, bankroll,
            fractional=1.0  # Full Kelly
        )

        # Should be capped at 5% = $500
        max_bet = bankroll * 0.05
        assert bet_size <= max_bet


class TestBankrollManager:
    """Test BankrollManager class."""

    def test_initialization(self):
        """Test BankrollManager initialization."""
        manager = BankrollManager(initial_bankroll=10000.0)

        assert manager.current_bankroll == 10000.0
        assert manager.peak_bankroll == 10000.0
        assert manager.current_drawdown_pct == 0.0

    def test_update_bankroll_win(self):
        """Test updating bankroll after a win."""
        manager = BankrollManager(initial_bankroll=10000.0)

        manager.update_bankroll(pnl=100.0, bet_won=True)

        assert manager.current_bankroll == 10100.0
        assert manager.peak_bankroll == 10100.0
        assert manager.current_streak == 1

    def test_update_bankroll_loss(self):
        """Test updating bankroll after a loss."""
        manager = BankrollManager(initial_bankroll=10000.0)

        manager.update_bankroll(pnl=-110.0, bet_won=False)

        assert manager.current_bankroll == 9890.0
        assert manager.peak_bankroll == 10000.0
        assert manager.current_drawdown_pct > 0

    def test_drawdown_calculation(self):
        """Test drawdown calculation."""
        manager = BankrollManager(initial_bankroll=10000.0)

        # Win to increase peak
        manager.update_bankroll(pnl=1000.0, bet_won=True)
        assert manager.peak_bankroll == 11000.0

        # Lose
        manager.update_bankroll(pnl=-1100.0, bet_won=False)
        assert manager.current_bankroll == 9900.0

        # Drawdown should be (11000 - 9900) / 11000 = 10%
        expected_dd = (11000.0 - 9900.0) / 11000.0
        assert abs(manager.current_drawdown_pct - expected_dd) < 0.001

    def test_daily_exposure_tracking(self):
        """Test daily exposure tracking."""
        manager = BankrollManager(initial_bankroll=10000.0)

        # Place 3 bets
        manager.record_bet_placed(100.0)
        manager.record_bet_placed(150.0)
        manager.record_bet_placed(200.0)

        # Total exposure should be $450
        assert manager.get_daily_exposure() == 450.0

        # Settle one bet
        manager.record_bet_settled(100.0)
        assert manager.get_daily_exposure() == 350.0

    def test_can_place_bet_within_limits(self):
        """Test that bet within limits is allowed."""
        manager = BankrollManager(initial_bankroll=10000.0)

        can_place, reason = manager.can_place_bet(100.0)

        assert can_place is True
        assert reason == "OK"

    def test_can_place_bet_exceeds_single_limit(self):
        """Test that bet exceeding single bet limit is rejected."""
        manager = BankrollManager(
            initial_bankroll=10000.0,
            max_single_bet_pct=0.05  # 5% = $500
        )

        can_place, reason = manager.can_place_bet(600.0)

        assert can_place is False
        assert "exceeds limit" in reason

    def test_can_place_bet_exceeds_daily_exposure(self):
        """Test that bet exceeding daily exposure is rejected."""
        manager = BankrollManager(
            initial_bankroll=10000.0,
            max_daily_exposure_pct=0.20  # 20% = $2000
        )

        # Place $1800 in bets (approaching limit)
        manager.record_bet_placed(1800.0)

        # Try to place $300 more (would exceed $2000 limit: $1800 + $300 = $2100)
        can_place, reason = manager.can_place_bet(300.0)

        assert can_place is False
        assert "exposure" in reason.lower()

    def test_stop_loss_daily_limit(self):
        """Test daily loss limit halt."""
        manager = BankrollManager(
            initial_bankroll=10000.0,
            max_daily_loss_pct=0.05  # 5% = $500
        )

        # Lose $510 (exceeds daily limit)
        manager.update_bankroll(pnl=-510.0, bet_won=False)

        risk_status = manager.get_risk_status()

        assert risk_status.risk_level == RiskLevel.HALT
        assert risk_status.halt_reason == HaltReason.DAILY_LIMIT

    def test_stop_loss_drawdown_limit(self):
        """Test drawdown limit halt."""
        manager = BankrollManager(
            initial_bankroll=10000.0,
            max_drawdown_pct=0.25  # 25%
        )

        # Lose 30% of bankroll
        manager.update_bankroll(pnl=-3000.0, bet_won=False)

        risk_status = manager.get_risk_status()

        assert risk_status.risk_level == RiskLevel.HALT
        assert risk_status.halt_reason == HaltReason.DRAWDOWN_LIMIT

    def test_losing_streak_halt(self):
        """Test losing streak halt."""
        manager = BankrollManager(
            initial_bankroll=10000.0,
            losing_streak_halt=8,
            max_daily_loss_pct=0.20  # Increase to 20% so daily limit doesn't trigger first
        )

        # Simulate 8 consecutive losses (should trigger halt at -8 streak)
        # 8 × $110 = $880 = 8.8% of $10k (below 20% daily limit)
        for _i in range(8):
            manager.update_bankroll(pnl=-110.0, bet_won=False)

        # After 8 losses, streak should be -8, which triggers halt
        risk_status = manager.get_risk_status()

        assert risk_status.current_streak == -8
        assert risk_status.risk_level == RiskLevel.HALT
        assert risk_status.halt_reason == HaltReason.LOSING_STREAK

    def test_manual_halt(self):
        """Test manual halt."""
        manager = BankrollManager(initial_bankroll=10000.0)

        manager.set_manual_halt(True)

        risk_status = manager.get_risk_status()

        assert risk_status.risk_level == RiskLevel.HALT
        assert risk_status.halt_reason == HaltReason.MANUAL_HALT

        # Clear halt
        manager.set_manual_halt(False)
        risk_status = manager.get_risk_status()

        assert risk_status.is_betting_allowed()


class TestCLVTracking:
    """Test Closing Line Value tracking."""

    def test_clv_calculation_positive(self):
        """Test CLV calculation for bet with positive CLV."""
        tracker = CLVTracker()

        # Bet at -110, closing at -115 (line moved in our favor - now harder to get)
        # We got -110 but closing is -115, meaning we got better value
        # CLV = closing_prob - bet_prob
        # closing_prob(−115) = 115/215 = 0.5349
        # bet_prob(−110) = 110/210 = 0.5238
        # CLV = 0.5349 - 0.5238 = 0.0111 (positive - good!)
        bet_record = tracker.record_bet(
            bet_id="1",
            selection="Team A",
            opening_odds=-110,
            bet_odds=-110,
            closing_odds=-115,  # Worse odds at close = we got better value
            won=True
        )

        # CLV should be positive (we beat closing line)
        assert bet_record["clv"] > 0

    def test_clv_calculation_negative(self):
        """Test CLV calculation for bet with negative CLV."""
        tracker = CLVTracker()

        # Bet at -110, closing at -105 (line moved against us - easier to get now)
        # We got -110 but closing is -105, meaning we got worse value
        # CLV = closing_prob - bet_prob
        # closing_prob(−105) = 105/205 = 0.5122
        # bet_prob(−110) = 110/210 = 0.5238
        # CLV = 0.5122 - 0.5238 = -0.0116 (negative - bad)
        bet_record = tracker.record_bet(
            bet_id="1",
            selection="Team A",
            opening_odds=-110,
            bet_odds=-110,
            closing_odds=-105,  # Better odds at close = we got worse value
            won=False
        )

        # CLV should be negative
        assert bet_record["clv"] < 0

    def test_clv_summary(self):
        """Test CLV summary statistics."""
        tracker = CLVTracker()

        # Record some bets (corrected for proper CLV expectations)
        tracker.record_bet("1", "A", -110, -110, -115, won=True)   # +CLV, win
        tracker.record_bet("2", "B", -110, -110, -105, won=False)  # -CLV, loss
        tracker.record_bet("3", "C", -110, -110, -112, won=True)   # +CLV, win

        summary = tracker.get_clv_summary()

        assert summary["total_bets"] == 3
        assert summary["avg_clv"] > 0  # Overall positive CLV
        assert summary["positive_clv_count"] == 2


class TestRiskOfRuin:
    """Test Risk of Ruin calculations."""

    def test_ror_negative_ev_certain_ruin(self):
        """Test that negative EV leads to certain ruin."""
        ror = calculate_risk_of_ruin(
            win_probability=0.45,  # Negative edge
            win_payout=1.0,
            loss_amount=1.0,
            bankroll_units=100
        )

        assert ror == 1.0

    def test_ror_positive_ev_low_risk(self):
        """Test that strong positive EV has low risk of ruin."""
        ror = calculate_risk_of_ruin(
            win_probability=0.60,  # Strong edge
            win_payout=1.0,
            loss_amount=1.0,
            bankroll_units=100
        )

        assert ror < 0.5  # Should have <50% risk of ruin

    def test_ror_monte_carlo(self):
        """Test Monte Carlo RoR simulation."""
        result = calculate_risk_of_ruin_monte_carlo(
            win_probability=0.55,
            bet_size_fraction=0.05,  # 5% per bet
            decimal_odds=1.91,
            num_simulations=1000,
            num_bets=100
        )

        assert "risk_of_ruin" in result
        assert 0.0 <= result["risk_of_ruin"] <= 1.0
        assert "mean_final_bankroll" in result


class TestDynamicKellyCalculator:
    """Test DynamicKellyCalculator class."""

    def test_full_kelly(self):
        """Test full Kelly calculation."""
        calc = DynamicKellyCalculator(kelly_fraction=1.0)

        kelly = calc.calculate_kelly(win_probability=0.55, decimal_odds=2.0)

        # Expected: (1*0.55 - 0.45) / 1 = 0.10
        assert abs(kelly - 0.10) < 0.01

    def test_uncertainty_adjusted_kelly(self):
        """Test uncertainty-adjusted Kelly."""
        calc = DynamicKellyCalculator()

        base_kelly = calc.calculate_kelly(0.55, 1.91)

        uncertain_kelly = calc.calculate_uncertainty_adjusted_kelly(
            0.55, 1.91, probability_std=0.10
        )

        # Uncertainty should reduce Kelly
        assert uncertain_kelly < base_kelly

    def test_dynamic_kelly_all_adjustments(self):
        """Test dynamic Kelly with all adjustments."""
        calc = DynamicKellyCalculator(kelly_fraction=0.25)

        result = calc.calculate_dynamic_kelly(
            win_probability=0.55,
            decimal_odds=1.91,
            edge_quality_score=80.0,
            current_drawdown=0.10,
            num_same_day_bets=3
        )

        assert "full_kelly" in result
        assert "final_kelly" in result
        assert result["final_kelly"] < result["full_kelly"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
