"""
Tests for revenue forecasting (exponential smoothing).

Domain rules verified:
- Revenue forecast must not return negative values for a positive-trend input series
- Projections list length must equal the requested horizon
- Annual total must equal sum of monthly projections
- Short series (< 3 months) must raise ValueError
- Trend direction must be correctly classified
"""

from __future__ import annotations

import pytest

from src.forecasting.revenue_forecast import RevenueForecaster


class TestForecastOutputShape:
    """Verify the forecast returns the expected structure and length."""

    def test_default_horizon_produces_12_projections(self):
        monthly = [10_000, 12_000, 11_000, 13_000, 12_500, 14_000]
        result = RevenueForecaster().forecast(monthly)
        assert len(result["projections"]) == 12
        assert len(result["confidence_intervals"]) == 12

    def test_custom_horizon_respected(self):
        monthly = [5_000, 6_000, 5_500, 7_000, 6_500, 8_000]
        result = RevenueForecaster().forecast(monthly, horizon=6)
        assert len(result["projections"]) == 6
        assert len(result["confidence_intervals"]) == 6

    def test_annual_total_equals_sum_of_projections(self):
        monthly = [10_000] * 6
        result = RevenueForecaster().forecast(monthly, horizon=12)
        expected = round(sum(result["projections"]), 2)
        assert abs(result["annual_total_projection"] - expected) < 0.01

    def test_result_contains_required_keys(self):
        monthly = [8_000, 9_000, 8_500, 10_000, 9_500, 11_000]
        result = RevenueForecaster().forecast(monthly)
        required_keys = {
            "projections",
            "confidence_intervals",
            "annual_total_projection",
            "annual_total_lower",
            "annual_total_upper",
            "seasonal_pattern_detected",
            "spike_months",
            "method",
            "history_months",
            "trend_direction",
            "notes",
        }
        assert required_keys.issubset(result.keys())


class TestForecastNonNegativity:
    """Revenue forecast must never return negative values for positive-trend input."""

    def test_positive_trend_series_produces_no_negative_projections(self):
        """A steadily growing donation series must produce non-negative forecasts."""
        monthly = [10_000 + i * 500 for i in range(12)]
        result = RevenueForecaster().forecast(monthly)
        for i, p in enumerate(result["projections"]):
            assert p >= 0.0, f"Projection month {i} is negative ({p}) for a positive-trend series"

    def test_flat_series_produces_no_negative_projections(self):
        """A flat series of equal donations must produce non-negative forecasts."""
        monthly = [5_000] * 8
        result = RevenueForecaster().forecast(monthly)
        for p in result["projections"]:
            assert p >= 0.0

    def test_confidence_lower_bounds_non_negative(self):
        """Lower confidence bounds must be >= 0.0 even for volatile series."""
        monthly = [10_000, 12_000, 11_000, 13_000, 12_500, 14_000]
        result = RevenueForecaster().forecast(monthly)
        for ci in result["confidence_intervals"]:
            assert ci["lower"] >= 0.0, f"Lower CI bound is negative: {ci['lower']}"

    def test_annual_lower_bound_non_negative(self):
        monthly = [1_000, 1_200, 1_100, 1_300, 1_250, 1_400]
        result = RevenueForecaster().forecast(monthly)
        assert result["annual_total_lower"] >= 0.0


class TestForecastTrendDetection:
    """Verify the trend_direction output reflects the actual series direction."""

    def test_growing_series_trend_is_up(self):
        monthly = [5_000, 6_000, 7_000, 8_000, 9_000, 10_000, 11_000, 12_000]
        result = RevenueForecaster().forecast(monthly)
        assert result["trend_direction"] == "up", (
            f"Expected 'up' for growing series, got '{result['trend_direction']}'"
        )

    def test_declining_series_trend_is_down(self):
        monthly = [12_000, 11_000, 10_000, 9_000, 8_000, 7_000, 6_000, 5_000]
        result = RevenueForecaster().forecast(monthly)
        assert result["trend_direction"] == "down", (
            f"Expected 'down' for declining series, got '{result['trend_direction']}'"
        )

    def test_flat_series_trend_is_flat(self):
        monthly = [10_000] * 9
        result = RevenueForecaster().forecast(monthly)
        assert result["trend_direction"] == "flat"

    def test_history_months_reflects_input_length(self):
        monthly = [10_000] * 7
        result = RevenueForecaster().forecast(monthly)
        assert result["history_months"] == 7


class TestForecastValidation:
    """Input validation for the forecaster."""

    def test_too_short_series_raises_value_error(self):
        """Fewer than 3 months must raise ValueError."""
        with pytest.raises(ValueError, match="at least 3 months"):
            RevenueForecaster().forecast([10_000, 11_000])

    def test_minimum_series_length_accepted(self):
        """Exactly 3 months is the minimum valid input."""
        result = RevenueForecaster().forecast([10_000, 11_000, 12_000])
        assert len(result["projections"]) == 12


class TestForecastSpikes:
    """Spike detection identifies anomalous months in the history."""

    def test_spike_months_detected_for_anomalous_campaign_month(self):
        """A single very high month should be detected as a campaign-driven spike."""
        # 11 normal months + 1 campaign month at 5x
        monthly = [10_000] * 11 + [50_000]
        result = RevenueForecaster().forecast(monthly)
        assert len(result["spike_months"]) >= 1, (
            "A campaign-driven spike month (5x average) must be detected"
        )

    def test_uniform_series_has_no_spikes(self):
        """A perfectly flat series must report zero spike months."""
        monthly = [10_000] * 12
        result = RevenueForecaster().forecast(monthly)
        assert result["spike_months"] == []
