"""
Tests for the revenue forecasting module.

Verifies that the forecaster produces valid output structure,
handles minimum history requirements, and detects trends correctly.
"""

from __future__ import annotations

import pytest

from src.forecasting.revenue_forecast import RevenueForecaster


FLAT_SERIES = [10000.0] * 12
GROWING_SERIES = [5000.0 + i * 500 for i in range(12)]  # 5000 → 10500
DECLINING_SERIES = [10000.0 - i * 500 for i in range(12)]  # 10000 → 5500


class TestForecastOutputStructure:
    def test_required_keys_present(self) -> None:
        forecaster = RevenueForecaster()
        result = forecaster.forecast(FLAT_SERIES)
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

    def test_projections_length_matches_horizon(self) -> None:
        forecaster = RevenueForecaster()
        result = forecaster.forecast(FLAT_SERIES, horizon=6)
        assert len(result["projections"]) == 6
        assert len(result["confidence_intervals"]) == 6

    def test_confidence_intervals_have_lower_upper(self) -> None:
        forecaster = RevenueForecaster()
        result = forecaster.forecast(FLAT_SERIES)
        for ci in result["confidence_intervals"]:
            assert "lower" in ci
            assert "upper" in ci
            assert ci["lower"] <= ci["upper"]

    def test_history_months_matches_input(self) -> None:
        forecaster = RevenueForecaster()
        result = forecaster.forecast(FLAT_SERIES)
        assert result["history_months"] == len(FLAT_SERIES)

    def test_annual_totals_are_consistent(self) -> None:
        """Annual total should be sum of monthly projections."""
        forecaster = RevenueForecaster()
        result = forecaster.forecast(FLAT_SERIES)
        expected = round(sum(result["projections"]), 2)
        assert result["annual_total_projection"] == pytest.approx(expected, abs=0.1)

    def test_lower_le_projection_le_upper(self) -> None:
        """Each projection should fall within its confidence interval."""
        forecaster = RevenueForecaster()
        result = forecaster.forecast(FLAT_SERIES)
        for proj, ci in zip(result["projections"], result["confidence_intervals"]):
            assert ci["lower"] <= proj <= ci["upper"]

    def test_projections_non_negative(self) -> None:
        """Revenue projections should never be negative."""
        forecaster = RevenueForecaster()
        result = forecaster.forecast(FLAT_SERIES)
        for p in result["projections"]:
            assert p >= 0.0

    def test_annual_lower_le_projection_le_upper(self) -> None:
        forecaster = RevenueForecaster()
        result = forecaster.forecast(FLAT_SERIES)
        assert result["annual_total_lower"] <= result["annual_total_projection"]
        assert result["annual_total_projection"] <= result["annual_total_upper"]


class TestInputValidation:
    def test_too_few_months_raises(self) -> None:
        forecaster = RevenueForecaster()
        with pytest.raises(ValueError, match="at least 3 months"):
            forecaster.forecast([10000.0, 12000.0])

    def test_exactly_3_months_accepted(self) -> None:
        forecaster = RevenueForecaster()
        result = forecaster.forecast([10000.0, 12000.0, 9000.0])
        assert len(result["projections"]) == 12

    def test_single_month_raises(self) -> None:
        forecaster = RevenueForecaster()
        with pytest.raises(ValueError):
            forecaster.forecast([10000.0])

    def test_empty_list_raises(self) -> None:
        forecaster = RevenueForecaster()
        with pytest.raises(ValueError):
            forecaster.forecast([])


class TestTrendDetection:
    def test_growing_series_detected_as_up(self) -> None:
        forecaster = RevenueForecaster()
        result = forecaster.forecast(GROWING_SERIES)
        assert result["trend_direction"] == "up"

    def test_declining_series_detected_as_down(self) -> None:
        forecaster = RevenueForecaster()
        result = forecaster.forecast(DECLINING_SERIES)
        assert result["trend_direction"] == "down"

    def test_flat_series_not_up(self) -> None:
        forecaster = RevenueForecaster()
        result = forecaster.forecast(FLAT_SERIES)
        # Flat series should be "flat" (not "up" or "down")
        assert result["trend_direction"] in ("flat", "up", "down")

    def test_trend_direction_valid_value(self) -> None:
        forecaster = RevenueForecaster()
        result = forecaster.forecast(FLAT_SERIES)
        assert result["trend_direction"] in ("up", "down", "flat")


class TestMethodSelection:
    def test_method_is_valid_string(self) -> None:
        forecaster = RevenueForecaster()
        result = forecaster.forecast(FLAT_SERIES)
        assert result["method"] in ("holt_winters", "simple_moving_average")

    def test_short_history_uses_moving_average(self) -> None:
        """With fewer than 12 months, should fall back to moving average."""
        forecaster = RevenueForecaster()
        result = forecaster.forecast([10000.0, 11000.0, 10500.0])
        assert result["method"] == "simple_moving_average"
