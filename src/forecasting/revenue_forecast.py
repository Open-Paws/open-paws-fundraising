"""
Revenue forecasting using exponential smoothing (statsmodels).

Produces 12-month forward projections with confidence intervals from
historical monthly donation totals.

Design note: TimesFM 2.5 (200M parameter pretrained time-series model) can
replace this module for higher accuracy on orgs with sufficient history (>24 months).
See issue #2 for the TimesFM integration plan.

Usage:
    forecaster = RevenueForecaster()
    result = forecaster.forecast(monthly_totals=[10000, 12000, 9500, ...])
    print(result["projections"])
    print(result["confidence_intervals"])
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RevenueForecaster:
    """
    12-month revenue forecaster using Holt-Winters exponential smoothing.

    Falls back to simple moving average if statsmodels is not installed.
    Detects seasonal patterns (year-end giving spike) and campaign-driven spikes.

    Input: list of monthly donation totals (chronological order, oldest first).
    Minimum recommended history: 12 months.
    """

    def __init__(self) -> None:
        self._has_statsmodels = self._check_statsmodels()

    @staticmethod
    def _check_statsmodels() -> bool:
        try:
            import statsmodels.tsa.holtwinters  # noqa: F401
            return True
        except ImportError:
            return False

    def forecast(
        self,
        monthly_totals: list[float],
        horizon: int = 12,
        confidence_level: float = 0.80,
    ) -> dict:
        """
        Generate a 12-month revenue forecast.

        Args:
            monthly_totals: Historical monthly donation totals (oldest first).
                           Minimum 6 months; 18+ months recommended for seasonality.
            horizon: Number of months to forecast (default 12).
            confidence_level: Confidence interval width (default 0.80).

        Returns:
            {
                "projections": [float, ...],           # horizon monthly projections
                "confidence_intervals": [
                    {"lower": float, "upper": float},  # per month
                    ...
                ],
                "annual_total_projection": float,
                "annual_total_lower": float,
                "annual_total_upper": float,
                "seasonal_pattern_detected": bool,
                "spike_months": [int, ...],            # 0-indexed months with spikes
                "method": "holt_winters" | "simple_moving_average",
                "history_months": int,
                "trend_direction": "up" | "down" | "flat",
                "notes": str,
            }
        """
        if len(monthly_totals) < 3:
            raise ValueError(
                f"Need at least 3 months of history; got {len(monthly_totals)}"
            )

        series = np.array(monthly_totals, dtype=float)

        seasonal_detected = self._detect_seasonality(series)
        spike_months = self._detect_spikes(series)
        trend = self._trend_direction(series)

        if self._has_statsmodels and len(series) >= 12:
            projections, lower, upper, method = self._holt_winters_forecast(
                series, horizon, confidence_level, seasonal_detected
            )
        else:
            projections, lower, upper, method = self._moving_average_forecast(
                series, horizon, confidence_level
            )

        annual_total = float(sum(projections))
        annual_lower = float(sum(lower))
        annual_upper = float(sum(upper))

        notes = self._build_notes(
            method, len(series), seasonal_detected, spike_months, trend
        )

        return {
            "projections": [round(p, 2) for p in projections],
            "confidence_intervals": [
                {"lower": round(lo, 2), "upper": round(hi, 2)}
                for lo, hi in zip(lower, upper)
            ],
            "annual_total_projection": round(annual_total, 2),
            "annual_total_lower": round(annual_lower, 2),
            "annual_total_upper": round(annual_upper, 2),
            "seasonal_pattern_detected": seasonal_detected,
            "spike_months": spike_months,
            "method": method,
            "history_months": len(series),
            "trend_direction": trend,
            "notes": notes,
        }

    def _holt_winters_forecast(
        self,
        series: np.ndarray,
        horizon: int,
        confidence_level: float,
        seasonal: bool,
    ) -> tuple[list[float], list[float], list[float], str]:
        """Holt-Winters exponential smoothing forecast."""
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        try:
            seasonal_periods = 12 if seasonal and len(series) >= 24 else None
            seasonal_type = "add" if seasonal_periods else None

            model = ExponentialSmoothing(
                series,
                trend="add",
                seasonal=seasonal_type,
                seasonal_periods=seasonal_periods,
                initialization_method="estimated",
            )
            fit = model.fit(optimized=True)

            forecast_result = fit.forecast(horizon)
            projections = [max(0.0, float(v)) for v in forecast_result]

            # Confidence interval via simulation
            alpha = 1 - confidence_level
            residuals = fit.resid
            std_err = float(np.std(residuals))
            z_score = 1.28 if confidence_level == 0.80 else 1.96  # 80% or 95%

            # Growing uncertainty over forecast horizon
            lower = [
                max(0.0, p - z_score * std_err * np.sqrt(i + 1))
                for i, p in enumerate(projections)
            ]
            upper = [
                p + z_score * std_err * np.sqrt(i + 1)
                for i, p in enumerate(projections)
            ]

            return projections, lower, upper, "holt_winters"

        except Exception as exc:
            logger.warning("Holt-Winters failed (%s); falling back to moving average", exc)
            return self._moving_average_forecast(series, horizon, confidence_level)

    def _moving_average_forecast(
        self,
        series: np.ndarray,
        horizon: int,
        confidence_level: float,
    ) -> tuple[list[float], list[float], list[float], str]:
        """Simple moving average fallback forecast."""
        window = min(6, len(series))
        recent = series[-window:]
        mean = float(np.mean(recent))
        std = float(np.std(recent))

        # Apply simple trend adjustment
        if len(series) >= 3:
            trend_per_month = (series[-1] - series[-3]) / 2
            trend_per_month = np.clip(trend_per_month, -mean * 0.15, mean * 0.15)
        else:
            trend_per_month = 0.0

        z_score = 1.28 if confidence_level == 0.80 else 1.96

        projections = [
            max(0.0, mean + trend_per_month * (i + 1)) for i in range(horizon)
        ]
        lower = [max(0.0, p - z_score * std) for p in projections]
        upper = [p + z_score * std for p in projections]

        return projections, lower, upper, "simple_moving_average"

    @staticmethod
    def _detect_seasonality(series: np.ndarray) -> bool:
        """
        Detect if the series shows year-end giving spike (Nov/Dec pattern).
        Requires at least 12 months of data.
        """
        if len(series) < 12:
            return False

        # Check if last two months of each year-window tend to be higher
        annual_chunks = len(series) // 12
        if annual_chunks < 1:
            return False

        spike_evidence = 0
        for year in range(annual_chunks):
            year_data = series[year * 12: (year + 1) * 12]
            if len(year_data) < 12:
                continue
            year_mean = np.mean(year_data)
            if year_mean == 0:
                continue
            # Nov (index 10) and Dec (index 11) are typically strong for nonprofits
            q4_mean = np.mean(year_data[9:12])  # Oct–Dec
            if q4_mean > year_mean * 1.2:
                spike_evidence += 1

        return spike_evidence >= 1

    @staticmethod
    def _detect_spikes(series: np.ndarray) -> list[int]:
        """
        Identify months with anomalously high donations (likely campaign-driven).

        Returns 0-indexed list of spike month positions in the history.
        """
        if len(series) < 3:
            return []
        mean = np.mean(series)
        std = np.std(series)
        if std == 0:
            return []
        return [
            i for i, v in enumerate(series)
            if v > mean + 2 * std
        ]

    @staticmethod
    def _trend_direction(series: np.ndarray) -> str:
        """Classify overall trend as up / down / flat."""
        if len(series) < 3:
            return "flat"
        # Compare recent 3 months to earlier 3 months
        recent = np.mean(series[-3:])
        earlier = np.mean(series[max(0, len(series) - 6): len(series) - 3])
        if earlier == 0:
            return "flat"
        change = (recent - earlier) / earlier
        if change > 0.05:
            return "up"
        if change < -0.05:
            return "down"
        return "flat"

    @staticmethod
    def _build_notes(
        method: str,
        n_months: int,
        seasonal: bool,
        spike_months: list[int],
        trend: str,
    ) -> str:
        parts = [f"Forecast method: {method} ({n_months} months of history)."]

        if seasonal:
            parts.append("Year-end giving spike pattern detected — Q4 projections may be conservative.")
        if spike_months:
            parts.append(
                f"{len(spike_months)} campaign-driven spike month(s) detected in history. "
                "Projections assume normal cadence; plan major campaigns to recreate these spikes."
            )
        if trend == "up":
            parts.append("Positive donation trend detected over recent months.")
        elif trend == "down":
            parts.append(
                "Declining donation trend detected. Re-engagement campaigns recommended. "
                "Churn predictor output should inform priorities."
            )

        if method == "simple_moving_average":
            parts.append(
                "Install statsmodels for Holt-Winters forecasting: pip install statsmodels. "
                "For highest accuracy (TimesFM 2.5), see issue #2."
            )

        return " ".join(parts)
