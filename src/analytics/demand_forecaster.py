"""Demand Forecasting by Branch.

With only 4-5 months of data per branch, complex models (ARIMA, Prophet)
would overfit. Instead we use a transparent ensemble of:
  1. Linear trend extrapolation (OLS on trend_idx)
  2. Weighted moving average (recent months weighted more)
  3. Growth-rate projection (apply avg MoM growth to last value)

The final forecast is a weighted average of the three methods, with
confidence intervals derived from historical residual variance.

Outputs:
  - branch_forecasts: per-branch monthly forecasts for Jan-Mar 2026
  - forecast_summary: JSON-friendly executive summary
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from ..config import FEATURES_DIR

logger = logging.getLogger(__name__)

RESULTS_DIR_NAME = "analytics"
FORECAST_MONTHS = [
    (2026, 1, "January"),
    (2026, 2, "February"),
    (2026, 3, "March"),
]

# Ensemble weights: trend, WMA, growth-rate
ENSEMBLE_WEIGHTS = (0.35, 0.40, 0.25)


class DemandForecaster:
    """Forecast demand per branch for the next 1-3 months."""

    def __init__(
        self,
        features_dir: Path | str | None = None,
        output_dir: Path | str | None = None,
    ) -> None:
        self.features_dir = Path(features_dir) if features_dir else FEATURES_DIR
        self.output_dir = (
            Path(output_dir)
            if output_dir
            else FEATURES_DIR.parent / RESULTS_DIR_NAME
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, *, save: bool = True) -> Dict[str, Any]:
        monthly = pd.read_csv(self.features_dir / "demand_branch_monthly_features.csv")
        profile = pd.read_csv(self.features_dir / "demand_branch_profile.csv")

        monthly["date"] = pd.to_datetime(monthly["date"])

        forecasts = self._forecast_all_branches(monthly)
        summary = self._build_summary(forecasts, profile)

        results: Dict[str, Any] = {
            "branch_forecasts": forecasts,
            "forecast_summary": summary,
        }

        if save:
            forecasts.to_csv(self.output_dir / "demand_branch_forecasts.csv", index=False)
            with open(self.output_dir / "demand_forecast_summary.json", "w") as f:
                json.dump(summary, f, indent=2)
            logger.info(
                "Saved forecasts (%d rows) and summary to %s",
                len(forecasts), self.output_dir,
            )

        return results

    # ------------------------------------------------------------------
    # Core forecasting
    # ------------------------------------------------------------------

    def _forecast_all_branches(self, monthly: pd.DataFrame) -> pd.DataFrame:
        all_rows: List[Dict] = []

        for branch in sorted(monthly["branch"].unique()):
            bm = monthly[monthly["branch"] == branch].sort_values("date")
            actuals = bm["total"].values
            trend_idx = bm["trend_idx"].values

            # Fit models on historical data
            trend_pred = self._linear_trend(trend_idx, actuals)
            wma_pred = self._weighted_moving_avg(actuals)
            growth_pred = self._growth_rate_projection(actuals)

            # Historical residuals for confidence intervals
            residuals = self._compute_residuals(actuals, trend_idx)
            residual_std = np.std(residuals) if len(residuals) > 1 else actuals.std()

            # Forecast future months
            last_idx = int(trend_idx[-1])
            for i, (year, month_num, month_name) in enumerate(FORECAST_MONTHS):
                future_idx = last_idx + 1 + i
                steps_ahead = i + 1

                f_trend = self._extrapolate_trend(trend_idx, actuals, future_idx)
                f_wma = wma_pred * (1 + 0.02 * steps_ahead)  # slight growth assumption
                f_growth = growth_pred * ((1 + self._avg_growth_rate(actuals)) ** steps_ahead)

                # Clip negatives
                f_trend = max(f_trend, 0)
                f_wma = max(f_wma, 0)
                f_growth = max(f_growth, 0)

                # Ensemble
                w_trend, w_wma, w_growth = ENSEMBLE_WEIGHTS
                forecast = w_trend * f_trend + w_wma * f_wma + w_growth * f_growth

                # Confidence intervals widen with steps ahead
                ci_multiplier = 1.96 * math.sqrt(steps_ahead)
                ci_lower = max(forecast - ci_multiplier * residual_std, 0)
                ci_upper = forecast + ci_multiplier * residual_std

                all_rows.append({
                    "branch": branch,
                    "year": year,
                    "month": month_name,
                    "month_num": month_num,
                    "date": f"{year}-{month_num:02d}-01",
                    "forecast": round(forecast, 2),
                    "ci_lower": round(ci_lower, 2),
                    "ci_upper": round(ci_upper, 2),
                    "forecast_trend": round(f_trend, 2),
                    "forecast_wma": round(f_wma, 2),
                    "forecast_growth": round(f_growth, 2),
                    "residual_std": round(residual_std, 2),
                    "steps_ahead": steps_ahead,
                })

            # Also include historical actuals for context
            for _, row in bm.iterrows():
                all_rows.append({
                    "branch": branch,
                    "year": int(row["year"]),
                    "month": row["month"],
                    "month_num": int(row["month_num"]),
                    "date": str(row["date"].date()),
                    "forecast": round(row["total"], 2),
                    "ci_lower": round(row["total"], 2),
                    "ci_upper": round(row["total"], 2),
                    "forecast_trend": None,
                    "forecast_wma": None,
                    "forecast_growth": None,
                    "residual_std": None,
                    "steps_ahead": 0,
                })

        df = pd.DataFrame(all_rows).sort_values(["branch", "date"]).reset_index(drop=True)
        logger.info("Generated forecasts for %d branch-months", len(df))
        return df

    # ------------------------------------------------------------------
    # Forecasting methods
    # ------------------------------------------------------------------

    @staticmethod
    def _linear_trend(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit y = a + b*x via OLS, return fitted values."""
        n = len(x)
        if n < 2:
            return y.copy()
        x_mean, y_mean = x.mean(), y.mean()
        b = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
        a = y_mean - b * x_mean
        return a + b * x

    @staticmethod
    def _extrapolate_trend(x: np.ndarray, y: np.ndarray, future_x: float) -> float:
        """Extrapolate linear trend to a future x value."""
        n = len(x)
        if n < 2:
            return float(y[-1])
        x_mean, y_mean = x.mean(), y.mean()
        ss_xx = np.sum((x - x_mean) ** 2)
        if ss_xx == 0:
            return float(y_mean)
        b = np.sum((x - x_mean) * (y - y_mean)) / ss_xx
        a = y_mean - b * x_mean
        return float(a + b * future_x)

    @staticmethod
    def _weighted_moving_avg(values: np.ndarray, window: int = 3) -> float:
        """Weighted moving average giving more weight to recent values."""
        n = min(window, len(values))
        recent = values[-n:]
        weights = np.arange(1, n + 1, dtype=float)
        return float(np.average(recent, weights=weights))

    @staticmethod
    def _avg_growth_rate(values: np.ndarray) -> float:
        """Median month-over-month growth rate, clamped to [-0.3, 0.5].

        Using median instead of mean to be robust to outlier months,
        and a tight clamp to keep multi-step projections realistic.
        """
        if len(values) < 2:
            return 0.0
        rates = []
        for i in range(1, len(values)):
            if values[i - 1] > 0:
                rates.append((values[i] - values[i - 1]) / values[i - 1])
        if not rates:
            return 0.0
        med = float(np.median(rates))
        return float(np.clip(med, -0.3, 0.5))

    @staticmethod
    def _growth_rate_projection(values: np.ndarray) -> float:
        """Base value for growth-rate projection: last observed value."""
        return float(values[-1])

    def _compute_residuals(self, actuals: np.ndarray, trend_idx: np.ndarray) -> np.ndarray:
        """Residuals from the linear trend fit."""
        fitted = self._linear_trend(trend_idx, actuals)
        return actuals - fitted

    # ------------------------------------------------------------------
    # Executive summary
    # ------------------------------------------------------------------

    def _build_summary(
        self, forecasts: pd.DataFrame, profile: pd.DataFrame
    ) -> Dict[str, Any]:
        future = forecasts[forecasts["steps_ahead"] > 0]
        jan = future[future["month_num"] == 1]

        summary: Dict[str, Any] = {
            "forecast_horizon": "January - March 2026",
            "method": "Ensemble of linear trend, weighted moving average, and growth-rate projection",
            "branches": {},
        }

        for branch in sorted(future["branch"].unique()):
            bf = future[future["branch"] == branch]
            bp = profile[profile["branch"] == branch]

            jan_row = bf[bf["month_num"] == 1].iloc[0] if len(bf[bf["month_num"] == 1]) > 0 else None
            total_forecast = bf["forecast"].sum()
            avg_forecast = bf["forecast"].mean()

            hist_avg = bp["avg_monthly_revenue"].values[0] if not bp.empty else 0

            branch_info: Dict[str, Any] = {
                "jan_2026_forecast": round(float(jan_row["forecast"]), 2) if jan_row is not None else None,
                "jan_2026_ci": [
                    round(float(jan_row["ci_lower"]), 2) if jan_row is not None else None,
                    round(float(jan_row["ci_upper"]), 2) if jan_row is not None else None,
                ],
                "q1_2026_total_forecast": round(float(total_forecast), 2),
                "avg_monthly_forecast": round(float(avg_forecast), 2),
                "historical_avg_monthly": round(float(hist_avg), 2),
                "forecast_vs_historical": round(
                    float((avg_forecast - hist_avg) / hist_avg), 4
                ) if hist_avg > 0 else None,
            }
            summary["branches"][branch] = branch_info

        # Network total
        network_q1 = future.groupby("month_num")["forecast"].sum()
        summary["network_q1_2026_total"] = round(float(network_q1.sum()), 2)
        summary["network_monthly_avg"] = round(float(network_q1.mean()), 2)

        # Highest growth branch
        growth_rates = {
            b: info.get("forecast_vs_historical", 0) or 0
            for b, info in summary["branches"].items()
        }
        if growth_rates:
            best = max(growth_rates, key=growth_rates.get)
            summary["fastest_growing_branch"] = best
            summary["fastest_growth_rate"] = growth_rates[best]

        return summary
