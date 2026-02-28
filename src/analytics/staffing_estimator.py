"""Shift Staffing Estimator.

Estimates required employees per shift type per branch by combining:
  1. Observed staffing patterns (who works when, how long)
  2. Revenue-per-hour efficiency ratios
  3. Demand forecasts (projected revenue -> projected hours needed)
  4. Day-of-week patterns (weekday vs weekend)

Outputs:
  - staffing_recommendations: per-branch, per-shift-type staffing targets
  - scheduling_guidelines: daily scheduling template per branch
  - staffing_summary: JSON executive summary
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ..config import FEATURES_DIR

logger = logging.getLogger(__name__)

RESULTS_DIR_NAME = "analytics"

SHIFT_ORDER = ["morning", "afternoon", "evening", "overnight"]


class StaffingEstimator:
    """Estimate staffing needs per shift per branch."""

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
        profile = pd.read_csv(self.features_dir / "staffing_branch_staffing_profile.csv")
        daily = pd.read_csv(self.features_dir / "staffing_daily_coverage.csv")
        shifts = pd.read_csv(self.features_dir / "staffing_shift_patterns.csv")

        forecast_path = self.output_dir / "demand_forecast_summary.json"
        forecast = {}
        if forecast_path.exists():
            with open(forecast_path) as f:
                forecast = json.load(f)

        recommendations = self._build_shift_recommendations(shifts, profile, forecast)
        schedule = self._build_scheduling_guidelines(daily, shifts, profile)
        summary = self._build_summary(recommendations, schedule, profile, forecast)

        results: Dict[str, Any] = {
            "staffing_recommendations": recommendations,
            "scheduling_guidelines": schedule,
            "staffing_summary": summary,
        }

        if save:
            recommendations.to_csv(
                self.output_dir / "staffing_recommendations.csv", index=False
            )
            schedule.to_csv(
                self.output_dir / "staffing_scheduling_guidelines.csv", index=False
            )
            with open(self.output_dir / "staffing_summary.json", "w") as f:
                json.dump(summary, f, indent=2)
            logger.info("Saved staffing outputs to %s", self.output_dir)

        return results

    # ------------------------------------------------------------------
    # Shift-level recommendations
    # ------------------------------------------------------------------

    def _build_shift_recommendations(
        self,
        shifts: pd.DataFrame,
        profile: pd.DataFrame,
        forecast: Dict,
    ) -> pd.DataFrame:
        rows: List[Dict] = []

        for branch in sorted(shifts["branch"].dropna().unique()):
            bs = shifts[shifts["branch"] == branch]
            bp = profile[profile["branch"] == branch].iloc[0]

            total_shifts = len(bs)
            branch_rev_per_hour = bp["revenue_per_hour"]

            # Forecasted demand growth multiplier
            fc_info = forecast.get("branches", {}).get(branch, {})
            growth = fc_info.get("forecast_vs_historical", 0) or 0
            demand_multiplier = max(1.0 + growth, 0.5)  # floor at 0.5x

            for shift_type in SHIFT_ORDER:
                st = bs[bs["shift_type"] == shift_type]
                if st.empty:
                    continue

                shift_count = len(st)
                shift_share = shift_count / total_shifts
                avg_hours = st["work_duration_hours"].mean()
                median_hours = st["work_duration_hours"].median()
                employees_in_shift = st["emp_id"].nunique()

                # How many shifts per day for this shift type
                unique_days = st["date"].nunique()
                avg_per_day = shift_count / unique_days if unique_days > 0 else 0

                # Weekend vs weekday split
                weekend_shifts = st[st["is_weekend"] == 1]
                weekday_shifts = st[st["is_weekend"] == 0]
                weekend_avg = len(weekend_shifts) / max(st["date"][st["is_weekend"] == 1].nunique(), 1)
                weekday_avg = len(weekday_shifts) / max(st["date"][st["is_weekend"] == 0].nunique(), 1)

                # Projected staffing: scale current by demand multiplier
                projected_per_day = math.ceil(avg_per_day * demand_multiplier)
                # Cap at reasonable limits
                projected_per_day = min(projected_per_day, employees_in_shift + 3)

                rows.append({
                    "branch": branch,
                    "shift_type": shift_type,
                    "current_shift_count": shift_count,
                    "shift_share": round(shift_share, 4),
                    "employees_available": employees_in_shift,
                    "avg_hours_per_shift": round(avg_hours, 2),
                    "median_hours_per_shift": round(median_hours, 2),
                    "avg_staff_per_day": round(avg_per_day, 2),
                    "weekday_avg_staff": round(weekday_avg, 2),
                    "weekend_avg_staff": round(weekend_avg, 2),
                    "demand_multiplier": round(demand_multiplier, 4),
                    "recommended_staff_per_day": projected_per_day,
                    "recommended_shift_hours": round(median_hours, 1),
                })

        result = pd.DataFrame(rows)
        logger.info("Built shift recommendations: %d rows", len(result))
        return result

    # ------------------------------------------------------------------
    # Daily scheduling guidelines
    # ------------------------------------------------------------------

    def _build_scheduling_guidelines(
        self,
        daily: pd.DataFrame,
        shifts: pd.DataFrame,
        profile: pd.DataFrame,
    ) -> pd.DataFrame:
        """Per-branch, per-day-of-week scheduling template."""
        rows: List[Dict] = []

        for branch in sorted(daily["branch"].unique()):
            bd = daily[daily["branch"] == branch]
            bp = profile[profile["branch"] == branch].iloc[0]

            for dow_num in range(7):
                day_data = bd[bd["day_num"] == dow_num]
                if day_data.empty:
                    continue

                day_name = day_data["day_of_week"].iloc[0]
                is_weekend = dow_num in (5, 6)

                avg_employees = day_data["employees_on_duty"].mean()
                avg_hours = day_data["total_hours"].mean()
                avg_coverage = day_data["coverage_span_hours"].mean()
                avg_shifts = day_data["total_shifts"].mean()

                # Shift type distribution for this day
                day_shifts = shifts[
                    (shifts["branch"] == branch) & (shifts["day_num"] == dow_num)
                ]
                shift_dist = day_shifts["shift_type"].value_counts(normalize=True)

                # Peak shift
                peak_shift = shift_dist.idxmax() if not shift_dist.empty else "afternoon"

                # Recommended: round up from average, add 1 for weekends
                rec_employees = math.ceil(avg_employees)
                if is_weekend:
                    rec_employees = max(rec_employees, math.ceil(avg_employees + 0.5))

                rec_hours = round(avg_hours / rec_employees, 1) if rec_employees > 0 else 8.0

                rows.append({
                    "branch": branch,
                    "day_of_week": day_name,
                    "day_num": dow_num,
                    "is_weekend": int(is_weekend),
                    "observed_avg_employees": round(avg_employees, 2),
                    "observed_avg_total_hours": round(avg_hours, 2),
                    "observed_avg_coverage_span": round(avg_coverage, 2),
                    "peak_shift_type": peak_shift,
                    "recommended_employees": rec_employees,
                    "recommended_hours_per_employee": rec_hours,
                })

        result = pd.DataFrame(rows)
        logger.info("Built scheduling guidelines: %d rows", len(result))
        return result

    # ------------------------------------------------------------------
    # Executive summary
    # ------------------------------------------------------------------

    def _build_summary(
        self,
        recommendations: pd.DataFrame,
        schedule: pd.DataFrame,
        profile: pd.DataFrame,
        forecast: Dict,
    ) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "branches_analyzed": sorted(profile["branch"].tolist()),
            "data_period": "December 2025",
            "branch_staffing": {},
        }

        for branch in sorted(profile["branch"].tolist()):
            bp = profile[profile["branch"] == branch].iloc[0]
            br = recommendations[recommendations["branch"] == branch]
            bs = schedule[schedule["branch"] == branch]

            # Current state
            current: Dict[str, Any] = {
                "num_employees": int(bp["num_employees"]),
                "avg_shift_hours": round(float(bp["avg_shift_hours"]), 1),
                "avg_employees_per_day": round(float(bp["avg_employees_per_day"]), 1),
                "revenue_per_hour": round(float(bp["revenue_per_hour"]), 2),
                "revenue_per_employee": round(float(bp["revenue_per_employee"]), 2),
            }

            # Dominant shift
            if not br.empty:
                dominant = br.sort_values("shift_share", ascending=False).iloc[0]
                current["dominant_shift"] = dominant["shift_type"]
                current["dominant_shift_share"] = float(dominant["shift_share"])

            # Recommendations
            rec: Dict[str, Any] = {}
            for _, row in br.iterrows():
                rec[row["shift_type"]] = {
                    "recommended_staff": int(row["recommended_staff_per_day"]),
                    "recommended_hours": float(row["recommended_shift_hours"]),
                    "demand_multiplier": float(row["demand_multiplier"]),
                }

            # Weekly schedule summary
            if not bs.empty:
                weekday = bs[bs["is_weekend"] == 0]
                weekend = bs[bs["is_weekend"] == 1]
                weekly = {
                    "weekday_recommended_employees": int(weekday["recommended_employees"].mode().iloc[0]) if not weekday.empty else 0,
                    "weekend_recommended_employees": int(weekend["recommended_employees"].mode().iloc[0]) if not weekend.empty else 0,
                    "peak_day": bs.loc[bs["observed_avg_total_hours"].idxmax(), "day_of_week"] if not bs.empty else None,
                }
            else:
                weekly = {}

            # Forecast-driven hiring need
            fc_info = forecast.get("branches", {}).get(branch, {})
            growth = fc_info.get("forecast_vs_historical", 0) or 0
            if growth > 0.3:
                hiring = {
                    "action": "HIRE",
                    "reason": f"Demand projected to grow {growth:.0%}. Current staff may be insufficient.",
                    "suggested_additional": max(1, math.ceil(bp["num_employees"] * growth * 0.5)),
                }
            elif growth < -0.2:
                hiring = {
                    "action": "OPTIMIZE",
                    "reason": f"Demand projected to decline {abs(growth):.0%}. Reduce overtime, not headcount.",
                    "suggested_additional": 0,
                }
            else:
                hiring = {
                    "action": "MAINTAIN",
                    "reason": "Demand is stable. Current staffing levels are adequate.",
                    "suggested_additional": 0,
                }

            summary["branch_staffing"][branch] = {
                "current": current,
                "shift_recommendations": rec,
                "weekly_schedule": weekly,
                "hiring_recommendation": hiring,
            }

        # Network-wide insight
        total_emp = int(profile["num_employees"].sum())
        avg_rev_per_emp = profile["revenue_per_employee"].mean()
        most_efficient = profile.loc[profile["revenue_per_employee"].idxmax(), "branch"]
        summary["network_insights"] = {
            "total_employees": total_emp,
            "avg_revenue_per_employee": round(float(avg_rev_per_emp), 2),
            "most_efficient_branch": most_efficient,
            "key_finding": (
                f"{most_efficient} generates the highest revenue per employee. "
                "Other branches should study its shift scheduling and operational practices."
            ),
        }

        return summary
