"""Feature engineering for Shift Staffing Estimation.

Builds shift-level and branch-level staffing features from attendance logs
and monthly sales data to support estimating required employees per shift.

Key outputs:
  - shift_patterns: per-employee shift-level features (hours, time-of-day, day-of-week)
  - branch_staffing_profile: branch-level staffing summary with demand ratios
  - daily_coverage: daily staffing coverage per branch (employees on duty, total hours)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from ..config import CLEANED_DATA_DIR, FEATURES_DIR

logger = logging.getLogger(__name__)

SHIFT_BINS = [0, 6, 12, 17, 22, 24]
SHIFT_LABELS = ["overnight", "morning", "afternoon", "evening", "late_night"]


class StaffingFeatureBuilder:
    """Build staffing-estimation features from cleaned data."""

    def __init__(
        self,
        cleaned_dir: Path | str | None = None,
        output_dir: Path | str | None = None,
    ) -> None:
        self.cleaned_dir = Path(cleaned_dir) if cleaned_dir else CLEANED_DATA_DIR
        self.output_dir = Path(output_dir) if output_dir else FEATURES_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, *, save: bool = True) -> Dict[str, pd.DataFrame]:
        attendance = pd.read_csv(self.cleaned_dir / "attendance.csv")
        monthly = pd.read_csv(self.cleaned_dir / "monthly_sales.csv")

        attendance["punch_in"] = pd.to_datetime(attendance["punch_in"])
        attendance["punch_out"] = pd.to_datetime(attendance["punch_out"])
        monthly["date"] = pd.to_datetime(monthly["date"])

        shift_patterns = self._build_shift_patterns(attendance)
        branch_profile = self._build_branch_staffing_profile(attendance, monthly)
        daily_coverage = self._build_daily_coverage(attendance)

        results = {
            "shift_patterns": shift_patterns,
            "branch_staffing_profile": branch_profile,
            "daily_coverage": daily_coverage,
        }

        if save:
            for name, df in results.items():
                path = self.output_dir / f"staffing_{name}.csv"
                df.to_csv(path, index=False)
                logger.info("Saved %s -> %s (%d rows)", name, path, len(df))

        return results

    # ------------------------------------------------------------------
    # Shift-level patterns
    # ------------------------------------------------------------------

    def _build_shift_patterns(self, att: pd.DataFrame) -> pd.DataFrame:
        df = att.copy()

        df["date"] = df["punch_in"].dt.date
        df["day_of_week"] = df["punch_in"].dt.day_name()
        df["day_num"] = df["punch_in"].dt.dayofweek  # 0=Mon, 6=Sun
        df["hour_in"] = df["punch_in"].dt.hour
        df["hour_out"] = df["punch_out"].dt.hour

        # Classify shift by punch-in hour
        df["shift_type"] = pd.cut(
            df["hour_in"],
            bins=SHIFT_BINS,
            labels=SHIFT_LABELS,
            right=False,
            ordered=False,
        )

        df["is_weekend"] = df["day_num"].isin([5, 6]).astype(int)

        # Overnight flag: punch-out date > punch-in date
        df["crosses_midnight"] = (
            df["punch_out"].dt.date > df["punch_in"].dt.date
        ).astype(int)

        keep_cols = [
            "emp_id", "emp_name", "branch", "date", "day_of_week", "day_num",
            "hour_in", "hour_out", "shift_type", "is_weekend",
            "crosses_midnight", "work_duration_hours",
        ]
        result = df[keep_cols].copy()

        logger.info("Built shift patterns: %d shifts", len(result))
        return result

    # ------------------------------------------------------------------
    # Branch staffing profile
    # ------------------------------------------------------------------

    def _build_branch_staffing_profile(
        self, att: pd.DataFrame, monthly: pd.DataFrame
    ) -> pd.DataFrame:
        rows: List[Dict] = []

        # Only branches that have attendance data
        branches_with_att = att["branch"].dropna().unique()

        for branch in branches_with_att:
            ba = att[att["branch"] == branch]
            bm = monthly[monthly["branch"] == branch]

            num_employees = ba["emp_id"].nunique()
            total_shifts = len(ba)
            total_hours = ba["work_duration_hours"].sum()
            avg_shift_hours = ba["work_duration_hours"].mean()
            median_shift_hours = ba["work_duration_hours"].median()
            std_shift_hours = ba["work_duration_hours"].std()

            # Shifts per employee
            shifts_per_emp = ba.groupby("emp_id").size()
            avg_shifts_per_emp = shifts_per_emp.mean()
            min_shifts = shifts_per_emp.min()
            max_shifts = shifts_per_emp.max()

            # Time-of-day distribution
            hours_in = ba["punch_in"].dt.hour
            shift_types = pd.cut(hours_in, bins=SHIFT_BINS, labels=SHIFT_LABELS, right=False, ordered=False)
            shift_dist = shift_types.value_counts(normalize=True)

            # Day-of-week distribution
            dow = ba["punch_in"].dt.dayofweek
            weekend_share = (dow.isin([5, 6])).mean()

            # Unique working days
            working_days = ba["punch_in"].dt.date.nunique()
            avg_employees_per_day = total_shifts / working_days if working_days > 0 else 0.0

            # Revenue-to-staffing ratios
            dec_revenue = bm[bm["date"].dt.month == 12]["total"].sum()
            revenue_per_hour = dec_revenue / total_hours if total_hours > 0 else 0.0
            revenue_per_shift = dec_revenue / total_shifts if total_shifts > 0 else 0.0
            revenue_per_employee = dec_revenue / num_employees if num_employees > 0 else 0.0

            rows.append({
                "branch": branch,
                "num_employees": num_employees,
                "total_shifts": total_shifts,
                "total_hours": round(total_hours, 2),
                "avg_shift_hours": round(avg_shift_hours, 2),
                "median_shift_hours": round(median_shift_hours, 2),
                "std_shift_hours": round(std_shift_hours, 2),
                "avg_shifts_per_employee": round(avg_shifts_per_emp, 2),
                "min_shifts_per_employee": int(min_shifts),
                "max_shifts_per_employee": int(max_shifts),
                "working_days": working_days,
                "avg_employees_per_day": round(avg_employees_per_day, 2),
                "weekend_share": round(weekend_share, 4),
                "morning_share": round(shift_dist.get("morning", 0), 4),
                "afternoon_share": round(shift_dist.get("afternoon", 0), 4),
                "evening_share": round(shift_dist.get("evening", 0), 4),
                "overnight_share": round(shift_dist.get("overnight", 0), 4),
                "dec_revenue": round(dec_revenue, 2),
                "revenue_per_hour": round(revenue_per_hour, 2),
                "revenue_per_shift": round(revenue_per_shift, 2),
                "revenue_per_employee": round(revenue_per_employee, 2),
            })

        result = pd.DataFrame(rows)
        logger.info("Built staffing profiles for %d branches", len(result))
        return result

    # ------------------------------------------------------------------
    # Daily coverage
    # ------------------------------------------------------------------

    def _build_daily_coverage(self, att: pd.DataFrame) -> pd.DataFrame:
        """Per-branch, per-day: how many employees worked and total hours."""
        df = att.copy()
        df["date"] = df["punch_in"].dt.date
        df["day_of_week"] = df["punch_in"].dt.day_name()
        df["day_num"] = df["punch_in"].dt.dayofweek

        daily = (
            df.groupby(["branch", "date"])
            .agg(
                employees_on_duty=("emp_id", "nunique"),
                total_shifts=("emp_id", "count"),
                total_hours=("work_duration_hours", "sum"),
                avg_shift_hours=("work_duration_hours", "mean"),
                earliest_in=("punch_in", "min"),
                latest_out=("punch_out", "max"),
            )
            .reset_index()
        )

        daily["day_of_week"] = pd.to_datetime(daily["date"]).dt.day_name()
        daily["day_num"] = pd.to_datetime(daily["date"]).dt.dayofweek
        daily["is_weekend"] = daily["day_num"].isin([5, 6]).astype(int)

        # Coverage span: hours between earliest in and latest out
        daily["coverage_span_hours"] = (
            (daily["latest_out"] - daily["earliest_in"]).dt.total_seconds() / 3600
        ).round(2)

        daily["total_hours"] = daily["total_hours"].round(2)
        daily["avg_shift_hours"] = daily["avg_shift_hours"].round(2)

        daily.drop(columns=["earliest_in", "latest_out"], inplace=True)

        logger.info("Built daily coverage: %d branch-days", len(daily))
        return daily
