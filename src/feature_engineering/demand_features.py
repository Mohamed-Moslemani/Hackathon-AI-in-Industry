"""Feature engineering for Demand Forecasting by Branch.

Builds time-series and cross-sectional features from monthly sales,
division summaries, average sales, and customer order data to support
per-branch demand forecasting.

Key outputs:
  - branch_monthly_features: enriched monthly time-series per branch
  - branch_profile: static branch-level profile with channel mix, growth, etc.
  - forecast_dataset: ready-to-model dataset with lag/trend features
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from ..config import CLEANED_DATA_DIR, FEATURES_DIR

logger = logging.getLogger(__name__)


class DemandFeatureBuilder:
    """Build demand-forecasting features from cleaned data."""

    def __init__(
        self,
        cleaned_dir: Path | str | None = None,
        output_dir: Path | str | None = None,
    ) -> None:
        self.cleaned_dir = Path(cleaned_dir) if cleaned_dir else CLEANED_DATA_DIR
        self.output_dir = Path(output_dir) if output_dir else FEATURES_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, *, save: bool = True) -> Dict[str, pd.DataFrame]:
        """Execute the full demand feature pipeline."""
        monthly = pd.read_csv(self.cleaned_dir / "monthly_sales.csv")
        division = pd.read_csv(self.cleaned_dir / "division_summary.csv")
        avg_sales = pd.read_csv(self.cleaned_dir / "avg_sales.csv")
        orders = pd.read_csv(self.cleaned_dir / "customer_orders.csv")

        monthly["date"] = pd.to_datetime(monthly["date"])

        branch_monthly = self._build_monthly_features(monthly)
        branch_profile = self._build_branch_profile(monthly, division, avg_sales, orders)
        forecast_ds = self._build_forecast_dataset(branch_monthly, branch_profile)

        results = {
            "branch_monthly_features": branch_monthly,
            "branch_profile": branch_profile,
            "forecast_dataset": forecast_ds,
        }

        if save:
            for name, df in results.items():
                path = self.output_dir / f"demand_{name}.csv"
                df.to_csv(path, index=False)
                logger.info("Saved %s -> %s (%d rows)", name, path, len(df))

        return results

    # ------------------------------------------------------------------
    # Monthly time-series features
    # ------------------------------------------------------------------

    def _build_monthly_features(self, monthly: pd.DataFrame) -> pd.DataFrame:
        """Enrich the monthly sales with lag, growth, and trend features."""
        df = monthly.sort_values(["branch", "date"]).copy()

        features = []
        for branch, grp in df.groupby("branch"):
            grp = grp.sort_values("date").reset_index(drop=True)

            # Lag features
            grp["lag_1"] = grp["total"].shift(1)
            grp["lag_2"] = grp["total"].shift(2)

            # Month-over-month growth rate
            grp["mom_growth"] = grp["total"].pct_change()

            # Rolling averages (2-month and 3-month where possible)
            grp["rolling_avg_2"] = grp["total"].rolling(2, min_periods=1).mean()
            grp["rolling_avg_3"] = grp["total"].rolling(3, min_periods=1).mean()

            # Cumulative total
            grp["cumulative_total"] = grp["total"].cumsum()

            # Share of total across all months for this branch
            branch_total = grp["total"].sum()
            grp["month_share"] = grp["total"] / branch_total if branch_total > 0 else 0.0

            # Simple linear trend index (0, 1, 2, ...)
            grp["trend_idx"] = range(len(grp))

            # Month-of-year for seasonality
            grp["month_of_year"] = grp["date"].dt.month

            features.append(grp)

        result = pd.concat(features, ignore_index=True)
        logger.info("Built monthly features: %d rows x %d cols", len(result), len(result.columns))
        return result

    # ------------------------------------------------------------------
    # Static branch profile
    # ------------------------------------------------------------------

    def _build_branch_profile(
        self,
        monthly: pd.DataFrame,
        division: pd.DataFrame,
        avg_sales: pd.DataFrame,
        orders: pd.DataFrame,
    ) -> pd.DataFrame:
        """Build a cross-sectional profile for each branch."""
        profiles = []

        for branch in monthly["branch"].unique():
            bm = monthly[monthly["branch"] == branch].sort_values("date")
            bd = division[division["branch"] == branch]
            ba = avg_sales[avg_sales["branch"] == branch]
            bo = orders[orders["branch"] == branch]

            total_revenue = bm["total"].sum()
            num_months = len(bm)
            avg_monthly = total_revenue / num_months if num_months > 0 else 0.0

            # Growth: compare last month to first month
            if num_months >= 2:
                first_val = bm["total"].iloc[0]
                last_val = bm["total"].iloc[-1]
                overall_growth = (last_val - first_val) / first_val if first_val > 0 else 0.0
                avg_mom_growth = bm["total"].pct_change().mean()
            else:
                overall_growth = 0.0
                avg_mom_growth = 0.0

            # Volatility (coefficient of variation)
            cv = bm["total"].std() / bm["total"].mean() if bm["total"].mean() > 0 else 0.0

            # Peak month
            peak_month = bm.loc[bm["total"].idxmax(), "month"] if num_months > 0 else None

            # Channel mix from division summary (ITEMS division = main product sales)
            items_row = bd[bd["division"] == "ITEMS"]
            if not items_row.empty:
                row = items_row.iloc[0]
                channel_total = row["delivery"] + row["table"] + row["take_away"]
                delivery_share = row["delivery"] / channel_total if channel_total > 0 else 0.0
                table_share = row["table"] / channel_total if channel_total > 0 else 0.0
                takeaway_share = row["take_away"] / channel_total if channel_total > 0 else 0.0
            else:
                delivery_share = table_share = takeaway_share = 0.0

            # Customer metrics from avg_sales
            total_customers = ba["num_customers"].sum() if not ba.empty else 0
            weighted_avg_ticket = (
                (ba["sales"].sum() / ba["num_customers"].sum())
                if not ba.empty and ba["num_customers"].sum() > 0
                else 0.0
            )

            # Delivery customer metrics from orders
            delivery_customers = bo["customer"].nunique() if not bo.empty else 0
            delivery_orders = bo["num_orders"].sum() if not bo.empty else 0
            avg_orders_per_customer = (
                delivery_orders / delivery_customers
                if delivery_customers > 0
                else 0.0
            )

            # Division diversity: how many product divisions have sales > 0
            active_divisions = len(bd[bd["total"] > 0])

            profiles.append({
                "branch": branch,
                "total_revenue": round(total_revenue, 2),
                "num_months": num_months,
                "avg_monthly_revenue": round(avg_monthly, 2),
                "overall_growth": round(overall_growth, 4),
                "avg_mom_growth": round(avg_mom_growth, 4),
                "revenue_cv": round(cv, 4),
                "peak_month": peak_month,
                "delivery_share": round(delivery_share, 4),
                "table_share": round(table_share, 4),
                "takeaway_share": round(takeaway_share, 4),
                "total_customers": int(total_customers),
                "avg_ticket_size": round(weighted_avg_ticket, 2),
                "delivery_customers": delivery_customers,
                "delivery_orders": int(delivery_orders),
                "avg_orders_per_delivery_customer": round(avg_orders_per_customer, 2),
                "active_divisions": active_divisions,
            })

        result = pd.DataFrame(profiles)
        logger.info("Built branch profiles for %d branches", len(result))
        return result

    # ------------------------------------------------------------------
    # Forecast-ready dataset
    # ------------------------------------------------------------------

    def _build_forecast_dataset(
        self,
        branch_monthly: pd.DataFrame,
        branch_profile: pd.DataFrame,
    ) -> pd.DataFrame:
        """Merge time-series and static features into a model-ready dataset."""
        df = branch_monthly.merge(branch_profile, on="branch", how="left", suffixes=("", "_profile"))

        # Normalized revenue (relative to branch average) for cross-branch comparability
        df["revenue_ratio"] = df["total"] / df["avg_monthly_revenue"]

        # Branch-level encoding: revenue rank among branches
        branch_rank = (
            branch_profile
            .sort_values("total_revenue", ascending=False)
            .reset_index(drop=True)
        )
        branch_rank["branch_revenue_rank"] = range(1, len(branch_rank) + 1)
        df = df.merge(
            branch_rank[["branch", "branch_revenue_rank"]],
            on="branch",
            how="left",
        )

        # Drop columns that would leak future info or aren't useful for modeling
        drop_cols = ["cumulative_total", "peak_month"]
        df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

        logger.info("Built forecast dataset: %d rows x %d cols", len(df), len(df.columns))
        return df
