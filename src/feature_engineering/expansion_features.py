"""Feature engineering for Expansion Feasibility analysis.

Builds branch-level scorecards that compare existing branches across
financial, operational, and customer dimensions to evaluate whether
the business can support a new location and identify what a successful
branch profile looks like.

Key outputs:
  - branch_scorecard: multi-dimensional performance scores per branch
  - expansion_benchmarks: network-wide benchmarks and thresholds
  - branch_comparison: pairwise branch comparison matrix
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from ..config import CLEANED_DATA_DIR, FEATURES_DIR

logger = logging.getLogger(__name__)


def _min_max_scale(series: pd.Series) -> pd.Series:
    """Scale a series to [0, 1]. Returns 0 if constant."""
    rng = series.max() - series.min()
    if rng == 0:
        return pd.Series(0.0, index=series.index)
    return (series - series.min()) / rng


class ExpansionFeatureBuilder:
    """Build expansion-feasibility features from cleaned data."""

    def __init__(
        self,
        cleaned_dir: Path | str | None = None,
        output_dir: Path | str | None = None,
    ) -> None:
        self.cleaned_dir = Path(cleaned_dir) if cleaned_dir else CLEANED_DATA_DIR
        self.output_dir = Path(output_dir) if output_dir else FEATURES_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, *, save: bool = True) -> Dict[str, pd.DataFrame]:
        monthly = pd.read_csv(self.cleaned_dir / "monthly_sales.csv")
        division = pd.read_csv(self.cleaned_dir / "division_summary.csv")
        avg_sales = pd.read_csv(self.cleaned_dir / "avg_sales.csv")
        orders = pd.read_csv(self.cleaned_dir / "customer_orders.csv")
        tax = pd.read_csv(self.cleaned_dir / "tax_report.csv")
        attendance = pd.read_csv(self.cleaned_dir / "attendance.csv")

        monthly["date"] = pd.to_datetime(monthly["date"])

        scorecard = self._build_scorecard(monthly, division, avg_sales, orders, tax, attendance)
        benchmarks = self._build_benchmarks(scorecard)
        comparison = self._build_comparison(scorecard)

        results = {
            "branch_scorecard": scorecard,
            "expansion_benchmarks": benchmarks,
            "branch_comparison": comparison,
        }

        if save:
            for name, df in results.items():
                path = self.output_dir / f"expansion_{name}.csv"
                df.to_csv(path, index=False)
                logger.info("Saved %s -> %s (%d rows)", name, path, len(df))

        return results

      # Branch scorecard
  
    def _build_scorecard(
        self,
        monthly: pd.DataFrame,
        division: pd.DataFrame,
        avg_sales: pd.DataFrame,
        orders: pd.DataFrame,
        tax: pd.DataFrame,
        attendance: pd.DataFrame,
    ) -> pd.DataFrame:
        branches = sorted(monthly["branch"].unique())
        rows: List[Dict] = []

        for branch in branches:
            bm = monthly[monthly["branch"] == branch].sort_values("date")
            bd = division[division["branch"] == branch]
            ba = avg_sales[avg_sales["branch"] == branch]
            bo = orders[orders["branch"] == branch]
            bt = tax[tax["branch"] == branch]
            batt = attendance[attendance["branch"] == branch]

            # --- Revenue metrics ---
            total_revenue = bm["total"].sum()
            num_months = len(bm)
            avg_monthly_rev = total_revenue / num_months if num_months > 0 else 0.0

            # Recent momentum: last 2 months avg vs first 2 months avg
            if num_months >= 4:
                early = bm["total"].iloc[:2].mean()
                late = bm["total"].iloc[-2:].mean()
            elif num_months >= 2:
                early = bm["total"].iloc[0]
                late = bm["total"].iloc[-1]
            else:
                early = late = avg_monthly_rev
            momentum = (late - early) / early if early > 0 else 0.0

            # Revenue stability (inverse of CV — lower CV = more stable)
            cv = bm["total"].std() / bm["total"].mean() if bm["total"].mean() > 0 else 1.0

            # --- Tax contribution ---
            vat = bt["vat_11_pct"].sum() if not bt.empty else 0.0

            # --- Customer metrics ---
            total_customers = ba["num_customers"].sum() if not ba.empty else 0
            avg_ticket = (
                ba["sales"].sum() / total_customers
                if total_customers > 0
                else 0.0
            )

            # Channel diversity (number of distinct channels with customers)
            channels_active = len(ba[ba["num_customers"] > 0]) if not ba.empty else 0

            # Delivery penetration
            delivery_customers = bo["customer"].nunique() if not bo.empty else 0
            delivery_orders = int(bo["num_orders"].sum()) if not bo.empty else 0
            repeat_rate = (
                (bo["num_orders"] > 1).mean() if not bo.empty and len(bo) > 0 else 0.0
            )

            # Revenue per customer
            revenue_per_customer = total_revenue / total_customers if total_customers > 0 else 0.0

            # --- Operational metrics ---
            num_employees = batt["emp_id"].nunique() if not batt.empty else 0
            total_shifts = len(batt) if not batt.empty else 0
            avg_shift_hours = batt["work_duration_hours"].mean() if not batt.empty else 0.0
            revenue_per_employee = total_revenue / num_employees if num_employees > 0 else 0.0

            # --- Product mix ---
            items_row = bd[bd["division"] == "ITEMS"]
            if not items_row.empty:
                r = items_row.iloc[0]
                channel_total = r["delivery"] + r["table"] + r["take_away"]
                delivery_share = r["delivery"] / channel_total if channel_total > 0 else 0.0
                table_share = r["table"] / channel_total if channel_total > 0 else 0.0
                takeaway_share = r["take_away"] / channel_total if channel_total > 0 else 0.0
            else:
                delivery_share = table_share = takeaway_share = 0.0

            # Beverage vs food ratio (coffee + frappes + shakes + drinks vs ITEMS)
            bev_divs = {"Hot-Coffee Based", "Frappes", "Shakes", "Hot and Cold Drinks"}
            bev_total = bd[bd["division"].isin(bev_divs)]["total"].sum()
            items_total = items_row["total"].values[0] if not items_row.empty else 0.0
            beverage_ratio = bev_total / items_total if items_total > 0 else 0.0

            rows.append({
                "branch": branch,
                # Revenue
                "total_revenue": round(total_revenue, 2),
                "avg_monthly_revenue": round(avg_monthly_rev, 2),
                "momentum": round(momentum, 4),
                "revenue_cv": round(cv, 4),
                "vat_contribution": round(vat, 2),
                # Customers
                "total_customers": int(total_customers),
                "avg_ticket_size": round(avg_ticket, 2),
                "revenue_per_customer": round(revenue_per_customer, 2),
                "channels_active": channels_active,
                "delivery_customers": delivery_customers,
                "delivery_orders": delivery_orders,
                "repeat_rate": round(repeat_rate, 4),
                # Operations
                "num_employees": num_employees,
                "total_shifts": total_shifts,
                "avg_shift_hours": round(avg_shift_hours, 2),
                "revenue_per_employee": round(revenue_per_employee, 2),
                # Channel mix
                "delivery_share": round(delivery_share, 4),
                "table_share": round(table_share, 4),
                "takeaway_share": round(takeaway_share, 4),
                "beverage_ratio": round(beverage_ratio, 4),
            })

        scorecard = pd.DataFrame(rows)

        # Compute composite scores (0-1 scale, higher = better)
        scorecard["score_revenue"] = _min_max_scale(scorecard["avg_monthly_revenue"])
        scorecard["score_growth"] = _min_max_scale(scorecard["momentum"])
        scorecard["score_stability"] = 1 - _min_max_scale(scorecard["revenue_cv"])
        scorecard["score_customers"] = _min_max_scale(scorecard["total_customers"])
        scorecard["score_ticket"] = _min_max_scale(scorecard["avg_ticket_size"])
        scorecard["score_efficiency"] = _min_max_scale(scorecard["revenue_per_employee"])

        # Weighted composite: what makes a branch "expansion-worthy"
        weights = {
            "score_revenue": 0.20,
            "score_growth": 0.25,
            "score_stability": 0.15,
            "score_customers": 0.15,
            "score_ticket": 0.10,
            "score_efficiency": 0.15,
        }
        scorecard["composite_score"] = sum(
            scorecard[col] * w for col, w in weights.items()
        ).round(4)

        scorecard = scorecard.sort_values("composite_score", ascending=False).reset_index(drop=True)
        logger.info("Built scorecard for %d branches", len(scorecard))
        return scorecard

      # Network benchmarks
  
    def _build_benchmarks(self, scorecard: pd.DataFrame) -> pd.DataFrame:
        """Compute network-wide benchmarks for expansion planning."""
        numeric_cols = scorecard.select_dtypes(include=[np.number]).columns.tolist()

        stats = []
        for col in numeric_cols:
            stats.append({
                "metric": col,
                "mean": round(scorecard[col].mean(), 4),
                "median": round(scorecard[col].median(), 4),
                "min": round(scorecard[col].min(), 4),
                "max": round(scorecard[col].max(), 4),
                "std": round(scorecard[col].std(), 4),
                "best_branch": scorecard.loc[scorecard[col].idxmax(), "branch"],
                "worst_branch": scorecard.loc[scorecard[col].idxmin(), "branch"],
            })

        result = pd.DataFrame(stats)
        logger.info("Built benchmarks for %d metrics", len(result))
        return result

      # Pairwise branch comparison
  
    def _build_comparison(self, scorecard: pd.DataFrame) -> pd.DataFrame:
        """Pairwise comparison: for each pair, which branch is stronger and by how much."""
        branches = scorecard["branch"].tolist()
        score_cols = [c for c in scorecard.columns if c.startswith("score_")]
        rows = []

        for i, b1 in enumerate(branches):
            for b2 in branches[i + 1:]:
                r1 = scorecard[scorecard["branch"] == b1].iloc[0]
                r2 = scorecard[scorecard["branch"] == b2].iloc[0]

                wins_b1 = sum(1 for c in score_cols if r1[c] > r2[c])
                wins_b2 = sum(1 for c in score_cols if r2[c] > r1[c])

                comp1 = r1["composite_score"]
                comp2 = r2["composite_score"]
                gap = abs(comp1 - comp2)
                stronger = b1 if comp1 > comp2 else b2

                rows.append({
                    "branch_a": b1,
                    "branch_b": b2,
                    "dimension_wins_a": wins_b1,
                    "dimension_wins_b": wins_b2,
                    "composite_a": round(comp1, 4),
                    "composite_b": round(comp2, 4),
                    "composite_gap": round(gap, 4),
                    "stronger_branch": stronger,
                })

        result = pd.DataFrame(rows)
        logger.info("Built %d pairwise comparisons", len(result))
        return result
