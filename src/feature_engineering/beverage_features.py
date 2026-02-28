"""Feature engineering for Coffee and Milkshake Growth Strategy.

Analyzes coffee and milkshake sales across branches to identify growth
opportunities, underperforming locations, and product-level gaps.

Key outputs:
  - beverage_branch_summary: branch-level coffee/milkshake/other beverage metrics
  - product_performance: per-product sales with category tags and branch ranking
  - growth_gaps: specific opportunities where a branch underperforms the network
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import pandas as pd

from ..config import CLEANED_DATA_DIR, FEATURES_DIR

logger = logging.getLogger(__name__)

COFFEE_KEYWORDS: List[str] = [
    "COFFEE", "ESPRESSO", "LATTE", "CAPPUCCINO", "MOCHA", "AMERICANO",
    "MACCHIATO", "FLAT WHITE", "AFFOGATO",
]

MILKSHAKE_KEYWORDS: List[str] = [
    "MILKSHAKE",
]

FRAPPE_KEYWORDS: List[str] = [
    "FRAPPE",
]

BEVERAGE_DIVISIONS: Set[str] = {
    "Hot-Coffee Based", "Frappes", "Shakes", "Hot and Cold Drinks", "Bev Add-ons",
}


def _classify_beverage(desc: str, division: str) -> str:
    """Classify an item into coffee / milkshake / frappe / other_beverage / non_beverage."""
    upper = desc.upper()
    if any(kw in upper for kw in COFFEE_KEYWORDS):
        return "coffee"
    if any(kw in upper for kw in MILKSHAKE_KEYWORDS):
        return "milkshake"
    if any(kw in upper for kw in FRAPPE_KEYWORDS):
        return "frappe"
    if division in BEVERAGE_DIVISIONS:
        return "other_beverage"
    return "non_beverage"


class BeverageFeatureBuilder:
    """Build coffee & milkshake growth features from cleaned data."""

    def __init__(
        self,
        cleaned_dir: Path | str | None = None,
        output_dir: Path | str | None = None,
    ) -> None:
        self.cleaned_dir = Path(cleaned_dir) if cleaned_dir else CLEANED_DATA_DIR
        self.output_dir = Path(output_dir) if output_dir else FEATURES_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, *, save: bool = True) -> Dict[str, pd.DataFrame]:
        items = pd.read_csv(self.cleaned_dir / "sales_by_item.csv")
        division = pd.read_csv(self.cleaned_dir / "division_summary.csv")
        avg_sales = pd.read_csv(self.cleaned_dir / "avg_sales.csv")

        items["bev_category"] = items.apply(
            lambda r: _classify_beverage(r["description"], r["division"]), axis=1
        )

        branch_summary = self._build_branch_summary(items, division, avg_sales)
        product_perf = self._build_product_performance(items)
        gaps = self._build_growth_gaps(items, branch_summary)

        results = {
            "beverage_branch_summary": branch_summary,
            "product_performance": product_perf,
            "growth_gaps": gaps,
        }

        if save:
            for name, df in results.items():
                path = self.output_dir / f"beverage_{name}.csv"
                df.to_csv(path, index=False)
                logger.info("Saved %s -> %s (%d rows)", name, path, len(df))

        return results

    # ------------------------------------------------------------------
    # Branch-level beverage summary
    # ------------------------------------------------------------------

    def _build_branch_summary(
        self,
        items: pd.DataFrame,
        division: pd.DataFrame,
        avg_sales: pd.DataFrame,
    ) -> pd.DataFrame:
        rows: List[Dict] = []

        for branch in sorted(items["branch"].unique()):
            bi = items[items["branch"] == branch]
            bd = division[division["branch"] == branch]
            ba = avg_sales[avg_sales["branch"] == branch]

            # Total branch revenue (from ITEMS division)
            items_div = bd[bd["division"] == "ITEMS"]
            branch_total = items_div["total"].values[0] if not items_div.empty else bi["total_amount"].sum()

            # Category breakdowns
            for cat in ["coffee", "milkshake", "frappe", "other_beverage"]:
                cat_items = bi[bi["bev_category"] == cat]
                cat_qty = cat_items["qty"].sum()
                cat_rev = cat_items["total_amount"].sum()
                cat_products = cat_items["description"].nunique()
                cat_avg_price = cat_rev / cat_qty if cat_qty > 0 else 0.0

                rows.append({
                    "branch": branch,
                    "category": cat,
                    "num_products": int(cat_products),
                    "total_qty": cat_qty,
                    "total_revenue": round(cat_rev, 2),
                    "avg_unit_price": round(cat_avg_price, 2),
                    "revenue_share": round(cat_rev / branch_total, 6) if branch_total > 0 else 0.0,
                    "qty_per_product": round(cat_qty / cat_products, 2) if cat_products > 0 else 0.0,
                })

            # Total customers for penetration
            total_cust = ba["num_customers"].sum() if not ba.empty else 0

            # Beverage total
            bev_items = bi[bi["bev_category"] != "non_beverage"]
            bev_qty = bev_items["qty"].sum()
            bev_rev = bev_items["total_amount"].sum()

            rows.append({
                "branch": branch,
                "category": "all_beverages",
                "num_products": int(bev_items["description"].nunique()),
                "total_qty": bev_qty,
                "total_revenue": round(bev_rev, 2),
                "avg_unit_price": round(bev_rev / bev_qty, 2) if bev_qty > 0 else 0.0,
                "revenue_share": round(bev_rev / branch_total, 6) if branch_total > 0 else 0.0,
                "qty_per_product": round(bev_qty / bev_items["description"].nunique(), 2) if bev_items["description"].nunique() > 0 else 0.0,
            })

        result = pd.DataFrame(rows)
        logger.info("Built beverage branch summary: %d rows", len(result))
        return result

    # ------------------------------------------------------------------
    # Product-level performance
    # ------------------------------------------------------------------

    def _build_product_performance(self, items: pd.DataFrame) -> pd.DataFrame:
        """Per-product performance with cross-branch ranking."""
        bev = items[items["bev_category"].isin(["coffee", "milkshake", "frappe"])].copy()

        if bev.empty:
            return pd.DataFrame()

        # Rank within branch by revenue
        bev["branch_rank"] = bev.groupby("branch")["total_amount"].rank(
            ascending=False, method="dense"
        ).astype(int)

        # Network-wide aggregation per product
        network = (
            bev.groupby("description")
            .agg(
                network_qty=("qty", "sum"),
                network_revenue=("total_amount", "sum"),
                branches_sold=("branch", "nunique"),
                avg_qty_per_branch=("qty", "mean"),
            )
            .reset_index()
        )

        bev = bev.merge(network, on="description", how="left")

        # Penetration: is this product sold at all 4 branches?
        total_branches = items["branch"].nunique()
        bev["branch_penetration"] = bev["branches_sold"] / total_branches

        # Performance vs network average
        bev["qty_vs_network_avg"] = bev.apply(
            lambda r: (r["qty"] - r["avg_qty_per_branch"]) / r["avg_qty_per_branch"]
            if r["avg_qty_per_branch"] > 0 else 0.0,
            axis=1,
        ).round(4)

        keep_cols = [
            "branch", "description", "bev_category", "division", "group",
            "qty", "total_amount", "unit_price", "branch_rank",
            "network_qty", "network_revenue", "branches_sold",
            "branch_penetration", "qty_vs_network_avg",
        ]
        result = bev[keep_cols].sort_values(
            ["bev_category", "network_revenue"], ascending=[True, False]
        ).reset_index(drop=True)

        logger.info("Built product performance: %d rows", len(result))
        return result

    # ------------------------------------------------------------------
    # Growth gaps
    # ------------------------------------------------------------------

    def _build_growth_gaps(
        self, items: pd.DataFrame, branch_summary: pd.DataFrame
    ) -> pd.DataFrame:
        """Identify specific growth opportunities per branch."""
        gaps: List[Dict] = []
        branches = sorted(items["branch"].unique())

        for cat in ["coffee", "milkshake", "frappe"]:
            cat_summary = branch_summary[branch_summary["category"] == cat]
            if cat_summary.empty:
                continue

            network_avg_share = cat_summary["revenue_share"].mean()
            network_avg_qty = cat_summary["total_qty"].mean()
            best_branch = cat_summary.loc[cat_summary["revenue_share"].idxmax(), "branch"]
            best_share = cat_summary["revenue_share"].max()

            for _, row in cat_summary.iterrows():
                branch = row["branch"]
                share = row["revenue_share"]
                qty = row["total_qty"]

                share_gap = network_avg_share - share
                qty_gap = network_avg_qty - qty

                # Only flag if below network average
                if share_gap <= 0 and qty_gap <= 0:
                    continue

                # Estimate potential revenue uplift if branch matched best
                potential_uplift_share = best_share - share

                # Missing products: items sold at best branch but not here
                best_items = set(
                    items[
                        (items["branch"] == best_branch)
                        & (items["bev_category"] == cat)
                    ]["description"].unique()
                )
                branch_items = set(
                    items[
                        (items["branch"] == branch)
                        & (items["bev_category"] == cat)
                    ]["description"].unique()
                )
                missing = best_items - branch_items

                gaps.append({
                    "branch": branch,
                    "category": cat,
                    "current_revenue_share": round(share, 6),
                    "network_avg_share": round(network_avg_share, 6),
                    "best_branch_share": round(best_share, 6),
                    "share_gap_vs_avg": round(share_gap, 6),
                    "share_gap_vs_best": round(potential_uplift_share, 6),
                    "current_qty": qty,
                    "network_avg_qty": round(network_avg_qty, 2),
                    "qty_gap": round(qty_gap, 2),
                    "benchmark_branch": best_branch,
                    "missing_products": len(missing),
                    "missing_product_list": "; ".join(sorted(missing)) if missing else "",
                })

        result = pd.DataFrame(gaps)
        if not result.empty:
            result = result.sort_values("share_gap_vs_best", ascending=False).reset_index(drop=True)
        logger.info("Identified %d growth gap opportunities", len(result))
        return result
