"""Coffee and Milkshake Growth Strategy.

Turns beverage feature data into a prioritized action plan per branch:
  1. Identifies the biggest revenue-share gaps vs. the best-performing branch
  2. Recommends specific products to introduce or promote
  3. Estimates revenue uplift potential
  4. Produces a ranked strategy list per branch

Outputs:
  - strategy_actions: prioritized action items per branch
  - product_opportunities: specific products to add or boost
  - beverage_summary: JSON executive summary
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ..config import FEATURES_DIR

logger = logging.getLogger(__name__)

RESULTS_DIR_NAME = "analytics"


class BeverageStrategist:
    """Generate coffee & milkshake growth strategies per branch."""

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
        summary_df = pd.read_csv(self.features_dir / "beverage_beverage_branch_summary.csv")
        gaps = pd.read_csv(self.features_dir / "beverage_growth_gaps.csv")
        products = pd.read_csv(self.features_dir / "beverage_product_performance.csv")

        actions = self._build_strategy_actions(summary_df, gaps)
        opportunities = self._build_product_opportunities(products, gaps)
        exec_summary = self._build_summary(summary_df, gaps, actions, opportunities)

        results: Dict[str, Any] = {
            "strategy_actions": actions,
            "product_opportunities": opportunities,
            "beverage_summary": exec_summary,
        }

        if save:
            actions.to_csv(self.output_dir / "beverage_strategy_actions.csv", index=False)
            opportunities.to_csv(self.output_dir / "beverage_product_opportunities.csv", index=False)
            with open(self.output_dir / "beverage_summary.json", "w") as f:
                json.dump(exec_summary, f, indent=2)
            logger.info("Saved beverage strategy outputs to %s", self.output_dir)

        return results

       # Strategy actions
   
    def _build_strategy_actions(
        self, summary_df: pd.DataFrame, gaps: pd.DataFrame
    ) -> pd.DataFrame:
        rows: List[Dict] = []
        priority_counter = 0

        for _, gap in gaps.iterrows():
            branch = gap["branch"]
            category = gap["category"]
            share_gap = gap["share_gap_vs_best"]
            qty_gap = gap["qty_gap"]
            missing = gap["missing_products"]
            benchmark = gap["benchmark_branch"]

            # Get current and benchmark revenue
            current_row = summary_df[
                (summary_df["branch"] == branch) & (summary_df["category"] == category)
            ]
            bench_row = summary_df[
                (summary_df["branch"] == benchmark) & (summary_df["category"] == category)
            ]

            current_rev = current_row["total_revenue"].values[0] if not current_row.empty else 0
            bench_rev = bench_row["total_revenue"].values[0] if not bench_row.empty else 0
            current_qty = current_row["total_qty"].values[0] if not current_row.empty else 0
            bench_qty = bench_row["total_qty"].values[0] if not bench_row.empty else 0

            # Estimated uplift if branch matched 75% of benchmark performance
            target_qty = current_qty + qty_gap * 0.75
            avg_price = current_row["avg_unit_price"].values[0] if not current_row.empty else 0
            estimated_uplift = qty_gap * 0.75 * avg_price if qty_gap > 0 else 0

            # Priority: larger share gap = higher priority
            priority_counter += 1

            # Action type
            if missing > 0 and qty_gap > 20:
                action = "INTRODUCE_AND_PROMOTE"
                description = (
                    f"Add {missing} missing {category} products and run promotions "
                    f"to close the {qty_gap:.0f}-unit gap vs {benchmark}."
                )
            elif missing > 0:
                action = "INTRODUCE"
                description = (
                    f"Add {missing} missing {category} products from {benchmark}'s menu."
                )
            elif qty_gap > 20:
                action = "PROMOTE"
                description = (
                    f"Run targeted promotions on existing {category} products "
                    f"to close the {qty_gap:.0f}-unit gap vs {benchmark}."
                )
            else:
                action = "OPTIMIZE"
                description = (
                    f"Fine-tune {category} pricing and placement to close "
                    f"the small gap vs {benchmark}."
                )

            rows.append({
                "priority": priority_counter,
                "branch": branch,
                "category": category,
                "action_type": action,
                "description": description,
                "current_revenue_share": round(gap["current_revenue_share"], 6),
                "target_revenue_share": round(gap["best_branch_share"] * 0.75, 6),
                "share_gap_vs_best": round(share_gap, 6),
                "current_qty": current_qty,
                "target_qty": round(target_qty, 0),
                "estimated_revenue_uplift": round(estimated_uplift, 2),
                "missing_products": missing,
                "benchmark_branch": benchmark,
            })

        result = pd.DataFrame(rows)
        logger.info("Built %d strategy actions", len(result))
        return result

       # Product-level opportunities
   
    def _build_product_opportunities(
        self, products: pd.DataFrame, gaps: pd.DataFrame
    ) -> pd.DataFrame:
        rows: List[Dict] = []

        branches = products["branch"].unique()
        all_products_by_cat: Dict[str, set] = {}
        for cat in ["coffee", "milkshake", "frappe"]:
            all_products_by_cat[cat] = set(
                products[products["bev_category"] == cat]["description"].unique()
            )

        for branch in sorted(branches):
            bp = products[products["branch"] == branch]

            for cat in ["coffee", "milkshake", "frappe"]:
                cat_products = bp[bp["bev_category"] == cat]
                branch_items = set(cat_products["description"].unique())
                all_items = all_products_by_cat[cat]
                missing = all_items - branch_items

                # Underperforming existing products (below network average)
                underperformers = cat_products[cat_products["qty_vs_network_avg"] < -0.3]

                for _, prod in underperformers.iterrows():
                    rows.append({
                        "branch": branch,
                        "category": cat,
                        "product": prod["description"],
                        "opportunity_type": "BOOST_EXISTING",
                        "current_qty": prod["qty"],
                        "network_avg_qty": round(prod["network_qty"] / prod["branches_sold"], 1),
                        "qty_gap_pct": round(prod["qty_vs_network_avg"] * 100, 1),
                        "current_revenue": round(prod["total_amount"], 2),
                        "unit_price": round(prod["unit_price"], 2),
                    })

                # Missing products — get their network stats
                for item in sorted(missing):
                    net = products[products["description"] == item]
                    if net.empty:
                        continue
                    net_agg = net.iloc[0]
                    avg_qty = net["qty"].mean()
                    avg_rev = net["total_amount"].mean()

                    rows.append({
                        "branch": branch,
                        "category": cat,
                        "product": item,
                        "opportunity_type": "ADD_NEW",
                        "current_qty": 0,
                        "network_avg_qty": round(avg_qty, 1),
                        "qty_gap_pct": -100.0,
                        "current_revenue": 0.0,
                        "unit_price": round(float(net_agg["unit_price"]), 2),
                    })

        result = pd.DataFrame(rows)
        if not result.empty:
            result = result.sort_values(
                ["branch", "category", "opportunity_type", "qty_gap_pct"]
            ).reset_index(drop=True)

        logger.info("Identified %d product opportunities", len(result))
        return result

       # Executive summary
   
    def _build_summary(
        self,
        summary_df: pd.DataFrame,
        gaps: pd.DataFrame,
        actions: pd.DataFrame,
        opportunities: pd.DataFrame,
    ) -> Dict[str, Any]:
        # Network-wide beverage performance
        all_bev = summary_df[summary_df["category"] == "all_beverages"]
        best_bev_branch = all_bev.loc[all_bev["revenue_share"].idxmax(), "branch"]
        worst_bev_branch = all_bev.loc[all_bev["revenue_share"].idxmin(), "branch"]

        exec_summary: Dict[str, Any] = {
            "network_overview": {
                "best_beverage_branch": best_bev_branch,
                "best_beverage_share": round(float(all_bev["revenue_share"].max()), 4),
                "worst_beverage_branch": worst_bev_branch,
                "worst_beverage_share": round(float(all_bev["revenue_share"].min()), 4),
                "network_avg_beverage_share": round(float(all_bev["revenue_share"].mean()), 4),
            },
            "total_actions": len(actions),
            "total_product_opportunities": len(opportunities),
            "total_estimated_uplift": round(float(actions["estimated_revenue_uplift"].sum()), 2),
        }

        # Per-branch strategy summary
        branch_strategies: Dict[str, Any] = {}
        for branch in sorted(actions["branch"].unique()):
            ba = actions[actions["branch"] == branch]
            bo = opportunities[opportunities["branch"] == branch]

            top_action = ba.iloc[0] if not ba.empty else None
            branch_strategies[branch] = {
                "num_actions": len(ba),
                "estimated_uplift": round(float(ba["estimated_revenue_uplift"].sum()), 2),
                "products_to_add": int(bo[bo["opportunity_type"] == "ADD_NEW"].shape[0]),
                "products_to_boost": int(bo[bo["opportunity_type"] == "BOOST_EXISTING"].shape[0]),
                "top_priority": {
                    "category": top_action["category"] if top_action is not None else None,
                    "action": top_action["action_type"] if top_action is not None else None,
                    "description": top_action["description"] if top_action is not None else None,
                } if top_action is not None else None,
            }

        exec_summary["branch_strategies"] = branch_strategies

        # Key insights
        insights = []

        # Biggest gap
        if not gaps.empty:
            biggest = gaps.iloc[0]
            insights.append(
                f"{biggest['branch']} has the largest coffee/milkshake gap: "
                f"{biggest['category']} revenue share is {biggest['current_revenue_share']:.1%} "
                f"vs best-in-class {biggest['best_branch_share']:.1%} at {biggest['benchmark_branch']}."
            )

        # Benchmark branch
        insights.append(
            f"{best_bev_branch} is the beverage leader with {all_bev['revenue_share'].max():.1%} "
            f"of revenue from beverages. Other branches should benchmark against it."
        )

        # Quick wins
        add_new = opportunities[opportunities["opportunity_type"] == "ADD_NEW"]
        if not add_new.empty:
            total_new = add_new["product"].nunique()
            insights.append(
                f"There are {total_new} unique beverage products that could be introduced "
                f"at underperforming branches for immediate menu expansion."
            )

        exec_summary["key_insights"] = insights

        return exec_summary
