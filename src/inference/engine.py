"""Inference engine — service layer that answers business questions.

This is the programmatic API that OpenClaw (or any other consumer) calls.
It loads pre-computed analytics results and provides typed query methods
for each of the five business objectives.

Usage:
    engine = InferenceEngine()
    engine.load()
    answer = engine.query("What combos should we offer?")
    # or use typed methods directly:
    combos = engine.get_combo_recommendations(branch="Conut Jnah", top_n=5)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from ..config import ANALYTICS_DIR, FEATURES_DIR

logger = logging.getLogger(__name__)


class InferenceEngine:
    """Load analytics outputs and serve business queries."""

    def __init__(
        self,
        analytics_dir: Path | str | None = None,
        features_dir: Path | str | None = None,
    ) -> None:
        self.analytics_dir = Path(analytics_dir) if analytics_dir else ANALYTICS_DIR
        self.features_dir = Path(features_dir) if features_dir else FEATURES_DIR
        self._data: Dict[str, Any] = {}

     # Loading
 
    def load(self) -> None:
        """Load all pre-computed analytics artifacts into memory."""
        a = self.analytics_dir

        self._data["combo_2item"] = self._read_csv(a / "combo_combo_recommendations_2item.csv")
        self._data["combo_3item"] = self._read_csv(a / "combo_combo_recommendations_3item.csv")
        self._data["combo_branch"] = self._read_csv(a / "combo_branch_combo_recommendations.csv")
        self._data["combo_summary"] = self._read_json(a / "combo_combo_summary.json")

        self._data["demand_forecasts"] = self._read_csv(a / "demand_branch_forecasts.csv")
        self._data["demand_summary"] = self._read_json(a / "demand_forecast_summary.json")

        self._data["expansion_assessment"] = self._read_csv(a / "expansion_assessment.csv")
        self._data["expansion_summary"] = self._read_json(a / "expansion_summary.json")

        self._data["staffing_recs"] = self._read_csv(a / "staffing_recommendations.csv")
        self._data["staffing_schedule"] = self._read_csv(a / "staffing_scheduling_guidelines.csv")
        self._data["staffing_summary"] = self._read_json(a / "staffing_summary.json")

        self._data["beverage_actions"] = self._read_csv(a / "beverage_strategy_actions.csv")
        self._data["beverage_opportunities"] = self._read_csv(a / "beverage_product_opportunities.csv")
        self._data["beverage_summary"] = self._read_json(a / "beverage_summary.json")

        logger.info("Inference engine loaded %d artifacts", len(self._data))

     # 1. Combo Optimization
 
    def get_combo_recommendations(
        self, *, branch: Optional[str] = None, top_n: int = 5
    ) -> Dict[str, Any]:
        """Return top combo recommendations, optionally filtered by branch."""
        if branch:
            df = self._data["combo_branch"]
            df = df[df["branch"].str.lower() == branch.lower()]
            combos = df.head(top_n).to_dict("records")
            return {
                "branch": branch,
                "top_combos": combos,
                "count": len(combos),
            }

        summary = self._data["combo_summary"]
        return {
            "top_2item_combos": summary.get("top_5_combos", [])[:top_n],
            "top_3item_combos": summary.get("top_3_triple_combos", [])[:top_n],
            "total_2item": summary.get("total_2item_combos"),
            "total_3item": summary.get("total_3item_combos"),
            "key_insight": summary.get("key_insight"),
        }

     # 2. Demand Forecasting
 
    def get_demand_forecast(
        self, *, branch: Optional[str] = None
    ) -> Dict[str, Any]:
        """Return demand forecasts, optionally for a specific branch."""
        summary = self._data["demand_summary"]

        if branch:
            branches = summary.get("branches", {})
            key = self._fuzzy_branch(branch, list(branches.keys()))
            if key is None:
                return {"error": f"Branch '{branch}' not found. Available: {list(branches.keys())}"}
            info = branches[key]
            return {
                "branch": key,
                "forecast_horizon": summary.get("forecast_horizon"),
                "jan_2026_forecast": info["jan_2026_forecast"],
                "q1_2026_total_forecast": info["q1_2026_total_forecast"],
                "avg_monthly_forecast": info["avg_monthly_forecast"],
                "historical_avg_monthly": info["historical_avg_monthly"],
                "forecast_vs_historical": info["forecast_vs_historical"],
                "confidence_interval": info.get("jan_2026_ci"),
            }

        return {
            "forecast_horizon": summary.get("forecast_horizon"),
            "method": summary.get("method"),
            "network_q1_total": summary.get("network_q1_2026_total"),
            "fastest_growing": summary.get("fastest_growing_branch"),
            "branches": summary.get("branches"),
        }

     # 3. Expansion Feasibility
 
    def get_expansion_assessment(self) -> Dict[str, Any]:
        """Return the expansion feasibility verdict and supporting data."""
        s = self._data["expansion_summary"]
        return {
            "verdict": s.get("verdict"),
            "rationale": s.get("rationale"),
            "network_health_score": s.get("network_health_score"),
            "strong_branches": s.get("strong_branches"),
            "weak_branches": s.get("weak_branches"),
            "risks": s.get("risks"),
            "new_branch_targets": s.get("new_branch_targets"),
            "model_to_replicate": s.get("model_to_replicate"),
            "location_insight": s.get("location_insight"),
        }

     # 4. Staffing Recommendations
 
    def get_staffing_recommendation(
        self, *, branch: Optional[str] = None
    ) -> Dict[str, Any]:
        """Return staffing recommendations, optionally for a specific branch."""
        summary = self._data["staffing_summary"]
        branch_data = summary.get("branch_staffing", {})

        if branch:
            key = self._fuzzy_branch(branch, list(branch_data.keys()))
            if key is None:
                return {"error": f"Branch '{branch}' not found. Available: {list(branch_data.keys())}"}
            info = branch_data[key]
            return {
                "branch": key,
                "current_staff": info["current"]["num_employees"],
                "avg_shift_hours": info["current"]["avg_shift_hours"],
                "revenue_per_employee": info["current"]["revenue_per_employee"],
                "dominant_shift": info["current"]["dominant_shift"],
                "shift_recommendations": info["shift_recommendations"],
                "weekly_schedule": info["weekly_schedule"],
                "hiring": info["hiring_recommendation"],
            }

        return {
            "branches_analyzed": summary.get("branches_analyzed"),
            "network_insights": summary.get("network_insights"),
            "branch_staffing": {
                b: {
                    "current_staff": d["current"]["num_employees"],
                    "hiring_action": d["hiring_recommendation"]["action"],
                    "suggested_additional": d["hiring_recommendation"]["suggested_additional"],
                }
                for b, d in branch_data.items()
            },
        }

     # 5. Beverage Growth Strategy
 
    def get_beverage_strategy(
        self, *, branch: Optional[str] = None
    ) -> Dict[str, Any]:
        """Return beverage growth strategy, optionally for a specific branch."""
        summary = self._data["beverage_summary"]

        if branch:
            strategies = summary.get("branch_strategies", {})
            key = self._fuzzy_branch(branch, list(strategies.keys()))
            if key is None:
                return {"error": f"Branch '{branch}' not found. Available: {list(strategies.keys())}"}
            info = strategies[key]
            actions_df = self._data["beverage_actions"]
            actions = actions_df[actions_df["branch"].str.lower() == key.lower()].to_dict("records")
            return {
                "branch": key,
                "num_actions": info["num_actions"],
                "estimated_uplift": info["estimated_uplift"],
                "products_to_add": info["products_to_add"],
                "products_to_boost": info["products_to_boost"],
                "top_priority": info["top_priority"],
                "actions": actions,
            }

        return {
            "network_overview": summary.get("network_overview"),
            "total_actions": summary.get("total_actions"),
            "total_product_opportunities": summary.get("total_product_opportunities"),
            "total_estimated_uplift": summary.get("total_estimated_uplift"),
            "key_insights": summary.get("key_insights"),
            "branch_strategies": summary.get("branch_strategies"),
        }

     # Natural-language query router
 
    QUERY_KEYWORDS = {
        "combo": ["combo", "bundle", "pair", "cross-sell", "upsell", "together"],
        "demand": ["demand", "forecast", "predict", "sales projection", "next month", "q1"],
        "expansion": ["expand", "new branch", "new location", "feasibility", "open"],
        "staffing": ["staff", "employee", "shift", "schedule", "hire", "hiring", "workforce"],
        "beverage": ["coffee", "milkshake", "frappe", "beverage", "drink", "growth strategy"],
    }

    def query(self, question: str, *, branch: Optional[str] = None) -> Dict[str, Any]:
        """Route a natural-language question to the appropriate handler."""
        q = question.lower()

        # Detect branch name in the question if not explicitly provided
        if branch is None:
            branch = self._extract_branch(q)

        scores = {
            topic: sum(1 for kw in keywords if kw in q)
            for topic, keywords in self.QUERY_KEYWORDS.items()
        }
        best_topic = max(scores, key=scores.get)  # type: ignore[arg-type]

        if scores[best_topic] == 0:
            return self._full_summary()

        dispatch = {
            "combo": lambda: self.get_combo_recommendations(branch=branch),
            "demand": lambda: self.get_demand_forecast(branch=branch),
            "expansion": lambda: self.get_expansion_assessment(),
            "staffing": lambda: self.get_staffing_recommendation(branch=branch),
            "beverage": lambda: self.get_beverage_strategy(branch=branch),
        }

        result = dispatch[best_topic]()
        result["_topic"] = best_topic
        result["_query"] = question
        return result

    def _full_summary(self) -> Dict[str, Any]:
        """Return a high-level summary across all objectives."""
        combo = self._data["combo_summary"]
        demand = self._data["demand_summary"]
        expansion = self._data["expansion_summary"]
        staffing = self._data["staffing_summary"]
        beverage = self._data["beverage_summary"]

        return {
            "_topic": "full_summary",
            "combo": {
                "top_combo": combo["top_5_combos"][0]["combo"] if combo.get("top_5_combos") else None,
                "total_combos": combo.get("total_2item_combos", 0) + combo.get("total_3item_combos", 0),
            },
            "demand": {
                "forecast_horizon": demand.get("forecast_horizon"),
                "network_q1_total": demand.get("network_q1_2026_total"),
                "fastest_growing": demand.get("fastest_growing_branch"),
            },
            "expansion": {
                "verdict": expansion.get("verdict"),
                "health_score": expansion.get("network_health_score"),
            },
            "staffing": {
                "total_employees": staffing.get("network_insights", {}).get("total_employees"),
                "most_efficient": staffing.get("network_insights", {}).get("most_efficient_branch"),
            },
            "beverage": {
                "best_branch": beverage.get("network_overview", {}).get("best_beverage_branch"),
                "total_uplift": beverage.get("total_estimated_uplift"),
            },
        }

     # Helpers
 
    KNOWN_BRANCHES = [
        "Conut - Tyre", "Conut Jnah", "Main Street Coffee", "Conut",
    ]

    def _extract_branch(self, text: str) -> Optional[str]:
        text_lower = text.lower()
        for b in self.KNOWN_BRANCHES:
            if b.lower() in text_lower:
                return b
        return None

    @staticmethod
    def _fuzzy_branch(query: str, candidates: List[str]) -> Optional[str]:
        q = query.lower().strip()
        for c in candidates:
            if c.lower() == q:
                return c
        for c in candidates:
            if q in c.lower() or c.lower() in q:
                return c
        return None

    @staticmethod
    def _read_csv(path: Path) -> pd.DataFrame:
        if path.exists():
            return pd.read_csv(path)
        logger.warning("File not found: %s", path)
        return pd.DataFrame()

    @staticmethod
    def _read_json(path: Path) -> Dict[str, Any]:
        if path.exists():
            with open(path) as f:
                return json.load(f)
        logger.warning("File not found: %s", path)
        return {}
