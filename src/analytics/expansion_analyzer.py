"""Expansion Feasibility Analyzer.

Evaluates whether Conut should open a new branch by:
  1. Assessing network health (are existing branches strong enough?)
  2. Defining the ideal new-branch profile based on top performers
  3. Identifying risks and prerequisites
  4. Producing a go / conditional-go / no-go recommendation

Uses the branch scorecards, benchmarks, forecasts, and comparisons
from the feature engineering layer.

Outputs:
  - expansion_assessment: detailed feasibility report (DataFrame)
  - expansion_summary: JSON executive summary with verdict and rationale
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


class ExpansionAnalyzer:
    """Analyze expansion feasibility and produce recommendations."""

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
        scorecard = pd.read_csv(self.features_dir / "expansion_branch_scorecard.csv")
        benchmarks = pd.read_csv(self.features_dir / "expansion_expansion_benchmarks.csv")
        comparison = pd.read_csv(self.features_dir / "expansion_branch_comparison.csv")

        forecast_path = self.output_dir / "demand_forecast_summary.json"
        forecast_summary = {}
        if forecast_path.exists():
            with open(forecast_path) as f:
                forecast_summary = json.load(f)

        network_health = self._assess_network_health(scorecard, forecast_summary)
        ideal_profile = self._define_ideal_profile(scorecard, benchmarks)
        risks = self._identify_risks(scorecard, forecast_summary)
        verdict = self._produce_verdict(network_health, risks)
        assessment = self._build_assessment_table(
            scorecard, network_health, ideal_profile, risks, verdict
        )
        summary = self._build_summary(
            network_health, ideal_profile, risks, verdict, scorecard, forecast_summary
        )

        results: Dict[str, Any] = {
            "expansion_assessment": assessment,
            "expansion_summary": summary,
        }

        if save:
            assessment.to_csv(
                self.output_dir / "expansion_assessment.csv", index=False
            )
            with open(self.output_dir / "expansion_summary.json", "w") as f:
                json.dump(summary, f, indent=2)
            logger.info("Saved expansion assessment and summary to %s", self.output_dir)

        return results

      # Network health assessment
  
    def _assess_network_health(
        self, scorecard: pd.DataFrame, forecast: Dict
    ) -> Dict[str, Any]:
        health: Dict[str, Any] = {}

        # How many branches are performing well (composite > 0.5)?
        strong = scorecard[scorecard["composite_score"] >= 0.5]
        weak = scorecard[scorecard["composite_score"] < 0.5]
        health["strong_branches"] = len(strong)
        health["weak_branches"] = len(weak)
        health["strong_branch_names"] = strong["branch"].tolist()
        health["weak_branch_names"] = weak["branch"].tolist()

        # Average composite score
        health["avg_composite"] = round(float(scorecard["composite_score"].mean()), 4)

        # Revenue concentration: does any branch dominate?
        total_rev = scorecard["total_revenue"].sum()
        scorecard_copy = scorecard.copy()
        scorecard_copy["rev_share"] = scorecard_copy["total_revenue"] / total_rev
        max_share = scorecard_copy["rev_share"].max()
        health["max_revenue_share"] = round(float(max_share), 4)
        health["revenue_concentrated"] = max_share > 0.35

        # Growth trajectory from forecasts
        if forecast.get("branches"):
            growing = sum(
                1 for b in forecast["branches"].values()
                if (b.get("forecast_vs_historical") or 0) > 0
            )
            declining = sum(
                1 for b in forecast["branches"].values()
                if (b.get("forecast_vs_historical") or 0) < 0
            )
            health["branches_growing"] = growing
            health["branches_declining"] = declining
        else:
            health["branches_growing"] = 0
            health["branches_declining"] = 0

        # Overall network health score (0-1)
        factors = [
            health["strong_branches"] / len(scorecard),
            1 - health["weak_branches"] / len(scorecard),
            min(health["avg_composite"] / 0.7, 1.0),
            health["branches_growing"] / max(len(scorecard), 1),
        ]
        health["network_health_score"] = round(float(np.mean(factors)), 4)

        logger.info("Network health score: %.2f", health["network_health_score"])
        return health

      # Ideal new-branch profile
  
    def _define_ideal_profile(
        self, scorecard: pd.DataFrame, benchmarks: pd.DataFrame
    ) -> Dict[str, Any]:
        """Define what a successful new branch should look like based on top performers."""
        top = scorecard.nlargest(2, "composite_score")

        profile: Dict[str, Any] = {
            "based_on": top["branch"].tolist(),
            "target_metrics": {},
        }

        key_metrics = [
            ("avg_monthly_revenue", "Target monthly revenue (first year)"),
            ("total_customers", "Target customer base"),
            ("avg_ticket_size", "Target average ticket size"),
            ("num_employees", "Recommended starting staff"),
            ("avg_shift_hours", "Average shift length (hours)"),
            ("beverage_ratio", "Target beverage revenue share"),
            ("repeat_rate", "Target customer repeat rate"),
        ]

        for metric, label in key_metrics:
            if metric in top.columns:
                top_avg = float(top[metric].mean())
                network_avg = float(scorecard[metric].mean())
                bm_row = benchmarks[benchmarks["metric"] == metric]
                bm_median = float(bm_row["median"].values[0]) if not bm_row.empty else network_avg

                profile["target_metrics"][metric] = {
                    "label": label,
                    "top_performer_avg": round(top_avg, 2),
                    "network_average": round(network_avg, 2),
                    "benchmark_median": round(bm_median, 2),
                    "recommended_target": round((top_avg + bm_median) / 2, 2),
                }

        # Channel recommendation
        top_table = top["table_share"].mean()
        top_takeaway = top["takeaway_share"].mean()
        top_delivery = top["delivery_share"].mean()
        if top_table > 0.7:
            profile["recommended_channel_focus"] = "TABLE (dine-in)"
        elif top_takeaway > 0.7:
            profile["recommended_channel_focus"] = "TAKE AWAY"
        else:
            profile["recommended_channel_focus"] = "MIXED (table + takeaway)"

        profile["recommended_channel_mix"] = {
            "table": round(float(top_table), 4),
            "takeaway": round(float(top_takeaway), 4),
            "delivery": round(float(top_delivery), 4),
        }

        logger.info("Defined ideal profile based on %s", profile["based_on"])
        return profile

      # Risk identification
  
    def _identify_risks(
        self, scorecard: pd.DataFrame, forecast: Dict
    ) -> List[Dict[str, str]]:
        risks: List[Dict[str, str]] = []

        # Risk: weak existing branch
        weak = scorecard[scorecard["composite_score"] < 0.4]
        if not weak.empty:
            names = weak["branch"].tolist()
            risks.append({
                "risk": "Underperforming existing branch",
                "severity": "HIGH",
                "detail": (
                    f"{', '.join(names)} has a composite score below 0.4. "
                    "Expanding while an existing branch struggles may spread "
                    "resources too thin."
                ),
                "mitigation": f"Stabilize {', '.join(names)} before expanding, or consider relocating.",
            })

        # Risk: high revenue volatility across network
        avg_cv = scorecard["revenue_cv"].mean()
        if avg_cv > 0.7:
            risks.append({
                "risk": "High revenue volatility",
                "severity": "MEDIUM",
                "detail": (
                    f"Average revenue coefficient of variation is {avg_cv:.2f}. "
                    "Demand is unpredictable, making ROI projections uncertain."
                ),
                "mitigation": "Build conservative financial projections with wide margins.",
            })

        # Risk: limited data history
        risks.append({
            "risk": "Short data history",
            "severity": "MEDIUM",
            "detail": (
                "Only 4-5 months of sales data available. Seasonal patterns "
                "and long-term trends cannot be reliably estimated."
            ),
            "mitigation": "Collect 12+ months of data before committing major capital.",
        })

        # Risk: staffing gaps
        no_staff = scorecard[scorecard["num_employees"] == 0]
        if not no_staff.empty:
            risks.append({
                "risk": "Missing operational data",
                "severity": "LOW",
                "detail": (
                    f"No attendance/staffing data for {no_staff['branch'].tolist()}. "
                    "Staffing benchmarks may be incomplete."
                ),
                "mitigation": "Ensure all branches track attendance before using staffing models.",
            })

        # Risk: declining branch
        if forecast.get("branches"):
            declining = [
                b for b, info in forecast["branches"].items()
                if (info.get("forecast_vs_historical") or 0) < -0.1
            ]
            if declining:
                risks.append({
                    "risk": "Declining branch revenue",
                    "severity": "HIGH",
                    "detail": (
                        f"{', '.join(declining)} is forecasted to decline. "
                        "Network-wide demand may not support a new location."
                    ),
                    "mitigation": "Investigate root cause of decline before expansion.",
                })

        logger.info("Identified %d risks", len(risks))
        return risks

      # Verdict
  
    def _produce_verdict(
        self, health: Dict[str, Any], risks: List[Dict]
    ) -> Dict[str, str]:
        high_risks = sum(1 for r in risks if r["severity"] == "HIGH")
        health_score = health["network_health_score"]

        if health_score >= 0.7 and high_risks == 0:
            verdict = "GO"
            rationale = (
                "The network is healthy with strong growth across branches. "
                "No high-severity risks identified. Expansion is recommended."
            )
        elif health_score >= 0.5 and high_risks <= 1:
            verdict = "CONDITIONAL GO"
            rationale = (
                "The network shows moderate health with some risks. "
                "Expansion is feasible but should address identified risks first. "
                "Recommend a phased approach."
            )
        else:
            verdict = "NOT RECOMMENDED"
            rationale = (
                "The network has significant weaknesses or multiple high-severity "
                "risks. Focus on strengthening existing operations before expanding."
            )

        return {
            "verdict": verdict,
            "rationale": rationale,
            "health_score": str(round(health_score, 2)),
            "high_risks": str(high_risks),
        }

      # Assessment table
  
    def _build_assessment_table(
        self,
        scorecard: pd.DataFrame,
        health: Dict,
        profile: Dict,
        risks: List[Dict],
        verdict: Dict,
    ) -> pd.DataFrame:
        rows = []

        rows.append({
            "category": "Verdict",
            "metric": "Expansion Recommendation",
            "value": verdict["verdict"],
            "detail": verdict["rationale"],
        })
        rows.append({
            "category": "Network Health",
            "metric": "Health Score",
            "value": str(health["network_health_score"]),
            "detail": f"{health['strong_branches']} strong, {health['weak_branches']} weak branches",
        })
        rows.append({
            "category": "Network Health",
            "metric": "Growing Branches",
            "value": str(health["branches_growing"]),
            "detail": f"{health['branches_declining']} declining",
        })

        for metric, info in profile.get("target_metrics", {}).items():
            rows.append({
                "category": "Ideal Profile",
                "metric": info["label"],
                "value": str(info["recommended_target"]),
                "detail": f"Top performer avg: {info['top_performer_avg']}, Network avg: {info['network_average']}",
            })

        rows.append({
            "category": "Ideal Profile",
            "metric": "Recommended Channel Focus",
            "value": profile.get("recommended_channel_focus", "N/A"),
            "detail": str(profile.get("recommended_channel_mix", {})),
        })

        for risk in risks:
            rows.append({
                "category": "Risk",
                "metric": risk["risk"],
                "value": risk["severity"],
                "detail": risk["detail"],
            })

        return pd.DataFrame(rows)

      # Executive summary
  
    def _build_summary(
        self,
        health: Dict,
        profile: Dict,
        risks: List[Dict],
        verdict: Dict,
        scorecard: pd.DataFrame,
        forecast: Dict,
    ) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "verdict": verdict["verdict"],
            "rationale": verdict["rationale"],
            "network_health_score": health["network_health_score"],
            "strong_branches": health["strong_branch_names"],
            "weak_branches": health["weak_branch_names"],
            "branches_growing": health["branches_growing"],
            "branches_declining": health["branches_declining"],
        }

        # Ideal profile highlights
        targets = profile.get("target_metrics", {})
        summary["new_branch_targets"] = {
            k: v["recommended_target"] for k, v in targets.items()
        }
        summary["recommended_channel_focus"] = profile.get("recommended_channel_focus")
        summary["recommended_channel_mix"] = profile.get("recommended_channel_mix")

        # Risk summary
        summary["risks"] = [
            {"risk": r["risk"], "severity": r["severity"], "mitigation": r["mitigation"]}
            for r in risks
        ]
        summary["high_risk_count"] = sum(1 for r in risks if r["severity"] == "HIGH")

        # Location recommendation based on channel gaps
        # Branches with only TABLE channel suggest a takeaway/delivery gap in the market
        table_only = scorecard[
            (scorecard["table_share"] > 0.9) & (scorecard["takeaway_share"] < 0.05)
        ]
        if not table_only.empty:
            summary["location_insight"] = (
                f"Branches {table_only['branch'].tolist()} are 100% dine-in. "
                "A new location near these areas with a takeaway/delivery focus "
                "could capture unserved demand without cannibalizing existing sales."
            )
        else:
            summary["location_insight"] = (
                "Consider areas with high foot traffic and limited existing coverage. "
                "A dine-in focused location aligns with the strongest branch profiles."
            )

        # Best-performing model to replicate
        best = scorecard.iloc[0]
        summary["model_to_replicate"] = {
            "branch": best["branch"],
            "composite_score": float(best["composite_score"]),
            "key_strengths": [],
        }
        if best["score_growth"] > 0.8:
            summary["model_to_replicate"]["key_strengths"].append("Strong growth momentum")
        if best["score_revenue"] > 0.8:
            summary["model_to_replicate"]["key_strengths"].append("High revenue")
        if best["score_efficiency"] > 0.8:
            summary["model_to_replicate"]["key_strengths"].append("High operational efficiency")
        if best["score_customers"] > 0.8:
            summary["model_to_replicate"]["key_strengths"].append("Large customer base")

        return summary
