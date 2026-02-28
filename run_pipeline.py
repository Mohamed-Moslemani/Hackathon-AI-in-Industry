#!/usr/bin/env python3
"""
Conut Bakery — Chief of Operations Agent
=========================================
End-to-end pipeline: data cleaning -> feature engineering -> analytics -> reporting.

Run:
    python run_pipeline.py              # full pipeline
    python run_pipeline.py --report     # report only (skip recomputation)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

  # Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ANALYTICS_DIR, CLEANED_DATA_DIR, FEATURES_DIR
from src.data_cleaning.pipeline import DataPipeline
from src.feature_engineering.pipeline import FeaturePipeline
from src.analytics.pipeline import AnalyticsPipeline
from src.inference.engine import InferenceEngine

  # Formatting helpers
  
SEP = "=" * 72
THIN = "-" * 72


def header(title: str) -> str:
    return f"\n{SEP}\n  {title}\n{SEP}"


def section(title: str) -> str:
    return f"\n{THIN}\n  {title}\n{THIN}"


def fmt_money(val: float) -> str:
    """Format scaled monetary values into readable form."""
    if abs(val) >= 1_000_000_000:
        return f"{val / 1_000_000_000:,.2f}B"
    if abs(val) >= 1_000_000:
        return f"{val / 1_000_000:,.2f}M"
    if abs(val) >= 1_000:
        return f"{val / 1_000:,.2f}K"
    return f"{val:,.2f}"


def fmt_pct(val: float) -> str:
    return f"{val * 100:.1f}%"


  # Report generators
  
def report_combo(engine: InferenceEngine) -> None:
    print(header("1. COMBO OPTIMIZATION"))
    data = engine.get_combo_recommendations()

    print("\n  Top 2-Item Combos:")
    for c in data.get("top_2item_combos", []):
        print(f"    #{c['rank']}  {c['combo']}")
        print(f"         Lift: {c['lift']:.1f}x | Confidence: {c['confidence']:.0%} "
              f"| Price: {fmt_money(c['combo_price'])} | Discount: {c['suggested_discount_pct']}%")

    print("\n  Top 3-Item Combos:")
    for c in data.get("top_3item_combos", []):
        print(f"    #{c['rank']}  {c['combo']}")
        print(f"         Occurrences: {c['occurrences']} | Price: {fmt_money(c['combo_price'])}")

    # Per-branch top combo
    combo_summary = engine._data["combo_summary"]
    branch_combos = combo_summary.get("top_combo_per_branch", {})
    if branch_combos:
        print("\n  Best Combo per Branch:")
        for branch, info in branch_combos.items():
            print(f"    {branch:25s}  {info['combo']}  "
                  f"(co-purchases: {info['co_purchase_count']}, price: {fmt_money(info['combo_price'])})")

    print(f"\n  Key Insight: {data.get('key_insight', 'N/A')}")
    print(f"  Total combos discovered: {data.get('total_2item', 0)} (2-item) + {data.get('total_3item', 0)} (3-item)")


def report_demand(engine: InferenceEngine) -> None:
    print(header("2. DEMAND FORECASTING BY BRANCH"))
    data = engine.get_demand_forecast()

    print(f"\n  Forecast Horizon : {data.get('forecast_horizon')}")
    print(f"  Method           : {data.get('method')}")
    print(f"  Network Q1 Total : {fmt_money(data.get('network_q1_total', 0))}")
    print(f"  Fastest Growing  : {data.get('fastest_growing')}")

    branches = data.get("branches", {})
    print(f"\n  {'Branch':25s} {'Jan 2026':>14s} {'Q1 Total':>14s} {'Hist. Avg':>14s} {'vs Hist.':>10s}")
    print(f"  {'-'*25} {'-'*14} {'-'*14} {'-'*14} {'-'*10}")
    for branch, info in branches.items():
        jan = fmt_money(info["jan_2026_forecast"])
        q1 = fmt_money(info["q1_2026_total_forecast"])
        hist = fmt_money(info["historical_avg_monthly"])
        vs = fmt_pct(info["forecast_vs_historical"])
        print(f"  {branch:25s} {jan:>14s} {q1:>14s} {hist:>14s} {vs:>10s}")


def report_expansion(engine: InferenceEngine) -> None:
    print(header("3. EXPANSION FEASIBILITY"))
    data = engine.get_expansion_assessment()

    verdict = data.get("verdict", "N/A")
    verdict_marker = ">>>" if verdict == "NOT RECOMMENDED" else "+++"
    print(f"\n  {verdict_marker} VERDICT: {verdict}")
    print(f"  Rationale: {data.get('rationale')}")
    print(f"  Network Health Score: {data.get('network_health_score', 0):.2f} / 1.00")

    print(f"\n  Strong Branches: {', '.join(data.get('strong_branches', []))}")
    print(f"  Weak Branches  : {', '.join(data.get('weak_branches', []))}")

    risks = data.get("risks", [])
    if risks:
        print("\n  Risks:")
        for r in risks:
            sev = r["severity"]
            marker = "!!!" if sev == "HIGH" else " ! " if sev == "MEDIUM" else " . "
            print(f"    {marker} [{sev:6s}] {r['risk']}")
            print(f"              Mitigation: {r['mitigation']}")

    targets = data.get("new_branch_targets", {})
    if targets:
        print("\n  Ideal New Branch Profile (if expansion proceeds):")
        print(f"    Avg Monthly Revenue : {fmt_money(targets.get('avg_monthly_revenue', 0))}")
        print(f"    Total Customers     : {targets.get('total_customers', 0):,.0f}")
        print(f"    Avg Ticket Size     : {fmt_money(targets.get('avg_ticket_size', 0))}")
        print(f"    Employees           : {targets.get('num_employees', 0)}")
        print(f"    Beverage Ratio      : {fmt_pct(targets.get('beverage_ratio', 0))}")

    model = data.get("model_to_replicate")
    if model:
        print(f"\n  Model to Replicate: {model['branch']} (score: {model['composite_score']:.2f})")
        print(f"    Strengths: {', '.join(model.get('key_strengths', []))}")

    print(f"\n  Location Insight: {data.get('location_insight', 'N/A')}")


def report_staffing(engine: InferenceEngine) -> None:
    print(header("4. SHIFT STAFFING OPTIMIZATION"))
    data = engine.get_staffing_recommendation()

    network = data.get("network_insights", {})
    print(f"\n  Total Employees       : {network.get('total_employees')}")
    print(f"  Avg Revenue/Employee  : {fmt_money(network.get('avg_revenue_per_employee', 0))}")
    print(f"  Most Efficient Branch : {network.get('most_efficient_branch')}")
    print(f"  Key Finding: {network.get('key_finding', 'N/A')}")

    branch_data = data.get("branch_staffing", {})
    for branch, info in branch_data.items():
        print(section(f"  {branch}"))
        print(f"    Current Staff    : {info['current_staff']}")
        print(f"    Hiring Action    : {info['hiring_action']}")
        if info["suggested_additional"] > 0:
            print(f"    Hire Additional  : {info['suggested_additional']}")

    # Detailed shift recommendations
    full_summary = engine._data["staffing_summary"]
    for branch, info in full_summary.get("branch_staffing", {}).items():
        print(f"\n    {branch} — Shift Breakdown:")
        shifts = info.get("shift_recommendations", {})
        print(f"      {'Shift':12s} {'Staff':>6s} {'Hours':>6s}")
        print(f"      {'-'*12} {'-'*6} {'-'*6}")
        for shift, rec in shifts.items():
            print(f"      {shift:12s} {rec['recommended_staff']:6d} {rec['recommended_hours']:6.1f}")
        sched = info.get("weekly_schedule", {})
        print(f"      Weekday: {sched.get('weekday_recommended_employees')} staff | "
              f"Weekend: {sched.get('weekend_recommended_employees')} staff | "
              f"Peak: {sched.get('peak_day')}")
        hire = info.get("hiring_recommendation", {})
        print(f"      Hiring: {hire.get('action')} — {hire.get('reason')}")


def report_beverage(engine: InferenceEngine) -> None:
    print(header("5. COFFEE & MILKSHAKE GROWTH STRATEGY"))
    data = engine.get_beverage_strategy()

    overview = data.get("network_overview", {})
    print(f"\n  Best Beverage Branch  : {overview.get('best_beverage_branch')} "
          f"({fmt_pct(overview.get('best_beverage_share', 0))} of revenue)")
    print(f"  Worst Beverage Branch : {overview.get('worst_beverage_branch')} "
          f"({fmt_pct(overview.get('worst_beverage_share', 0))} of revenue)")
    print(f"  Network Average       : {fmt_pct(overview.get('network_avg_beverage_share', 0))}")
    print(f"  Total Estimated Uplift: {fmt_money(data.get('total_estimated_uplift', 0))}")

    strategies = data.get("branch_strategies", {})
    for branch, info in strategies.items():
        print(section(f"  {branch}"))
        print(f"    Actions          : {info['num_actions']}")
        print(f"    Estimated Uplift : {fmt_money(info['estimated_uplift'])}")
        print(f"    Products to Add  : {info['products_to_add']}")
        print(f"    Products to Boost: {info['products_to_boost']}")
        if info.get("top_priority"):
            tp = info["top_priority"]
            print(f"    Top Priority     : [{tp['action']}] {tp['description']}")

    insights = data.get("key_insights", [])
    if insights:
        print("\n  Key Insights:")
        for i, insight in enumerate(insights, 1):
            print(f"    {i}. {insight}")


def report_inference_demo(engine: InferenceEngine) -> None:
    """Demonstrate the natural-language query interface."""
    print(header("INFERENCE ENGINE — QUERY DEMO"))

    sample_queries = [
        "What combos should we offer at Conut Jnah?",
        "What is the demand forecast for Main Street Coffee?",
        "Should we expand to a new location?",
        "How many staff do we need at Conut - Tyre?",
        "What coffee products should Conut add?",
        "Give me a full summary of the business.",
    ]

    for q in sample_queries:
        print(f"\n  Q: \"{q}\"")
        result = engine.query(q)
        topic = result.pop("_topic", "unknown")
        result.pop("_query", None)
        print(f"  Topic: {topic}")

        # Print a compact version of the answer
        compact = json.dumps(result, indent=2, default=str)
        for line in compact.split("\n")[:12]:
            print(f"    {line}")
        if compact.count("\n") > 12:
            print(f"    ... ({compact.count(chr(10)) - 12} more lines)")


  # Pipeline stages
  
def run_cleaning() -> None:
    print(header("STAGE 1: DATA CLEANING"))
    t0 = time.time()
    pipeline = DataPipeline()
    pipeline.run(save=True)
    summary = pipeline.summary()
    print(f"\n  Cleaned {len(summary)} datasets in {time.time() - t0:.1f}s")
    print(f"  {'Dataset':25s} {'Rows':>8s} {'Cols':>6s}")
    print(f"  {'-'*25} {'-'*8} {'-'*6}")
    for _, row in summary.iterrows():
        print(f"  {row['dataset']:25s} {row['rows']:8d} {row['columns']:6d}")
    print(f"  Output: {CLEANED_DATA_DIR}")


def run_features() -> None:
    print(header("STAGE 2: FEATURE ENGINEERING"))
    t0 = time.time()
    pipeline = FeaturePipeline()
    pipeline.run(save=True)
    summary = pipeline.summary()
    print(f"\n  Built {len(summary)} feature datasets in {time.time() - t0:.1f}s")
    print(f"  {'Builder':12s} {'Dataset':40s} {'Rows':>6s} {'Cols':>6s}")
    print(f"  {'-'*12} {'-'*40} {'-'*6} {'-'*6}")
    for _, row in summary.iterrows():
        print(f"  {row['builder']:12s} {row['dataset']:40s} {row['rows']:6d} {row['columns']:6d}")
    print(f"  Output: {FEATURES_DIR}")


def run_analytics() -> None:
    print(header("STAGE 3: MODELING & ANALYTICS"))
    t0 = time.time()
    pipeline = AnalyticsPipeline()
    pipeline.run(save=True)
    elapsed = time.time() - t0
    print(f"\n  Completed {len(pipeline.results)} analytics modules in {elapsed:.1f}s")
    for name, results in pipeline.results.items():
        keys = [k for k in results.keys()]
        print(f"    {name:12s} -> {', '.join(keys)}")
    print(f"  Output: {ANALYTICS_DIR}")


def run_report() -> None:
    print(header("STAGE 4: INFERENCE & REPORTING"))
    engine = InferenceEngine()
    engine.load()

    report_combo(engine)
    report_demand(engine)
    report_expansion(engine)
    report_staffing(engine)
    report_beverage(engine)
    report_inference_demo(engine)

    print(header("PIPELINE COMPLETE"))
    print(f"""
  All outputs saved to:
    Cleaned data  : {CLEANED_DATA_DIR}
    Features      : {FEATURES_DIR}
    Analytics     : {ANALYTICS_DIR}

  To query the system programmatically:
    from src.inference import InferenceEngine
    engine = InferenceEngine()
    engine.load()
    result = engine.query("What combos should we offer?")
""")


  # Main
  
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Conut Bakery — Chief of Operations Agent Pipeline"
    )
    parser.add_argument(
        "--report", action="store_true",
        help="Skip recomputation; only generate the report from existing outputs",
    )
    parser.add_argument(
        "--stage", choices=["clean", "features", "analytics", "report"],
        help="Run a single stage only",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    print(SEP)
    print("  CONUT BAKERY — CHIEF OF OPERATIONS AGENT")
    print("  AI-Driven Analytics Pipeline")
    print(SEP)

    if args.stage:
        {"clean": run_cleaning, "features": run_features,
         "analytics": run_analytics, "report": run_report}[args.stage]()
        return

    if args.report:
        run_report()
        return

    run_cleaning()
    run_features()
    run_analytics()
    run_report()


if __name__ == "__main__":
    main()
