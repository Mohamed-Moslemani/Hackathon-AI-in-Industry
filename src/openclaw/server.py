"""
Exposes the Chief of Operations Agent as MCP tools that OpenClaw can invoke.

Tools:
  1. get_combo_recommendations  — Combo optimization suggestions
  2. get_demand_forecast        — Branch-level demand forecasting
  3. get_expansion_assessment   — Expansion feasibility verdict
  4. get_staffing_recommendation— Shift staffing recommendations
  5. get_beverage_strategy      — Coffee & milkshake growth strategy
  6. ask_operations_agent       — Natural-language query router

Run:
    python -m src.openclaw.server          # stdio mode (for OpenClaw local)
    fastmcp run src/openclaw/server.py     # alternative
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

from fastmcp import FastMCP

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.engine import InferenceEngine

 # Initialize MCP server and inference engine
 
mcp = FastMCP("ConutBakeryOps")

_engine: Optional[InferenceEngine] = None


def _get_engine() -> InferenceEngine:
    global _engine
    if _engine is None:
        _engine = InferenceEngine()
        _engine.load()
    return _engine


# Tool 1: Combo Optimization

@mcp.tool()
def get_combo_recommendations(
    branch: Optional[str] = None,
    top_n: int = 5,
) -> str:
    """Get product combo recommendations for Conut Bakery.

    Returns the top-performing 2-item and 3-item combos ranked by
    lift, confidence, and estimated revenue. Optionally filter by branch.

    Args:
        branch: Branch name to filter (e.g. "Conut Jnah", "Main Street Coffee").
                If omitted, returns network-wide recommendations.
        top_n: Number of top combos to return (default 5).
    """
    engine = _get_engine()
    result = engine.get_combo_recommendations(branch=branch, top_n=top_n)
    return json.dumps(result, indent=2, default=str)


 # Tool 2: Demand Forecasting
 
@mcp.tool()
def get_demand_forecast(
    branch: Optional[str] = None,
) -> str:
    """Get demand forecasts for Conut Bakery branches.

    Returns revenue forecasts for Q1 2026 (January–March) using an ensemble
    of linear trend, weighted moving average, and growth-rate projection.

    Args:
        branch: Branch name (e.g. "Conut", "Conut - Tyre", "Conut Jnah",
                "Main Street Coffee"). If omitted, returns all branches.
    """
    engine = _get_engine()
    result = engine.get_demand_forecast(branch=branch)
    return json.dumps(result, indent=2, default=str)


 # Tool 3: Expansion Feasibility
 
@mcp.tool()
def get_expansion_assessment() -> str:
    """Assess whether Conut Bakery should expand to a new location.

    Returns a go/no-go verdict with network health score, risk analysis,
    ideal new-branch profile, and the best model branch to replicate.
    """
    engine = _get_engine()
    result = engine.get_expansion_assessment()
    return json.dumps(result, indent=2, default=str)


 # Tool 4: Staffing Recommendations
 
@mcp.tool()
def get_staffing_recommendation(
    branch: Optional[str] = None,
) -> str:
    """Get shift staffing recommendations for Conut Bakery branches.

    Returns recommended employees per shift (morning, afternoon, evening),
    weekly scheduling guidelines, and hiring recommendations based on
    demand forecasts and historical efficiency.

    Args:
        branch: Branch name. If omitted, returns network-wide summary.
    """
    engine = _get_engine()
    result = engine.get_staffing_recommendation(branch=branch)
    return json.dumps(result, indent=2, default=str)


 # Tool 5: Beverage Growth Strategy
 
@mcp.tool()
def get_beverage_strategy(
    branch: Optional[str] = None,
) -> str:
    """Get coffee and milkshake growth strategy for Conut Bakery.

    Returns prioritized actions to increase beverage sales: products to
    introduce, products to promote, estimated revenue uplift, and
    benchmarking against the best-performing branch.

    Args:
        branch: Branch name. If omitted, returns network-wide strategy.
    """
    engine = _get_engine()
    result = engine.get_beverage_strategy(branch=branch)
    return json.dumps(result, indent=2, default=str)


 # Tool 6: Natural-language query (catch-all)
 
@mcp.tool()
def ask_operations_agent(
    question: str,
    branch: Optional[str] = None,
) -> str:
    """Ask the Conut Bakery Chief of Operations Agent any business question.

    Routes your question to the appropriate analytics module automatically.
    Supports questions about combos, demand, expansion, staffing, and beverages.

    Examples:
        "What combos should we offer at Conut Jnah?"
        "What is the demand forecast for Main Street Coffee?"
        "Should we expand to a new location?"
        "How many staff do we need at Conut - Tyre?"
        "What coffee products should Conut add?"
        "Give me a full business summary."

    Args:
        question: Your business question in natural language.
        branch: Optional branch name to focus the answer on.
    """
    engine = _get_engine()
    result = engine.query(question, branch=branch)
    result.pop("_topic", None)
    result.pop("_query", None)
    return json.dumps(result, indent=2, default=str)


 # Resources: expose key data for agent context
 
@mcp.resource("conut://branches")
def list_branches() -> str:
    """List all Conut Bakery branches."""
    return json.dumps([
        "Conut", "Conut - Tyre", "Conut Jnah", "Main Street Coffee"
    ])


@mcp.resource("conut://summary")
def business_summary() -> str:
    """Get a high-level summary of all five business objectives."""
    engine = _get_engine()
    result = engine._full_summary()
    return json.dumps(result, indent=2, default=str)


 # Prompts: reusable templates for common agent interactions
 
@mcp.prompt()
def daily_operations_briefing() -> str:
    """Generate a daily operations briefing for the Conut Bakery management team."""
    return (
        "You are the Conut Bakery Chief of Operations Agent. "
        "Generate a concise daily briefing covering:\n"
        "1. Top combo recommendations and estimated revenue impact\n"
        "2. Demand forecast for the next month per branch\n"
        "3. Any staffing adjustments needed\n"
        "4. Beverage strategy priorities\n"
        "5. Expansion status\n\n"
        "Use the available tools to gather data, then present findings "
        "in a clear executive summary format."
    )


@mcp.prompt()
def branch_deep_dive(branch: str) -> str:
    """Generate a deep-dive analysis for a specific branch."""
    return (
        f"You are the Conut Bakery Chief of Operations Agent. "
        f"Perform a comprehensive analysis of the '{branch}' branch covering:\n"
        f"1. Demand forecast and trend\n"
        f"2. Staffing levels and recommendations\n"
        f"3. Best-performing combos at this branch\n"
        f"4. Beverage strategy and growth opportunities\n"
        f"5. Comparison with network benchmarks\n\n"
        f"Use all available tools to gather data for '{branch}', "
        f"then present a detailed report with actionable recommendations."
    )


 # Entry point
 
if __name__ == "__main__":
    mcp.run()
