"""
Conut Bakery — OpenClaw MCP Server
====================================
Exposes the Chief of Operations Agent as MCP tools that OpenClaw can invoke.

Tools:
  1. get_combo_recommendations  — Combo optimization suggestions
  2. get_demand_forecast        — Branch-level demand forecasting
  3. get_expansion_assessment   — Expansion feasibility verdict
  4. get_staffing_recommendation— Shift staffing recommendations
  5. get_beverage_strategy      — Coffee & milkshake growth strategy
  6. ask_operations_agent       — Natural-language query router
  7. health_check               — Verify server connectivity and data status

Resources:
  - conut://branches            — List of all branches
  - conut://summary             — High-level business summary
  - conut://system-info         — Server version and capabilities

Prompts:
  - daily_operations_briefing   — Daily management briefing template
  - branch_deep_dive            — Deep-dive analysis for a branch

Run:
    python -m src.openclaw          # stdio mode (for OpenClaw local)
    fastmcp run src/openclaw/server.py
"""

from __future__ import annotations

import json
import logging
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from fastmcp import FastMCP

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.engine import InferenceEngine

logger = logging.getLogger(__name__)

# ── MCP Server ──────────────────────────────────────────────────────────

mcp = FastMCP(
    "ConutBakeryOps",
    instructions=(
        "You are the Conut Bakery Chief of Operations Agent. "
        "You have access to analytics tools covering five business objectives: "
        "combo optimization, demand forecasting, expansion feasibility, "
        "shift staffing, and beverage growth strategy. "
        "Use these tools to answer operational questions about Conut Bakery. "
        "Available branches: Conut, Conut - Tyre, Conut Jnah, Main Street Coffee."
    ),
)

_engine: Optional[InferenceEngine] = None

SERVER_VERSION = "1.0.0"
AVAILABLE_BRANCHES = ["Conut", "Conut - Tyre", "Conut Jnah", "Main Street Coffee"]


def _get_engine() -> InferenceEngine:
    """Lazy-load the inference engine with pre-computed analytics data."""
    global _engine
    if _engine is None:
        _engine = InferenceEngine()
        _engine.load()
    return _engine


def _safe_json(data: Any) -> str:
    """Serialize to JSON, handling pandas types and NaN values."""
    return json.dumps(data, indent=2, default=str, ensure_ascii=False)


def _error_response(tool_name: str, error: Exception) -> str:
    """Return a structured error response."""
    logger.error("Tool %s failed: %s", tool_name, error, exc_info=True)
    return _safe_json({
        "error": True,
        "tool": tool_name,
        "message": str(error),
        "hint": "Ensure the pipeline has been run: python run_pipeline.py",
    })


# ── Tool 1: Combo Optimization ─────────────────────────────────────────

@mcp.tool()
def get_combo_recommendations(
    branch: Optional[str] = None,
    top_n: int = 5,
) -> str:
    """Get product combo recommendations for Conut Bakery.

    Returns the top-performing 2-item and 3-item product combos ranked by
    lift, confidence, and estimated revenue. Use this to identify which
    products should be bundled together for promotions and menu combos.

    Args:
        branch: Branch name to filter (e.g. "Conut Jnah", "Main Street Coffee").
                If omitted, returns network-wide recommendations.
                Available branches: Conut, Conut - Tyre, Conut Jnah, Main Street Coffee.
        top_n: Number of top combos to return (default 5, max 20).
    """
    try:
        engine = _get_engine()
        top_n = min(max(top_n, 1), 20)
        result = engine.get_combo_recommendations(branch=branch, top_n=top_n)
        return _safe_json(result)
    except Exception as e:
        return _error_response("get_combo_recommendations", e)


# ── Tool 2: Demand Forecasting ─────────────────────────────────────────

@mcp.tool()
def get_demand_forecast(
    branch: Optional[str] = None,
) -> str:
    """Get demand forecasts for Conut Bakery branches.

    Returns revenue forecasts for Q1 2026 (January-March) using an ensemble
    of linear trend, weighted moving average, and growth-rate projection.
    Use this to support inventory planning and supply chain decisions.

    Args:
        branch: Branch name (e.g. "Conut", "Conut - Tyre", "Conut Jnah",
                "Main Street Coffee"). If omitted, returns all branches.
    """
    try:
        engine = _get_engine()
        result = engine.get_demand_forecast(branch=branch)
        return _safe_json(result)
    except Exception as e:
        return _error_response("get_demand_forecast", e)


# ── Tool 3: Expansion Feasibility ──────────────────────────────────────

@mcp.tool()
def get_expansion_assessment() -> str:
    """Assess whether Conut Bakery should expand to a new location.

    Returns a go/no-go verdict with network health score, risk analysis,
    ideal new-branch profile, and the best model branch to replicate.
    Use this to evaluate expansion feasibility and recommend candidate locations.
    """
    try:
        engine = _get_engine()
        result = engine.get_expansion_assessment()
        return _safe_json(result)
    except Exception as e:
        return _error_response("get_expansion_assessment", e)


# ── Tool 4: Staffing Recommendations ───────────────────────────────────

@mcp.tool()
def get_staffing_recommendation(
    branch: Optional[str] = None,
) -> str:
    """Get shift staffing recommendations for Conut Bakery branches.

    Returns recommended employees per shift (morning, afternoon, evening),
    weekly scheduling guidelines, and hiring recommendations based on
    demand forecasts and historical efficiency data.

    Args:
        branch: Branch name. If omitted, returns network-wide staffing summary
                with per-branch hiring actions.
    """
    try:
        engine = _get_engine()
        result = engine.get_staffing_recommendation(branch=branch)
        return _safe_json(result)
    except Exception as e:
        return _error_response("get_staffing_recommendation", e)


# ── Tool 5: Beverage Growth Strategy ───────────────────────────────────

@mcp.tool()
def get_beverage_strategy(
    branch: Optional[str] = None,
) -> str:
    """Get coffee and milkshake growth strategy for Conut Bakery.

    Returns prioritized actions to increase beverage sales: products to
    introduce, products to promote, estimated revenue uplift, and
    benchmarking against the best-performing branch.

    Args:
        branch: Branch name. If omitted, returns network-wide strategy
                with per-branch breakdowns.
    """
    try:
        engine = _get_engine()
        result = engine.get_beverage_strategy(branch=branch)
        return _safe_json(result)
    except Exception as e:
        return _error_response("get_beverage_strategy", e)


# ── Tool 6: Natural-language query (catch-all) ─────────────────────────

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
    try:
        engine = _get_engine()
        result = engine.query(question, branch=branch)
        result.pop("_topic", None)
        result.pop("_query", None)
        return _safe_json(result)
    except Exception as e:
        return _error_response("ask_operations_agent", e)


# ── Tool 7: Health Check ───────────────────────────────────────────────

@mcp.tool()
def health_check() -> str:
    """Check the health and readiness of the Conut Bakery Operations Agent.

    Returns server version, data availability status, and a list of
    available tools. Use this to verify the system is operational before
    running queries.
    """
    from src.config import ANALYTICS_DIR

    analytics_files = list(ANALYTICS_DIR.glob("*.csv")) + list(ANALYTICS_DIR.glob("*.json"))
    data_ready = len(analytics_files) > 0

    engine_loaded = _engine is not None
    if not engine_loaded and data_ready:
        try:
            _get_engine()
            engine_loaded = True
        except Exception as e:
            logger.warning("Engine load failed during health check: %s", e)

    return _safe_json({
        "status": "healthy" if data_ready and engine_loaded else "degraded",
        "server_version": SERVER_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data_ready": data_ready,
        "analytics_artifacts": len(analytics_files),
        "engine_loaded": engine_loaded,
        "available_branches": AVAILABLE_BRANCHES,
        "available_tools": [
            "get_combo_recommendations",
            "get_demand_forecast",
            "get_expansion_assessment",
            "get_staffing_recommendation",
            "get_beverage_strategy",
            "ask_operations_agent",
            "health_check",
        ],
        "hint": None if data_ready else "Run 'python run_pipeline.py' to generate analytics data.",
    })


# ── Resources ───────────────────────────────────────────────────────────

@mcp.resource("conut://branches")
def list_branches() -> str:
    """List all Conut Bakery branches with basic metadata."""
    engine = _get_engine()
    branch_info = []
    for b in AVAILABLE_BRANCHES:
        info: Dict[str, Any] = {"name": b}
        forecast = engine.get_demand_forecast(branch=b)
        if "error" not in forecast:
            info["q1_2026_forecast"] = forecast.get("q1_2026_total_forecast")
            info["forecast_vs_historical"] = forecast.get("forecast_vs_historical")
        branch_info.append(info)
    return _safe_json(branch_info)


@mcp.resource("conut://summary")
def business_summary() -> str:
    """Get a high-level summary of all five business objectives."""
    engine = _get_engine()
    result = engine._full_summary()
    return _safe_json(result)


@mcp.resource("conut://system-info")
def system_info() -> str:
    """Server version, capabilities, and configuration."""
    return _safe_json({
        "server": "ConutBakeryOps",
        "version": SERVER_VERSION,
        "protocol": "MCP (Model Context Protocol)",
        "framework": "FastMCP",
        "description": "AI-Driven Chief of Operations Agent for Conut Bakery",
        "business_objectives": [
            "Combo Optimization",
            "Demand Forecasting by Branch",
            "Expansion Feasibility",
            "Shift Staffing Estimation",
            "Coffee and Milkshake Growth Strategy",
        ],
        "branches": AVAILABLE_BRANCHES,
        "tools_count": 7,
        "resources_count": 3,
        "prompts_count": 2,
    })


# ── Prompts ─────────────────────────────────────────────────────────────

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
        "in a clear executive summary format with actionable recommendations."
    )


@mcp.prompt()
def branch_deep_dive(branch: str) -> str:
    """Generate a deep-dive analysis for a specific Conut Bakery branch.

    Args:
        branch: The branch to analyze (e.g. "Conut Jnah", "Main Street Coffee").
    """
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


# ── Entry point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()
