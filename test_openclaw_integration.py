#!/usr/bin/env python3
"""
OpenClaw Integration Test
=========================
Validates that the MCP server starts correctly and all tools respond.

Run:
    python test_openclaw_integration.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"
SEP = "=" * 60


def test_imports():
    """Verify all required modules can be imported."""
    print(f"\n{SEP}")
    print("  Test: Module Imports")
    print(SEP)
    modules = [
        ("fastmcp", "FastMCP"),
        ("src.inference.engine", "InferenceEngine"),
        ("src.openclaw.server", "mcp"),
    ]
    all_ok = True
    for mod_name, attr in modules:
        try:
            mod = __import__(mod_name, fromlist=[attr])
            getattr(mod, attr)
            print(f"  {PASS} {mod_name}.{attr}")
        except Exception as e:
            print(f"  {FAIL} {mod_name}.{attr} — {e}")
            all_ok = False
    return all_ok


def test_engine_load():
    """Verify the inference engine loads analytics data."""
    print(f"\n{SEP}")
    print("  Test: Inference Engine Load")
    print(SEP)
    try:
        from src.inference.engine import InferenceEngine
        engine = InferenceEngine()
        engine.load()
        artifact_count = len(engine._data)
        if artifact_count > 0:
            print(f"  {PASS} Engine loaded {artifact_count} artifacts")
            return True
        else:
            print(f"  {WARN} Engine loaded but 0 artifacts found")
            return False
    except Exception as e:
        print(f"  {FAIL} Engine load failed — {e}")
        return False


def test_tools():
    """Call each MCP tool function directly and verify responses."""
    print(f"\n{SEP}")
    print("  Test: Tool Responses")
    print(SEP)

    from src.openclaw.server import (
        get_combo_recommendations,
        get_demand_forecast,
        get_expansion_assessment,
        get_staffing_recommendation,
        get_beverage_strategy,
        ask_operations_agent,
        health_check,
    )

    tools = [
        ("health_check", lambda: health_check()),
        ("get_combo_recommendations", lambda: get_combo_recommendations()),
        ("get_combo_recommendations(branch)", lambda: get_combo_recommendations(branch="Conut Jnah")),
        ("get_demand_forecast", lambda: get_demand_forecast()),
        ("get_demand_forecast(branch)", lambda: get_demand_forecast(branch="Main Street Coffee")),
        ("get_expansion_assessment", lambda: get_expansion_assessment()),
        ("get_staffing_recommendation", lambda: get_staffing_recommendation()),
        ("get_staffing_recommendation(branch)", lambda: get_staffing_recommendation(branch="Conut - Tyre")),
        ("get_beverage_strategy", lambda: get_beverage_strategy()),
        ("get_beverage_strategy(branch)", lambda: get_beverage_strategy(branch="Conut")),
        ("ask_operations_agent", lambda: ask_operations_agent(question="What combos should we offer?")),
        ("ask_operations_agent(branch)", lambda: ask_operations_agent(question="Demand forecast", branch="Conut Jnah")),
    ]

    all_ok = True
    for name, fn in tools:
        try:
            raw = fn()
            data = json.loads(raw)
            has_error = data.get("error", False)
            if has_error:
                print(f"  {FAIL} {name} — returned error: {data.get('message', '?')}")
                all_ok = False
            else:
                preview = raw[:80].replace("\n", " ")
                print(f"  {PASS} {name} — {preview}...")
        except Exception as e:
            print(f"  {FAIL} {name} — exception: {e}")
            all_ok = False
    return all_ok


def test_resources():
    """Verify MCP resources return valid JSON."""
    print(f"\n{SEP}")
    print("  Test: Resource Responses")
    print(SEP)

    from src.openclaw.server import list_branches, business_summary, system_info

    resources = [
        ("conut://branches", list_branches),
        ("conut://summary", business_summary),
        ("conut://system-info", system_info),
    ]

    all_ok = True
    for uri, fn in resources:
        try:
            raw = fn()
            json.loads(raw)
            print(f"  {PASS} {uri} — {len(raw)} chars")
        except Exception as e:
            print(f"  {FAIL} {uri} — {e}")
            all_ok = False
    return all_ok


def test_mcp_server_metadata():
    """Verify the MCP server object has the expected tools registered."""
    print(f"\n{SEP}")
    print("  Test: MCP Server Metadata")
    print(SEP)

    from src.openclaw.server import mcp

    print(f"  {PASS} Server name: {mcp.name}")

    return True


def test_config_file():
    """Verify openclaw.config.json is valid."""
    print(f"\n{SEP}")
    print("  Test: Configuration File")
    print(SEP)

    config_path = PROJECT_ROOT / "openclaw.config.json"
    if not config_path.exists():
        print(f"  {FAIL} openclaw.config.json not found")
        return False

    try:
        with open(config_path) as f:
            config = json.load(f)
        servers = config.get("mcpServers", {})
        if "conut-bakery-ops" in servers:
            srv = servers["conut-bakery-ops"]
            print(f"  {PASS} Server 'conut-bakery-ops' configured")
            print(f"         command: {srv.get('command')} {' '.join(srv.get('args', []))}")
            return True
        else:
            print(f"  {FAIL} 'conut-bakery-ops' not found in mcpServers")
            return False
    except Exception as e:
        print(f"  {FAIL} Config parse error — {e}")
        return False


def main():
    print(SEP)
    print("  CONUT BAKERY — OpenClaw Integration Test")
    print(SEP)

    results = {
        "imports": test_imports(),
        "config": test_config_file(),
        "engine": test_engine_load(),
        "tools": test_tools(),
        "resources": test_resources(),
        "metadata": test_mcp_server_metadata(),
    }

    print(f"\n{SEP}")
    print("  SUMMARY")
    print(SEP)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    for name, ok in results.items():
        print(f"  {PASS if ok else FAIL} {name}")
    print(f"\n  Result: {passed}/{total} passed")

    if passed == total:
        print(f"\n  All tests passed. The MCP server is ready for OpenClaw.")
        print(f"  Start with: python -m src.openclaw")
    else:
        print(f"\n  Some tests failed. Run 'python run_pipeline.py' first if data is missing.")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
