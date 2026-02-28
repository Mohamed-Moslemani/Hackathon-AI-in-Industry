#!/usr/bin/env python3
"""
OpenClaw End-to-End Test
========================
Launches the ConutBakeryOps MCP server and exercises every tool, resource,
and prompt through the FastMCP Client — exactly the way OpenClaw would.

Two modes:
  --in-memory   (default) Import the server object and test in-process.
  --subprocess   Launch `python -m src.openclaw` as a child process and
                 communicate over stdio, mirroring real OpenClaw behaviour.

Run:
    python test_openclaw_e2e.py                 # in-memory
    python test_openclaw_e2e.py --subprocess    # true subprocess over stdio
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import textwrap
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PASS = "[PASS]"
FAIL = "[FAIL]"
SKIP = "[SKIP]"
SEP = "=" * 68
SUB_SEP = "-" * 68

passed = 0
failed = 0
skipped = 0


def _record(ok: bool | None):
    global passed, failed, skipped
    if ok is True:
        passed += 1
    elif ok is False:
        failed += 1
    else:
        skipped += 1


def _preview(text: str, width: int = 90) -> str:
    flat = " ".join(text.split())
    return flat[:width] + ("..." if len(flat) > width else "")


def _parse_json(raw: str) -> dict | list | None:
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None


# -- Scenario helpers --------------------------------------------------------

async def scenario_ping(client):
    """Verify basic connectivity to the MCP server."""
    print(f"\n{SUB_SEP}")
    print("  Scenario: Ping / Connectivity")
    print(SUB_SEP)
    try:
        await client.ping()
        print(f"  {PASS} Server responded to ping")
        _record(True)
    except Exception as e:
        print(f"  {FAIL} Ping failed — {e}")
        _record(False)


async def scenario_list_tools(client):
    """Discover all registered tools and verify the expected set."""
    print(f"\n{SUB_SEP}")
    print("  Scenario: List Tools")
    print(SUB_SEP)
    expected = {
        "get_combo_recommendations",
        "get_demand_forecast",
        "get_expansion_assessment",
        "get_staffing_recommendation",
        "get_beverage_strategy",
        "ask_operations_agent",
        "health_check",
    }
    try:
        tools = await client.list_tools()
        names = {t.name for t in tools}
        print(f"  Discovered {len(tools)} tools: {sorted(names)}")
        missing = expected - names
        extra = names - expected
        if missing:
            print(f"  {FAIL} Missing tools: {missing}")
            _record(False)
        elif extra:
            print(f"  {PASS} All expected tools present (+{len(extra)} extra)")
            _record(True)
        else:
            print(f"  {PASS} All 7 expected tools registered")
            _record(True)
    except Exception as e:
        print(f"  {FAIL} list_tools failed — {e}")
        _record(False)


async def scenario_list_resources(client):
    """Discover all registered resources."""
    print(f"\n{SUB_SEP}")
    print("  Scenario: List Resources")
    print(SUB_SEP)
    try:
        resources = await client.list_resources()
        uris = [str(r.uri) for r in resources]
        print(f"  Discovered {len(resources)} resources: {uris}")
        ok = len(resources) >= 1
        print(f"  {PASS if ok else FAIL} Resource count: {len(resources)}")
        _record(ok)
    except Exception as e:
        print(f"  {FAIL} list_resources failed — {e}")
        _record(False)


async def scenario_list_prompts(client):
    """Discover all registered prompts."""
    print(f"\n{SUB_SEP}")
    print("  Scenario: List Prompts")
    print(SUB_SEP)
    try:
        prompts = await client.list_prompts()
        names = [p.name for p in prompts]
        print(f"  Discovered {len(prompts)} prompts: {names}")
        ok = len(prompts) >= 2
        print(f"  {PASS if ok else FAIL} Prompt count: {len(prompts)}")
        _record(ok)
    except Exception as e:
        print(f"  {FAIL} list_prompts failed — {e}")
        _record(False)


async def scenario_health_check(client):
    """Call health_check and verify the server reports healthy."""
    print(f"\n{SUB_SEP}")
    print("  Scenario: Health Check Tool")
    print(SUB_SEP)
    try:
        result = await client.call_tool("health_check", {})
        text = result.data if isinstance(result.data, str) else str(result)
        data = _parse_json(text)
        if data and isinstance(data, dict):
            status = data.get("status")
            version = data.get("server_version")
            engine = data.get("engine_loaded")
            print(f"  Status: {status}  |  Version: {version}  |  Engine loaded: {engine}")
            ok = status == "healthy" and engine is True
            print(f"  {PASS if ok else FAIL} Health check {'passed' if ok else 'degraded'}")
            _record(ok)
        else:
            print(f"  {FAIL} Unexpected response: {_preview(text)}")
            _record(False)
    except Exception as e:
        print(f"  {FAIL} health_check failed — {e}")
        _record(False)


async def _call_tool_scenario(client, tool_name: str, args: dict, label: str, validate=None):
    """Generic helper: call a tool, parse JSON, optionally run a validator."""
    try:
        result = await client.call_tool(tool_name, args)
        text = result.data if isinstance(result.data, str) else str(result)
        data = _parse_json(text)
        if data is None:
            print(f"  {FAIL} {label} — response is not valid JSON")
            _record(False)
            return
        if isinstance(data, dict) and data.get("error"):
            print(f"  {FAIL} {label} — server error: {data.get('message', '?')}")
            _record(False)
            return
        if validate:
            ok, detail = validate(data)
        else:
            ok, detail = True, _preview(text)
        print(f"  {PASS if ok else FAIL} {label} — {detail}")
        _record(ok)
    except Exception as e:
        print(f"  {FAIL} {label} — exception: {e}")
        _record(False)


async def scenario_combo_tools(client):
    """Test combo recommendations: network-wide and per-branch."""
    print(f"\n{SUB_SEP}")
    print("  Scenario: Combo Optimization Tools")
    print(SUB_SEP)

    await _call_tool_scenario(
        client, "get_combo_recommendations", {},
        "Network-wide combos",
        lambda d: (bool(d.get("top_2item_combos")), f"{len(d.get('top_2item_combos', []))} 2-item combos returned"),
    )
    await _call_tool_scenario(
        client, "get_combo_recommendations", {"branch": "Conut Jnah", "top_n": 3},
        "Branch combos (Conut Jnah, top 3)",
        lambda d: (d.get("branch") == "Conut Jnah", f"branch={d.get('branch')}, count={d.get('count')}"),
    )


async def scenario_demand_tools(client):
    """Test demand forecasting: all branches and single branch."""
    print(f"\n{SUB_SEP}")
    print("  Scenario: Demand Forecasting Tools")
    print(SUB_SEP)

    await _call_tool_scenario(
        client, "get_demand_forecast", {},
        "Network-wide forecast",
        lambda d: (d.get("network_q1_total") is not None, f"Q1 total={d.get('network_q1_total')}"),
    )
    await _call_tool_scenario(
        client, "get_demand_forecast", {"branch": "Main Street Coffee"},
        "Branch forecast (Main Street Coffee)",
        lambda d: ("Main Street" in str(d.get("branch", "")), f"branch={d.get('branch')}, Q1={d.get('q1_2026_total_forecast')}"),
    )


async def scenario_expansion_tool(client):
    """Test expansion feasibility assessment."""
    print(f"\n{SUB_SEP}")
    print("  Scenario: Expansion Feasibility Tool")
    print(SUB_SEP)

    await _call_tool_scenario(
        client, "get_expansion_assessment", {},
        "Expansion verdict",
        lambda d: (d.get("verdict") is not None, f"verdict={d.get('verdict')}, health={d.get('network_health_score')}"),
    )


async def scenario_staffing_tools(client):
    """Test staffing recommendations: network-wide and per-branch."""
    print(f"\n{SUB_SEP}")
    print("  Scenario: Staffing Recommendation Tools")
    print(SUB_SEP)

    await _call_tool_scenario(
        client, "get_staffing_recommendation", {},
        "Network-wide staffing",
        lambda d: (d.get("branch_staffing") is not None, f"{len(d.get('branch_staffing', {}))} branches"),
    )
    await _call_tool_scenario(
        client, "get_staffing_recommendation", {"branch": "Conut - Tyre"},
        "Branch staffing (Conut - Tyre)",
        lambda d: ("Tyre" in str(d.get("branch", "")), f"branch={d.get('branch')}, staff={d.get('current_staff')}"),
    )


async def scenario_beverage_tools(client):
    """Test beverage growth strategy: network-wide and per-branch."""
    print(f"\n{SUB_SEP}")
    print("  Scenario: Beverage Strategy Tools")
    print(SUB_SEP)

    await _call_tool_scenario(
        client, "get_beverage_strategy", {},
        "Network-wide beverage strategy",
        lambda d: (d.get("total_estimated_uplift") is not None, f"uplift={d.get('total_estimated_uplift')}"),
    )
    await _call_tool_scenario(
        client, "get_beverage_strategy", {"branch": "Conut"},
        "Branch beverage strategy (Conut)",
        lambda d: (d.get("branch") == "Conut", f"branch={d.get('branch')}, actions={d.get('num_actions')}"),
    )


async def scenario_natural_language(client):
    """Test the ask_operations_agent catch-all with various questions."""
    print(f"\n{SUB_SEP}")
    print("  Scenario: Natural-Language Query Router")
    print(SUB_SEP)

    questions = [
        ("What combos should we offer?", "combo"),
        ("What is the demand forecast for next quarter?", "demand"),
        ("Should we expand to a new location?", "expansion"),
        ("How many staff do we need at Conut Jnah?", "staffing"),
        ("What coffee products should we introduce?", "beverage"),
        ("Give me a full business summary.", "summary"),
    ]
    for question, expected_topic in questions:
        await _call_tool_scenario(
            client, "ask_operations_agent", {"question": question},
            f"NL: \"{question[:50]}\"",
        )


async def scenario_read_resources(client):
    """Read each MCP resource and validate the response."""
    print(f"\n{SUB_SEP}")
    print("  Scenario: Read MCP Resources")
    print(SUB_SEP)

    resource_uris = [
        "conut://branches",
        "conut://summary",
        "conut://system-info",
    ]
    for uri in resource_uris:
        try:
            content = await client.read_resource(uri)
            text = content[0].text if hasattr(content[0], "text") else str(content)
            data = _parse_json(text)
            ok = data is not None
            print(f"  {PASS if ok else FAIL} {uri} — {len(text)} chars, valid JSON: {ok}")
            _record(ok)
        except Exception as e:
            print(f"  {FAIL} {uri} — {e}")
            _record(False)


async def scenario_get_prompts(client):
    """Render each MCP prompt and verify the output."""
    print(f"\n{SUB_SEP}")
    print("  Scenario: Render MCP Prompts")
    print(SUB_SEP)

    try:
        result = await client.get_prompt("daily_operations_briefing", {})
        text = str(result.messages[0].content) if result.messages else ""
        ok = len(text) > 20
        print(f"  {PASS if ok else FAIL} daily_operations_briefing — {len(text)} chars")
        _record(ok)
    except Exception as e:
        print(f"  {FAIL} daily_operations_briefing — {e}")
        _record(False)

    try:
        result = await client.get_prompt("branch_deep_dive", {"branch": "Conut Jnah"})
        text = str(result.messages[0].content) if result.messages else ""
        ok = "Conut Jnah" in text
        print(f"  {PASS if ok else FAIL} branch_deep_dive(Conut Jnah) — branch name in prompt: {ok}")
        _record(ok)
    except Exception as e:
        print(f"  {FAIL} branch_deep_dive — {e}")
        _record(False)


async def scenario_config_dynamic(client):
    """Verify openclaw.config.json uses a relative (non-hardcoded) cwd."""
    print(f"\n{SUB_SEP}")
    print("  Scenario: Config — Dynamic Path Check")
    print(SUB_SEP)

    config_path = PROJECT_ROOT / "openclaw.config.json"
    if not config_path.exists():
        print(f"  {FAIL} openclaw.config.json not found")
        _record(False)
        return

    with open(config_path) as f:
        config = json.load(f)

    srv = config.get("mcpServers", {}).get("conut-bakery-ops", {})
    cwd_value = srv.get("cwd", "")

    is_absolute = Path(cwd_value).is_absolute() if cwd_value else False
    if is_absolute:
        print(f"  {FAIL} cwd is an absolute path: {cwd_value}")
        print(f"         Config should use a relative path (e.g. \".\") for portability.")
        _record(False)
    else:
        print(f"  {PASS} cwd is relative: \"{cwd_value}\"")
        _record(True)

    has_pythonpath = "PYTHONPATH" in srv.get("env", {})
    print(f"  {PASS if has_pythonpath else FAIL} PYTHONPATH set in env: {has_pythonpath}")
    _record(has_pythonpath)


# -- Main orchestrator -------------------------------------------------------

async def run_all_scenarios(client):
    """Execute every scenario in order."""
    await scenario_ping(client)
    await scenario_config_dynamic(client)
    await scenario_list_tools(client)
    await scenario_list_resources(client)
    await scenario_list_prompts(client)
    await scenario_health_check(client)
    await scenario_combo_tools(client)
    await scenario_demand_tools(client)
    await scenario_expansion_tool(client)
    await scenario_staffing_tools(client)
    await scenario_beverage_tools(client)
    await scenario_natural_language(client)
    await scenario_read_resources(client)
    await scenario_get_prompts(client)


async def run_in_memory():
    """Import the server object and test in-process (fast, no subprocess)."""
    from fastmcp import Client
    from src.openclaw.server import mcp as server

    client = Client(server)
    async with client:
        await run_all_scenarios(client)


async def run_subprocess():
    """Launch the MCP server as a subprocess and test over stdio."""
    from fastmcp import Client
    from fastmcp.client.transports import PythonStdioTransport

    transport = PythonStdioTransport(
        script_path="-m",
        args=["src.openclaw"],
        cwd=str(PROJECT_ROOT),
        env={"PYTHONPATH": "."},
    )
    client = Client(transport)
    async with client:
        await run_all_scenarios(client)


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end test for the ConutBakeryOps MCP server",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--in-memory", action="store_true", default=True,
        help="Test by importing the server in-process (default)",
    )
    mode.add_argument(
        "--subprocess", action="store_true",
        help="Test by launching the server as a child process over stdio",
    )
    args = parser.parse_args()

    print(SEP)
    print("  CONUT BAKERY — OpenClaw End-to-End Test")
    transport_label = "subprocess (stdio)" if args.subprocess else "in-memory"
    print(f"  Transport: {transport_label}")
    print(SEP)

    t0 = time.perf_counter()

    if args.subprocess:
        asyncio.run(run_subprocess())
    else:
        asyncio.run(run_in_memory())

    elapsed = time.perf_counter() - t0

    print(f"\n{SEP}")
    print("  SUMMARY")
    print(SEP)
    total = passed + failed + skipped
    print(f"  Passed:  {passed}")
    print(f"  Failed:  {failed}")
    print(f"  Skipped: {skipped}")
    print(f"  Total:   {total}")
    print(f"  Time:    {elapsed:.2f}s")

    if failed == 0:
        print(f"\n  {PASS} All scenarios passed. The MCP server is ready for OpenClaw.")
        print(f"  Start with: python -m src.openclaw")
    else:
        print(f"\n  {FAIL} {failed} scenario(s) failed.")
        print(f"  Ensure 'python run_pipeline.py' has been run to generate analytics data.")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
