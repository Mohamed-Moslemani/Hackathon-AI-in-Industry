# Conut Bakery — Chief of Operations Agent

## About
AI-driven operations agent for Conut Bakery. Provides data-backed answers for combo optimization, demand forecasting, expansion feasibility, shift staffing, and beverage growth strategy across four branches.

## Branches
- **Conut** — Main branch
- **Conut - Tyre** — Tyre location
- **Conut Jnah** — Jnah location
- **Main Street Coffee** — Coffee-focused branch

## Tools

### `health_check`
Call first to verify the system is operational and data is loaded.

### `get_combo_recommendations`
Product bundle suggestions. Pass `branch` to filter by location, `top_n` to control count.

### `get_demand_forecast`
Q1 2026 revenue forecasts per branch. Pass `branch` for a single branch or omit for all.

### `get_expansion_assessment`
Go/no-go verdict on opening a new branch. No parameters needed.

### `get_staffing_recommendation`
Shift staffing and hiring recommendations. Pass `branch` for details or omit for network summary.

### `get_beverage_strategy`
Coffee and milkshake growth actions. Pass `branch` for branch-specific strategy.

### `ask_operations_agent`
Natural-language catch-all. Pass any business question as `question` and optionally a `branch`.

## Prompts

### `daily_operations_briefing`
Generates a management briefing covering all five objectives.

### `branch_deep_dive`
Deep analysis for a single branch. Requires `branch` parameter.

## Workflow
1. Call `health_check` to confirm readiness
2. Use specific tools for targeted questions
3. Use `ask_operations_agent` for open-ended questions
4. Use prompts for structured reporting
