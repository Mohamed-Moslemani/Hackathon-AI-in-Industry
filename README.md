# Conut Bakery — AI-Driven Chief of Operations Agent

An end-to-end AI analytics system that turns Conut Bakery's operational data into actionable business decisions, integrated with **OpenClaw** via the Model Context Protocol (MCP).

---

## Business Problem

Conut is a growing sweets and beverages business operating across multiple branches. Management needs data-driven answers to five operational questions:

1. **Combo Optimization** — Which products should be bundled together?
2. **Demand Forecasting** — What will each branch sell next quarter?
3. **Expansion Feasibility** — Should we open a new location?
4. **Shift Staffing** — How many employees per shift at each branch?
5. **Beverage Growth** — How do we grow coffee and milkshake sales?

This system ingests raw POS data, cleans it, engineers features, runs analytics, and exposes the results through an inference engine that OpenClaw can query in natural language.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        OpenClaw Agent                           │
│              (asks questions, receives answers)                  │
└──────────────────────────┬──────────────────────────────────────┘
                           │ MCP (stdio)
┌──────────────────────────▼──────────────────────────────────────┐
│                   ConutBakeryOps MCP Server                     │
│              src/openclaw/server.py (FastMCP)                   │
│   7 tools · 3 resources · 2 prompts                            │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                     Inference Engine                             │
│                  src/inference/engine.py                         │
│   Loads pre-computed analytics · Routes queries · Fuzzy match   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                    Analytics Pipeline                            │
│                  src/analytics/pipeline.py                       │
│   ComboRecommender · DemandForecaster · ExpansionAnalyzer       │
│   StaffingEstimator · BeverageStrategist                        │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                 Feature Engineering Pipeline                     │
│              src/feature_engineering/pipeline.py                 │
│   ComboFeatures · DemandFeatures · ExpansionFeatures            │
│   StaffingFeatures · BeverageFeatures                           │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                   Data Cleaning Pipeline                         │
│                src/data_cleaning/pipeline.py                     │
│   8 cleaners for report-style POS CSVs                          │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                      Raw Data (CSVs)                            │
│               Conut bakery Scaled Data/                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
Hackathon/
├── run_pipeline.py                  # Main entry point — runs the full pipeline
├── test_openclaw_integration.py     # Quick validation of MCP server and tools
├── test_openclaw_e2e.py             # Full end-to-end MCP scenario test suite
├── openclaw.config.json             # OpenClaw MCP server configuration
├── requirements.txt                 # Python dependencies
├── CONUT_AI_ENGINEERING_HACKATHON.md
│
├── Conut bakery Scaled Data/        # Raw input CSVs (9 files)
│   ├── REP_S_00136_SMRY.csv        #   Division/menu channel summary
│   ├── REP_S_00194_SMRY.csv        #   Tax summary by branch
│   ├── REP_S_00461.csv             #   Time and attendance logs
│   ├── REP_S_00502.csv             #   Sales by customer (line-item)
│   ├── rep_s_00150.csv             #   Customer orders with timestamps
│   ├── rep_s_00191_SMRY.csv        #   Sales by items and groups
│   ├── rep_s_00334_1_SMRY.csv      #   Monthly sales by branch
│   ├── rep_s_00435_SMRY.csv        #   Average sales by menu
│   └── rep_s_00435_SMRY (1).csv    #   Duplicate report version
│
├── data/                            # Pipeline outputs (auto-generated)
│   ├── cleaned/                     #   8 cleaned CSVs
│   ├── features/                    #   16 feature datasets
│   └── analytics/                   #   14 analytics artifacts (CSV + JSON)
│
├── src/
│   ├── config.py                    # Paths and raw file mapping
│   │
│   ├── data_cleaning/               # Stage 1: Data ingestion & cleaning
│   │   ├── pipeline.py              #   DataPipeline orchestrator
│   │   ├── base_cleaner.py          #   Base class (noise removal, parsing)
│   │   ├── division_summary_cleaner.py
│   │   ├── tax_report_cleaner.py
│   │   ├── attendance_cleaner.py
│   │   ├── sales_by_customer_cleaner.py
│   │   ├── customer_orders_cleaner.py
│   │   ├── sales_by_item_cleaner.py
│   │   ├── monthly_sales_cleaner.py
│   │   └── avg_sales_cleaner.py
│   │
│   ├── feature_engineering/         # Stage 2: Feature engineering
│   │   ├── pipeline.py              #   FeaturePipeline orchestrator
│   │   ├── combo_features.py        #   Basket analysis, co-purchase pairs
│   │   ├── demand_features.py       #   Monthly trends, branch profiles
│   │   ├── expansion_features.py    #   Branch scorecards, benchmarks
│   │   ├── staffing_features.py     #   Shift patterns, daily coverage
│   │   └── beverage_features.py     #   Beverage share, growth gaps
│   │
│   ├── analytics/                   # Stage 3: Modeling & analytics
│   │   ├── pipeline.py              #   AnalyticsPipeline orchestrator
│   │   ├── combo_recommender.py     #   Association rules, combo ranking
│   │   ├── demand_forecaster.py     #   Ensemble forecasting (Q1 2026)
│   │   ├── expansion_analyzer.py    #   Go/no-go verdict, risk analysis
│   │   ├── staffing_estimator.py    #   Shift staffing, hiring recs
│   │   └── beverage_strategist.py   #   Growth actions, product opportunities
│   │
│   ├── inference/                   # Stage 4: Query & inference layer
│   │   └── engine.py               #   InferenceEngine (loads analytics, routes queries)
│   │
│   └── openclaw/                    # OpenClaw MCP integration
│       ├── server.py               #   FastMCP server (7 tools, 3 resources, 2 prompts)
│       ├── __main__.py             #   python -m src.openclaw entry point
│       └── SKILL.md                #   OpenClaw skill documentation
│
└── tests/
    └── test_cleaners.py            # Unit tests for data cleaners
```

---

## Prerequisites

- **Python 3.10+**
- **OpenClaw** installed and running (for the MCP integration)

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline

```bash
python run_pipeline.py
```

This executes all four stages sequentially and prints a full report:

| Stage | What it does | Output directory |
|-------|-------------|-----------------|
| **Stage 1: Data Cleaning** | Ingests 8 raw POS CSVs, removes noise/headers, parses numbers and dates | `data/cleaned/` |
| **Stage 2: Feature Engineering** | Builds 16 feature datasets for all 5 business objectives | `data/features/` |
| **Stage 3: Analytics** | Runs 5 analytics modules, produces forecasts, recommendations, and verdicts | `data/analytics/` |
| **Stage 4: Reporting** | Loads analytics into the inference engine and prints a formatted report | (console output) |

### 3. Verify OpenClaw integration

```bash
python test_openclaw_e2e.py                # in-memory (fast, no subprocess)
python test_openclaw_e2e.py --subprocess   # launches the MCP server as a child process over stdio
```

Runs a full end-to-end scenario suite: pings the server, discovers tools/resources/prompts, calls every tool with realistic arguments, reads all resources, renders prompts, and validates the config file uses portable paths.

### 4. Start the MCP server

```bash
python -m src.openclaw
```

The server starts in **stdio mode** and waits for MCP protocol messages from OpenClaw.

### 5. Connect OpenClaw

Copy the MCP server config into your OpenClaw configuration file (`~/.openclaw/openclaw.json`), replacing `cwd` with the absolute path to this project on your machine:

```json
{
  "mcpServers": {
    "conut-bakery-ops": {
      "command": "python",
      "args": ["-m", "src.openclaw"],
      "cwd": "/absolute/path/to/Hackathon",
      "env": {
        "PYTHONPATH": "."
      }
    }
  }
}
```

> **Note:** The project's `openclaw.config.json` uses a relative `cwd` of `"."` so it works from any location. When registering with OpenClaw globally, replace `cwd` with the actual absolute path.

Then restart the OpenClaw gateway. OpenClaw will auto-launch the MCP server and discover all tools.

---

## Running Individual Stages

You can run any single stage independently:

```bash
python run_pipeline.py --stage clean       # Stage 1 only
python run_pipeline.py --stage features    # Stage 2 only
python run_pipeline.py --stage analytics   # Stage 3 only
python run_pipeline.py --stage report      # Stage 4 only (report from existing data)
```

Shortcut to regenerate just the report without recomputing:

```bash
python run_pipeline.py --report
```

Enable verbose logging for debugging:

```bash
python run_pipeline.py -v
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Pipeline Details

### Stage 1: Data Cleaning

The raw CSVs are Omega POS report-style exports with repeated headers, page markers, and inconsistent formatting. Each cleaner handles a specific report type:

| Cleaner | Input File | Output |
|---------|-----------|--------|
| `DivisionSummaryCleaner` | `REP_S_00136_SMRY.csv` | `division_summary.csv` |
| `TaxReportCleaner` | `REP_S_00194_SMRY.csv` | `tax_report.csv` |
| `AttendanceCleaner` | `REP_S_00461.csv` | `attendance.csv` |
| `SalesByCustomerCleaner` | `REP_S_00502.csv` | `sales_by_customer.csv` |
| `CustomerOrdersCleaner` | `rep_s_00150.csv` | `customer_orders.csv` |
| `SalesByItemCleaner` | `rep_s_00191_SMRY.csv` | `sales_by_item.csv` |
| `MonthlySalesCleaner` | `rep_s_00334_1_SMRY.csv` | `monthly_sales.csv` |
| `AvgSalesCleaner` | `rep_s_00435_SMRY.csv` | `avg_sales.csv` |

### Stage 2: Feature Engineering

Each builder reads from `data/cleaned/` and writes to `data/features/`:

| Builder | Features Produced |
|---------|------------------|
| **ComboFeatures** | Baskets, co-purchase pairs, association rules, item affinity |
| **DemandFeatures** | Branch monthly trends, branch profiles, forecast dataset |
| **ExpansionFeatures** | Branch scorecards, expansion benchmarks, branch comparison |
| **StaffingFeatures** | Shift patterns, daily coverage, branch staffing profiles |
| **BeverageFeatures** | Beverage branch summary, growth gaps, product performance |

### Stage 3: Analytics

Each module reads from `data/features/` and writes to `data/analytics/`:

| Module | What it produces |
|--------|-----------------|
| **ComboRecommender** | Top 2-item and 3-item combos by lift/confidence, branch-level combos |
| **DemandForecaster** | Q1 2026 forecasts per branch (ensemble of linear trend, WMA, growth-rate) |
| **ExpansionAnalyzer** | Go/no-go verdict, network health score, risk matrix, ideal branch profile |
| **StaffingEstimator** | Per-shift staffing recommendations, weekly schedules, hiring actions |
| **BeverageStrategist** | Strategy actions (INTRODUCE/PROMOTE/OPTIMIZE), product opportunities, uplift estimates |

### Stage 4: Inference & Reporting

The `InferenceEngine` loads all 14 analytics artifacts and provides:

- **Typed methods**: `get_combo_recommendations()`, `get_demand_forecast()`, `get_expansion_assessment()`, `get_staffing_recommendation()`, `get_beverage_strategy()`
- **Natural-language router**: `query("What combos should we offer at Conut Jnah?")` — keyword-matches to the right module
- **Full summary**: `_full_summary()` — high-level overview across all 5 objectives

---

## OpenClaw Integration

### How It Works

The MCP server (`src/openclaw/server.py`) wraps the `InferenceEngine` in a FastMCP server that speaks the **Model Context Protocol**. OpenClaw launches the server as a subprocess (stdio transport) and discovers all registered tools, resources, and prompts.

### Available MCP Tools

| Tool | Description | Parameters |
|------|------------|------------|
| `health_check` | Verify system readiness and data status | (none) |
| `get_combo_recommendations` | Product bundle suggestions | `branch?`, `top_n?` |
| `get_demand_forecast` | Q1 2026 revenue forecasts | `branch?` |
| `get_expansion_assessment` | Expansion go/no-go verdict | (none) |
| `get_staffing_recommendation` | Shift staffing and hiring recs | `branch?` |
| `get_beverage_strategy` | Coffee & milkshake growth actions | `branch?` |
| `ask_operations_agent` | Natural-language catch-all | `question`, `branch?` |

### Available MCP Resources

| URI | Description |
|-----|------------|
| `conut://branches` | List of all branches with forecast metadata |
| `conut://summary` | High-level summary across all 5 objectives |
| `conut://system-info` | Server version, capabilities, and configuration |

### Available MCP Prompts

| Prompt | Description | Parameters |
|--------|------------|------------|
| `daily_operations_briefing` | Management briefing template covering all objectives | (none) |
| `branch_deep_dive` | Deep-dive analysis for a single branch | `branch` |

### Example OpenClaw Queries

Once connected, you can ask OpenClaw:

- *"What combos should we offer at Conut Jnah?"*
- *"What is the demand forecast for Main Street Coffee?"*
- *"Should we expand to a new location?"*
- *"How many staff do we need at Conut - Tyre for the morning shift?"*
- *"What coffee products should we introduce?"*
- *"Give me a daily operations briefing."*

### Branches

The system covers four Conut Bakery branches:

- **Conut** — Main branch
- **Conut - Tyre** — Tyre location
- **Conut Jnah** — Jnah location
- **Main Street Coffee** — Coffee-focused branch

---

## Data Notes

- Numeric values in the raw data are intentionally transformed to arbitrary units (scaled dataset).
- All analysis focuses on **patterns, ratios, and relative comparisons**, not absolute values.
- Customer and employee names are anonymized.

---

## Key Results

| Objective | Key Finding |
|-----------|------------|
| **Combos** | Top combo is Triple Chocolate + related items; branch-specific combos identified |
| **Demand** | Q1 2026 forecasts generated for all branches; fastest-growing branch identified |
| **Expansion** | Current verdict with risk analysis and ideal new-branch profile |
| **Staffing** | Per-shift recommendations with hiring actions per branch |
| **Beverages** | Prioritized INTRODUCE/PROMOTE/OPTIMIZE actions with revenue uplift estimates |

Run `python run_pipeline.py --report` to see the full formatted report with all numbers.

---

## Troubleshooting

| Problem | Solution |
|---------|---------|
| `FileNotFoundError` on raw CSVs | Ensure `Conut bakery Scaled Data/` folder is in the project root |
| Tools return `"error": true` | Run `python run_pipeline.py` to generate analytics data first |
| OpenClaw can't find the server | When registering globally, set `cwd` to the absolute path of this project |
| `ModuleNotFoundError` | Run from the project root, or set `PYTHONPATH=.` |
| MCP server won't start | Verify `pip install -r requirements.txt` completed successfully |
