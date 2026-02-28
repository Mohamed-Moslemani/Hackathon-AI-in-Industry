"""Orchestrator that runs all cleaners and persists cleaned DataFrames."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import pandas as pd

from ..config import CLEANED_DATA_DIR, RAW_FILES
from .attendance_cleaner import AttendanceCleaner
from .avg_sales_cleaner import AvgSalesCleaner
from .customer_orders_cleaner import CustomerOrdersCleaner
from .division_summary_cleaner import DivisionSummaryCleaner
from .monthly_sales_cleaner import MonthlySalesCleaner
from .sales_by_customer_cleaner import SalesByCustomerCleaner
from .sales_by_item_cleaner import SalesByItemCleaner
from .tax_report_cleaner import TaxReportCleaner

logger = logging.getLogger(__name__)

CLEANER_MAP = {
    "division_summary": DivisionSummaryCleaner,
    "tax_report": TaxReportCleaner,
    "attendance": AttendanceCleaner,
    "sales_by_customer": SalesByCustomerCleaner,
    "customer_orders": CustomerOrdersCleaner,
    "sales_by_item": SalesByItemCleaner,
    "monthly_sales": MonthlySalesCleaner,
    "avg_sales": AvgSalesCleaner,
}


class DataPipeline:
    """Run all cleaners, cache results, and optionally save to disk."""

    def __init__(self, output_dir: Path | str | None = None) -> None:
        self.output_dir = Path(output_dir) if output_dir else CLEANED_DATA_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.datasets: Dict[str, pd.DataFrame] = {}

    def run(self, *, save: bool = True) -> Dict[str, pd.DataFrame]:
        """Execute every cleaner and return a dict of DataFrames."""
        for name, cleaner_cls in CLEANER_MAP.items():
            filepath = RAW_FILES.get(name)
            if filepath is None or not filepath.exists():
                logger.warning("Skipping %s — file not found: %s", name, filepath)
                continue
            try:
                cleaner = cleaner_cls(filepath)
                df = cleaner.clean()
                self.datasets[name] = df
                if save:
                    out_path = self.output_dir / f"{name}.csv"
                    df.to_csv(out_path, index=False)
                    logger.info("Saved %s -> %s (%d rows)", name, out_path, len(df))
            except Exception:
                logger.exception("Failed to clean %s", name)

        return self.datasets

    def get(self, name: str) -> pd.DataFrame:
        if name not in self.datasets:
            raise KeyError(
                f"Dataset '{name}' not found. Available: {list(self.datasets.keys())}"
            )
        return self.datasets[name]

    def summary(self) -> pd.DataFrame:
        """Return a summary table of all cleaned datasets."""
        info = []
        for name, df in self.datasets.items():
            info.append(
                {
                    "dataset": name,
                    "rows": len(df),
                    "columns": len(df.columns),
                    "column_names": ", ".join(df.columns.tolist()),
                }
            )
        return pd.DataFrame(info)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )
    pipeline = DataPipeline()
    pipeline.run(save=True)
    print("\n" + "=" * 60)
    print("DATA CLEANING COMPLETE")
    print("=" * 60)
    summary = pipeline.summary()
    print(summary.to_string(index=False))
    print(f"\nCleaned files saved to: {pipeline.output_dir}")
