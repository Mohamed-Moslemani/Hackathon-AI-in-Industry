"""Orchestrator that runs all feature engineering builders and persists outputs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import pandas as pd

from ..config import CLEANED_DATA_DIR, FEATURES_DIR
from .combo_features import ComboFeatureBuilder
from .demand_features import DemandFeatureBuilder
from .expansion_features import ExpansionFeatureBuilder
from .staffing_features import StaffingFeatureBuilder
from .beverage_features import BeverageFeatureBuilder

logger = logging.getLogger(__name__)

BUILDER_MAP = {
    "combo": ComboFeatureBuilder,
    "demand": DemandFeatureBuilder,
    "expansion": ExpansionFeatureBuilder,
    "staffing": StaffingFeatureBuilder,
    "beverage": BeverageFeatureBuilder,
}


class FeaturePipeline:
    """Run all feature builders, cache results, and optionally save to disk."""

    def __init__(
        self,
        cleaned_dir: Path | str | None = None,
        output_dir: Path | str | None = None,
    ) -> None:
        self.cleaned_dir = Path(cleaned_dir) if cleaned_dir else CLEANED_DATA_DIR
        self.output_dir = Path(output_dir) if output_dir else FEATURES_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: Dict[str, Dict[str, pd.DataFrame]] = {}

    def run(self, *, save: bool = True) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Execute every feature builder and return a nested dict of DataFrames."""
        for name, builder_cls in BUILDER_MAP.items():
            try:
                builder = builder_cls(
                    cleaned_dir=self.cleaned_dir,
                    output_dir=self.output_dir,
                )
                result = builder.run(save=save)
                self.results[name] = result
                logger.info("Completed %s feature builder", name)
            except Exception:
                logger.exception("Failed to run %s feature builder", name)

        return self.results

    def summary(self) -> pd.DataFrame:
        """Return a summary table of all feature outputs."""
        info = []
        for builder_name, datasets in self.results.items():
            for dataset_name, df in datasets.items():
                if not isinstance(df, pd.DataFrame):
                    continue
                info.append({
                    "builder": builder_name,
                    "dataset": dataset_name,
                    "rows": len(df),
                    "columns": len(df.columns),
                    "column_names": ", ".join(df.columns.tolist()),
                })
        return pd.DataFrame(info)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )
    pipeline = FeaturePipeline()
    pipeline.run(save=True)
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 60)
    summary = pipeline.summary()
    print(summary[["builder", "dataset", "rows", "columns"]].to_string(index=False))
    print(f"\nFeature files saved to: {pipeline.output_dir}")
