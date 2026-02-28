"""Orchestrator that runs all analytics modules and persists outputs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

from ..config import ANALYTICS_DIR, FEATURES_DIR
from .combo_recommender import ComboRecommender
from .demand_forecaster import DemandForecaster
from .expansion_analyzer import ExpansionAnalyzer
from .staffing_estimator import StaffingEstimator
from .beverage_strategist import BeverageStrategist

logger = logging.getLogger(__name__)

ANALYTICS_MAP = {
    "combo": ComboRecommender,
    "demand": DemandForecaster,
    "expansion": ExpansionAnalyzer,
    "staffing": StaffingEstimator,
    "beverage": BeverageStrategist,
}


class AnalyticsPipeline:
    """Run all analytics modules, cache results, and optionally save to disk."""

    def __init__(
        self,
        features_dir: Path | str | None = None,
        output_dir: Path | str | None = None,
    ) -> None:
        self.features_dir = Path(features_dir) if features_dir else FEATURES_DIR
        self.output_dir = Path(output_dir) if output_dir else ANALYTICS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: Dict[str, Dict[str, Any]] = {}

    def run(self, *, save: bool = True) -> Dict[str, Dict[str, Any]]:
        for name, cls in ANALYTICS_MAP.items():
            try:
                module = cls(
                    features_dir=self.features_dir,
                    output_dir=self.output_dir,
                )
                result = module.run(save=save)
                self.results[name] = result
                logger.info("Completed %s analytics module", name)
            except Exception:
                logger.exception("Failed to run %s analytics module", name)

        return self.results

    def get(self, name: str) -> Dict[str, Any]:
        if name not in self.results:
            raise KeyError(
                f"Analytics '{name}' not found. Available: {list(self.results.keys())}"
            )
        return self.results[name]


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )
    pipeline = AnalyticsPipeline()
    pipeline.run(save=True)
    print("\n" + "=" * 60)
    print("ANALYTICS COMPLETE")
    print("=" * 60)
    for name in pipeline.results:
        keys = list(pipeline.results[name].keys())
        print(f"  {name}: {keys}")
    print(f"\nAnalytics files saved to: {pipeline.output_dir}")
