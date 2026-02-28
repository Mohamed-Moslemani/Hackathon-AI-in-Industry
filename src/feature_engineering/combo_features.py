"""Feature engineering for Combo Optimization.

Builds customer transaction baskets from line-item sales data and computes
co-purchase statistics and association rules for product combo recommendations.

Data source: cleaned sales_by_customer.csv (line-item delivery orders)
             cleaned sales_by_item.csv    (item metadata with group/division)
"""

from __future__ import annotations

import logging
import re
from itertools import combinations
from pathlib import Path
from typing import Dict, FrozenSet, List, Set, Tuple

import pandas as pd

from ..config import CLEANED_DATA_DIR, FEATURES_DIR

logger = logging.getLogger(__name__)

# Groups that represent free modifiers, toppings, or options — not standalone
# products a customer would intentionally "combo".
MODIFIER_GROUPS: Set[str] = {
    "Free Dressing",
    "Free Whipped Cream",
    "Free Chimney Cake Spreads",
    "Free Conut Spreads",
    "FREE CHIMNEY TOP",
    "FREE CONUT TOP",
    "FREE MINI TOP",
    "FREE MINI SPREAD",
    "CHIMNEY CAKE OPTIONS",
    "OPTIONS ICE CREAM",
    "MARSHMALLOW OPTIONS",
    "MILK OPTIONS",
    "DRINK TYPE",
    "coffee type",
    "free dip",
    "TEA FLAVORS",
    "Holders",
}

# Description patterns that are modifiers/service charges, not real products.
MODIFIER_PATTERNS: List[re.Pattern] = [
    re.compile(r"^\[.*\]$"),                  # [CHOCOLATE DRESSING], [NO DRESSING]
    re.compile(r"DELIVERY CHARGE", re.I),
    re.compile(r"^NO\s", re.I),               # NO TOPPINGS, NO SPREAD, NO WHIPPED CREAM
    re.compile(r"^PRESSED$", re.I),
    re.compile(r"^REGULAR\.?$", re.I),
    re.compile(r"WHIPPED CREAM\.\.\.", re.I),
    re.compile(r"FULL FAT MILK$", re.I),
    re.compile(r"^WATER$", re.I),
]


class ComboFeatureBuilder:
    """Build combo-optimization features from cleaned sales data."""

    def __init__(
        self,
        cleaned_dir: Path | str | None = None,
        output_dir: Path | str | None = None,
    ) -> None:
        self.cleaned_dir = Path(cleaned_dir) if cleaned_dir else CLEANED_DATA_DIR
        self.output_dir = Path(output_dir) if output_dir else FEATURES_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._sales: pd.DataFrame | None = None
        self._item_meta: pd.DataFrame | None = None
        self._baskets: pd.DataFrame | None = None

    def run(self, *, save: bool = True) -> Dict[str, pd.DataFrame]:
        """Execute the full combo feature pipeline."""
        self._load_data()
        modifier_set = self._build_modifier_set()
        transactions = self._build_baskets(modifier_set)
        copurchase = self._compute_copurchase_matrix(transactions)
        rules = self._compute_association_rules(transactions)
        item_affinity = self._compute_item_affinity(transactions)

        results = {
            "baskets": transactions,
            "copurchase_pairs": copurchase,
            "association_rules": rules,
            "item_affinity": item_affinity,
        }

        if save:
            for name, df in results.items():
                path = self.output_dir / f"combo_{name}.csv"
                df.to_csv(path, index=False)
                logger.info("Saved %s -> %s (%d rows)", name, path, len(df))

        return results

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_data(self) -> None:
        self._sales = pd.read_csv(self.cleaned_dir / "sales_by_customer.csv")
        self._item_meta = pd.read_csv(self.cleaned_dir / "sales_by_item.csv")
        logger.info(
            "Loaded sales_by_customer (%d rows) and sales_by_item (%d rows)",
            len(self._sales),
            len(self._item_meta),
        )

    # ------------------------------------------------------------------
    # Modifier detection
    # ------------------------------------------------------------------

    def _build_modifier_set(self) -> Set[str]:
        """Return a set of item descriptions that are modifiers, not products."""
        modifiers: Set[str] = set()

        if self._item_meta is not None:
            mod_items = self._item_meta[
                self._item_meta["group"].isin(MODIFIER_GROUPS)
            ]
            modifiers.update(
                mod_items["description"].str.upper().str.strip().unique()
            )

        logger.info("Identified %d modifier item names from item metadata", len(modifiers))
        return modifiers

    @staticmethod
    def _is_modifier(desc: str, modifier_set: Set[str]) -> bool:
        """Check if a description is a modifier/topping rather than a product."""
        upper = desc.upper().strip()
        if upper in modifier_set:
            return True
        return any(p.search(desc) for p in MODIFIER_PATTERNS)

    # ------------------------------------------------------------------
    # Basket construction
    # ------------------------------------------------------------------

    def _build_baskets(self, modifier_set: Set[str]) -> pd.DataFrame:
        """Group line items into per-customer baskets of real products."""
        df = self._sales.copy()

        # Remove cancellations
        df = df[~df["is_cancellation"]].copy()

        # Remove modifiers
        df["_is_mod"] = df["description"].apply(
            lambda d: self._is_modifier(d, modifier_set)
        )
        products = df[~df["_is_mod"]].copy()
        products.drop(columns=["_is_mod"], inplace=True)

        logger.info(
            "After filtering: %d product line-items from %d total (removed %d cancellations + %d modifiers)",
            len(products),
            len(self._sales),
            self._sales["is_cancellation"].sum(),
            df["_is_mod"].sum(),
        )

        # Normalize descriptions for consistent matching
        products["item"] = products["description"].str.upper().str.strip()
        products["item"] = products["item"].str.replace(r"[.,]+$", "", regex=True)
        products["item"] = products["item"].str.replace(r"\s+", " ", regex=True)

        # Build basket: unique items per (branch, customer)
        baskets = (
            products.groupby(["branch", "customer"])["item"]
            .apply(lambda x: sorted(set(x)))
            .reset_index()
        )
        baskets.rename(columns={"item": "items"}, inplace=True)
        baskets["basket_size"] = baskets["items"].apply(len)

        # Only keep baskets with 2+ distinct products (needed for combos)
        baskets = baskets[baskets["basket_size"] >= 2].reset_index(drop=True)

        logger.info(
            "Built %d multi-product baskets (min size 2) from %d customers",
            len(baskets),
            products["customer"].nunique(),
        )

        self._baskets = baskets
        return baskets

    # ------------------------------------------------------------------
    # Co-purchase analysis
    # ------------------------------------------------------------------

    def _compute_copurchase_matrix(
        self, baskets: pd.DataFrame
    ) -> pd.DataFrame:
        """Count how often each pair of items appears in the same basket."""
        pair_counts: Dict[Tuple[str, str], int] = {}

        for items in baskets["items"]:
            for a, b in combinations(items, 2):
                pair = (a, b) if a < b else (b, a)
                pair_counts[pair] = pair_counts.get(pair, 0) + 1

        rows = [
            {"item_a": a, "item_b": b, "co_purchase_count": cnt}
            for (a, b), cnt in pair_counts.items()
        ]
        df = pd.DataFrame(rows)
        if df.empty:
            return df

        n_baskets = len(baskets)
        df["support"] = df["co_purchase_count"] / n_baskets
        df = df.sort_values("co_purchase_count", ascending=False).reset_index(drop=True)

        logger.info("Computed %d co-purchase pairs", len(df))
        return df

    def _compute_association_rules(
        self,
        baskets: pd.DataFrame,
        min_support: float = 0.02,
        min_confidence: float = 0.1,
    ) -> pd.DataFrame:
        """Compute association rules (A -> B) with support, confidence, and lift.

        Uses a lightweight custom implementation instead of mlxtend to avoid
        an extra dependency.
        """
        n_baskets = len(baskets)
        if n_baskets == 0:
            return pd.DataFrame()

        # Item frequency
        item_freq: Dict[str, int] = {}
        pair_freq: Dict[FrozenSet[str], int] = {}

        for items in baskets["items"]:
            item_set = set(items)
            for item in item_set:
                item_freq[item] = item_freq.get(item, 0) + 1
            for a, b in combinations(item_set, 2):
                pair = frozenset([a, b])
                pair_freq[pair] = pair_freq.get(pair, 0) + 1

        rules = []
        for pair, pair_count in pair_freq.items():
            pair_support = pair_count / n_baskets
            if pair_support < min_support:
                continue

            items = list(pair)
            for antecedent, consequent in [(items[0], items[1]), (items[1], items[0])]:
                ant_support = item_freq[antecedent] / n_baskets
                cons_support = item_freq[consequent] / n_baskets
                confidence = pair_count / item_freq[antecedent]
                lift = confidence / cons_support if cons_support > 0 else 0.0

                if confidence < min_confidence:
                    continue

                rules.append({
                    "antecedent": antecedent,
                    "consequent": consequent,
                    "support": round(pair_support, 4),
                    "confidence": round(confidence, 4),
                    "lift": round(lift, 4),
                    "pair_count": pair_count,
                    "antecedent_count": item_freq[antecedent],
                    "consequent_count": item_freq[consequent],
                })

        df = pd.DataFrame(rules)
        if not df.empty:
            df = df.sort_values("lift", ascending=False).reset_index(drop=True)

        logger.info("Generated %d association rules (min_support=%.2f, min_confidence=%.2f)",
                     len(df), min_support, min_confidence)
        return df

    # ------------------------------------------------------------------
    # Item affinity scores
    # ------------------------------------------------------------------

    def _compute_item_affinity(self, baskets: pd.DataFrame) -> pd.DataFrame:
        """Per-item summary: how often it appears, avg basket size, top partners."""
        item_stats: Dict[str, Dict] = {}

        for _, row in baskets.iterrows():
            items = row["items"]
            for item in items:
                if item not in item_stats:
                    item_stats[item] = {
                        "appearances": 0,
                        "total_basket_size": 0,
                        "partners": {},
                    }
                item_stats[item]["appearances"] += 1
                item_stats[item]["total_basket_size"] += len(items)
                for other in items:
                    if other != item:
                        item_stats[item]["partners"][other] = (
                            item_stats[item]["partners"].get(other, 0) + 1
                        )

        rows = []
        for item, stats in item_stats.items():
            partners = stats["partners"]
            top_3 = sorted(partners.items(), key=lambda x: -x[1])[:3]
            rows.append({
                "item": item,
                "basket_appearances": stats["appearances"],
                "avg_basket_size": round(stats["total_basket_size"] / stats["appearances"], 2),
                "unique_partners": len(partners),
                "top_partner_1": top_3[0][0] if len(top_3) > 0 else None,
                "top_partner_1_count": top_3[0][1] if len(top_3) > 0 else 0,
                "top_partner_2": top_3[1][0] if len(top_3) > 1 else None,
                "top_partner_2_count": top_3[1][1] if len(top_3) > 1 else 0,
                "top_partner_3": top_3[2][0] if len(top_3) > 2 else None,
                "top_partner_3_count": top_3[2][1] if len(top_3) > 2 else 0,
            })

        df = pd.DataFrame(rows).sort_values("basket_appearances", ascending=False).reset_index(drop=True)
        logger.info("Computed affinity stats for %d items", len(df))
        return df
