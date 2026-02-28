"""Combo Optimization — Recommender.

Turns raw association rules and co-purchase data into actionable combo
recommendations: ranked 2-item and 3-item combos per branch and network-wide,
scored by a composite of lift, support, and estimated revenue.

Outputs:
  - combo_recommendations: ranked combo suggestions (2-item and 3-item)
  - branch_combo_recommendations: per-branch combo suggestions
  - combo_summary: executive summary dict for the agent layer
"""

from __future__ import annotations

import json
import logging
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Set, Tuple

import pandas as pd

from ..config import CLEANED_DATA_DIR, FEATURES_DIR

logger = logging.getLogger(__name__)

RESULTS_DIR_NAME = "analytics"


class ComboRecommender:
    """Generate actionable combo recommendations from association rules."""

    def __init__(
        self,
        features_dir: Path | str | None = None,
        cleaned_dir: Path | str | None = None,
        output_dir: Path | str | None = None,
    ) -> None:
        self.features_dir = Path(features_dir) if features_dir else FEATURES_DIR
        self.cleaned_dir = Path(cleaned_dir) if cleaned_dir else CLEANED_DATA_DIR
        self.output_dir = (
            Path(output_dir)
            if output_dir
            else FEATURES_DIR.parent / RESULTS_DIR_NAME
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, *, save: bool = True) -> Dict[str, Any]:
        rules = pd.read_csv(self.features_dir / "combo_association_rules.csv")
        pairs = pd.read_csv(self.features_dir / "combo_copurchase_pairs.csv")
        affinity = pd.read_csv(self.features_dir / "combo_item_affinity.csv")
        baskets = pd.read_csv(self.features_dir / "combo_baskets.csv")
        items = pd.read_csv(self.cleaned_dir / "sales_by_item.csv")

        price_map = self._build_price_map(items)

        two_item = self._rank_two_item_combos(rules, pairs, price_map)
        three_item = self._find_three_item_combos(baskets, pairs, price_map)
        branch_combos = self._branch_level_combos(baskets, pairs, price_map)
        summary = self._build_summary(two_item, three_item, branch_combos)

        results = {
            "combo_recommendations_2item": two_item,
            "combo_recommendations_3item": three_item,
            "branch_combo_recommendations": branch_combos,
            "combo_summary": summary,
        }

        if save:
            for name, data in results.items():
                path = self.output_dir / f"combo_{name}.csv"
                if isinstance(data, pd.DataFrame):
                    data.to_csv(path, index=False)
                else:
                    path = path.with_suffix(".json")
                    with open(path, "w") as f:
                        json.dump(data, f, indent=2)
                logger.info("Saved %s -> %s", name, path)

        return results

       # Price lookup
   
    def _build_price_map(self, items: pd.DataFrame) -> Dict[str, float]:
        """Average unit price per product (upper-cased, stripped).

        Also builds a reverse index keyed by sorted word-sets so that
        items with reordered words (e.g. "PISTACHIO CONUT" vs
        "CONUT PISTACHIO") can still be matched.
        """
        df = items[items["unit_price"] > 0].copy()
        df["key"] = df["description"].str.upper().str.strip()
        df["key"] = df["key"].str.replace(r"[.,]+$", "", regex=True)
        df["key"] = df["key"].str.replace(r"\s+", " ", regex=True)
        exact = df.groupby("key")["unit_price"].mean().to_dict()

        # Word-set index for fuzzy matching reordered names
        self._word_set_index: Dict[FrozenSet[str], float] = {}
        for key, price in exact.items():
            ws = frozenset(key.split())
            self._word_set_index[ws] = price

        return exact

    def _lookup_price(self, item: str, price_map: Dict[str, float]) -> float:
        """Look up price by exact key, then fall back to word-set match."""
        if item in price_map:
            return price_map[item]
        ws = frozenset(item.split())
        return self._word_set_index.get(ws, 0.0)

    def _combo_price(self, item_list: List[str], price_map: Dict[str, float]) -> float:
        return sum(self._lookup_price(item, price_map) for item in item_list)

       # 2-item combos
   
    def _rank_two_item_combos(
        self,
        rules: pd.DataFrame,
        pairs: pd.DataFrame,
        price_map: Dict[str, float],
    ) -> pd.DataFrame:
        if rules.empty:
            return pd.DataFrame()

        # Deduplicate rules to unique pairs (keep the direction with higher confidence)
        rules = rules.sort_values("confidence", ascending=False)
        seen: Set[FrozenSet[str]] = set()
        deduped = []
        for _, row in rules.iterrows():
            pair = frozenset([row["antecedent"], row["consequent"]])
            if pair in seen:
                continue
            seen.add(pair)
            deduped.append(row)
        df = pd.DataFrame(deduped)

        # Compute combo score: weighted composite of lift, support, confidence
        df["combo_price"] = df.apply(
            lambda r: self._combo_price([r["antecedent"], r["consequent"]], price_map),
            axis=1,
        )

        # Normalize metrics to 0-1 for scoring
        for col in ["lift", "support", "confidence"]:
            rng = df[col].max() - df[col].min()
            df[f"{col}_norm"] = (df[col] - df[col].min()) / rng if rng > 0 else 0.0

        # Combo score: lift matters most (shows genuine affinity), then confidence, then support
        df["combo_score"] = (
            0.45 * df["lift_norm"]
            + 0.30 * df["confidence_norm"]
            + 0.25 * df["support_norm"]
        ).round(4)

        # Estimated combo revenue = combo_price * pair_count (observed volume)
        df["estimated_revenue"] = (df["combo_price"] * df["pair_count"]).round(2)

        # Discount suggestion: higher lift = less discount needed
        df["suggested_discount_pct"] = df["lift"].apply(
            lambda l: 5 if l > 10 else (10 if l > 5 else (15 if l > 2 else 20))
        )

        df["combo_name"] = df.apply(
            lambda r: f"{r['antecedent']} + {r['consequent']}", axis=1
        )

        keep = [
            "combo_name", "antecedent", "consequent",
            "support", "confidence", "lift", "pair_count",
            "combo_price", "combo_score", "estimated_revenue",
            "suggested_discount_pct",
        ]
        result = df[keep].sort_values("combo_score", ascending=False).reset_index(drop=True)
        result.index.name = "rank"
        result.index = result.index + 1

        logger.info("Ranked %d 2-item combos", len(result))
        return result.reset_index().rename(columns={"index": "rank"})

       # 3-item combos
   
    def _find_three_item_combos(
        self,
        baskets: pd.DataFrame,
        pairs: pd.DataFrame,
        price_map: Dict[str, float],
    ) -> pd.DataFrame:
        """Find 3-item combos that appear together frequently."""
        import ast

        # Parse basket item lists
        basket_items = baskets["items"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )

        # Count 3-item combinations
        triple_counts: Dict[Tuple[str, ...], int] = {}
        for items in basket_items:
            if len(items) < 3:
                continue
            for triple in combinations(sorted(items), 3):
                triple_counts[triple] = triple_counts.get(triple, 0) + 1

        # Filter to triples appearing 2+ times
        n_baskets = len(baskets)
        rows = []
        for triple, count in triple_counts.items():
            if count < 2:
                continue
            support = count / n_baskets
            combo_price = self._combo_price(list(triple), price_map)
            rows.append({
                "item_1": triple[0],
                "item_2": triple[1],
                "item_3": triple[2],
                "combo_name": f"{triple[0]} + {triple[1]} + {triple[2]}",
                "co_occurrence": count,
                "support": round(support, 4),
                "combo_price": round(combo_price, 2),
                "estimated_revenue": round(combo_price * count, 2),
                "suggested_discount_pct": 10 if count >= 5 else 15,
            })

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        df = df.sort_values("co_occurrence", ascending=False).reset_index(drop=True)
        df.index = df.index + 1
        df = df.reset_index().rename(columns={"index": "rank"})

        logger.info("Found %d 3-item combos (min 2 occurrences)", len(df))
        return df

       # Branch-level combos
   
    def _branch_level_combos(
        self,
        baskets: pd.DataFrame,
        pairs: pd.DataFrame,
        price_map: Dict[str, float],
    ) -> pd.DataFrame:
        """Top combo recommendations per branch."""
        import ast

        rows = []
        for branch in sorted(baskets["branch"].unique()):
            bb = baskets[baskets["branch"] == branch]
            n = len(bb)
            if n == 0:
                continue

            pair_counts: Dict[Tuple[str, str], int] = {}
            for items_raw in bb["items"]:
                items = ast.literal_eval(items_raw) if isinstance(items_raw, str) else items_raw
                for a, b in combinations(sorted(items), 2):
                    pair_counts[(a, b)] = pair_counts.get((a, b), 0) + 1

            for (a, b), count in sorted(pair_counts.items(), key=lambda x: -x[1])[:10]:
                combo_price = self._combo_price([a, b], price_map)
                rows.append({
                    "branch": branch,
                    "item_a": a,
                    "item_b": b,
                    "combo_name": f"{a} + {b}",
                    "co_purchase_count": count,
                    "branch_support": round(count / n, 4),
                    "combo_price": round(combo_price, 2),
                    "estimated_revenue": round(combo_price * count, 2),
                })

        result = pd.DataFrame(rows)
        logger.info("Built branch-level combos: %d recommendations across %d branches",
                     len(result), result["branch"].nunique() if not result.empty else 0)
        return result

       # Executive summary
   
    def _build_summary(
        self,
        two_item: pd.DataFrame,
        three_item: pd.DataFrame,
        branch_combos: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Build a JSON-serializable executive summary."""
        summary: Dict[str, Any] = {
            "total_2item_combos": len(two_item),
            "total_3item_combos": len(three_item),
        }

        # Top 5 network-wide 2-item combos
        if not two_item.empty:
            top5 = two_item.head(5)
            summary["top_5_combos"] = [
                {
                    "rank": int(row["rank"]),
                    "combo": row["combo_name"],
                    "lift": float(row["lift"]),
                    "confidence": float(row["confidence"]),
                    "support": float(row["support"]),
                    "combo_price": float(row["combo_price"]),
                    "suggested_discount_pct": int(row["suggested_discount_pct"]),
                }
                for _, row in top5.iterrows()
            ]

        # Top 3 3-item combos
        if not three_item.empty:
            top3 = three_item.head(3)
            summary["top_3_triple_combos"] = [
                {
                    "rank": int(row["rank"]),
                    "combo": row["combo_name"],
                    "occurrences": int(row["co_occurrence"]),
                    "combo_price": float(row["combo_price"]),
                }
                for _, row in top3.iterrows()
            ]

        # Per-branch top combo
        if not branch_combos.empty:
            branch_top = {}
            for branch in branch_combos["branch"].unique():
                bb = branch_combos[branch_combos["branch"] == branch].iloc[0]
                branch_top[branch] = {
                    "combo": bb["combo_name"],
                    "co_purchase_count": int(bb["co_purchase_count"]),
                    "combo_price": float(bb["combo_price"]),
                }
            summary["top_combo_per_branch"] = branch_top

        # Key insight
        if not two_item.empty:
            best = two_item.iloc[0]
            summary["key_insight"] = (
                f"The strongest combo is '{best['combo_name']}' with a lift of "
                f"{best['lift']:.1f}x (customers who buy {best['antecedent']} are "
                f"{best['lift']:.0f}x more likely to also buy {best['consequent']}). "
                f"Recommend offering this as a bundled deal with ~{int(best['suggested_discount_pct'])}% discount."
            )

        return summary
