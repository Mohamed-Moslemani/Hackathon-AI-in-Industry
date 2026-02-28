"""Cleaner for rep_s_00435_SMRY.csv — Average Sales by Menu.

Raw structure (single page, very clean):
  Branch header: ``<branch_name>,,,,``
  Menu rows:     ``<menu_name>,<num_customers>,<sales>,<avg_customer>,``
  Branch total:  ``Total By Branch:,<num>,<sales>,<avg>,``
  Grand total:   ``Total :,<num>,<sales>,<avg>,``
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

import pandas as pd

from .base_cleaner import BaseCleaner


class AvgSalesCleaner(BaseCleaner):

    KNOWN_BRANCHES = {"Conut - Tyre", "Conut", "Conut Jnah", "Main Street Coffee"}

    def _parse_clean(self, lines: List[str]) -> pd.DataFrame:
        rows: List[Dict] = []
        current_branch: Optional[str] = None

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if self._is_page_header(stripped):
                continue
            if "Average Sales" in stripped:
                continue
            if re.match(r"^\d{2}-\w{3}-\d{2}", stripped):
                continue
            if stripped.startswith(",Year:"):
                continue
            if stripped.startswith("Menu Name"):
                continue

            fields = stripped.split(",")
            col0 = fields[0].strip() if fields else ""

            # Skip total rows
            if col0.startswith("Total"):
                continue

            # Branch header
            if col0 in self.KNOWN_BRANCHES:
                current_branch = col0
                continue

            # Menu row: name, num_cust, sales, avg
            if not col0:
                continue

            menu_name = col0
            num_cust = self.parse_number(fields[1]) if len(fields) > 1 else None
            sales = self.parse_number(fields[2]) if len(fields) > 2 else None
            avg_customer = self.parse_number(fields[3]) if len(fields) > 3 else None

            if num_cust is None:
                continue

            rows.append(
                {
                    "branch": current_branch,
                    "menu_channel": menu_name,
                    "num_customers": num_cust,
                    "sales": sales or 0.0,
                    "avg_per_customer": avg_customer or 0.0,
                }
            )

        return pd.DataFrame(rows)

    def _coerce_types(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in ("num_customers", "sales", "avg_per_customer"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        if "num_customers" in df.columns:
            df["num_customers"] = df["num_customers"].astype(int)
        return df
