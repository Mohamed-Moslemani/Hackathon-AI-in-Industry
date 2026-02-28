"""Cleaner for rep_s_00191_SMRY.csv — Sales by Items by Group.

Raw structure (45 pages, grouped by branch > division > group):
  Branch header:  ``Branch: <name>,,,,``
  Division header: ``Division: <name>,,,,``
  Group header:   ``Group: <name>,,,,``
  Item rows:      ``<description>,<barcode>,<qty>,<total_amount>,``
  Group total:    ``Total by Group: <name>,,<qty>,<total>,``
  Division total: ``Total by Division: <name>,,<qty>,<total>,``
  Branch total:   ``Total by Branch: <name>,,<qty>,<total>,``
  Column headers: ``Description,Barcode,Qty,Total Amount,``
"""

from __future__ import annotations

import csv
import io
import re
from typing import Dict, List, Optional

import pandas as pd

from .base_cleaner import BaseCleaner


class SalesByItemCleaner(BaseCleaner):

    BRANCH_PATTERN = re.compile(r"^Branch:\s*(.+)", re.IGNORECASE)
    DIVISION_PATTERN = re.compile(r"^Division:\s*(.+)", re.IGNORECASE)
    GROUP_PATTERN = re.compile(r"^Group:\s*(.+)", re.IGNORECASE)
    TOTAL_PATTERN = re.compile(r"^Total by (Group|Division|Branch)", re.IGNORECASE)
    COLUMN_HEADER = re.compile(r"^Description\s*,\s*Barcode", re.IGNORECASE)

    def _parse_clean(self, lines: List[str]) -> pd.DataFrame:
        rows: List[Dict] = []
        current_branch: Optional[str] = None
        current_division: Optional[str] = None
        current_group: Optional[str] = None

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if self._is_page_header(stripped):
                continue
            if self.COLUMN_HEADER.match(stripped):
                continue
            if "Sales by Items" in stripped:
                continue
            if re.match(r"^\d{2}-\w{3}-\d{2}", stripped):
                continue

            reader = csv.reader(io.StringIO(stripped))
            fields = next(reader)
            col0 = fields[0].strip().replace("''", "'") if fields else ""

            # Branch header
            m = self.BRANCH_PATTERN.match(col0)
            if m:
                current_branch = m.group(1).strip()
                continue

            # Division header
            m = self.DIVISION_PATTERN.match(col0)
            if m:
                current_division = m.group(1).strip()
                continue

            # Group header
            m = self.GROUP_PATTERN.match(col0)
            if m:
                current_group = m.group(1).strip()
                continue

            # Total rows — skip
            if self.TOTAL_PATTERN.match(col0):
                continue

            # Item row: description, barcode, qty, total_amount
            if len(fields) < 3:
                continue

            description = self.strip_description(col0)
            if not description:
                continue

            qty = self.parse_number(fields[2]) if len(fields) > 2 else None
            total_amount = self.parse_number(fields[3]) if len(fields) > 3 else None

            if qty is None:
                continue

            rows.append(
                {
                    "branch": current_branch,
                    "division": current_division,
                    "group": current_group,
                    "description": description,
                    "qty": qty,
                    "total_amount": total_amount or 0.0,
                }
            )

        return pd.DataFrame(rows)

    def _coerce_types(self, df: pd.DataFrame) -> pd.DataFrame:
        if "qty" in df.columns:
            df["qty"] = pd.to_numeric(df["qty"], errors="coerce").fillna(0.0)
        if "total_amount" in df.columns:
            df["total_amount"] = pd.to_numeric(df["total_amount"], errors="coerce").fillna(0.0)
        # Compute unit price where qty > 0
        if "qty" in df.columns and "total_amount" in df.columns:
            df["unit_price"] = df.apply(
                lambda r: r["total_amount"] / r["qty"] if r["qty"] > 0 else 0.0, axis=1
            )
        return df
