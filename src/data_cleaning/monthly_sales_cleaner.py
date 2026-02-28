"""Cleaner for rep_s_00334_1_SMRY.csv — Monthly Sales by Branch.

Raw structure (2 pages, very clean):
  Branch header: ``Branch Name: <name>,,,,``
  Month rows:    ``<month>,,<year>,<total>,``
  Branch total:  ``,,Total for    <year>,<total>,``
  Grand total:   ``,,Grand Total:,<total>,``
"""

from __future__ import annotations

import csv
import io
import re
from typing import Dict, List, Optional

import pandas as pd

from .base_cleaner import BaseCleaner


class MonthlySalesCleaner(BaseCleaner):

    BRANCH_PATTERN = re.compile(r"^Branch Name:\s*(.+)", re.IGNORECASE)

    MONTH_MAP = {
        "January": 1, "February": 2, "March": 3, "April": 4,
        "May": 5, "June": 6, "July": 7, "August": 8,
        "September": 9, "October": 10, "November": 11, "December": 12,
    }

    def _parse_clean(self, lines: List[str]) -> pd.DataFrame:
        rows: List[Dict] = []
        current_branch: Optional[str] = None

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if self._is_page_header(stripped):
                continue
            if "Monthly Sales" in stripped:
                continue
            if re.match(r"^\d{2}-\w{3}-\d{2}", stripped):
                continue
            if stripped.startswith(",Year:"):
                continue
            if stripped.startswith("Month"):
                continue

            reader = csv.reader(io.StringIO(stripped))
            fields = next(reader)
            col0 = fields[0].strip() if fields else ""

            # Branch header
            m = self.BRANCH_PATTERN.match(col0)
            if m:
                current_branch = m.group(1).strip()
                continue

            # Skip total rows
            if any("Total" in f or "Grand" in f for f in fields):
                continue

            # Month row
            if col0 in self.MONTH_MAP:
                month_num = self.MONTH_MAP[col0]
                year = None
                total = None
                for f in fields[1:]:
                    f = f.strip()
                    if re.match(r"^\d{4}$", f):
                        year = int(f)
                    elif f:
                        parsed = self.parse_number(f)
                        if parsed is not None:
                            total = parsed

                rows.append(
                    {
                        "branch": current_branch,
                        "month": col0,
                        "month_num": month_num,
                        "year": year,
                        "total": total or 0.0,
                    }
                )

        return pd.DataFrame(rows)

    def _coerce_types(self, df: pd.DataFrame) -> pd.DataFrame:
        if "total" in df.columns:
            df["total"] = pd.to_numeric(df["total"], errors="coerce").fillna(0.0)
        if "year" in df.columns:
            df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(0).astype(int)
        if "month_num" in df.columns:
            df["month_num"] = df["month_num"].astype(int)
        # Create a proper date column for time-series work
        if {"year", "month_num"}.issubset(df.columns):
            df["date"] = pd.to_datetime(
                df[["year"]].assign(month=df["month_num"], day=1)
            )
        return df
