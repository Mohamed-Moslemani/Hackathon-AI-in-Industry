"""Cleaner for REP_S_00194_SMRY.csv — Tax summary by branch.

Raw structure (very simple, single page):
  Row pattern:
    ``Branch Name:  <name>``
    ``Total By Branch, <VAT>, 0, 0, 0, , 0, 0, <Total>``
"""

from __future__ import annotations

import csv
import io
import re
from typing import Dict, List, Optional

import pandas as pd

from .base_cleaner import BaseCleaner


class TaxReportCleaner(BaseCleaner):

    def _parse_clean(self, lines: List[str]) -> pd.DataFrame:
        rows: List[Dict[str, Optional[float | str]]] = []
        current_branch: Optional[str] = None

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if self._is_page_header(stripped):
                continue
            if stripped.startswith("Tax Report"):
                continue
            if re.match(r"^\d{2}-\w{3}-\d{2}", stripped):
                continue
            if stripped.startswith(",Year:"):
                continue
            if stripped.upper().startswith("TAX DESCRIPTION"):
                continue

            reader = csv.reader(io.StringIO(stripped))
            fields = next(reader)

            col0 = fields[0].strip() if fields else ""

            # Branch name row
            branch_match = re.match(r"Branch Name:\s*(.+)", col0, re.IGNORECASE)
            if branch_match:
                current_branch = branch_match.group(1).strip()
                continue

            # Total row
            if col0.startswith("Total By Branch") and current_branch:
                vat = self.parse_number(fields[1]) if len(fields) > 1 else 0.0
                total = self.parse_number(fields[-1]) if len(fields) > 1 else 0.0
                rows.append(
                    {
                        "branch": current_branch,
                        "vat_11_pct": vat or 0.0,
                        "total_tax": total or 0.0,
                    }
                )

        return pd.DataFrame(rows)

    def _coerce_types(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in ("vat_11_pct", "total_tax"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        return df
