"""Cleaner for REP_S_00502.csv — Sales by customer in detail (delivery).

Raw structure (63 pages):
  Branch header:  ``Branch :<branch_name>,,,,``
  Customer header: ``<customer_name>,,,,``  (or ``0 <name>`` / ``1 <name>`` / ``90 <name>``)
  Line items:     ``,<qty>,  <description>,<price>,``
  Customer total: ``Total :,<qty>,,<total>,``
  Branch total:   ``Total Branch:,<qty>,,<total>,``
  Page headers repeat with date/title/column headers.

Key challenges:
  - Descriptions contain commas inside quotes
  - Negative quantities represent order cancellations/refunds
  - Customer names sometimes have numeric prefixes (``0 Person_0017``)
"""

from __future__ import annotations

import csv
import io
import re
from typing import Dict, List, Optional

import pandas as pd

from .base_cleaner import BaseCleaner


class SalesByCustomerCleaner(BaseCleaner):

    BRANCH_PATTERN = re.compile(r"^Branch\s*:\s*(.+)", re.IGNORECASE)
    TOTAL_PATTERN = re.compile(r"^Total\s*(Branch)?\s*:", re.IGNORECASE)
    COLUMN_HEADER = re.compile(r"^Full Name\s*,\s*Qty", re.IGNORECASE)

    def _parse_clean(self, lines: List[str]) -> pd.DataFrame:
        rows: List[Dict] = []
        current_branch: Optional[str] = None
        current_customer: Optional[str] = None

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if self._is_page_header(stripped):
                continue
            if self.COLUMN_HEADER.match(stripped):
                continue
            if re.match(r"^Sales by customer", stripped, re.IGNORECASE):
                continue
            if re.match(r"^\d{2}-\w{3}-\d{2}", stripped):
                continue

            # Parse with CSV reader to handle quoted commas
            reader = csv.reader(io.StringIO(stripped))
            fields = next(reader)

            col0 = fields[0].strip() if fields else ""

            # Branch header
            branch_match = self.BRANCH_PATTERN.match(col0)
            if branch_match:
                current_branch = branch_match.group(1).strip()
                continue

            # Total rows — skip
            if self.TOTAL_PATTERN.match(col0):
                continue
            if len(fields) > 1 and self.TOTAL_PATTERN.match(fields[0].strip() + fields[1].strip()):
                continue

            # Customer header: col0 has a name, other cols are mostly empty
            if col0 and not col0.startswith(","):
                # Check if this looks like a customer name (not a line-item)
                col1 = fields[1].strip() if len(fields) > 1 else ""
                if not col1 or col1 == "":
                    # Strip numeric prefix from customer names like "0 Person_0017"
                    current_customer = re.sub(r"^\d+\s+", "", col0).strip()
                    continue

            # Line item row: ``,<qty>,  <description>,<price>,``
            col1 = fields[1].strip() if len(fields) > 1 else ""
            col2 = fields[2].strip() if len(fields) > 2 else ""
            col3 = fields[3].strip() if len(fields) > 3 else ""

            qty = self.parse_number(col1)
            if qty is None:
                continue

            description = self.strip_description(col2)
            price = self.parse_number(col3) or 0.0

            rows.append(
                {
                    "branch": current_branch,
                    "customer": current_customer,
                    "qty": qty,
                    "description": description,
                    "price": price,
                }
            )

        return pd.DataFrame(rows)

    def _coerce_types(self, df: pd.DataFrame) -> pd.DataFrame:
        if "qty" in df.columns:
            df["qty"] = pd.to_numeric(df["qty"], errors="coerce").fillna(0.0)
        if "price" in df.columns:
            df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)
        # Flag cancellation rows (negative qty)
        if "qty" in df.columns:
            df["is_cancellation"] = df["qty"] < 0
        return df
