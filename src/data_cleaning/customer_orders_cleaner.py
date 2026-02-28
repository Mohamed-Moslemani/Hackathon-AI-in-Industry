"""Cleaner for rep_s_00150.csv — Customer Orders (Delivery).

Raw structure (15 pages, grouped by branch):
  Branch header:  ``<branch_name>,,,,,,,,,``
  Column headers: ``Customer Name,Address,Phone Number,First Order,,Last Order,,Total,No. of Orders,``
  Data rows:      ``<name>,<addr>,<phone>,<first_order>,,<last_order>,,<total>,<num_orders>,``
  Branch total:   ``,,Total By Branch,<first>,,<last>,,<total>,<count>,``
  Page headers repeat.

Quirk: The last page has an extra empty column shifting the layout.
"""

from __future__ import annotations

import csv
import io
import re
from typing import Dict, List, Optional

import pandas as pd

from .base_cleaner import BaseCleaner


class CustomerOrdersCleaner(BaseCleaner):

    COLUMN_HEADER = re.compile(r"^Customer Name", re.IGNORECASE)
    TOTAL_PATTERN = re.compile(r"Total By Branch", re.IGNORECASE)

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
            if self.COLUMN_HEADER.match(stripped):
                continue
            if "Customer Orders" in stripped:
                continue
            if re.match(r"^\d{2}-\w{3}-\d{2}", stripped):
                continue

            reader = csv.reader(io.StringIO(stripped))
            fields = next(reader)

            col0 = fields[0].strip() if fields else ""

            # Branch header: a known branch name alone on a line
            if col0 in self.KNOWN_BRANCHES:
                current_branch = col0
                continue

            # Total row — skip
            if any(self.TOTAL_PATTERN.search(f) for f in fields):
                continue

            # Data row: customer name starts with "Person_"
            if not re.match(r"Person_\d+", col0):
                continue

            customer_name = col0
            phone = fields[2].strip() if len(fields) > 2 else ""

            # Handle the variable column layout (last page has extra col)
            # Find the first field that looks like a datetime
            first_order = None
            last_order = None
            total = None
            num_orders = None

            # Scan fields for datetime patterns
            dt_indices = []
            for i, f in enumerate(fields):
                if re.match(r"\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:", f.strip()):
                    dt_indices.append(i)

            if len(dt_indices) >= 2:
                first_order = fields[dt_indices[0]].strip()
                last_order = fields[dt_indices[1]].strip()
                # Total and num_orders follow after last_order
                remaining = fields[dt_indices[1] + 1 :]
                nums = [self.parse_number(f) for f in remaining if f.strip()]
                # Filter out empty cols
                nums = [n for n in nums if n is not None]
                if len(nums) >= 2:
                    total = nums[0]
                    num_orders = int(nums[1])
                elif len(nums) == 1:
                    total = nums[0]

            rows.append(
                {
                    "branch": current_branch,
                    "customer": customer_name,
                    "phone": phone.strip(),
                    "first_order": first_order,
                    "last_order": last_order,
                    "total": total or 0.0,
                    "num_orders": num_orders or 0,
                }
            )

        return pd.DataFrame(rows)

    def _coerce_types(self, df: pd.DataFrame) -> pd.DataFrame:
        if "first_order" in df.columns:
            df["first_order"] = pd.to_datetime(
                df["first_order"], format="%Y-%m-%d %H:%M:", errors="coerce"
            )
        if "last_order" in df.columns:
            df["last_order"] = pd.to_datetime(
                df["last_order"], format="%Y-%m-%d %H:%M:", errors="coerce"
            )
        if "total" in df.columns:
            df["total"] = pd.to_numeric(df["total"], errors="coerce").fillna(0.0)
        if "num_orders" in df.columns:
            df["num_orders"] = pd.to_numeric(df["num_orders"], errors="coerce").fillna(0).astype(int)
        return df
