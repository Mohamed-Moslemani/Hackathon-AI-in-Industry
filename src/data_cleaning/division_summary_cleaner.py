"""Cleaner for REP_S_00136_SMRY.csv — Summary by Division / Menu Channel.

Raw structure (multi-page, two different column layouts across pages):
  Page 1-style: branch, division, DELIVERY, TABLE, (empty), TAKE AWAY, TOTAL
  Page 2-style: branch, division, DELIVERY, TABLE, TAKE AWAY, TOTAL

Rows alternate between branch-header rows (branch name in col 0) and
division data rows (division name in col 1). A ``TOTAL`` row marks the
end of each branch block.
"""

from __future__ import annotations

import csv
import io
import re
from typing import Dict, List, Optional

import pandas as pd

from .base_cleaner import BaseCleaner


class DivisionSummaryCleaner(BaseCleaner):
    """Parse the division/channel summary into a flat table."""

    COLUMN_HEADER_PATTERN = re.compile(
        r"^,*\s*(DELIVERY|TABLE|TAKE\s*AWAY|TOTAL)", re.IGNORECASE
    )

    def _parse_clean(self, lines: List[str]) -> pd.DataFrame:
        rows: List[Dict[str, Optional[float | str]]] = []
        current_branch: Optional[str] = None

        # The file has two distinct column layouts separated by page breaks.
        # We detect column-header rows and skip them, tracking the branch
        # context from the first non-empty cell.

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            if self._is_page_header(stripped):
                continue

            # Skip report title rows
            if stripped.startswith("Summary By Division"):
                continue
            if re.match(r"^\d{2}-\w{3}-\d{2}", stripped):
                continue

            # Skip column-header rows
            if self.COLUMN_HEADER_PATTERN.match(stripped):
                continue

            # Parse CSV fields
            reader = csv.reader(io.StringIO(stripped))
            fields = next(reader)

            # Trim trailing empty fields
            while fields and not fields[-1].strip():
                fields.pop()

            if not fields:
                continue

            col0 = fields[0].strip().replace("''", "'")
            col1 = fields[1].strip().replace("''", "'") if len(fields) > 1 else ""

            # Branch header row: col0 is non-empty, col1 is a division name
            # or col0 is a known branch name
            if col0 and col0 not in ("", " "):
                # Could be a branch name or a branch+division row
                if col1 and col1.upper() == "TOTAL":
                    # TOTAL row for a branch — skip
                    continue
                if col1:
                    # Branch name in col0, division in col1
                    current_branch = col0
                    division = col1
                    numeric_fields = fields[2:]
                else:
                    # Might be a standalone branch header
                    current_branch = col0
                    continue
            elif col1:
                division = col1
                numeric_fields = fields[2:]
            else:
                continue

            if division.upper() == "TOTAL":
                continue

            # Parse the numeric columns — layout varies; empty columns
            # create None gaps that must be filtered before positional mapping.
            delivery, table, take_away, total = (
                None, None, None, None,
            )
            nums = [
                self.parse_number(f)
                for f in numeric_fields
                if f.strip()
            ]
            if len(nums) >= 4:
                delivery = nums[0]
                table = nums[1]
                take_away = nums[2]
                total = nums[3]
            elif len(nums) == 3:
                delivery = nums[0]
                table = nums[1]
                take_away = nums[2]

            rows.append(
                {
                    "branch": current_branch,
                    "division": self.clean_text(division),
                    "delivery": delivery or 0.0,
                    "table": table or 0.0,
                    "take_away": take_away or 0.0,
                    "total": total or 0.0,
                }
            )

        df = pd.DataFrame(rows)
        # Forward-fill branch names
        if not df.empty:
            df["branch"] = df["branch"].ffill()
        return df

    def _coerce_types(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in ("delivery", "table", "take_away", "total"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        return df
