"""Cleaner for REP_S_00461.csv — Time & Attendance logs.

Raw structure (multi-page, grouped by employee):
  Employee header: ``,EMP ID :<id>,NAME :<name>,,,``
  Branch row:      ``,<branch_name>,,,,``
  Punch rows:      ``<date>,,<punch_in_time>,<date>,<punch_out_time>,<duration>``
  Total row:       ``,,,,Total :,<HH:MM:SS>``
  Page headers repeat every ~23 rows with date/title/column headers.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd

from .base_cleaner import BaseCleaner


class AttendanceCleaner(BaseCleaner):

    EMP_PATTERN = re.compile(r"EMP ID\s*:\s*([\d.]+)\s*,\s*NAME\s*:\s*([^,\s]+)")
    DATE_PATTERN = re.compile(r"^\d{2}-\w{3}-\d{2}$")
    TOTAL_PATTERN = re.compile(r"Total\s*:", re.IGNORECASE)
    COLUMN_HEADER_PATTERN = re.compile(r"PUNCH\s+IN|PUNCH\s+OUT|Work Duration", re.IGNORECASE)

    def _parse_clean(self, lines: List[str]) -> pd.DataFrame:
        rows: List[Dict] = []
        current_emp_id: Optional[str] = None
        current_emp_name: Optional[str] = None
        current_branch: Optional[str] = None

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if self._is_page_header(stripped):
                continue

            # Skip report title
            if "Time & Attendance" in stripped:
                continue

            # Skip column header rows
            if self.COLUMN_HEADER_PATTERN.search(stripped):
                # But check if this also contains an employee header
                emp_match = self.EMP_PATTERN.search(stripped)
                if emp_match:
                    current_emp_id = emp_match.group(1).replace(".0", "")
                    current_emp_name = emp_match.group(2)
                continue

            # Employee header
            emp_match = self.EMP_PATTERN.search(stripped)
            if emp_match:
                current_emp_id = emp_match.group(1).replace(".0", "")
                current_emp_name = emp_match.group(2)
                continue

            # Total row — skip
            if self.TOTAL_PATTERN.search(stripped):
                continue

            # Date line at top of page (report date) — skip
            if re.match(r"^\d{2}-\w{3}-\d{2}\s*$", stripped):
                continue

            # Split by comma
            fields = stripped.split(",")

            # Branch row: ``,<branch_name>,,,,``
            if (
                len(fields) >= 2
                and not fields[0].strip()
                and fields[1].strip()
                and not any(self.DATE_PATTERN.match(f.strip()) for f in fields if f.strip())
                and not re.match(r"\d{2}\.\d{2}\.\d{2}", fields[1].strip())
                and not fields[1].strip().startswith("EMP")
                and not re.match(r"^\d{2}-\w{3}-\d{2}", fields[1].strip())
            ):
                candidate = fields[1].strip()
                if candidate and not re.match(r"^\d", candidate):
                    current_branch = candidate
                    continue

            # Punch row: date,,time,date,time,duration
            col0 = fields[0].strip() if fields else ""
            if not self.DATE_PATTERN.match(col0):
                continue

            punch_in_date = col0
            punch_in_time = fields[2].strip() if len(fields) > 2 else ""
            punch_out_date = fields[3].strip() if len(fields) > 3 else ""
            punch_out_time = fields[4].strip() if len(fields) > 4 else ""
            work_duration = fields[5].strip() if len(fields) > 5 else ""

            punch_in_dt = self._parse_datetime(punch_in_date, punch_in_time)
            punch_out_dt = self._parse_datetime(punch_out_date, punch_out_time)
            duration_td = self._parse_duration(work_duration)

            rows.append(
                {
                    "emp_id": current_emp_id,
                    "emp_name": current_emp_name,
                    "branch": current_branch,
                    "punch_in": punch_in_dt,
                    "punch_out": punch_out_dt,
                    "work_duration": duration_td,
                }
            )

        return pd.DataFrame(rows)

    @staticmethod
    def _parse_datetime(date_str: str, time_str: str) -> Optional[datetime]:
        if not date_str or not time_str:
            return None
        time_str = time_str.replace(".", ":")
        try:
            return datetime.strptime(f"{date_str} {time_str}", "%d-%b-%y %H:%M:%S")
        except ValueError:
            return None

    @staticmethod
    def _parse_duration(dur_str: str) -> Optional[timedelta]:
        if not dur_str:
            return None
        dur_str = dur_str.replace(".", ":")
        parts = dur_str.split(":")
        if len(parts) != 3:
            return None
        try:
            return timedelta(
                hours=int(parts[0]), minutes=int(parts[1]), seconds=int(parts[2])
            )
        except ValueError:
            return None

    def _coerce_types(self, df: pd.DataFrame) -> pd.DataFrame:
        if "punch_in" in df.columns:
            df["punch_in"] = pd.to_datetime(df["punch_in"], errors="coerce")
        if "punch_out" in df.columns:
            df["punch_out"] = pd.to_datetime(df["punch_out"], errors="coerce")
        if "work_duration" in df.columns:
            df["work_duration_hours"] = df["work_duration"].apply(
                lambda td: td.total_seconds() / 3600 if pd.notna(td) and td is not None else 0.0
            )
        return df
