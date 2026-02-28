"""Base cleaner with shared utilities for Conut report-style CSVs."""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class BaseCleaner(ABC):
    """Abstract base for all Conut CSV cleaners.

    Every raw CSV exported from Omega POS shares common noise:
      - Repeated page headers (``Page X of Y``)
      - Report title rows at the top of each page
      - Copyright / footer lines (``Copyright © 2026 Omega Software …``)
      - Column-header rows that repeat on every page
      - Numeric values formatted with commas inside quotes

    Subclasses implement ``_parse_clean`` to handle file-specific structure.
    """

    COPYRIGHT_PATTERN = re.compile(r"Copyright\s*©", re.IGNORECASE)
    PAGE_PATTERN = re.compile(r"Page\s+\d+\s+of", re.IGNORECASE)
    REPORT_ID_PATTERN = re.compile(r"^REP_S_\d+", re.IGNORECASE)
    URL_PATTERN = re.compile(r"www\.omegapos\.com", re.IGNORECASE)

    def __init__(self, filepath: Path | str) -> None:
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"Data file not found: {self.filepath}")
        self._raw_lines: Optional[List[str]] = None
        self._df: Optional[pd.DataFrame] = None

    # Public API

    def clean(self) -> pd.DataFrame:
        """Run the full cleaning pipeline and return a tidy DataFrame."""
        self._raw_lines = self._read_raw_lines()
        cleaned_lines = self._remove_noise(self._raw_lines)
        self._df = self._parse_clean(cleaned_lines)
        self._df = self._coerce_types(self._df)
        logger.info(
            "%s cleaned %s -> %d rows, %d cols",
            self.__class__.__name__,
            self.filepath.name,
            len(self._df),
            len(self._df.columns),
        )
        return self._df

    @property
    def dataframe(self) -> pd.DataFrame:
        if self._df is None:
            raise RuntimeError("Call .clean() first.")
        return self._df

    # Shared utilities

    def _read_raw_lines(self) -> List[str]:
        with open(self.filepath, encoding="utf-8", errors="replace") as fh:
            return fh.readlines()

    def _is_noise_line(self, line: str) -> bool:
        """Return True if a line is a page header, footer, or copyright."""
        stripped = line.strip()
        if not stripped:
            return True
        if self.COPYRIGHT_PATTERN.search(stripped):
            return True
        if self.REPORT_ID_PATTERN.match(stripped):
            return True
        if self.URL_PATTERN.search(stripped):
            return True
        return False

    def _is_page_header(self, line: str) -> bool:
        return bool(self.PAGE_PATTERN.search(line))

    def _remove_noise(self, lines: List[str]) -> List[str]:
        """Strip copyright lines and blank lines. Keep page-header lines
        so subclasses can decide how to handle them."""
        return [ln for ln in lines if not self._is_noise_line(ln)]

    @staticmethod
    def parse_number(value: str) -> Optional[float]:
        """Parse a formatted number string like ``"1,251,486.48"`` or ``-893,918.92``."""
        if value is None:
            return None
        cleaned = str(value).strip().strip('"').replace(",", "")
        if not cleaned or cleaned == "-":
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None

    @staticmethod
    def clean_text(value: str) -> str:
        """Normalize whitespace and strip leading/trailing junk from text fields."""
        if not isinstance(value, str):
            return value
        value = value.strip().strip('"').strip()
        value = re.sub(r"\s+", " ", value)
        # Remove trailing dots/commas that are formatting artifacts
        value = re.sub(r"[.,]+$", "", value)
        # Remove leading dots/commas
        value = re.sub(r"^[.,]+\s*", "", value)
        return value.strip()

    @staticmethod
    def strip_description(desc: str) -> str:
        """Normalize product description: strip leading spaces, trailing
        punctuation artifacts like ``.``, ``,(R)``, ``(R)`` etc., and collapse
        whitespace."""
        if not isinstance(desc, str):
            return desc
        desc = desc.strip()
        # Remove leading/trailing whitespace and special chars
        desc = re.sub(r"^\s+", "", desc)
        desc = re.sub(r"\s+$", "", desc)
        # Collapse internal whitespace
        desc = re.sub(r"\s+", " ", desc)
        return desc

    # Template methods for subclasses

    @abstractmethod
    def _parse_clean(self, lines: List[str]) -> pd.DataFrame:
        """Parse the noise-free lines into a structured DataFrame."""
        ...

    def _coerce_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optional hook for subclasses to coerce column types."""
        return df
