"""Comprehensive unit tests for all Conut data cleaners.

Tests are split into:
  1. BaseCleaner utility tests (parse_number, clean_text, noise detection)
  2. Per-cleaner integration tests that run against the real CSV files
  3. Edge-case tests with synthetic data via tmp_path fixtures
"""

from __future__ import annotations

import textwrap
from datetime import timedelta
from pathlib import Path

import pandas as pd
import pytest

# Resolve project root so imports work regardless of cwd
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RAW_FILES
from src.data_cleaning.base_cleaner import BaseCleaner
from src.data_cleaning.division_summary_cleaner import DivisionSummaryCleaner
from src.data_cleaning.tax_report_cleaner import TaxReportCleaner
from src.data_cleaning.attendance_cleaner import AttendanceCleaner
from src.data_cleaning.sales_by_customer_cleaner import SalesByCustomerCleaner
from src.data_cleaning.customer_orders_cleaner import CustomerOrdersCleaner
from src.data_cleaning.sales_by_item_cleaner import SalesByItemCleaner
from src.data_cleaning.monthly_sales_cleaner import MonthlySalesCleaner
from src.data_cleaning.avg_sales_cleaner import AvgSalesCleaner
from src.data_cleaning.pipeline import DataPipeline


 # Helpers
 
def _write_csv(tmp_path: Path, name: str, content: str) -> Path:
    """Write a CSV string to a temp file and return its path."""
    p = tmp_path / name
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return p


 # 1. BaseCleaner utility tests
 
class TestParseNumber:
    def test_plain_integer(self):
        assert BaseCleaner.parse_number("42") == 42.0

    def test_comma_formatted(self):
        assert BaseCleaner.parse_number("1,251,486.48") == 1251486.48

    def test_quoted_comma_formatted(self):
        assert BaseCleaner.parse_number('"893,918.92"') == 893918.92

    def test_negative(self):
        assert BaseCleaner.parse_number("-893,918.92") == -893918.92

    def test_zero(self):
        assert BaseCleaner.parse_number("0.00") == 0.0

    def test_empty_string(self):
        assert BaseCleaner.parse_number("") is None

    def test_none(self):
        assert BaseCleaner.parse_number(None) is None

    def test_non_numeric(self):
        assert BaseCleaner.parse_number("hello") is None

    def test_dash(self):
        assert BaseCleaner.parse_number("-") is None

    def test_whitespace(self):
        assert BaseCleaner.parse_number("  1234.56  ") == 1234.56


class TestCleanText:
    def test_strips_whitespace(self):
        assert BaseCleaner.clean_text("  hello world  ") == "hello world"

    def test_strips_quotes(self):
        assert BaseCleaner.clean_text('"some text"') == "some text"

    def test_collapses_spaces(self):
        assert BaseCleaner.clean_text("a   b   c") == "a b c"

    def test_removes_trailing_dots(self):
        assert BaseCleaner.clean_text("NUTELLA SPREAD CHIMNEY.") == "NUTELLA SPREAD CHIMNEY"

    def test_removes_trailing_commas(self):
        assert BaseCleaner.clean_text("CARAMEL SAUCE,") == "CARAMEL SAUCE"


class TestNoiseDetection:
    """Test the _is_noise_line method via a concrete subclass."""

    class _DummyCleaner(BaseCleaner):
        def _parse_clean(self, lines):
            return pd.DataFrame()

    def _make_cleaner(self, tmp_path: Path):
        p = tmp_path / "dummy.csv"
        p.write_text("a,b,c\n", encoding="utf-8")
        return self._DummyCleaner(p)

    def test_copyright_line(self, tmp_path):
        c = self._make_cleaner(tmp_path)
        assert c._is_noise_line('REP_S_00136,,"Copyright © 2026 Omega Software, Inc."')

    def test_url_line(self, tmp_path):
        c = self._make_cleaner(tmp_path)
        assert c._is_noise_line(",,,,www.omegapos.com,")

    def test_report_id_line(self, tmp_path):
        c = self._make_cleaner(tmp_path)
        assert c._is_noise_line("REP_S_00461,,something")

    def test_blank_line(self, tmp_path):
        c = self._make_cleaner(tmp_path)
        assert c._is_noise_line("")
        assert c._is_noise_line("   ")

    def test_data_line_not_noise(self, tmp_path):
        c = self._make_cleaner(tmp_path)
        assert not c._is_noise_line("Conut,Bev Add-ons,,0.00,1197189.17")


class TestFileNotFound:
    def test_raises_on_missing_file(self):
        class _Dummy(BaseCleaner):
            def _parse_clean(self, lines):
                return pd.DataFrame()

        with pytest.raises(FileNotFoundError):
            _Dummy("/nonexistent/path.csv")


 # 2. Integration tests against real CSV files
 
def _skip_if_missing(name: str):
    path = RAW_FILES.get(name)
    if path is None or not path.exists():
        pytest.skip(f"Raw file for '{name}' not found at {path}")
    return path


class TestDivisionSummaryCleaner:
    def test_produces_dataframe(self):
        path = _skip_if_missing("division_summary")
        df = DivisionSummaryCleaner(path).clean()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_expected_columns(self):
        path = _skip_if_missing("division_summary")
        df = DivisionSummaryCleaner(path).clean()
        expected = {"branch", "division", "delivery", "table", "take_away", "total"}
        assert expected.issubset(set(df.columns))

    def test_all_branches_present(self):
        path = _skip_if_missing("division_summary")
        df = DivisionSummaryCleaner(path).clean()
        branches = set(df["branch"].unique())
        for b in ["Conut", "Conut - Tyre", "Conut Jnah", "Main Street Coffee"]:
            assert b in branches, f"Missing branch: {b}"

    def test_numeric_columns_are_float(self):
        path = _skip_if_missing("division_summary")
        df = DivisionSummaryCleaner(path).clean()
        for col in ("delivery", "table", "take_away", "total"):
            assert df[col].dtype in ("float64", "float32"), f"{col} is not float"

    def test_no_total_rows_in_output(self):
        path = _skip_if_missing("division_summary")
        df = DivisionSummaryCleaner(path).clean()
        assert not df["division"].str.upper().str.contains("TOTAL").any()


class TestTaxReportCleaner:
    def test_produces_dataframe(self):
        path = _skip_if_missing("tax_report")
        df = TaxReportCleaner(path).clean()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 4  # 4 branches

    def test_expected_columns(self):
        path = _skip_if_missing("tax_report")
        df = TaxReportCleaner(path).clean()
        assert set(df.columns) == {"branch", "vat_11_pct", "total_tax"}

    def test_all_branches_present(self):
        path = _skip_if_missing("tax_report")
        df = TaxReportCleaner(path).clean()
        branches = set(df["branch"].unique())
        assert len(branches) == 4

    def test_tax_values_positive(self):
        path = _skip_if_missing("tax_report")
        df = TaxReportCleaner(path).clean()
        assert (df["total_tax"] > 0).all()


class TestAttendanceCleaner:
    def test_produces_dataframe(self):
        path = _skip_if_missing("attendance")
        df = AttendanceCleaner(path).clean()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_expected_columns(self):
        path = _skip_if_missing("attendance")
        df = AttendanceCleaner(path).clean()
        expected = {"emp_id", "emp_name", "branch", "punch_in", "punch_out", "work_duration", "work_duration_hours"}
        assert expected.issubset(set(df.columns))

    def test_multiple_employees(self):
        path = _skip_if_missing("attendance")
        df = AttendanceCleaner(path).clean()
        assert df["emp_id"].nunique() > 1

    def test_branches_present(self):
        path = _skip_if_missing("attendance")
        df = AttendanceCleaner(path).clean()
        branches = set(df["branch"].dropna().unique())
        assert len(branches) >= 2

    def test_duration_hours_reasonable(self):
        path = _skip_if_missing("attendance")
        df = AttendanceCleaner(path).clean()
        # Most shifts should be < 24 hours (with some exceptions for data quirks)
        assert df["work_duration_hours"].median() < 12


class TestSalesByCustomerCleaner:
    def test_produces_dataframe(self):
        path = _skip_if_missing("sales_by_customer")
        df = SalesByCustomerCleaner(path).clean()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_expected_columns(self):
        path = _skip_if_missing("sales_by_customer")
        df = SalesByCustomerCleaner(path).clean()
        expected = {"branch", "customer", "qty", "description", "price", "is_cancellation"}
        assert expected.issubset(set(df.columns))

    def test_has_cancellations(self):
        path = _skip_if_missing("sales_by_customer")
        df = SalesByCustomerCleaner(path).clean()
        assert df["is_cancellation"].any(), "Expected some cancellation rows (negative qty)"

    def test_branches_present(self):
        path = _skip_if_missing("sales_by_customer")
        df = SalesByCustomerCleaner(path).clean()
        branches = set(df["branch"].dropna().unique())
        assert len(branches) >= 3

    def test_descriptions_cleaned(self):
        path = _skip_if_missing("sales_by_customer")
        df = SalesByCustomerCleaner(path).clean()
        # No descriptions should be empty after cleaning
        non_empty = df["description"].str.strip().str.len() > 0
        assert non_empty.sum() > 0


class TestCustomerOrdersCleaner:
    def test_produces_dataframe(self):
        path = _skip_if_missing("customer_orders")
        df = CustomerOrdersCleaner(path).clean()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_expected_columns(self):
        path = _skip_if_missing("customer_orders")
        df = CustomerOrdersCleaner(path).clean()
        expected = {"branch", "customer", "phone", "first_order", "last_order", "total", "num_orders"}
        assert expected.issubset(set(df.columns))

    def test_all_branches_present(self):
        path = _skip_if_missing("customer_orders")
        df = CustomerOrdersCleaner(path).clean()
        branches = set(df["branch"].dropna().unique())
        for b in ["Conut - Tyre", "Conut", "Conut Jnah", "Main Street Coffee"]:
            assert b in branches, f"Missing branch: {b}"

    def test_datetime_columns_parsed(self):
        path = _skip_if_missing("customer_orders")
        df = CustomerOrdersCleaner(path).clean()
        assert pd.api.types.is_datetime64_any_dtype(df["first_order"])


class TestSalesByItemCleaner:
    def test_produces_dataframe(self):
        path = _skip_if_missing("sales_by_item")
        df = SalesByItemCleaner(path).clean()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_expected_columns(self):
        path = _skip_if_missing("sales_by_item")
        df = SalesByItemCleaner(path).clean()
        expected = {"branch", "division", "group", "description", "qty", "total_amount", "unit_price"}
        assert expected.issubset(set(df.columns))

    def test_all_branches_present(self):
        path = _skip_if_missing("sales_by_item")
        df = SalesByItemCleaner(path).clean()
        branches = set(df["branch"].dropna().unique())
        assert len(branches) == 4

    def test_has_coffee_items(self):
        path = _skip_if_missing("sales_by_item")
        df = SalesByItemCleaner(path).clean()
        coffee_items = df[df["division"].str.contains("Coffee", case=False, na=False)]
        assert len(coffee_items) > 0

    def test_has_shake_items(self):
        path = _skip_if_missing("sales_by_item")
        df = SalesByItemCleaner(path).clean()
        shake_items = df[df["division"].str.contains("Shake", case=False, na=False)]
        assert len(shake_items) > 0

    def test_no_total_rows(self):
        path = _skip_if_missing("sales_by_item")
        df = SalesByItemCleaner(path).clean()
        assert not df["description"].str.contains("Total by", case=False, na=False).any()


class TestMonthlySalesCleaner:
    def test_produces_dataframe(self):
        path = _skip_if_missing("monthly_sales")
        df = MonthlySalesCleaner(path).clean()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_expected_columns(self):
        path = _skip_if_missing("monthly_sales")
        df = MonthlySalesCleaner(path).clean()
        expected = {"branch", "month", "month_num", "year", "total", "date"}
        assert expected.issubset(set(df.columns))

    def test_all_branches_present(self):
        path = _skip_if_missing("monthly_sales")
        df = MonthlySalesCleaner(path).clean()
        branches = set(df["branch"].unique())
        assert len(branches) == 4

    def test_months_in_range(self):
        path = _skip_if_missing("monthly_sales")
        df = MonthlySalesCleaner(path).clean()
        assert df["month_num"].min() >= 1
        assert df["month_num"].max() <= 12

    def test_date_column_is_datetime(self):
        path = _skip_if_missing("monthly_sales")
        df = MonthlySalesCleaner(path).clean()
        assert pd.api.types.is_datetime64_any_dtype(df["date"])

    def test_main_street_starts_september(self):
        path = _skip_if_missing("monthly_sales")
        df = MonthlySalesCleaner(path).clean()
        msc = df[df["branch"] == "Main Street Coffee"]
        assert msc["month_num"].min() == 9  # September


class TestAvgSalesCleaner:
    def test_produces_dataframe(self):
        path = _skip_if_missing("avg_sales")
        df = AvgSalesCleaner(path).clean()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_expected_columns(self):
        path = _skip_if_missing("avg_sales")
        df = AvgSalesCleaner(path).clean()
        expected = {"branch", "menu_channel", "num_customers", "sales", "avg_per_customer"}
        assert expected.issubset(set(df.columns))

    def test_all_branches_present(self):
        path = _skip_if_missing("avg_sales")
        df = AvgSalesCleaner(path).clean()
        branches = set(df["branch"].unique())
        assert len(branches) == 4

    def test_menu_channels(self):
        path = _skip_if_missing("avg_sales")
        df = AvgSalesCleaner(path).clean()
        channels = set(df["menu_channel"].unique())
        assert "TABLE" in channels or "DELIVERY" in channels or "TAKE AWAY" in channels


 # 3. Edge-case tests with synthetic data
 
class TestTaxReportSynthetic:
    def test_parses_minimal_tax_report(self, tmp_path):
        content = """\
        Conut - Tyre,,,,,,,,,
        Tax Report,,,,,,,,,
        30-Jan-26,,,,,,,,Page 1 of, 1
        ,Year: 2025 - All Months,,,,,,,,
        TAX DESCRIPTION,VAT 11 %,Tax 2,Tax 3,Tax 4,Tax 5,,Service,Total,
        Branch Name:  TestBranch,,,,,,,,,
        Total By Branch,"100,000.00",0.00,0.00,0.00,,0.00,0.00,"100,000.00",
        REP_S_00194,"Copyright © 2026 Omega Software, Inc.",,,,,,,www.omegapos.com,
        """
        p = _write_csv(tmp_path, "tax.csv", content)
        df = TaxReportCleaner(p).clean()
        assert len(df) == 1
        assert df.iloc[0]["branch"] == "TestBranch"
        assert df.iloc[0]["total_tax"] == 100000.0


class TestMonthlySalesSynthetic:
    def test_parses_minimal_monthly(self, tmp_path):
        content = """\
        Conut - Tyre,,,,
        Monthly Sales,,,,
        30-Jan-26,,,,
        ,Year: 2025,,Page 1 of, 1
        Month,,Year,Total,
        Branch Name: TestBranch,,,,
        October,,2025,"500,000.00",
        November,,2025,"600,000.00",
        ,,Total for    2025,"1,100,000.00",
        ,,Grand Total:,"1,100,000.00",
        REP_S_00334_1,Copyright © 2026,,www.omegapos.com,
        """
        p = _write_csv(tmp_path, "monthly.csv", content)
        df = MonthlySalesCleaner(p).clean()
        assert len(df) == 2
        assert set(df["month"].tolist()) == {"October", "November"}
        assert df[df["month"] == "October"].iloc[0]["total"] == 500000.0


class TestAvgSalesSynthetic:
    def test_parses_minimal_avg_sales(self, tmp_path):
        content = """\
        Conut - Tyre,,,,
        Average Sales By Menu,,,,
        ,Year: 2025 - All Months,,Page 1 of, 1
        30-Jan-26,,,,
        Menu Name,# Cust,Sales,Avg Customer,
        Conut,,,,
        TABLE,100.00,5000000.00,50000.00,
        Total By Branch:,100.00,5000000.00,50000.00,
        Total :,100.00,5000000.00,50000.00,
        REP_S_00435,Copyright © 2026,,www.omegapos.com,
        """
        p = _write_csv(tmp_path, "avg.csv", content)
        df = AvgSalesCleaner(p).clean()
        assert len(df) == 1
        assert df.iloc[0]["branch"] == "Conut"
        assert df.iloc[0]["menu_channel"] == "TABLE"
        assert df.iloc[0]["num_customers"] == 100


class TestAttendanceSynthetic:
    def test_parses_minimal_attendance(self, tmp_path):
        content = """\
        Conut - Tyre,,,,,
        Time & Attendance Report,,,,,
        ,30-Jan-26,From Date: 01-Dec-2025 30-Dec-2025,,,
        ,PUNCH IN,,PUNCH OUT,,Work Duration
        ,EMP ID :1.0,NAME :Person_0001,,,
        ,Main Street Coffee,,,,
        01-Dec-25,,07.39.35,01-Dec-25,19.37.56,11.58.21
        02-Dec-25,,15.14.59,02-Dec-25,23.51.33,08.36.34
        ,,,,Total :,20:34:55
        REP_S_00461,,Copyright © 2026,,,www.omegapos.com
        """
        p = _write_csv(tmp_path, "attendance.csv", content)
        df = AttendanceCleaner(p).clean()
        assert len(df) == 2
        assert df.iloc[0]["emp_id"] == "1"
        assert df.iloc[0]["emp_name"] == "Person_0001"
        assert df.iloc[0]["branch"] == "Main Street Coffee"
        assert df.iloc[0]["work_duration_hours"] == pytest.approx(11 + 58/60 + 21/3600, abs=0.01)


class TestSalesByItemSynthetic:
    def test_parses_minimal_items(self, tmp_path):
        content = """\
        Conut - Tyre,,,,
        Sales by Items By Group,,,,
        30-Jan-26,Years:2025 Months:0,,Page 1 of, 1
        Description,Barcode,Qty,Total Amount,
        Branch: TestBranch,,,,
        Division: Hot-Coffee Based,,,,
        Group: Hot-Coffee Based,,,,
        CAFFE LATTE,,47.0,"19,606,621.36",
        DOUBLE ESPRESSO,,55.0,"18,063,783.51",
        Total by Group: Hot-Coffee Based,,102.0,"37,670,404.87",
        Total by Division: Hot-Coffee Based,,102.0,"37,670,404.87",
        Total by Branch: TestBranch,,102.0,"37,670,404.87",
        REP_S_00191,Copyright © 2026,,,
        """
        p = _write_csv(tmp_path, "items.csv", content)
        df = SalesByItemCleaner(p).clean()
        assert len(df) == 2
        assert set(df["description"].tolist()) == {"CAFFE LATTE", "DOUBLE ESPRESSO"}
        assert df[df["description"] == "CAFFE LATTE"].iloc[0]["qty"] == 47.0
        latte = df[df["description"] == "CAFFE LATTE"].iloc[0]
        assert latte["unit_price"] == pytest.approx(19606621.36 / 47.0, rel=0.01)


 # 4. Pipeline integration test
 
class TestDataPipeline:
    def test_pipeline_runs(self, tmp_path):
        pipeline = DataPipeline(output_dir=tmp_path / "out")
        results = pipeline.run(save=True)
        assert isinstance(results, dict)
        # At least some datasets should have been cleaned
        assert len(results) > 0

    def test_pipeline_summary(self, tmp_path):
        pipeline = DataPipeline(output_dir=tmp_path / "out")
        pipeline.run(save=False)
        summary = pipeline.summary()
        assert isinstance(summary, pd.DataFrame)
        assert "dataset" in summary.columns
        assert "rows" in summary.columns

    def test_pipeline_get(self, tmp_path):
        pipeline = DataPipeline(output_dir=tmp_path / "out")
        pipeline.run(save=False)
        if pipeline.datasets:
            name = list(pipeline.datasets.keys())[0]
            df = pipeline.get(name)
            assert isinstance(df, pd.DataFrame)

    def test_pipeline_get_missing_raises(self, tmp_path):
        pipeline = DataPipeline(output_dir=tmp_path / "out")
        pipeline.run(save=False)
        with pytest.raises(KeyError):
            pipeline.get("nonexistent_dataset")

    def test_saved_files_exist(self, tmp_path):
        out_dir = tmp_path / "out"
        pipeline = DataPipeline(output_dir=out_dir)
        pipeline.run(save=True)
        for name in pipeline.datasets:
            assert (out_dir / f"{name}.csv").exists()
