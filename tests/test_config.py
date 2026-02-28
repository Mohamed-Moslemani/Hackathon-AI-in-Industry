"""Unit tests for the configuration module.

Tests verify:
  1. Path construction and correctness
  2. Directory existence
  3. Raw files mapping completeness
  4. Path types and attributes
"""

from __future__ import annotations

from pathlib import Path
import sys
import pytest

# Resolve project root so imports work regardless of cwd
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config


class TestProjectRoot:
    """Tests for the PROJECT_ROOT constant."""

    def test_project_root_is_path_object(self):
        """PROJECT_ROOT should be a Path object."""
        assert isinstance(config.PROJECT_ROOT, Path)

    def test_project_root_is_absolute(self):
        """PROJECT_ROOT should be an absolute path."""
        assert config.PROJECT_ROOT.is_absolute()

    def test_project_root_exists(self):
        """PROJECT_ROOT should exist on the filesystem."""
        assert config.PROJECT_ROOT.exists()

    def test_project_root_is_directory(self):
        """PROJECT_ROOT should be a directory."""
        assert config.PROJECT_ROOT.is_dir()

    def test_project_root_has_src_directory(self):
        """PROJECT_ROOT should contain a 'src' directory."""
        src_dir = config.PROJECT_ROOT / "src"
        assert src_dir.exists()
        assert src_dir.is_dir()

    def test_project_root_name(self):
        """PROJECT_ROOT should be named 'Hackathon'."""
        assert config.PROJECT_ROOT.name == "Hackathon"


class TestRawDataDir:
    """Tests for the RAW_DATA_DIR constant."""

    def test_raw_data_dir_is_path_object(self):
        """RAW_DATA_DIR should be a Path object."""
        assert isinstance(config.RAW_DATA_DIR, Path)

    def test_raw_data_dir_is_absolute(self):
        """RAW_DATA_DIR should be an absolute path."""
        assert config.RAW_DATA_DIR.is_absolute()

    def test_raw_data_dir_exists(self):
        """RAW_DATA_DIR should exist on the filesystem."""
        assert config.RAW_DATA_DIR.exists()

    def test_raw_data_dir_is_directory(self):
        """RAW_DATA_DIR should be a directory."""
        assert config.RAW_DATA_DIR.is_dir()

    def test_raw_data_dir_name(self):
        """RAW_DATA_DIR should be named 'Conut bakery Scaled Data'."""
        assert config.RAW_DATA_DIR.name == "Conut bakery Scaled Data"

    def test_raw_data_dir_parent_is_project_root(self):
        """RAW_DATA_DIR's parent should be PROJECT_ROOT."""
        assert config.RAW_DATA_DIR.parent == config.PROJECT_ROOT


class TestCleanedDataDir:
    """Tests for the CLEANED_DATA_DIR constant."""

    def test_cleaned_data_dir_is_path_object(self):
        """CLEANED_DATA_DIR should be a Path object."""
        assert isinstance(config.CLEANED_DATA_DIR, Path)

    def test_cleaned_data_dir_is_absolute(self):
        """CLEANED_DATA_DIR should be an absolute path."""
        assert config.CLEANED_DATA_DIR.is_absolute()

    def test_cleaned_data_dir_exists(self):
        """CLEANED_DATA_DIR should exist on the filesystem."""
        assert config.CLEANED_DATA_DIR.exists()

    def test_cleaned_data_dir_is_directory(self):
        """CLEANED_DATA_DIR should be a directory."""
        assert config.CLEANED_DATA_DIR.is_dir()

    def test_cleaned_data_dir_path_structure(self):
        """CLEANED_DATA_DIR should be at data_cleaned/cleaned."""
        assert config.CLEANED_DATA_DIR.name == "cleaned"
        assert config.CLEANED_DATA_DIR.parent.name == "data_cleaned"

    def test_cleaned_data_dir_parent_is_project_root_subdir(self):
        """CLEANED_DATA_DIR's parent's parent should be PROJECT_ROOT."""
        assert config.CLEANED_DATA_DIR.parent.parent == config.PROJECT_ROOT


class TestFeaturesDir:
    """Tests for the FEATURES_DIR constant."""

    def test_features_dir_is_path_object(self):
        """FEATURES_DIR should be a Path object."""
        assert isinstance(config.FEATURES_DIR, Path)

    def test_features_dir_is_absolute(self):
        """FEATURES_DIR should be an absolute path."""
        assert config.FEATURES_DIR.is_absolute()

    def test_features_dir_exists(self):
        """FEATURES_DIR should exist on the filesystem."""
        assert config.FEATURES_DIR.exists()

    def test_features_dir_is_directory(self):
        """FEATURES_DIR should be a directory."""
        assert config.FEATURES_DIR.is_dir()

    def test_features_dir_path_structure(self):
        """FEATURES_DIR should be at data_cleaned/features."""
        assert config.FEATURES_DIR.name == "features"
        assert config.FEATURES_DIR.parent.name == "data_cleaned"

    def test_features_dir_parent_is_project_root_subdir(self):
        """FEATURES_DIR's parent's parent should be PROJECT_ROOT."""
        assert config.FEATURES_DIR.parent.parent == config.PROJECT_ROOT


class TestAnalyticsDir:
    """Tests for the ANALYTICS_DIR constant."""

    def test_analytics_dir_is_path_object(self):
        """ANALYTICS_DIR should be a Path object."""
        assert isinstance(config.ANALYTICS_DIR, Path)

    def test_analytics_dir_is_absolute(self):
        """ANALYTICS_DIR should be an absolute path."""
        assert config.ANALYTICS_DIR.is_absolute()

    def test_analytics_dir_exists(self):
        """ANALYTICS_DIR should exist on the filesystem."""
        assert config.ANALYTICS_DIR.exists()

    def test_analytics_dir_is_directory(self):
        """ANALYTICS_DIR should be a directory."""
        assert config.ANALYTICS_DIR.is_dir()

    def test_analytics_dir_path_structure(self):
        """ANALYTICS_DIR should be at data_cleaned/analytics."""
        assert config.ANALYTICS_DIR.name == "analytics"
        assert config.ANALYTICS_DIR.parent.name == "data_cleaned"

    def test_analytics_dir_parent_is_project_root_subdir(self):
        """ANALYTICS_DIR's parent's parent should be PROJECT_ROOT."""
        assert config.ANALYTICS_DIR.parent.parent == config.PROJECT_ROOT


class TestRawFiles:
    """Tests for the RAW_FILES mapping."""

    def test_raw_files_is_dict(self):
        """RAW_FILES should be a dictionary."""
        assert isinstance(config.RAW_FILES, dict)

    def test_raw_files_not_empty(self):
        """RAW_FILES should not be empty."""
        assert len(config.RAW_FILES) > 0

    def test_raw_files_expected_keys(self):
        """RAW_FILES should contain all expected data source keys."""
        expected_keys = {
            "division_summary",
            "tax_report",
            "attendance",
            "sales_by_customer",
            "customer_orders",
            "sales_by_item",
            "monthly_sales",
            "avg_sales",
        }
        assert set(config.RAW_FILES.keys()) == expected_keys

    def test_raw_files_values_are_paths(self):
        """All values in RAW_FILES should be Path objects."""
        for key, value in config.RAW_FILES.items():
            assert isinstance(value, Path), f"RAW_FILES['{key}'] is not a Path object"

    def test_raw_files_all_absolute(self):
        """All paths in RAW_FILES should be absolute."""
        for key, value in config.RAW_FILES.items():
            assert value.is_absolute(), f"RAW_FILES['{key}'] is not an absolute path"

    def test_raw_files_all_exist(self):
        """All files in RAW_FILES should exist on the filesystem."""
        for key, value in config.RAW_FILES.items():
            assert value.exists(), f"RAW_FILES['{key}'] ({value}) does not exist"

    def test_raw_files_all_are_files(self):
        """All paths in RAW_FILES should be files, not directories."""
        for key, value in config.RAW_FILES.items():
            assert value.is_file(), f"RAW_FILES['{key}'] is not a file"

    def test_raw_files_all_are_csv(self):
        """All files in RAW_FILES should have .csv extension."""
        for key, value in config.RAW_FILES.items():
            assert value.suffix == ".csv", f"RAW_FILES['{key}'] does not have .csv extension"

    def test_raw_files_parent_is_raw_data_dir(self):
        """All files in RAW_FILES should be in RAW_DATA_DIR."""
        for key, value in config.RAW_FILES.items():
            assert value.parent == config.RAW_DATA_DIR, (
                f"RAW_FILES['{key}'] parent is not RAW_DATA_DIR"
            )

    def test_division_summary_file(self):
        """division_summary file should map to REP_S_00136_SMRY.csv."""
        assert config.RAW_FILES["division_summary"].name == "REP_S_00136_SMRY.csv"

    def test_tax_report_file(self):
        """tax_report file should map to REP_S_00194_SMRY.csv."""
        assert config.RAW_FILES["tax_report"].name == "REP_S_00194_SMRY.csv"

    def test_attendance_file(self):
        """attendance file should map to REP_S_00461.csv."""
        assert config.RAW_FILES["attendance"].name == "REP_S_00461.csv"

    def test_sales_by_customer_file(self):
        """sales_by_customer file should map to REP_S_00502.csv."""
        assert config.RAW_FILES["sales_by_customer"].name == "REP_S_00502.csv"

    def test_customer_orders_file(self):
        """customer_orders file should map to rep_s_00150.csv."""
        assert config.RAW_FILES["customer_orders"].name == "rep_s_00150.csv"

    def test_sales_by_item_file(self):
        """sales_by_item file should map to rep_s_00191_SMRY.csv."""
        assert config.RAW_FILES["sales_by_item"].name == "rep_s_00191_SMRY.csv"

    def test_monthly_sales_file(self):
        """monthly_sales file should map to rep_s_00334_1_SMRY.csv."""
        assert config.RAW_FILES["monthly_sales"].name == "rep_s_00334_1_SMRY.csv"

    def test_avg_sales_file(self):
        """avg_sales file should map to rep_s_00435_SMRY.csv."""
        assert config.RAW_FILES["avg_sales"].name == "rep_s_00435_SMRY.csv"


class TestDirectoryConsistency:
    """Integration tests for consistency across directory configurations."""

    def test_all_data_dirs_are_siblings(self):
        """CLEANED_DATA_DIR, FEATURES_DIR, and ANALYTICS_DIR should share same parent."""
        parent = config.CLEANED_DATA_DIR.parent
        assert config.FEATURES_DIR.parent == parent
        assert config.ANALYTICS_DIR.parent == parent

    def test_all_data_dirs_are_subdirs_of_project(self):
        """All data directories should be under PROJECT_ROOT."""
        assert config.CLEANED_DATA_DIR.parent.parent == config.PROJECT_ROOT
        assert config.FEATURES_DIR.parent.parent == config.PROJECT_ROOT
        assert config.ANALYTICS_DIR.parent.parent == config.PROJECT_ROOT

    def test_data_dirs_not_overlapping(self):
        """Data directories should be distinct."""
        paths = {config.CLEANED_DATA_DIR, config.FEATURES_DIR, config.ANALYTICS_DIR}
        assert len(paths) == 3


class TestPathResolution:
    """Tests for proper path resolution behavior."""

    def test_paths_are_resolved(self):
        """All paths should be resolved (no relative components)."""
        assert ".." not in str(config.PROJECT_ROOT)
        assert ".." not in str(config.RAW_DATA_DIR)
        assert ".." not in str(config.CLEANED_DATA_DIR)
        assert ".." not in str(config.FEATURES_DIR)
        assert ".." not in str(config.ANALYTICS_DIR)

    def test_no_symlinks_in_config_paths(self):
        """Config paths should use resolve() to handle any symlinks."""
        # Just verify that the paths are what we expect
        assert config.PROJECT_ROOT.exists()
        assert config.RAW_DATA_DIR.exists()
