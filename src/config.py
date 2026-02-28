from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "Conut bakery Scaled Data"
CLEANED_DATA_DIR = PROJECT_ROOT / "data" / "cleaned"

RAW_FILES = {
    "division_summary": RAW_DATA_DIR / "REP_S_00136_SMRY.csv",
    "tax_report": RAW_DATA_DIR / "REP_S_00194_SMRY.csv",
    "attendance": RAW_DATA_DIR / "REP_S_00461.csv",
    "sales_by_customer": RAW_DATA_DIR / "REP_S_00502.csv",
    "customer_orders": RAW_DATA_DIR / "rep_s_00150.csv",
    "sales_by_item": RAW_DATA_DIR / "rep_s_00191_SMRY.csv",
    "monthly_sales": RAW_DATA_DIR / "rep_s_00334_1_SMRY.csv",
    "avg_sales": RAW_DATA_DIR / "rep_s_00435_SMRY.csv",
}
