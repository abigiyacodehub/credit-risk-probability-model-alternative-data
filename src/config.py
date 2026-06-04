from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Data" / "data.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "customer_features.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "credit_risk_pd_model.joblib"
REPORT_PATH = PROJECT_ROOT / "reports" / "credit_risk_probability_model_report.md"
RANDOM_STATE = 42
