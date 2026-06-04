# Credit Risk Probability Model for Alternative Data

This project builds a probability-of-default style credit-risk model from Xente transaction data. Because the dataset has no true loan-default outcome, the target is a documented proxy label:

- high risk when a customer belongs to the weakest RFMS segment;
- high risk when the customer has at least one observed fraud transaction;
- otherwise low risk.

The model uses customer-level transaction behavior, product/channel usage, recency, frequency, monetary value, and activity consistency. `FraudResult` is used only to construct the proxy target and is excluded from model inputs.

## Credit Scoring Business Understanding

Basel II requires financial institutions to measure credit risk in a way that is transparent, auditable, and explainable to regulators and internal risk committees. For this reason, a credit scoring workflow should preserve interpretable evidence such as feature definitions, Weight of Evidence (WoE), Information Value (IV), model coefficients or feature importances, validation metrics, and documented assumptions. A highly accurate black-box model is not enough if the lender cannot explain why a customer was approved, declined, or routed for review.

The Xente transaction dataset does not contain an actual loan-default or delinquency outcome, so a proxy target variable is necessary for model development. In this project, `is_high_risk` is based on RFM customer behavior and fraud history. This allows supervised learning to proceed, but it creates business risk: the model may learn behavioral weakness or fraud exposure rather than true repayment default. Before production use, the proxy must be validated against real repayment, arrears, charge-off, or collections outcomes.

There is also a trade-off between interpretable models and high-performance models. Logistic Regression with WoE features is easier to explain and aligns well with traditional scorecard practice, while Random Forest can capture nonlinear behavioral patterns and scored best in this project. A responsible lender may use the high-performance model for ranking risk while keeping an interpretable challenger model and feature-level explanations for governance.

## Project Structure

```text
data/raw/Data/data.csv                 Raw Xente transactions
data/processed/customer_features.csv   Customer-level modeling table
notebooks/eda.ipynb                    Executed Xente EDA notebook
models/credit_risk_pd_model.joblib     Trained model artifact
models/metrics.json                    Evaluation metrics
src/data_processing.py                 Rubric-aligned preprocessing, RFM, WoE/IV, pipeline code
src/features.py                        Feature engineering and proxy target
src/train.py                           Model training and evaluation
src/predict.py                         Batch/single-customer scoring
src/api/main.py                        FastAPI serving app
src/api/pydantic_models.py             Pydantic API schemas
tests/                                 Unit tests
.github/workflows/ci.yml               Lint and test workflow
reports/credit_risk_probability_model_report.md
```

## Run

```bash
pip install -r requirements.txt
python -m src.train
pytest
uvicorn src.api.main:app --reload
```

## Latest Training Result

The best model from the included run is Random Forest:

- ROC-AUC: 0.9985
- F1: 0.9518
- Precision: 0.9655
- Recall: 0.9385
- Proxy high-risk rate: 23.94%

Treat these as proxy-label evaluation metrics, not verified real-world default performance.

## Git & GitHub Workflow

Recommended branch sequence for submission:

- `task-1-business-understanding`
- `task-2-eda`
- `task-3-feature-engineering`
- `task-4-proxy-target`
- `task-5-model-training`
- `task-6-deployment-ci`

Each task branch should be merged to `main` with a pull request after tests pass. The included `.github/workflows/ci.yml` runs flake8 and pytest on pushes and pull requests to `main`.
