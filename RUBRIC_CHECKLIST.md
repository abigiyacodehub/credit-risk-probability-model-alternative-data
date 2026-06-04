# Rubric Checklist

## Task 1 and 2: Business Understanding and EDA

- Standard project structure is present.
- `README.md` contains `Credit Scoring Business Understanding`.
- `notebooks/eda.ipynb` has executed cells with visible outputs.
- EDA covers overview, summary statistics, numerical/categorical distributions, correlation, missing values, and outliers.
- The notebook ends with 5 key insights.

## Task 3 and 4: Feature Engineering and Proxy Target Variable

- `src/data_processing.py` computes total amount, average amount, transaction count, and standard deviation per `CustomerId`.
- Datetime features include hour, day, month, and year.
- Categorical encoding uses `OneHotEncoder`.
- Missing values use `SimpleImputer`.
- Numerical scaling uses `StandardScaler`.
- WoE/IV is implemented with `calculate_woe_iv` and `WoEEncoder`.
- Transformations are chained with `sklearn.pipeline.Pipeline`.
- RFM, K-Means clustering, `is_high_risk`, and processed dataset output are implemented.

## Task 5 and 6: Model Training, Tracking, and Deployment

- `src/train.py` trains Logistic Regression and Random Forest.
- Train/test split uses `random_state`.
- Hyperparameter tuning uses `GridSearchCV`.
- MLflow code logs params, metrics, artifacts, model, and registers a model when MLflow is available.
- `tests/test_data_processing.py` includes multiple helper-function tests.
- `src/api/main.py` provides FastAPI `/predict`; `src/api/pydantic_models.py` defines schemas.
- `Dockerfile`, `docker-compose.yml`, and `.github/workflows/ci.yml` are present.
- CI workflow includes flake8 and pytest.
