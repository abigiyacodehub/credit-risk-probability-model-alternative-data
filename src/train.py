from __future__ import annotations

import json

import joblib
import pandas as pd
try:
    import mlflow
    import mlflow.sklearn
except Exception:
    mlflow = None
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import MODEL_PATH, PROCESSED_DATA_PATH, RANDOM_STATE, RAW_DATA_PATH
from src.data_processing import TARGET_COL, create_processed_dataset


def feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {"CustomerId", TARGET_COL, "fraud_txn_count", "rfm_cluster", "rfms_cluster"}
    return [col for col in df.columns if col not in excluded]


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )


def _evaluate(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict:
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    return {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y, y_prob),
        "average_precision": average_precision_score(y, y_prob),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
        "classification_report": classification_report(y, y_pred, zero_division=0),
    }


def train_models(df: pd.DataFrame) -> tuple[dict, Pipeline]:
    cols = feature_columns(df)
    X = df[cols]
    y = df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    preprocessor = _build_preprocessor(X_train)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    candidates = {
        "logistic_regression": (
            LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE),
            {"model__C": [0.1, 1.0, 10.0]},
        ),
        "random_forest": (
            RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
            {
                "model__n_estimators": [200, 400],
                "model__max_depth": [4, 8, None],
                "model__min_samples_leaf": [1, 5],
            },
        ),
    }

    results = {}
    fitted = {}
    for name, (estimator, params) in candidates.items():
        pipe = Pipeline([("preprocess", preprocessor), ("model", estimator)])
        search = GridSearchCV(
            pipe,
            params,
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            refit=True,
        )
        search.fit(X_train, y_train)
        metrics = _evaluate(search.best_estimator_, X_test, y_test)
        metrics["best_params"] = search.best_params_
        metrics["cv_roc_auc"] = search.best_score_
        results[name] = metrics
        fitted[name] = search.best_estimator_

    best_name = max(results, key=lambda n: results[n]["roc_auc"])
    results["best_model"] = best_name
    results["n_customers"] = int(len(df))
    results["target_rate"] = float(y.mean())
    return results, fitted[best_name]


def main() -> None:
    raw_df = pd.read_csv(RAW_DATA_PATH)
    df = create_processed_dataset(raw_df)
    target_metadata = {
        "target_definition": (
            "1 when customer belongs to the highest-risk RFM cluster or has at least "
            "one observed fraud transaction; 0 otherwise."
        ),
        "target_rate": float(df[TARGET_COL].mean()),
    }
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)

    results, best_model = train_models(df)
    artifact = {
        "model": best_model,
        "feature_columns": feature_columns(df),
        "target_metadata": target_metadata,
        "metrics": results,
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)

    metrics_path = MODEL_PATH.parent / "metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    if mlflow is not None:
        try:
            mlflow.set_experiment("Alternative Data Credit Risk PD")
            with mlflow.start_run(run_name="credit-risk-alternative-data"):
                mlflow.log_param("best_model", results["best_model"])
                mlflow.log_param("target_definition", target_metadata["target_definition"])
                best_metrics = results[results["best_model"]]
                for metric_name in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
                    mlflow.log_metric(metric_name, best_metrics[metric_name])
                mlflow.log_artifact(str(metrics_path))
                mlflow.sklearn.log_model(best_model, artifact_path="model")
                model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
                mlflow.register_model(model_uri, "AlternativeDataCreditRiskPD")
        except Exception as exc:
            print(f"MLflow tracking skipped: {exc}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
