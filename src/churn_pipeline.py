from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "customer_churn_sample.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
METRICS_PATH = PROCESSED_DIR / "churn_model_metrics.csv"
SCORED_PATH = PROCESSED_DIR / "churn_scored_customers.csv"
TARGET = "churned"
ID_COLUMN = "customer_id"

NUMERIC_FEATURES = [
    "tenure_months",
    "monthly_charges",
    "support_tickets_90d",
    "payment_failures_90d",
    "usage_score",
]

CATEGORICAL_FEATURES = [
    "contract_type",
    "region",
]

REQUIRED_COLUMNS = [ID_COLUMN, *NUMERIC_FEATURES, *CATEGORICAL_FEATURES, TARGET]


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Add a churn dataset under data/raw/ before running the pipeline."
        )
    return pd.read_csv(path)


def validate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: " + ", ".join(missing_columns)
        )

    validated = df.copy()
    validated[TARGET] = pd.to_numeric(validated[TARGET], errors="coerce")

    if validated[TARGET].isna().any():
        raise ValueError("Target column 'churned' must contain only numeric 0/1 values.")

    unique_target_values = set(validated[TARGET].astype(int).unique())
    if not unique_target_values.issubset({0, 1}) or len(unique_target_values) < 2:
        raise ValueError(
            "Target column 'churned' must contain both classes 0 and 1 for model training."
        )

    return validated


def build_pipeline() -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    model = LogisticRegression(max_iter=1000, class_weight="balanced")

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def evaluate_model(y_true: pd.Series, y_pred: pd.Series, y_prob: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"metric": "accuracy", "value": round(float(accuracy_score(y_true, y_pred)), 4)},
            {"metric": "precision", "value": round(float(precision_score(y_true, y_pred, zero_division=0)), 4)},
            {"metric": "recall", "value": round(float(recall_score(y_true, y_pred, zero_division=0)), 4)},
            {"metric": "f1", "value": round(float(f1_score(y_true, y_pred, zero_division=0)), 4)},
            {"metric": "roc_auc", "value": round(float(roc_auc_score(y_true, y_prob)), 4)},
            {
                "metric": "average_precision",
                "value": round(float(average_precision_score(y_true, y_prob)), 4),
            },
            {"metric": "test_rows", "value": int(len(y_true))},
            {"metric": "positive_rate_test", "value": round(float(pd.Series(y_true).mean()), 4)},
        ]
    )


def score_dataset(model: Pipeline, df: pd.DataFrame) -> pd.DataFrame:
    features = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    scored = df[[ID_COLUMN]].copy() if ID_COLUMN in df.columns else pd.DataFrame(index=df.index)
    scored["churn_probability"] = model.predict_proba(features)[:, 1]
    scored["predicted_churn"] = (scored["churn_probability"] >= 0.5).astype(int)
    scored["risk_band"] = pd.cut(
        scored["churn_probability"],
        bins=[-0.01, 0.3, 0.7, 1.0],
        labels=["low", "medium", "high"],
    )
    return scored.sort_values("churn_probability", ascending=False)


def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = validate_dataset(load_dataset())

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    test_probabilities = pipeline.predict_proba(X_test)[:, 1]
    test_predictions = (test_probabilities >= 0.5).astype(int)

    metrics = evaluate_model(y_test, test_predictions, test_probabilities)
    scored = score_dataset(pipeline, df)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(METRICS_PATH, index=False)
    scored.to_csv(SCORED_PATH, index=False)

    return metrics, scored


if __name__ == "__main__":
    metrics_df, scored_df = run()
    print(f"Saved metrics to {METRICS_PATH}")
    print(metrics_df.to_string(index=False))
    print(f"Saved scored customers to {SCORED_PATH} ({len(scored_df)} rows)")
