from __future__ import annotations

from pathlib import Path
import csv

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "customer_churn_sample.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
METRICS_PATH = PROCESSED_DIR / "churn_model_metrics.csv"
SCORED_PATH = PROCESSED_DIR / "churn_scored_customers.csv"
TARGET = "churned"


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


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


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

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )


def calculate_metrics(y_true: pd.Series, y_pred: pd.Series, y_score: pd.Series) -> list[dict[str, float]]:
    return [
        {"metric": "accuracy", "value": round(float(accuracy_score(y_true, y_pred)), 4)},
        {"metric": "precision", "value": round(float(precision_score(y_true, y_pred, zero_division=0)), 4)},
        {"metric": "recall", "value": round(float(recall_score(y_true, y_pred, zero_division=0)), 4)},
        {"metric": "f1", "value": round(float(f1_score(y_true, y_pred, zero_division=0)), 4)},
        {"metric": "roc_auc", "value": round(float(roc_auc_score(y_true, y_score)), 4)},
    ]


def save_metrics(rows: list[dict[str, float]], path: Path = METRICS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(rows)


def save_scored_customers(frame: pd.DataFrame, path: Path = SCORED_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def main() -> None:
    df = load_dataset()

    missing_columns = [
        column for column in NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET, "customer_id"]
        if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(f"Dataset missing required columns: {', '.join(missing_columns)}")

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X,
        y,
        df["customer_id"],
        test_size=0.3,
        random_state=42,
        stratify=y,
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_score = pipeline.predict_proba(X_test)[:, 1]

    metrics = calculate_metrics(y_test, y_pred, y_score)
    save_metrics(metrics)

    scored = X_test.copy()
    scored.insert(0, "customer_id", id_test.values)
    scored["actual_churned"] = y_test.values
    scored["predicted_churned"] = y_pred
    scored["churn_probability"] = y_score.round(4)
    scored = scored.sort_values("churn_probability", ascending=False)
    save_scored_customers(scored)

    print(f"Saved model metrics to {METRICS_PATH}")
    print(f"Saved scored customers to {SCORED_PATH}")


if __name__ == "__main__":
    main()
