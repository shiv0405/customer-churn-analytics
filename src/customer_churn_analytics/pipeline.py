from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
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

from .config import ProjectPaths


TARGET = "churned"
ID_COLUMN = "customer_id"
NUMERIC_FEATURES = [
    "age",
    "tenure_months",
    "monthly_charges",
    "avg_monthly_usage_gb",
    "support_tickets_90d",
    "payment_failures_90d",
    "service_incidents_30d",
    "late_deliveries_90d",
    "discount_pct",
    "engagement_score",
    "nps_score",
    "days_since_last_login",
    "cross_sell_products",
]
BOOLEAN_FEATURES = [
    "autopay_enabled",
    "paperless_billing",
    "premium_support",
    "streaming_bundle",
]
CATEGORICAL_FEATURES = [
    "region",
    "contract_type",
    "payment_method",
    "internet_tier",
]
FEATURE_COLUMNS = [*NUMERIC_FEATURES, *BOOLEAN_FEATURES, *CATEGORICAL_FEATURES]
REQUIRED_COLUMNS = [ID_COLUMN, *FEATURE_COLUMNS, TARGET]


@dataclass
class ModelSelectionResult:
    model_name: str
    threshold: float
    validation_metrics: dict[str, float]


@dataclass
class TrainingRunResult:
    model_name: str
    threshold: float
    metrics: dict[str, float]
    dataset_rows: int
    churn_rate: float
    artifacts: dict[str, str]


def load_dataset(paths: ProjectPaths) -> pd.DataFrame:
    if not paths.raw_data_path.exists():
        raise FileNotFoundError(f"Expected dataset at {paths.raw_data_path}")
    return pd.read_csv(paths.raw_data_path)


def validate_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in REQUIRED_COLUMNS if column not in dataset.columns]
    if missing:
        raise ValueError("Dataset is missing required columns: " + ", ".join(missing))

    validated = dataset.copy()
    validated[TARGET] = pd.to_numeric(validated[TARGET], errors="coerce")
    if validated[TARGET].isna().any():
        raise ValueError("Target column must contain only 0/1 values.")

    target_values = set(validated[TARGET].astype(int).unique())
    if not target_values.issubset({0, 1}) or len(target_values) < 2:
        raise ValueError("Target column must contain both classes 0 and 1.")

    return validated


def build_preprocessor() -> ColumnTransformer:
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    boolean_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", encoder),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("boolean", boolean_pipeline, BOOLEAN_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        sparse_threshold=0.0,
    )


def build_candidate_models() -> dict[str, Pipeline]:
    preprocessor = build_preprocessor()
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", LogisticRegression(max_iter=1500, class_weight="balanced", C=0.8)),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=360,
                        max_depth=12,
                        min_samples_leaf=8,
                        class_weight="balanced_subsample",
                        random_state=42,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
    }


def run_training_pipeline(paths: ProjectPaths | None = None) -> TrainingRunResult:
    resolved_paths = paths or ProjectPaths.from_root()
    resolved_paths.ensure_directories()
    dataset = validate_dataset(load_dataset(resolved_paths))

    X = dataset[FEATURE_COLUMNS]
    y = dataset[TARGET].astype(int)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.36,
        random_state=42,
        stratify=y,
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.56,
        random_state=42,
        stratify=y_temp,
    )

    selections: list[ModelSelectionResult] = []
    for model_name, pipeline in build_candidate_models().items():
        pipeline.fit(X_train, y_train)
        valid_probabilities = pipeline.predict_proba(X_valid)[:, 1]
        threshold = select_threshold(y_valid, valid_probabilities)
        metrics = evaluate_predictions(y_valid, valid_probabilities, threshold)
        selections.append(
            ModelSelectionResult(
                model_name=model_name,
                threshold=threshold,
                validation_metrics=metrics,
            )
        )

    leaderboard = pd.DataFrame(
        [
            {
                "model_name": item.model_name,
                "threshold": item.threshold,
                **item.validation_metrics,
                "selection_score": selection_score(item.validation_metrics),
            }
            for item in selections
        ]
    ).sort_values(["selection_score", "roc_auc", "average_precision"], ascending=False)
    champion_name = str(leaderboard.iloc[0]["model_name"])
    champion_threshold = float(leaderboard.iloc[0]["threshold"])

    champion = build_candidate_models()[champion_name]
    X_train_full = pd.concat([X_train, X_valid], axis=0)
    y_train_full = pd.concat([y_train, y_valid], axis=0)
    champion.fit(X_train_full, y_train_full)

    test_probabilities = champion.predict_proba(X_test)[:, 1]
    test_metrics = evaluate_predictions(y_test, test_probabilities, champion_threshold)
    feature_importance = extract_feature_importance(champion)
    scored_customers = score_customers(champion, dataset, champion_threshold)
    segment_risk = build_segment_risk_table(scored_customers)

    leaderboard.to_csv(resolved_paths.leaderboard_path, index=False)
    feature_importance.to_csv(resolved_paths.feature_importance_path, index=False)
    scored_customers.to_csv(resolved_paths.scored_output_path, index=False)
    resolved_paths.metrics_path.write_text(
        json.dumps(
            {
                "model_name": champion_name,
                "threshold": round(champion_threshold, 4),
                "dataset_rows": int(len(dataset)),
                "churn_rate": round(float(dataset[TARGET].mean()), 4),
                "test_metrics": test_metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    resolved_paths.summary_markdown_path.write_text(
        build_summary_markdown(champion_name, champion_threshold, test_metrics, feature_importance, segment_risk),
        encoding="utf-8",
    )
    resolved_paths.executive_html_path.write_text(
        build_executive_html(champion_name, champion_threshold, test_metrics, feature_importance, segment_risk),
        encoding="utf-8",
    )
    with resolved_paths.model_path.open("wb") as handle:
        pickle.dump(champion, handle)

    return TrainingRunResult(
        model_name=champion_name,
        threshold=round(champion_threshold, 4),
        metrics=test_metrics,
        dataset_rows=int(len(dataset)),
        churn_rate=round(float(dataset[TARGET].mean()), 4),
        artifacts={
            "model_path": str(resolved_paths.model_path),
            "metrics_path": str(resolved_paths.metrics_path),
            "leaderboard_path": str(resolved_paths.leaderboard_path),
            "feature_importance_path": str(resolved_paths.feature_importance_path),
            "summary_markdown_path": str(resolved_paths.summary_markdown_path),
            "executive_html_path": str(resolved_paths.executive_html_path),
            "scored_output_path": str(resolved_paths.scored_output_path),
        },
    )


def select_threshold(y_true: pd.Series, probabilities: pd.Series) -> float:
    best_threshold = 0.5
    best_score = -1.0
    for candidate in [round(value, 2) for value in list(pd.Series(range(25, 76)).div(100))]:
        predictions = (probabilities >= candidate).astype(int)
        precision = precision_score(y_true, predictions, zero_division=0)
        recall = recall_score(y_true, predictions, zero_division=0)
        f1 = f1_score(y_true, predictions, zero_division=0)
        score = f1 + recall * 0.15 + (0.08 if precision >= 0.58 else 0.0)
        if score > best_score:
            best_score = score
            best_threshold = candidate
    return best_threshold


def evaluate_predictions(y_true: pd.Series, probabilities: pd.Series, threshold: float) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(int)
    return {
        "accuracy": round(float(accuracy_score(y_true, predictions)), 4),
        "precision": round(float(precision_score(y_true, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, predictions, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, probabilities)), 4),
        "average_precision": round(float(average_precision_score(y_true, probabilities)), 4),
    }


def selection_score(metrics: dict[str, float]) -> float:
    return round(metrics["roc_auc"] * 0.55 + metrics["average_precision"] * 0.30 + metrics["f1"] * 0.15, 6)


def extract_feature_importance(pipeline: Pipeline) -> pd.DataFrame:
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    feature_names = preprocessor.get_feature_names_out()

    if hasattr(classifier, "feature_importances_"):
        raw_values = classifier.feature_importances_
    else:
        raw_values = classifier.coef_[0]

    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": pd.Series(raw_values).abs().round(6),
        }
    ).sort_values("importance", ascending=False)
    return importance.head(25).reset_index(drop=True)


def score_customers(pipeline: Pipeline, dataset: pd.DataFrame, threshold: float) -> pd.DataFrame:
    scores = dataset[[ID_COLUMN, "region", "contract_type", "monthly_charges", TARGET]].copy()
    scores["churn_probability"] = pipeline.predict_proba(dataset[FEATURE_COLUMNS])[:, 1]
    scores["predicted_churn"] = (scores["churn_probability"] >= threshold).astype(int)
    scores["risk_band"] = pd.cut(
        scores["churn_probability"],
        bins=[-0.01, 0.30, 0.55, 0.75, 1.0],
        labels=["low", "watch", "elevated", "critical"],
    )
    return scores.sort_values("churn_probability", ascending=False).reset_index(drop=True)


def build_segment_risk_table(scored_customers: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        scored_customers.groupby(["region", "contract_type"], observed=False)
        .agg(
            customers=("customer_id", "count"),
            average_probability=("churn_probability", "mean"),
            critical_accounts=("risk_band", lambda series: int((series == "critical").sum())),
        )
        .reset_index()
        .sort_values(["average_probability", "critical_accounts"], ascending=False)
    )
    grouped["average_probability"] = grouped["average_probability"].round(4)
    return grouped.head(10)


def build_summary_markdown(
    model_name: str,
    threshold: float,
    metrics: dict[str, float],
    feature_importance: pd.DataFrame,
    segment_risk: pd.DataFrame,
) -> str:
    top_features = "\n".join(f"- `{row.feature}`: {row.importance}" for row in feature_importance.head(10).itertuples())
    top_segments = "\n".join(
        f"- `{row.region} / {row.contract_type}`: probability {row.average_probability}, critical accounts {row.critical_accounts}"
        for row in segment_risk.itertuples()
    )
    metric_lines = "\n".join(f"- `{name}`: {value}" for name, value in metrics.items())
    return "\n".join(
        [
            "# Model Summary",
            "",
            f"- Champion model: `{model_name}`",
            f"- Operating threshold: `{threshold}`",
            "",
            "## Test Metrics",
            "",
            metric_lines,
            "",
            "## Top Features",
            "",
            top_features,
            "",
            "## Highest-Risk Segments",
            "",
            top_segments,
        ]
    )


def build_executive_html(
    model_name: str,
    threshold: float,
    metrics: dict[str, float],
    feature_importance: pd.DataFrame,
    segment_risk: pd.DataFrame,
) -> str:
    metric_cards = "".join(
        f"<div class='card'><span>{label.replace('_', ' ').title()}</span><strong>{value}</strong></div>"
        for label, value in metrics.items()
    )
    feature_rows = "".join(
        f"<tr><td>{row.feature}</td><td>{row.importance}</td></tr>" for row in feature_importance.head(12).itertuples()
    )
    segment_rows = "".join(
        f"<tr><td>{row.region}</td><td>{row.contract_type}</td><td>{row.average_probability}</td><td>{row.critical_accounts}</td></tr>"
        for row in segment_risk.itertuples()
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Customer Churn Executive Summary</title>
  <style>
    body {{ font-family: "Segoe UI", sans-serif; margin: 0; padding: 28px; background: #f5efe7; color: #1c2733; }}
    .hero, .panel {{ background: #fffdf9; border: 1px solid #ebdfd3; border-radius: 24px; padding: 24px; margin-bottom: 18px; }}
    h1, h2 {{ margin-top: 0; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }}
    .card {{ background: #f8f1e7; border-radius: 18px; padding: 16px; }}
    .card span {{ display: block; color: #627080; font-size: 12px; text-transform: uppercase; margin-bottom: 6px; }}
    .card strong {{ font-size: 28px; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ text-align: left; padding: 10px; border-bottom: 1px solid #eee5da; }}
    .kicker {{ text-transform: uppercase; font-size: 12px; letter-spacing: 0.12em; color: #9b6331; font-weight: 700; }}
  </style>
</head>
<body>
  <section class="hero">
    <div class="kicker">Customer Retention Intelligence</div>
    <h1>Churn Analytics Executive Summary</h1>
    <p>Champion model: <strong>{model_name}</strong>. Operating threshold: <strong>{threshold}</strong>.</p>
    <div class="cards">{metric_cards}</div>
  </section>
  <section class="panel">
    <h2>Top Drivers</h2>
    <table>
      <thead><tr><th>Feature</th><th>Importance</th></tr></thead>
      <tbody>{feature_rows}</tbody>
    </table>
  </section>
  <section class="panel">
    <h2>Highest-Risk Segments</h2>
    <table>
      <thead><tr><th>Region</th><th>Contract</th><th>Average Churn Probability</th><th>Critical Accounts</th></tr></thead>
      <tbody>{segment_rows}</tbody>
    </table>
  </section>
</body>
</html>"""
