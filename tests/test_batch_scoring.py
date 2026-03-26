from __future__ import annotations

from pathlib import Path

import pandas as pd

from customer_churn_analytics.config import ProjectPaths
from customer_churn_analytics.data_generation import generate_customer_dataset, write_dataset
from customer_churn_analytics.inference import score_batch_file
from customer_churn_analytics.pipeline import run_training_pipeline


def test_batch_scoring_generates_intervention_fields(tmp_path: Path) -> None:
    paths = ProjectPaths.from_root(tmp_path)
    dataset = generate_customer_dataset(rows=2200, seed=13)
    write_dataset(dataset, paths.raw_data_path)
    result = run_training_pipeline(paths)

    scoring_input = dataset.drop(columns=["churned", "churn_probability_signal"]).head(25)
    input_path = tmp_path / "data" / "raw" / "manual_scoring_input.csv"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    scoring_input.to_csv(input_path, index=False)

    destination = score_batch_file(paths=paths, input_path=input_path, threshold=result.threshold)
    scored = pd.read_csv(destination)

    assert "retention_priority" in scored.columns
    assert "risk_summary" in scored.columns
    assert "recommended_action" in scored.columns
